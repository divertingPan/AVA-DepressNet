import torch
from torch import nn
import torch.nn.functional as F

from net.modules.transformer import TransformerEncoder


class MULTModel(nn.Module):
    def __init__(self, feature_size_a, feature_size_v):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.orig_d_a, self.orig_d_v = feature_size_a, feature_size_v
        self.d_a, self.d_v = 200, 200
        self.num_heads = 5
        self.layers = 8
        self.attn_dropout = 0
        self.attn_dropout_a = 0
        self.attn_dropout_v = 0
        self.relu_dropout = 0
        self.res_dropout = 0
        self.out_dropout = 0
        self.embed_dropout = 0
        self.attn_mask = False

        combined_dim = self.d_a + self.d_v

        output_dim = 1        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_a = self.get_network(self_type='va')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
            
    def forward(self, x_a, x_v):
        """
        audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_a = x_a.transpose(1, 2)  # [1, 1600, seq_len]
        x_v = x_v.transpose(1, 2)  # [1, 512, seq_len]
       
        # Project the textual/visual/audio features
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)  # [1, 30, seq_len]
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)  # [1, 30, seq_len]
        proj_x_a = proj_x_a.permute(2, 0, 1)  # [seq_len, 1, 30]
        proj_x_v = proj_x_v.permute(2, 0, 1)  # [seq_len, 1, 30]

        # A --> V
        h_vs = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)  # [seq_len, 1, 30]
        h_vs = self.trans_v_mem(h_vs)  # [seq_len, 1, 30]
        if type(h_vs) == tuple:
            h_vs = h_vs[0]

        # V --> A
        h_as = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)  # [seq_len, 1, 30]
        h_as = self.trans_a_mem(h_as)  # [seq_len, 1, 30]
        if type(h_as) == tuple:
            h_as = h_as[0]

        last_hs = torch.cat([h_as, h_vs], dim=2)[-1]  # [1, 60]
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)  # [1, 1]

        return output, last_hs


if __name__ == '__main__':
    a = torch.rand((1, 16, 1600))  # x: [1, seq_len, audio_feature_len]
    v = torch.rand((1, 16, 512))

    net = MULTModel(1600, 512)
    output, last_hs = net(a, v)
    print('output: {}, lash_hs: {}'.format(output, last_hs))
