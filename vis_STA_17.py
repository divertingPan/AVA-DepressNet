import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.backends.cudnn as cudnn

from net.sinc_ta_net import AudioNet
from net.net_x64_wgan import Generator
from net.emotion_net import EmotionNet
from net.fusion_net import UnalignedTransformer
from dataset import AVEC17_MM_Pack_Dataset, AVEC17_MM_Val_Pack_Dataset
from utils import read_conf, mult_weights_init
from path import root_path, model_root_path
torch.set_printoptions(profile="full")

if __name__ == '__main__':

    cudnn.benchmark = True
    TLEN = 1024  # number of temporal frames
    DATASET = 'avec17'
    STAGE = 'exp1_STA_TA_17'
    CUDA_NUM = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:{}'.format(CUDA_NUM) if torch.cuda.is_available() else 'cpu')

    mkdir_list = ['{}/weights/{}'.format(model_root_path, STAGE)]
    for path in mkdir_list:
        if not os.path.exists(path):
            os.makedirs(path)

    options = read_conf()

    ''' AVEC 17 '''
    df = pd.read_csv('dataset/{}/label.csv'.format(DATASET))
    landmark_path_list = df['audio'].values
    score_list = df['label'].values
    # total: 189 --- train-107 dev-35 test-47
    train_size = 107
    val_landmark_path_list = landmark_path_list[107:108]
    val_score_list = score_list[107:108]

    visual_feature_len = 512  # output size of latent visual feature
    fs = int(options.fs)
    cw_len = int(options.cw_len)
    cw_shift = int(options.cw_shift)
    wlen = int(fs * cw_len / 1000.00)
    wshift = int(fs * cw_shift / 1000.00)
    # audio_feature_len = int(options.class_lay)
    audio_feature_len = 2880

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)

    val_dataset = AVEC17_MM_Val_Pack_Dataset(data_list=val_landmark_path_list,
                                             label_list=val_score_list,
                                             wlen=wlen,
                                             t_len=TLEN)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True,
                                drop_last=False)

    model_G = Generator()
    model_G = torch.nn.DataParallel(model_G)
    model_G = model_G.to(device)
    model_G.load_state_dict(torch.load("{}/weights/exp1_STA_TA_17/model_G.pth".format(model_root_path),
                                       map_location='cuda:{}'.format(CUDA_NUM)))

    model_emo_D = EmotionNet()
    model_emo_D = torch.nn.DataParallel(model_emo_D)
    model_emo_D = model_emo_D.to(device)
    model_emo_D.load_state_dict(torch.load("{}/weights/exp1_STA_TA_17/model_emo_D.pth".format(model_root_path),
                                           map_location='cuda:{}'.format(CUDA_NUM)))

    model_audio = AudioNet(options)
    model_audio = torch.nn.DataParallel(model_audio)
    model_audio = model_audio.to(device)
    model_audio.load_state_dict(torch.load("{}/weights/exp1_STA_TA_17/model_audio.pth".format(model_root_path),
                                           map_location='cuda:{}'.format(CUDA_NUM)))

    fusion_net = UnalignedTransformer(dataset=DATASET, feature_size_a=audio_feature_len,
                                      feature_size_v=visual_feature_len)
    fusion_net = torch.nn.DataParallel(fusion_net)
    fusion_net = fusion_net.to(device)
    fusion_net.load_state_dict(torch.load("{}/weights/exp1_STA_TA_17/fusion_net.pth".format(model_root_path),
                                          map_location='cuda:{}'.format(CUDA_NUM)))

    mse_loss_func = nn.MSELoss()
    mae_loss_func = nn.L1Loss()

    print('======================== VAL AVEC 2017 ========================')
    model_emo_D.eval()
    model_G.eval()
    fusion_net.eval()
    model_audio.eval()

    with torch.no_grad():
        for step, (val_data_pack, val_audio_pack, val_label) in enumerate(val_dataloader):
            val_label = val_label.type(torch.FloatTensor).squeeze()

            i = 5
            val_data = val_data_pack[i]
            val_audio = val_audio_pack[i]
            val_data = val_data.type(torch.FloatTensor).to(device)
            val_audio = val_audio.type(torch.FloatTensor).to(device)

            val_data = val_data.view(-1, 136, 1, 1)  # torch.Size([TLEN*BATCH, 136, 1, 1])
            pred_v = model_G(val_data)  # torch.Size([TLEN*BATCH, 3, 64, 64])
            pred_v = pred_v.view(-1, TLEN, 3, 64, 64)
            pred_v = model_emo_D(pred_v).squeeze(4).squeeze(3)  # torch.Size([BATCH, TLEN, 512])

            val_audio = val_audio.view(-1, wlen)  # torch.Size([BATCH*TLEN, wlen])
            pred_a = model_audio(val_audio)  # torch.Size([BATCH*TLEN, audio_feature_len])
            pred_a = pred_a.view(-1, TLEN, audio_feature_len)  # torch.Size([BATCH, TLEN, audio_feature_len])

            pred = fusion_net(pred_a, pred_v)  # torch.Size([])

            print('Step: {:d} / {:d} | val label: {:.4f} | val predict: {:.4f}'.format(step + 1, len(val_dataloader), val_label, pred))

