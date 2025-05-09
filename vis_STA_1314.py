import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from net.sinc_ta_net import AudioNet
from net.net_x64_wgan import Generator
from net.emotion_net import EmotionNet
from net.fusion_net import UnalignedTransformer
from dataset import AVEC14_MM_Val_Pack_Dataset
from utils import read_conf
from path import root_path, model_root_path
torch.set_printoptions(profile="full")
import multiprocessing



def show_spatial(att_s, val_data_list, step, i):
    for feature in range(128):
        att_s_show = att_s[0,0,feature,0,:,:]   # [batch, 1, 128, 1, H, W]
        save_path_s = "./results/vis/{}/spatial/clip_{}".format(val_data_list[step], i)
        plt.matshow(att_s_show, cmap=plt.get_cmap('Greens'), alpha=0.5)  # , alpha=0.3
        if not os.path.exists(save_path_s):
            os.makedirs(save_path_s)
        plt.savefig("{}/feature_{}.png".format(save_path_s, feature))
        plt.close()


def show_temporal(att_t, val_data_list, step, i):
    for feature in range(128):
        att_t_show = att_t[0, 0, feature, :, 0, 0]  # [batch, 1, 128, T, 1, 1]
        att_t_show = att_t_show.reshape([32, 32])
        save_path_t = "./results/vis/{}/temporal/clip_{}".format(val_data_list[step], i)
        plt.matshow(att_t_show, cmap=plt.get_cmap('Greens'), alpha=0.5)  # , alpha=0.3
        if not os.path.exists(save_path_t):
            os.makedirs(save_path_t)
        plt.savefig("{}/feature_{}.png".format(save_path_t, feature))
        plt.close()


if __name__ == '__main__':

    cudnn.benchmark = True

    TLEN = 1024  # number of temporal frames
    DATASET = 'avec14'
    STAGE = 'exp1_STA_TA_1314'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    CUDA_NUM = 0
    device = torch.device('cuda:{}'.format(CUDA_NUM) if torch.cuda.is_available() else 'cpu')

    options = read_conf()

    ''' AVEC 14 '''
    df = pd.read_csv('{}/Depression_3D/datasets/avec14/label.csv'.format(root_path))
    data_list = df['path'].values
    label_list = df['label'].values
    # total: 300 --- train-100 dev-100 test-100
    val_data_list = data_list[100:200]
    val_label_list = label_list[100:200]

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

    val_dataset = AVEC14_MM_Val_Pack_Dataset(data_list=val_data_list,
                                             label_list=val_label_list,
                                             wlen=wlen,
                                             t_len=TLEN)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=False,
                                drop_last=False)

    model_G = Generator()
    model_G = torch.nn.DataParallel(model_G)
    model_G = model_G.to(device)
    model_G.load_state_dict(torch.load("{}/weights/exp1_STA_TA_1314/model_G.pth".format(model_root_path),
                                       map_location='cuda:{}'.format(CUDA_NUM)))

    model_emo_D = EmotionNet()
    model_emo_D = torch.nn.DataParallel(model_emo_D)
    model_emo_D = model_emo_D.to(device)
    model_emo_D.load_state_dict(torch.load("{}/weights/exp1_STA_TA_1314/model_emo_D.pth".format(model_root_path),
                                           map_location='cuda:{}'.format(CUDA_NUM)))

    model_audio = AudioNet(options)
    model_audio = torch.nn.DataParallel(model_audio)
    model_audio = model_audio.to(device)
    model_audio.load_state_dict(torch.load("{}/weights/exp1_STA_TA_1314/model_audio.pth".format(model_root_path),
                                           map_location='cuda:{}'.format(CUDA_NUM)))

    fusion_net = UnalignedTransformer(dataset=DATASET, feature_size_a=audio_feature_len,
                                      feature_size_v=visual_feature_len)
    fusion_net = torch.nn.DataParallel(fusion_net)
    fusion_net = fusion_net.to(device)
    fusion_net.load_state_dict(torch.load("{}/weights/exp1_STA_TA_1314/fusion_net.pth".format(model_root_path),
                                          map_location='cuda:{}'.format(CUDA_NUM)))

    print('======================== VAL AVEC 2014 ========================')
    model_emo_D.eval()
    model_G.eval()
    fusion_net.eval()
    model_audio.eval()
    with torch.no_grad():
        for step, (val_data_pack, val_audio_pack, val_label) in enumerate(val_dataloader):
            val_label = val_label.type(torch.FloatTensor).squeeze()

            for i in range(len(val_data_pack)):
                val_data = val_data_pack[i]
                val_audio = val_audio_pack[i]
                val_data = val_data.type(torch.FloatTensor).to(device)
                val_audio = val_audio.type(torch.FloatTensor).to(device)

                val_data = val_data.view(-1, 136, 1, 1)  # torch.Size([TLEN*BATCH, 136, 1, 1])
                pred_v = model_G(val_data)  # torch.Size([TLEN*BATCH, 3, 64, 64])
                pred_v = pred_v.view(-1, TLEN, 3, 64, 64)
                pred_v, att_s, att_t = model_emo_D(pred_v)
                pred_v = pred_v.squeeze(4).squeeze(3)  # torch.Size([BATCH, TLEN, 512])

                att_s = att_s.cpu()
                att_t = att_t.cpu()

                p1 = multiprocessing.Process(target=show_spatial, args=(att_s, val_data_list, step, i,))
                p2 = multiprocessing.Process(target=show_temporal, args=(att_t, val_data_list, step, i,))
                p1.start()
                p2.start()
                p1.join()
                p2.join()

                val_audio = val_audio.view(-1, wlen)  # torch.Size([BATCH*TLEN, wlen])
                pred_a = model_audio(val_audio)  # torch.Size([BATCH*TLEN, audio_feature_len])
                pred_a = pred_a.view(-1, TLEN, audio_feature_len)  # torch.Size([BATCH, TLEN, audio_feature_len])

                pred = fusion_net(pred_a, pred_v)  # torch.Size([])

                print('Step: {:d} / {:d} | Sample: {:d} / {:d} '.format(i + 1, len(val_data_pack),
                                                                        step + 1, len(val_dataloader)))
