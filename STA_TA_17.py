import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data.dataloader import DataLoader
import torch.backends.cudnn as cudnn
from scipy.stats import pearsonr
from torchvision import models
import itertools

from net.sinc_ta_net import AudioNet
from net.net_x64_wgan import Generator
from net.emotion_net import EmotionNet
from net.fusion_net import UnalignedTransformer
from dataset import AVEC17_MM_Pack_Dataset, AVEC17_MM_Val_Pack_Dataset
from utils import read_conf, mult_weights_init
from path import root_path, model_root_path


if __name__ == '__main__':

    cudnn.benchmark = True

    EPOCHS = 1000
    BATCHSIZE = 8
    TLEN = 1024  # number of temporal frames
    DATASET = 'avec17'
    LOG_STEP = 13
    STAGE = 'STA_TA_17'
    best_mae = 100
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    VAL_STEP = 1
    CUDA_NUM = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mkdir_list = ['{}/weights/{}'.format(model_root_path, STAGE)]
    for path in mkdir_list:
        if not os.path.exists(path):
            os.makedirs(path)

    options = read_conf()

    df = pd.read_csv('dataset/{}/label.csv'.format(DATASET))
    landmark_path_list = df['audio'].values
    score_list = df['label'].values
    # total: 189 --- train-107 dev-35 test-47
    train_landmark_path_list = landmark_path_list[:107]
    train_score_list = score_list[:107]
    val_landmark_path_list = landmark_path_list[107:142]
    val_score_list = score_list[107:142]
    test_landmark_path_list = landmark_path_list[142:]
    test_score_list = score_list[142:]

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

    train_dataset = AVEC17_MM_Pack_Dataset(data_list=train_landmark_path_list,
                                           label_list=train_score_list,
                                           wlen=wlen,
                                           t_len=TLEN)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCHSIZE,
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=True,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)
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
    test_dataset = AVEC17_MM_Val_Pack_Dataset(data_list=test_landmark_path_list,
                                              label_list=test_score_list,
                                              wlen=wlen,
                                              t_len=TLEN)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=8,
                                 pin_memory=True,
                                 drop_last=False)

    model_G = Generator()
    model_G.load_state_dict(torch.load('{}/weights/stage_2/model_G.pth'.format(model_root_path),
                                       map_location='cuda:{}'.format(CUDA_NUM)))
    model_G = torch.nn.DataParallel(model_G)
    model_G = model_G.to(device)

    model_emo_D = EmotionNet()
    model_emo_D.load_state_dict(torch.load('{}/weights/stage_2/model_emo_D.pth'.format(model_root_path),
                                           map_location='cuda:{}'.format(CUDA_NUM)))
    model_emo_D = torch.nn.DataParallel(model_emo_D)
    model_emo_D = model_emo_D.to(device)
    
    model_audio = AudioNet(options)
    model_audio = torch.nn.DataParallel(model_audio)
    model_audio = model_audio.to(device)

    fusion_net = UnalignedTransformer(dataset=DATASET, feature_size_a=audio_feature_len,
                                      feature_size_v=visual_feature_len)
    fusion_net = torch.nn.DataParallel(fusion_net)
    fusion_net = fusion_net.to(device)
    
    optimizer = Adam(itertools.chain(model_G.parameters(),
                                     model_emo_D.parameters(),
                                     model_audio.parameters(),
                                     fusion_net.parameters()), lr=0.00001)

    mse_loss_func = nn.MSELoss()
    mae_loss_func = nn.L1Loss()

    for epoch in range(EPOCHS):
        RMSE_loss = []
        MAE_loss = []
        model_emo_D.train()
        model_audio.train()
        model_G.train()
        fusion_net.train()
        for step, (train_data, train_audio, train_label) in enumerate(train_dataloader):
            train_data = train_data.type(torch.FloatTensor).to(device)
            train_label = train_label.type(torch.FloatTensor).to(device).squeeze()
            train_audio = train_audio.type(torch.FloatTensor).to(device)

            # train_data: torch.Size([BATCH, TLEN, 136, 1, 1])
            train_data = train_data.view(-1, 136, 1, 1)  # torch.Size([TLEN*BATCH, 136, 1, 1])
            pred_v = model_G(train_data)  # torch.Size([TLEN*BATCH, 3, 64, 64])
            pred_v = pred_v.view(-1, TLEN, 3, 64, 64)
            pred_v = model_emo_D(pred_v).squeeze(4).squeeze(3)  # torch.Size([BATCH, TLEN, 512])

            # train_audio: torch.Size([BATCH, TLEN, wlen(1600)])
            train_audio = train_audio.view(-1, wlen)  # torch.Size([BATCH*TLEN, wlen])
            pred_a = model_audio(train_audio)  # torch.Size([BATCH*TLEN, audio_feature_len])
            pred_a = pred_a.view(-1, TLEN, audio_feature_len)  # torch.Size([BATCH, TLEN, audio_feature_len])

            pred = fusion_net(pred_a, pred_v)  # torch.Size([])

            loss = mse_loss_func(pred, train_label)
            loss.backward()
            
            RMSE_loss.append(mse_loss_func(pred, train_label).item())
            MAE_loss.append(mae_loss_func(pred, train_label).item())
            mean_mae_loss = np.mean(MAE_loss)
            mean_rmse_loss = np.sqrt(np.mean(RMSE_loss))

            if (step + 1) % LOG_STEP == 0:
                print('Epoch: {:d}  Step: {:d} / {:d} | train MAE loss: {:.4f} | train RMSE loss: {:.4f}'.format(
                       epoch, step + 1, len(train_dataloader), mean_mae_loss, mean_rmse_loss))
                print('label: {}\npredict: {}'.format(train_label, pred))
                # print([x.grad for x in optimizer_2.param_groups[0]['params']])
                optimizer.step()
                optimizer.zero_grad()

        if (epoch+1) % VAL_STEP == 0:
            model_emo_D.eval()
            model_G.eval()
            fusion_net.eval()
            model_audio.eval()
            RMSE_loss = []
            MAE_loss = []
            f_list = []
            y_list = []
            with torch.no_grad():
                for step, (val_data_pack, val_audio_pack, val_label) in enumerate(val_dataloader):
                    val_label = val_label.type(torch.FloatTensor).squeeze()
                    predict_list = []
                    for i in range(len(val_data_pack)):
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

                        predict_list.append(pred.item())

                    predict = torch.tensor(np.mean(predict_list))

                    RMSE_loss.append(mse_loss_func(predict, val_label))
                    MAE_loss.append(mae_loss_func(predict, val_label))
                    if (step + 1) % 10 == 0:
                        print('Step: {:d} / {:d} | val label: {:.4f} | val predict: {:.4f}'.format(
                            step + 1, len(val_dataloader), val_label, predict))
                    f_list.append(predict)
                    y_list.append(val_label)

                mean_rmse_loss = np.sqrt(np.mean(RMSE_loss))
                mean_mae_loss = np.mean(MAE_loss)
                print('val MAE loss: {:.4f}    val RMSE loss: {:.4f}'.format(mean_mae_loss, mean_rmse_loss))

                f_mean = np.mean(f_list)
                f_std = np.std(f_list, ddof=1)
                y_mean = np.mean(y_list)
                y_std = np.std(y_list, ddof=1)
                PCC = pearsonr(f_list, y_list)[0]
                CCC = (2 * PCC * f_std * y_std) / (f_std ** 2 + y_std ** 2 + (f_mean - y_mean) ** 2)
                print('PCC: %.4f' % PCC)
                print('CCC: %.4f' % CCC)

                if mean_mae_loss < best_mae:
                    best_mae = mean_mae_loss
                    torch.save(model_emo_D.state_dict(), "{}/weights/{}/model_emo_D.pth".format(model_root_path, STAGE))
                    torch.save(model_G.state_dict(), "{}/weights/{}/model_G.pth".format(model_root_path, STAGE))
                    torch.save(fusion_net.state_dict(), "{}/weights/{}/fusion_net.pth".format(model_root_path, STAGE))
                    torch.save(model_audio.state_dict(), "{}/weights/{}/model_audio.pth".format(model_root_path, STAGE))
                    print('Current MAE: {:.4f}, model saved!'.format(mean_mae_loss))

                    for step, (val_data_pack, val_audio_pack, val_label) in enumerate(test_dataloader):
                        val_label = val_label.type(torch.FloatTensor).squeeze()
                        predict_list = []
                        for i in range(len(val_data_pack)):
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
                            pred_a = pred_a.view(-1, TLEN,
                                                 audio_feature_len)  # torch.Size([BATCH, TLEN, audio_feature_len])

                            pred = fusion_net(pred_a, pred_v)  # torch.Size([])

                            predict_list.append(pred.item())

                        predict = torch.tensor(np.mean(predict_list))

                        RMSE_loss.append(mse_loss_func(predict, val_label))
                        MAE_loss.append(mae_loss_func(predict, val_label))
                        if (step + 1) % 10 == 0:
                            print('Step: {:d} / {:d} | val label: {:.4f} | val predict: {:.4f}'.format(
                                step + 1, len(val_dataloader), val_label, predict))
                        f_list.append(predict)
                        y_list.append(val_label)

                    mean_rmse_loss = np.sqrt(np.mean(RMSE_loss))
                    mean_mae_loss = np.mean(MAE_loss)
                    print('test MAE loss: {:.4f}    test RMSE loss: {:.4f}'.format(mean_mae_loss, mean_rmse_loss))

                    f_mean = np.mean(f_list)
                    f_std = np.std(f_list, ddof=1)
                    y_mean = np.mean(y_list)
                    y_std = np.std(y_list, ddof=1)
                    PCC = pearsonr(f_list, y_list)[0]
                    CCC = (2 * PCC * f_std * y_std) / (f_std ** 2 + y_std ** 2 + (f_mean - y_mean) ** 2)
                    print('PCC: %.4f' % PCC)
                    print('CCC: %.4f' % CCC)
