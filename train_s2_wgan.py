# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim import Adam, RMSprop, SGD
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torchvision import models

from net.net_x64_wgan import Generator, Perceptual_D, Adversarial_D
from net.emotion_net import EmotionNet
from dataset import Stage_II_Dataset
from utils import weight_init, calc_gradient_penalty, save_results, LossCollector
from path import root_path


def train_s2():

    EPOCHS = 400
    BATCH_SIZE = 1024  # temporal length of video clips
    D_ENHANCE = 1
    G_ENHANCE = 1
    DATASET = 'avec14'
    LAMBDA_1 = 1
    LAMBDA_2 = 1
    LAMBDA_3 = 1
    N_ROW = 16
    LOG_STEP = 100
    STAGE = 'stage_2_D1G1_epoch400'
    CUDA_NUM = 0
    best_RMSE = 10000
    cudnn.benchmark = True
    device = torch.device('cuda:{}'.format(CUDA_NUM) if torch.cuda.is_available() else 'cpu')

    mkdir_list = ['results/{}/train'.format(STAGE),
                  'results/{}/val'.format(STAGE),
                  'weights/{}'.format(STAGE)]
    for path in mkdir_list:
        if not os.path.exists(path):
            os.makedirs(path)

    df = pd.read_csv('{}/dataset/avec14/label.csv'.format(root_path))
    image_path_list = df['path'].values
    score_list = df['label'].values

    train_image_path_list = image_path_list[:100]
    train_score_list = score_list[:100]
    val_image_path_list = image_path_list[100:200]
    val_score_list = score_list[100:200]

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = Stage_II_Dataset(dataset=DATASET,
                                     data_path=train_image_path_list,
                                     score=train_score_list,
                                     transform=transform,
                                     rescale=64,
                                     t_len=BATCH_SIZE)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=16,
                                  pin_memory=True,
                                  drop_last=False,
                                  worker_init_fn=worker_init_fn)
    val_dataset = Stage_II_Dataset(dataset=DATASET,
                                   data_path=val_image_path_list,
                                   score=val_score_list,
                                   transform=transform,
                                   rescale=64,
                                   t_len=BATCH_SIZE)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=1,
                                shuffle=True,
                                num_workers=16,
                                pin_memory=True,
                                drop_last=False)

    model_G = Generator().to(device)
    model_G.load_state_dict(torch.load('weights/stage_1_D5G1_epoch50/model_G.pth', map_location='cuda:{}'.format(CUDA_NUM)))
    optimizer_G = Adam(model_G.parameters(), lr=0.0001, betas=(0, 0.9))

    model_per_D = Perceptual_D().to(device)
    model_per_D.load_state_dict(torch.load('weights/stage_1_D5G1_epoch50/model_per_D.pth', map_location='cuda:{}'.format(CUDA_NUM)))
    optimizer_per_D = Adam(model_per_D.parameters(), lr=0.0001, betas=(0, 0.9))

    model_adv_D = Adversarial_D().to(device)
    model_adv_D.load_state_dict(torch.load('weights/stage_1_D5G1_epoch50/model_adv_D.pth', map_location='cuda:{}'.format(CUDA_NUM)))
    optimizer_adv_D = Adam(model_adv_D.parameters(), lr=0.0001, betas=(0, 0.9))

    model_emo_D = EmotionNet().to(device)
    optimizer_emo_D = Adam(model_emo_D.parameters(), lr=0.0001)

    perceptual_loss_func = nn.MSELoss()
    adversarial_loss_func = nn.BCELoss()
    content_loss_func = nn.MSELoss()
    emotional_loss_func = nn.MSELoss()
    mae_loss_func = nn.L1Loss()

    loss_collector = LossCollector(STAGE, ['D_real_adv_loss', 'D_real_per_loss', 'D_real_emo_loss', 'D_real_MAE', 'D_gen_adv_loss', 'D_gen_per_loss', 'D_gen_emo_loss', 'D_gen_MAE',
                                           'D_real_img_flag', 'D_gen_img_flag',
                                           'G_real_adv_loss', 'G_real_per_loss', 'G_real_emo_loss', 'G_real_MAE', 'G_gen_adv_loss', 'G_gen_per_loss', 'G_gen_emo_loss', 'G_gen_MAE', 'G_gen_content_loss',
                                           'G_real_img_flag', 'G_gen_img_flag',
                                           'real_label', 'gen_label',
                                           'val_real_adv_loss', 'val_real_per_loss', 'val_real_emo_loss', 'val_real_MAE', 'val_gen_adv_loss', 'val_gen_per_loss', 'val_gen_emo_loss', 'val_gen_MAE', 'val_gen_content_loss',
                                           'val_real_img_flag', 'val_gen_img_flag'])

    real_label = torch.ones(BATCH_SIZE).to(device)
    fake_label = -torch.ones(BATCH_SIZE).to(device)

    # model_G.train()
    # model_adv_D.train()
    # model_per_D.train()
    model_G.eval()
    model_adv_D.eval()
    model_per_D.eval()
    model_emo_D.train()

    for epoch in range(EPOCHS):
        for train_step, (train_img, train_landmark, train_score) in enumerate(train_dataloader):
            train_img = train_img.type(torch.FloatTensor).to(device)
            train_landmark = train_landmark + torch.randn(BATCH_SIZE, 136, 1, 1) / 256
            train_landmark = train_landmark.type(torch.FloatTensor).to(device)
            train_score = train_score.type(torch.FloatTensor).to(device).unsqueeze(1)
            # print('img: {}  landmark: {}'.format(train_img.shape, train_landmark.shape))
            # >>> img: torch.Size([4, 3, 256, 256])  landmark: torch.Size([4, 136, 1, 1])
            ############################
            # Update D
            ############################

            for D_epoch in range(D_ENHANCE):

                optimizer_per_D.zero_grad()
                optimizer_adv_D.zero_grad()
                optimizer_emo_D.zero_grad()

                real_img_landmark = model_per_D(train_img.squeeze(0))
                real_img_flag = model_adv_D(train_img.squeeze(0))
                real_img_emotion = torch.sigmoid(model_emo_D(train_img))
                real_perceptual_loss = perceptual_loss_func(real_img_landmark, train_landmark)
                real_adversarial_loss = real_img_flag.mean().view(1)  # W-GAN
                real_emotional_loss = emotional_loss_func(real_img_emotion * 63, train_score * 63)
                real_mae_loss = mae_loss_func(real_img_emotion * 63, train_score * 63)

                D_gen_img = model_G(train_landmark.squeeze(0))
                D_gen_img_landmark = model_per_D(D_gen_img.squeeze(0))
                D_gen_img_flag = model_adv_D(D_gen_img.squeeze(0))
                D_gen_img_emotion = torch.sigmoid(model_emo_D(D_gen_img.unsqueeze(0)))
                D_gen_perceptual_loss = perceptual_loss_func(D_gen_img_landmark, train_landmark)
                D_gen_adversarial_loss = D_gen_img_flag.mean().view(1)  # W-GAN
                D_gen_emotional_loss = emotional_loss_func(D_gen_img_emotion * 63, train_score * 63)
                D_gen_mae_loss = mae_loss_func(D_gen_img_emotion * 63, train_score * 63)
                # WGAN-GP
                gradient_penalty = calc_gradient_penalty(model_adv_D, train_img.data.squeeze(0), D_gen_img.data, device)

                '''
                D_loss = lambda_1 * adversarial_loss
                       + lambda_2 * perceptual_loss
                       + lambda_3 * emotional_loss
                '''
                D_loss = - LAMBDA_1 * real_adversarial_loss + LAMBDA_1 * D_gen_adversarial_loss \
                         + LAMBDA_2 * real_perceptual_loss \
                         + LAMBDA_3 * real_emotional_loss + gradient_penalty  # W-GAN
                D_loss.backward()

                optimizer_per_D.step()
                optimizer_adv_D.step()
                optimizer_emo_D.step()

            ############################
            # Update G
            ############################

            for G_epoch in range(G_ENHANCE):
                G_gen_img = model_G(train_landmark.squeeze(0))
                G_gen_img_landmark = model_per_D(G_gen_img.squeeze(0))
                G_gen_img_flag = model_adv_D(G_gen_img.squeeze(0))
                G_gen_img_emotion = torch.sigmoid(model_emo_D(G_gen_img.unsqueeze(0)))
                G_real_img_landmark = model_per_D(train_img.squeeze(0))
                G_real_img_flag = model_adv_D(train_img.squeeze(0))
                G_real_img_emotion = torch.sigmoid(model_emo_D(train_img))
                G_real_perceptual_loss = perceptual_loss_func(G_real_img_landmark, train_landmark)
                G_real_emotional_loss = emotional_loss_func(G_real_img_emotion * 63, train_score * 63)
                G_real_mae_loss = mae_loss_func(G_real_img_emotion * 63, train_score * 63)
                G_real_adversarial_loss = G_real_img_flag.mean().view(1)  # W-GAN
                G_gen_perceptual_loss = perceptual_loss_func(G_gen_img_landmark, train_landmark)
                G_gen_emotional_loss = emotional_loss_func(G_gen_img_emotion * 63, train_score * 63)
                G_gen_mae_loss = mae_loss_func(G_gen_img_emotion * 63, train_score * 63)
                G_gen_adversarial_loss = G_gen_img_flag.mean().view(1)  # W-GAN
                G_gen_content_loss = content_loss_func(G_gen_img, train_img)

                '''
                G_loss = lambda_1 * adversarial_loss
                       + lambda_2 * perceptual_loss
                       + lambda_3 * emotional_loss
                       + 0 * G_gen_content_loss
                '''
                G_loss = - LAMBDA_1 * G_gen_adversarial_loss \
                         + LAMBDA_2 * G_gen_perceptual_loss \
                         + LAMBDA_3 * G_gen_emotional_loss  # W-GAN

                optimizer_G.zero_grad()
                G_loss.backward()
                optimizer_G.step()

            # if epoch % 500 == 0 and train_step % 1000 == 0:
            if (train_step+1) % LOG_STEP == 0:
                model_G.eval()
                model_adv_D.eval()
                model_per_D.eval()
                model_emo_D.eval()
                with torch.no_grad():
                    print('\033[0;36;40m- - - epoch: {} / {}  step: {} / {} - - -\033[0m'.format(
                           epoch, EPOCHS, train_step, len(train_dataloader)))
                    print('D: real_adv_loss: {:.8f}  real_per_loss: {:.8f}  real_emo_loss: {:.8f}  real MAE: {:.8f} gen_adv_loss: {:.8f}  gen_per_loss: {:.8f}  gen_emo_loss: {:.8f}  gen MAE: {:.8f}'.format(
                            real_adversarial_loss.item(), real_perceptual_loss, real_emotional_loss, real_mae_loss, D_gen_adversarial_loss.item(), D_gen_perceptual_loss, D_gen_emotional_loss, D_gen_mae_loss))
                    print('D: real_img_flag: {:.4f}  gen_img_flag: {:.4f}'.format(
                           real_img_flag.mean(), D_gen_img_flag.mean()))

                    print('G: real_adv_loss: {:.8f}  real_per_loss: {:.8f}  real_emo_loss: {:.8f}  real MAE: {:.8f}  gen_adv_loss: {:.8f}  gen_per_loss: {:.8f}  gen_emo_loss: {:.8f} gen MAE: {:.8f}  content_loss: {:.8f}'.format(
                           G_real_adversarial_loss.item(), G_real_perceptual_loss, G_real_emotional_loss, G_real_mae_loss, G_gen_adversarial_loss.item(), G_gen_perceptual_loss, G_gen_emotional_loss, G_gen_mae_loss, G_gen_content_loss))
                    print('G: real_img_flag: {:.4f}  gen_img_flag: {:.4f}'.format(
                           G_real_img_flag.mean(), G_gen_img_flag.mean()))
                    print('real_label: {}  fake_label: {}'.format(real_label.mean(), fake_label.mean()))

                    save_results(train_img.squeeze(0), train_landmark.squeeze(0),
                                 real_img_landmark.squeeze(0), G_gen_img, G_gen_img_landmark,
                                 epoch, train_step, N_ROW, STAGE, 'train')

                    print('====== Val ======')
                    val_emotional_loss_list = []
                    real_emotion_loss_list = []
                    val_mae_loss_list = []
                    real_mae_loss_list = []
                    for val_step, (val_img, val_landmark, val_score) in enumerate(val_dataloader):
                        val_img = val_img.type(torch.FloatTensor).to(device)
                        val_landmark = val_landmark.type(torch.FloatTensor).to(device)
                        val_score = val_score.type(torch.FloatTensor).to(device).unsqueeze(1)
                        val_gen_img = model_G(val_landmark.squeeze(0))
                        val_gen_img_emotion = torch.sigmoid(model_emo_D(val_gen_img.unsqueeze(0)))
                        val_emotional_loss = emotional_loss_func(val_gen_img_emotion * 63, val_score * 63).cpu().numpy()
                        val_emotional_loss_list.append(val_emotional_loss)
                        val_mae_loss = mae_loss_func(val_gen_img_emotion * 63, val_score * 63).cpu().numpy()
                        val_mae_loss_list.append(val_mae_loss)
                        print('Val step: {} / {}  mean RMSE: {}  mean MAE: {}'.format(
                               val_step, len(val_dataloader), np.sqrt(np.mean(val_emotional_loss_list)), np.mean(val_mae_loss_list)))

                        val_gen_img_landmark = model_per_D(val_gen_img.squeeze(0))
                        val_real_img_landmark = model_per_D(val_img.squeeze(0))
                        val_gen_img_flag = model_adv_D(val_gen_img.squeeze(0))
                        val_real_img_flag = model_adv_D(val_img.squeeze(0))
                        val_real_img_emotion = torch.sigmoid(model_emo_D(val_img))

                        val_real_perceptual_loss = perceptual_loss_func(val_real_img_landmark, val_landmark)
                        val_real_emotional_loss = emotional_loss_func(val_real_img_emotion * 63, val_score * 63).cpu().numpy()
                        real_emotion_loss_list.append(val_real_emotional_loss)
                        real_mae_loss = mae_loss_func(val_real_img_emotion * 63, val_score * 63).cpu().numpy()
                        real_mae_loss_list.append(real_mae_loss)

                        val_real_adversarial_loss = val_real_img_flag.mean().view(1)  # W-GAN
                        val_perceptual_loss = perceptual_loss_func(val_gen_img_landmark, val_landmark)
                        val_adversarial_loss = val_gen_img_flag.mean().view(1)  # W-GAN
                        val_content_loss = content_loss_func(val_gen_img, val_img)
                        break

                    print('G: real_adv_loss: {:.8f}  real_per_loss: {:.8f}  real_emo_loss: {:.8f}  real MAE: {:.8f}  gen_adv_loss: {:.8f}  gen_per_loss: {:.8f}  gen_emo_loss: {:.8f}  gen MAE: {:.8f}  content_loss: {:.8f}'.format(
                           val_real_adversarial_loss.item(), val_real_perceptual_loss, np.sqrt(np.mean(real_emotion_loss_list)), np.mean(real_mae_loss_list), val_adversarial_loss.item(), val_perceptual_loss, np.sqrt(np.mean(val_emotional_loss_list)), np.mean(val_mae_loss_list), val_content_loss))
                    print('real_img_flag: {:.4f}  gen_img_flag: {:.4f}'.format(
                          val_real_img_flag.mean(), val_gen_img_flag.mean()))

                    save_results(val_img.squeeze(0), val_landmark.squeeze(0),
                                 val_real_img_landmark.squeeze(0), val_gen_img, val_gen_img_landmark,
                                 epoch, train_step, N_ROW, STAGE, 'val')

                    loss_collector.append(
                        [real_adversarial_loss.item(), real_perceptual_loss.item(), real_emotional_loss.item(), real_mae_loss.item(), D_gen_adversarial_loss.item(), D_gen_perceptual_loss.item(), D_gen_emotional_loss.item(), D_gen_mae_loss.item(),
                         real_img_flag.mean().item(), D_gen_img_flag.mean().item(),
                         G_real_adversarial_loss.item(), G_real_perceptual_loss.item(), G_real_emotional_loss.item(), G_real_mae_loss.item(), G_gen_adversarial_loss.item(), G_gen_perceptual_loss.item(), G_gen_emotional_loss.item(), G_gen_mae_loss.item(), G_gen_content_loss.item(),
                         G_real_img_flag.mean().item(), G_gen_img_flag.mean().item(),
                         real_label.mean().item(), fake_label.mean().item(),
                         val_real_adversarial_loss.item(), val_real_perceptual_loss.item(), np.sqrt(np.mean(real_emotion_loss_list)), np.mean(real_mae_loss_list), val_adversarial_loss.item(), val_perceptual_loss.item(), np.sqrt(np.mean(val_emotional_loss_list)), np.mean(val_mae_loss_list), val_content_loss.item(),
                         val_real_img_flag.mean().item(), val_gen_img_flag.mean().item()])

                    torch.save(model_G.state_dict(), "weights/{}/model_G.pth".format(STAGE))
                    torch.save(model_per_D.state_dict(), "weights/{}/model_per_D.pth".format(STAGE))
                    torch.save(model_adv_D.state_dict(), "weights/{}/model_adv_D.pth".format(STAGE))
                    torch.save(model_emo_D.state_dict(), "weights/{}/model_emo_D.pth".format(STAGE))


if __name__ == '__main__':
    train_s2()
