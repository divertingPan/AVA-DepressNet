# -*- coding: utf-8 -*-
import cv2
import os
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam, RMSprop, SGD
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from net.net_x64_wgan import Generator, Perceptual_D, Adversarial_D
from dataset import Adv_Dataset
from utils import weight_init, calc_gradient_penalty, save_results, LossCollector
from path import root_path


def train_s1():
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    EPOCHS = 50
    BATCH_SIZE = 256
    D_ENHANCE = 5
    G_ENHANCE = 1
    DATASET = 'celeba'
    LAMBDA_1 = 1
    LAMBDA_2 = 1
    N_ROW = 16
    LOG_STEP = 800
    STAGE = 'stage_1_D5G1_epoch50'

    mkdir_list = ['results/{}/train'.format(STAGE),
                  'results/{}/val'.format(STAGE),
                  'weights/{}'.format(STAGE)]
    for path in mkdir_list:
        if not os.path.exists(path):
            os.makedirs(path)

    df = pd.read_csv('{}/dataset/{}/label.csv'.format(root_path, DATASET))
    image_path_list = df['img'].values
    landmark_path_list = df['landmark'].values
    # total: 192366
    train_size = 192366 - BATCH_SIZE
    train_image_path_list = image_path_list[:train_size]
    train_landmark_path_list = landmark_path_list[:train_size]
    val_image_path_list = image_path_list[train_size:]
    val_landmark_path_list = landmark_path_list[train_size:]

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = Adv_Dataset(dataset=DATASET,
                                image_path=train_image_path_list,
                                landmark_path=train_landmark_path_list,
                                transform=transform,
                                rescale=64)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=16,
                                  pin_memory=True,
                                  drop_last=True)
    val_dataset = Adv_Dataset(dataset=DATASET,
                              image_path=val_image_path_list,
                              landmark_path=val_landmark_path_list,
                              transform=transform,
                              rescale=64)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=16,
                                pin_memory=True,
                                drop_last=True)

    model_G = Generator().to(device)
    model_G.apply(weight_init)
    optimizer_G = Adam(model_G.parameters(), lr=0.0001, betas=(0, 0.9))

    model_per_D = Perceptual_D().to(device)
    model_per_D.apply(weight_init)
    optimizer_per_D = Adam(model_per_D.parameters(), lr=0.0001, betas=(0, 0.9))

    model_adv_D = Adversarial_D().to(device)
    model_adv_D.apply(weight_init)
    optimizer_adv_D = Adam(model_adv_D.parameters(), lr=0.0001, betas=(0, 0.9))

    perceptual_loss_func = nn.MSELoss()
    adversarial_loss_func = nn.BCELoss()
    content_loss_func = nn.MSELoss()

    loss_collector = LossCollector(STAGE, ['D_real_adv_loss', 'D_real_per_loss', 'D_gen_adv_loss', 'D_gen_per_loss',
                                           'D_real_img_flag', 'D_gen_img_flag',
                                           'G_real_adv_loss', 'G_real_per_loss', 'G_gen_adv_loss', 'G_gen_per_loss', 'G_gen_content_loss',
                                           'G_real_img_flag', 'G_gen_img_flag',
                                           'real_label', 'gen_label',
                                           'val_real_adv_loss', 'val_real_per_loss', 'val_gen_adv_loss', 'val_gen_per_loss', 'val_gen_content_loss',
                                           'val_real_img_flag', 'val_gen_img_flag'])

    real_label = torch.ones(BATCH_SIZE).to(device)
    fake_label = -torch.ones(BATCH_SIZE).to(device)

    model_G.train()
    model_adv_D.train()
    model_per_D.train()

    for epoch in range(EPOCHS):
        for train_step, (train_img, train_landmark) in enumerate(train_dataloader):
            train_img = train_img.type(torch.FloatTensor).to(device)
            train_landmark = train_landmark + torch.randn(BATCH_SIZE, 136, 1, 1) / 256
            train_landmark = train_landmark.type(torch.FloatTensor).to(device)
            # print('img: {}  landmark: {}'.format(train_img.shape, train_landmark.shape))
            # >>> img: torch.Size([4, 3, 256, 256])  landmark: torch.Size([4, 136, 1, 1])
            ############################
            # Update D
            ############################

            for D_epoch in range(D_ENHANCE):

                optimizer_per_D.zero_grad()
                optimizer_adv_D.zero_grad()

                real_img_landmark = model_per_D(train_img)
                real_img_flag = model_adv_D(train_img)
                real_perceptual_loss = perceptual_loss_func(real_img_landmark, train_landmark)
                real_adversarial_loss = real_img_flag.mean().view(1)  # W-GAN

                D_gen_img = model_G(train_landmark)
                D_gen_img_landmark = model_per_D(D_gen_img)
                D_gen_img_flag = model_adv_D(D_gen_img)
                D_gen_perceptual_loss = perceptual_loss_func(D_gen_img_landmark, train_landmark)
                D_gen_adversarial_loss = D_gen_img_flag.mean().view(1)  # W-GAN
                # WGAN-GP
                gradient_penalty = calc_gradient_penalty(model_adv_D, train_img.data, D_gen_img.data, device)

                '''
                D_loss = lambda_1 * adversarial_loss
                       + lambda_2 * perceptual_loss
                '''
                D_loss = - LAMBDA_1 * real_adversarial_loss + LAMBDA_1 * D_gen_adversarial_loss \
                    + LAMBDA_2 * real_perceptual_loss - 0 * LAMBDA_2 * D_gen_perceptual_loss + gradient_penalty  # W-GAN
                D_loss.backward()

                optimizer_per_D.step()
                optimizer_adv_D.step()

            ############################
            # Update G
            ############################

            for G_epoch in range(G_ENHANCE):
                G_gen_img = model_G(train_landmark)
                G_gen_img_landmark = model_per_D(G_gen_img)
                G_gen_img_flag = model_adv_D(G_gen_img)
                G_real_img_landmark = model_per_D(train_img)
                G_real_img_flag = model_adv_D(train_img)
                G_real_perceptual_loss = perceptual_loss_func(G_real_img_landmark, train_landmark)
                G_real_adversarial_loss = G_real_img_flag.mean().view(1)  # W-GAN
                G_gen_perceptual_loss = perceptual_loss_func(G_gen_img_landmark, train_landmark)
                G_gen_adversarial_loss = G_gen_img_flag.mean().view(1)  # W-GAN
                G_gen_content_loss = content_loss_func(G_gen_img, train_img)

                '''
                G_loss = lambda_1 * adversarial_loss
                       + lambda_2 * perceptual_loss
                       + 0 * G_gen_content_loss
                '''
                G_loss = - LAMBDA_1 * G_gen_adversarial_loss + LAMBDA_2 * G_gen_perceptual_loss  # W-GAN

                optimizer_G.zero_grad()
                G_loss.backward()
                optimizer_G.step()

            # if epoch % 500 == 0 and train_step % 1000 == 0:
            if train_step % LOG_STEP == 0:
                model_G.eval()
                model_adv_D.eval()
                model_per_D.eval()
                with torch.no_grad():
                    print('\033[0;36;40m- - - epoch: {} / {}  step: {} / {} - - -\033[0m'.format(
                          epoch, EPOCHS, train_step, train_size//BATCH_SIZE))
                    print('D: real_adv_loss: {:.8f}  real_per_loss: {:.8f}  gen_adv_loss: {:.8f}  gen_per_loss: {:.8f}'.format(
                          real_adversarial_loss.item(), real_perceptual_loss, D_gen_adversarial_loss.item(), D_gen_perceptual_loss))
                    print('D: real_img_flag: {:.4f}  gen_img_flag: {:.4f}'.format(real_img_flag.mean(), D_gen_img_flag.mean()))

                    print('G: real_adv_loss: {:.8f}  real_per_loss: {:.8f}  gen_adv_loss: {:.8f}  gen_per_loss: {:.8f}  content_loss: {:.8f}'.format(
                          G_real_adversarial_loss.item(), G_real_perceptual_loss, G_gen_adversarial_loss.item(), G_gen_perceptual_loss, G_gen_content_loss))
                    print('G: real_img_flag: {:.4f}  gen_img_flag: {:.4f}'.format(G_real_img_flag.mean(), G_gen_img_flag.mean()))
                    print('real_label: {}  fake_label: {}'.format(real_label.mean(), fake_label.mean()))

                    save_results(train_img, train_landmark, real_img_landmark, G_gen_img, G_gen_img_landmark,
                                 epoch, train_step, N_ROW, STAGE, 'train')

                    print('====== Val ======')
                    for val_step, (val_img, val_landmark) in enumerate(val_dataloader):
                        val_img = val_img.type(torch.FloatTensor).to(device)
                        val_landmark = val_landmark.type(torch.FloatTensor).to(device)
                        val_gen_img = model_G(val_landmark)
                        val_gen_img_landmark = model_per_D(val_gen_img)
                        val_real_img_landmark = model_per_D(val_img)
                        val_gen_img_flag = model_adv_D(val_gen_img)
                        val_real_img_flag = model_adv_D(val_img)

                        val_real_perceptual_loss = perceptual_loss_func(val_real_img_landmark, val_landmark)
                        val_real_adversarial_loss = val_real_img_flag.mean().view(1)  # W-GAN
                        val_perceptual_loss = perceptual_loss_func(val_gen_img_landmark, val_landmark)
                        val_adversarial_loss = val_gen_img_flag.mean().view(1)  # W-GAN
                        val_content_loss = content_loss_func(val_gen_img, val_img)
                        print('G: real_adv_loss: {:.8f}  real_per_loss: {:.8f}  gen_adv_loss: {:.8f}  gen_per_loss: {:.8f}  content_loss: {:.8f}'.format(
                              val_real_adversarial_loss.item(), val_real_perceptual_loss, val_adversarial_loss.item(), val_perceptual_loss, val_content_loss))
                        print('real_img_flag: {:.4f}  gen_img_flag: {:.4f}'.format(
                              val_real_img_flag.mean(), val_gen_img_flag.mean()))

                        save_results(val_img, val_landmark, val_real_img_landmark, val_gen_img, val_gen_img_landmark,
                                     epoch, train_step, N_ROW, STAGE, 'val')

                    loss_collector.append([real_adversarial_loss.item(), real_perceptual_loss.item(), D_gen_adversarial_loss.item(), D_gen_perceptual_loss.item(),
                                           real_img_flag.mean().item(), D_gen_img_flag.mean().item(),
                                           G_real_adversarial_loss.item(), G_real_perceptual_loss.item(), G_gen_adversarial_loss.item(), G_gen_perceptual_loss.item(), G_gen_content_loss.item(),
                                           G_real_img_flag.mean().item(), G_gen_img_flag.mean().item(),
                                           real_label.mean().item(), fake_label.mean().item(),
                                           val_real_adversarial_loss.item(), val_real_perceptual_loss.item(), val_adversarial_loss.item(), val_perceptual_loss.item(), val_content_loss.item(),
                                           val_real_img_flag.mean().item(), val_gen_img_flag.mean().item()])

                    torch.save(model_G.state_dict(), "weights/{}/model_G.pth".format(STAGE))
                    torch.save(model_per_D.state_dict(), "weights/{}/model_per_D.pth".format(STAGE))
                    torch.save(model_adv_D.state_dict(), "weights/{}/model_adv_D.pth".format(STAGE))
