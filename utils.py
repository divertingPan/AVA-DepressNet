import cv2
import torch
from torch import autograd
from torch.nn import init
import numpy as np
import pandas as pd
from torchvision.utils import save_image
import configparser as ConfigParser
from optparse import OptionParser


def visualize_landmark(img, landmarks, rescale):
    """ plot landmarks on image
    :param img: tensor of a mini-batch image
    :param landmarks: landmarks list for facial parts as [x0, x1, ..., x67, y0, y1, ..., y67]
    :param rescale: image size
    :return: tensor of mini-batch images with landmarks on
    """
    img = img.permute(0, 2, 3, 1)
    img = ((img.cpu().detach().numpy() + 1) * 0.5 * 255).astype(np.uint8)
    landmarks = (landmarks.detach().squeeze().cpu().numpy() * rescale).astype(int)
    for idx in range(img.shape[0]):
        draw_img = np.ascontiguousarray(img[idx])
        for i in range(68):
            position = (landmarks[idx][i], landmarks[idx][i + 68])
            cv2.circle(draw_img, position, 1, (0, 0, 255), -1)
        img[idx] = draw_img

    img = torch.tensor(img / 255.).permute(0, 3, 1, 2)

    return img


def save_results(real_img, real_lm, real_img_lm, fake_img, fake_img_lm, epoch, step, n_row, stage, dataset):
    landmark_from_D = visualize_landmark((real_img - 0.5) * 2, real_img_lm, 64)
    save_image(landmark_from_D,
               "results/{}/{}/D_landmark_{}_{}.png".format(stage, dataset, epoch, step),
               nrow=n_row)
    real_img_with_landmarks = visualize_landmark((real_img - 0.5) * 2, real_lm, 64)
    save_image(real_img_with_landmarks,
               "results/{}/{}/real_{}_{}.png".format(stage, dataset, epoch, step),
               nrow=n_row)
    gen_img_with_landmarks = visualize_landmark((fake_img - 0.5) * 2, fake_img_lm, 64)
    save_image(gen_img_with_landmarks,
               "results/{}/{}/gen_landmark_{}_{}.png".format(stage, dataset, epoch, step),
               nrow=n_row)
    save_image(fake_img, "results/{}/{}/gen_img_{}_{}.png".format(stage, dataset, epoch, step),
               nrow=n_row)


def weight_init(m):
    # weight_initialization: important for W-GAN
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0, 0.02)
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)


def mult_weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv1d') or classname.find('Conv2d') or classname.find('Conv3d')) != -1:
        init.xavier_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)


def calc_gradient_penalty(netD, real_data, fake_data, device):
    BATCH_SIZE = real_data.shape[0]
    LAMBDA = 10

    alpha = torch.rand(BATCH_SIZE, 1, 1, 1)
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


class LossCollector:
    def __init__(self, stage, header_list):
        self.save_path = 'results/{}/loss.csv'.format(stage)
        self.df = pd.DataFrame(columns=header_list)

    def append(self, param_list):
        self.df.loc[len(self.df)] = param_list
        self.df.to_csv(self.save_path, index=False)


def read_conf():
    parser = OptionParser()
    (options, args) = parser.parse_args()
    Config = ConfigParser.ConfigParser()
    Config.read('./net/audio_net_config.cfg')

    # # [data]
    # options.tr_lst = Config.get('data', 'tr_lst')
    # options.te_lst = Config.get('data', 'te_lst')
    # options.lab_dict = Config.get('data', 'lab_dict')
    # options.data_folder = Config.get('data', 'data_folder')
    # options.output_folder = Config.get('data', 'output_folder')
    # options.pt_file = Config.get('data', 'pt_file')

    # [windowing]
    options.fs = Config.get('windowing', 'fs')
    options.cw_len = Config.get('windowing', 'cw_len')
    options.cw_shift = Config.get('windowing', 'cw_shift')

    # [cnn]
    options.cnn_N_filt = Config.get('cnn', 'cnn_N_filt')
    options.cnn_len_filt = Config.get('cnn', 'cnn_len_filt')
    options.cnn_max_pool_len = Config.get('cnn', 'cnn_max_pool_len')
    options.cnn_use_laynorm_inp = Config.get('cnn', 'cnn_use_laynorm_inp')
    options.cnn_use_batchnorm_inp = Config.get('cnn', 'cnn_use_batchnorm_inp')
    options.cnn_use_laynorm = Config.get('cnn', 'cnn_use_laynorm')
    options.cnn_use_batchnorm = Config.get('cnn', 'cnn_use_batchnorm')
    options.cnn_act = Config.get('cnn', 'cnn_act')
    options.cnn_drop = Config.get('cnn', 'cnn_drop')

    # [dnn]
    options.fc_lay = Config.get('dnn', 'fc_lay')
    options.fc_drop = Config.get('dnn', 'fc_drop')
    options.fc_use_laynorm_inp = Config.get('dnn', 'fc_use_laynorm_inp')
    options.fc_use_batchnorm_inp = Config.get('dnn', 'fc_use_batchnorm_inp')
    options.fc_use_batchnorm = Config.get('dnn', 'fc_use_batchnorm')
    options.fc_use_laynorm = Config.get('dnn', 'fc_use_laynorm')
    options.fc_act = Config.get('dnn', 'fc_act')

    # [class]
    options.class_lay = Config.get('class', 'class_lay')
    options.class_drop = Config.get('class', 'class_drop')
    options.class_use_laynorm_inp = Config.get('class', 'class_use_laynorm_inp')
    options.class_use_batchnorm_inp = Config.get('class', 'class_use_batchnorm_inp')
    options.class_use_batchnorm = Config.get('class', 'class_use_batchnorm')
    options.class_use_laynorm = Config.get('class', 'class_use_laynorm')
    options.class_act = Config.get('class', 'class_act')

    # [optimization]
    options.lr = Config.get('optimization', 'lr')
    options.batch_size = Config.get('optimization', 'batch_size')
    options.N_epochs = Config.get('optimization', 'N_epochs')
    options.N_batches = Config.get('optimization', 'N_batches')
    options.N_eval_epoch = Config.get('optimization', 'N_eval_epoch')
    options.seed = Config.get('optimization', 'seed')

    return options


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError
