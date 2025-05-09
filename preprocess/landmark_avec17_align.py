# -*- coding: utf-8 -*-
import os
from pandas.core.frame import DataFrame
from landmark_align import read_process_save


#### avec17 #####
subject_dir = os.listdir('./data')
save_path = './landmark_aligned'
if not os.path.exists(save_path):
    os.makedirs(save_path)


for i in subject_dir[31:]:
    print('===== processing {} ====='.format(i))
    identify = i.split('_')[0]
    read_process_save(256,
                      './data/{}/{}_CLNF_features.txt'.format(i, identify),
                      '{}/{}_landmark_aligned.csv'.format(save_path, identify),
                      sep=', ')


