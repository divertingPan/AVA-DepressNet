# -*- coding: utf-8 -*-
import os
from pandas.core.frame import DataFrame
from face_landmark_extract import extract_landmark


img_path = './celeba/img_align_celeba'
subject_dir = os.listdir(img_path)
subject_dir = sorted(subject_dir)
save_path = './celeba/landmark_extracted'
if not os.path.exists(save_path):
    os.makedirs(save_path)

for i in subject_dir:
    print('===== processing {} ====='.format(i))
    try:
        landmark_list_x, landmark_list_y = extract_landmark('{}/{}'.format(img_path, i))
    except:
        print('pass')
        continue
    c = {}
    for idx in range(68):
        c['x{}'.format(idx)] = landmark_list_x[idx]
        c['y{}'.format(idx)] = landmark_list_y[idx]
        
    face_landmarks_df = DataFrame([c])
    face_landmarks_df.to_csv('{}/{}.txt'.format(save_path, i.split('.')[0]),
                             sep=',', index=None)


#     read_process_save(256,
#                       './data/{}/{}_CLNF_features.txt'.format(i, identify),
#                       '{}/{}_landmark_aligned.csv'.format(save_path, identify))
