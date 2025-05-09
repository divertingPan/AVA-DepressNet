# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
from pandas.core.frame import DataFrame
from face_landmark_extract import extract_landmark
'''
main_path = '/home/pyc/桌面/Depression_3D/datasets/avec14'

split_dir = ['Training']  # ['Training', 'Testing', 'Development']

for task in ['Northwind']:  # ['Freeform', 'Northwind']
    for sets in split_dir:
        video_dir = os.listdir('{}/{}/{}'.format(main_path, sets, task))
        video_dir = sorted(video_dir)
        
        for video in video_dir:
            save_path = './avec14/landmark_aligned/{}/{}/{}'.format(sets, task, video)
            if os.path.exists(save_path):
                continue
            print('{}/{}/{}'.format(sets, task, video))
            
            try:    
                images = os.listdir('{}/{}/{}/{}'.format(main_path, sets, task, video))
                print('total: {}'.format(len(images)))
                save_path = './avec14/landmark_aligned/{}/{}/{}'.format(sets, task, video)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            except:
                continue
            images.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
            for idx, image in enumerate(images):
                
                print(image)
                try:
                    landmark_list_x, landmark_list_y = extract_landmark('{}/{}/{}/{}/{}'.format(main_path, sets, task, video, image))
                except:
                    print('pass')
                    continue
                c = {}
                for idx in range(68):
                    c['x{}'.format(idx)] = landmark_list_x[idx]
                    c['y{}'.format(idx)] = landmark_list_y[idx]
                    
                face_landmarks_df = DataFrame([c])
                face_landmarks_df.to_csv('{}/{}.csv'.format(save_path, image.split('.')[0]), index=None)
'''


main_path = './avec14/landmark_aligned'
split_dir = os.listdir(main_path)

label = {'img': [],
         'landmark': [],
         'score': []}

score_csv = pd.read_csv('/home/pyc/桌面/Depression_3D/datasets/avec14/label.csv')
score_item = score_csv['path'].values.tolist()
score_value = score_csv['label'].values.tolist()
score_dic = { i:score_value[score_item.index(i)] for i in score_item }

for sets in ['Training', 'Development', 'Testing']:
    for task in ['Freeform', 'Northwind']:
        video_dir = os.listdir('{}/{}/{}'.format(main_path, sets, task))
        for video in video_dir:
            try:    
                images = os.listdir('{}/{}/{}/{}'.format(main_path, sets, task, video))
                print('{}/{}/{}'.format(sets, task, video))
            except:
                continue
            for idx, image in enumerate(images):
                label['img'].append('{}/{}/{}/{}.png'.format(sets, task, video, image.split('.')[0]))
                label['landmark'].append('landmark_aligned/{}/{}/{}/{}.csv'.format(sets, task, video, image.split('.')[0]))
                
                
                label['score'].append(score_dic['{}/{}/{}'.format(sets, task, video)])



df = DataFrame(label)
df.to_csv('./avec14/label.csv', index=None)
