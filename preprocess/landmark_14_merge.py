import pandas as pd
import os
import re


dataset_path = 'avec14/landmark_aligned'
new_merged_path = 'avec14/landmark_aligned_merge'
path_1 = os.listdir(dataset_path)

for path in path_1:
    path_2 = os.listdir(dataset_path + '/' + path)
    for task in path_2:
        items = os.listdir(dataset_path + '/' + path + '/' + task)
        for data in items:
            file_list = os.listdir(dataset_path + '/' + path + '/' + task + '/' + data)
            file_list.sort(key=lambda f: int(re.match(r'(\d+)', f).group()))
            
            print('merging: ' + dataset_path + '/' + path + '/' + task + '/' + data)
            
            df = pd.read_csv(dataset_path + '/' + path + '/' + task + '/' + data + '/' + file_list[0])
            df.to_csv(new_merged_path + '/' + path + '/' + task + '/' + data + '.csv', index=False)
            for i in range(1, len(file_list)):
                df = pd.read_csv(dataset_path + '/' + path + '/' + task + '/' + data + '/' + file_list[i])
                df.to_csv(new_merged_path + '/' + path + '/' + task + '/' + data + '.csv',
                          index=False, header=False, mode='a+')
