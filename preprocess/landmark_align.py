# -*- coding: utf-8 -*-
"""
reference: https://zhuanlan.zhihu.com/p/55479744
"""
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import imageio


def visualize_landmark(landmarks):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    image_array = np.ones((IMG_SIZE, IMG_SIZE)) * 255
    origin_img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(origin_img)
    for facial_feature in landmarks.keys():
        draw.point(landmarks[facial_feature])
    # plt.imshow(origin_img)
    return origin_img

def rotate(origin, point, angle, row):
    """ rotate coordinates in image coordinate system
    :param origin: tuple of coordinates,the rotation center
    :param point: tuple of coordinates, points to rotate
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated coordinates of point
    """
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    if x != x :
        x = 0
    if y != y:
        y = 0
    return int(x), int(y)

def rotate_landmarks(landmarks, eye_center, angle, row):
    """ rotate landmarks to fit the aligned face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param eye_center: tuple of coordinates for eye center
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated_landmarks with the same structure with landmarks, but different values
    """
    rotated_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            rotated_landmark = rotate(origin=eye_center, point=landmark, angle=angle, row=row)
            rotated_landmarks[facial_feature].append(rotated_landmark)
    return rotated_landmarks


def align_face(landmarks):
    """ align faces according to eyes position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    rotated_img:  numpy array of aligned image
    eye_center: tuple of coordinates for eye center
    angle: degrees of rotation
    """
    # get list landmarks of left and right eye
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    # calculate the mean point of landmarks of left and right eye
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")
    # compute the angle between the eye centroids
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    # compute angle between the line of 2 centeroids and the horizontal line
    angle = math.atan2(dy, dx) * 180. / math.pi
    # calculate the center of 2 eyes
    eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)
    # at the eye_center, rotate the image by the angle
    # rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    # rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))
    return eye_center, angle

def corp_face(landmarks):
    """ crop face according to eye,mouth and chin position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    cropped_img: numpy array of cropped image
    """

    eye_landmark = np.concatenate([np.array(landmarks['left_eye']),
                                   np.array(landmarks['right_eye'])])
    eye_center = np.mean(eye_landmark, axis=0).astype("int")
    lip_landmark = np.concatenate([np.array(landmarks['top_lip']),
                                   np.array(landmarks['bottom_lip'])])
    lip_center = np.mean(lip_landmark, axis=0).astype("int")
    mid_part = lip_center[1] - eye_center[1]
    top = eye_center[1] - mid_part * 30 / 35
    bottom = lip_center[1] + mid_part

    w = bottom - top
    x_min = np.min(landmarks['chin'], axis=0)[0]
    x_max = np.max(landmarks['chin'], axis=0)[0]
    x_center = (x_max - x_min) / 2 + x_min
    left, right = (x_center - w / 2, x_center + w / 2)

    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    return left, top, w

def transfer_landmark(landmarks, left, top, size, landmark_size):
    """transfer landmarks to fit the cropped face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param left: left coordinates of cropping
    :param top: top coordinates of cropping
    :param size: output size
    :param landmark_size: cropped size according to the landmark
    :return: transferred_landmarks with the same structure with landmarks, but different values
    """
    rescale_factor = size / landmark_size
    transferred_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            transferred_landmark = ((landmark[0] - left) * rescale_factor, (landmark[1] - top) * rescale_factor)
            transferred_landmarks[facial_feature].append(transferred_landmark)
    return transferred_landmarks


def read_process_save(img_size, input_path, output_path, sep):

    face_landmarks_df = pd.read_csv(input_path, sep=sep)
    for df_index in range(face_landmarks_df.shape[0]):
    
        data_item = face_landmarks_df.loc[df_index]
        face_landmarks_dict = {'chin': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(17)],
                               'left_eyebrow': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(17, 22)],
                               'right_eyebrow': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(22, 27)],
                               'nose_bridge': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(27, 31)],
                               'nose_tip': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(31, 36)],
                               'left_eye': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(36, 42)],
                               'right_eye': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(42, 48)],
                               'top_lip': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(48, 60)],
                               'bottom_lip': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(60, 68)],
                              }
            
        eye_center, angle = align_face(landmarks=face_landmarks_dict)
        rotated_landmarks = rotate_landmarks(landmarks=face_landmarks_dict, eye_center=eye_center, angle=angle, row=768)
        
        left, top, w = corp_face(landmarks=rotated_landmarks)
        transferred_landmarks = transfer_landmark(landmarks=rotated_landmarks, left=left, top=top, size=img_size, landmark_size=w)
        
        for idx, i in enumerate(transferred_landmarks['chin']):
            data_item['x{}'.format(idx)] = i[0]
            data_item['y{}'.format(idx)] = i[1]
        for idx, i in enumerate(transferred_landmarks['left_eyebrow']):
            data_item['x{}'.format(idx+17)] = i[0]
            data_item['y{}'.format(idx+17)] = i[1]
        for idx, i in enumerate(transferred_landmarks['right_eyebrow']):
            data_item['x{}'.format(idx+22)] = i[0]
            data_item['y{}'.format(idx+22)] = i[1]
        for idx, i in enumerate(transferred_landmarks['nose_bridge']):
            data_item['x{}'.format(idx+27)] = i[0]
            data_item['y{}'.format(idx+27)] = i[1]
        for idx, i in enumerate(transferred_landmarks['nose_tip']):
            data_item['x{}'.format(idx+31)] = i[0]
            data_item['y{}'.format(idx+31)] = i[1]
        for idx, i in enumerate(transferred_landmarks['left_eye']):
            data_item['x{}'.format(idx+36)] = i[0]
            data_item['y{}'.format(idx+36)] = i[1]
        for idx, i in enumerate(transferred_landmarks['right_eye']):
            data_item['x{}'.format(idx+42)] = i[0]
            data_item['y{}'.format(idx+42)] = i[1]
        for idx, i in enumerate(transferred_landmarks['top_lip']):
            data_item['x{}'.format(idx+48)] = i[0]
            data_item['y{}'.format(idx+48)] = i[1]
        for idx, i in enumerate(transferred_landmarks['bottom_lip']):
            data_item['x{}'.format(idx+60)] = i[0]
            data_item['y{}'.format(idx+60)] = i[1]
        
        face_landmarks_df.loc[df_index] = data_item
        
        if df_index % 1000 == 0:
            print(df_index)
        
    face_landmarks_df.to_csv(output_path, index=None)





if __name__ == '__main__':
    
    IMG_SIZE = 64
    face_landmarks_df = pd.read_csv('./300_CLNF_features.txt', sep=', ')
    im_gif = []
    print('total frames: {}'.format(face_landmarks_df.shape[0]))
    
    for df_index in range(face_landmarks_df.shape[0]):
    
        data_item = face_landmarks_df.loc[df_index]
        face_landmarks_dict = {'chin': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(17)],
                               'left_eyebrow': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(17, 22)],
                               'right_eyebrow': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(22, 27)],
                               'nose_bridge': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(27, 31)],
                               'nose_tip': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(31, 36)],
                               'left_eye': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(36, 42)],
                               'right_eye': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(42, 48)],
                               'top_lip': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(48, 60)],
                               'bottom_lip': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(60, 68)],
                              }
            
            
        # print(face_landmarks_dict, end=" ")
        '''out
        {'chin': [(76, 102), (74, 115), (72, 128), (70, 142), (73, 155), (80, 165), (90, 175), (102, 183), (114, 188), (126, 189), (136, 183), (145, 176), (152, 168), (159, 159), (164, 149), (167, 138), (169, 128)],
        'left_eyebrow': [(94, 93), (103, 90), (113, 90), (122, 94), (129, 100)], 
        'right_eyebrow': [(147, 105), (155, 104), (162, 104), (168, 108), (171, 114)], 
        'nose_bridge': [(136, 113), (135, 121), (134, 129), (133, 137)], 
        'nose_tip': [(119, 137), (123, 140), (128, 144), (133, 144), (137, 143)],
        'left_eye': [(102, 105), (109, 105), (115, 107), (119, 111), (113, 110), (107, 108)], 
        'right_eye': [(145, 119), (152, 118), (157, 119), (161, 123), (156, 123), (151, 121)],
        'top_lip': [(99, 146), (109, 144), (119, 145), (125, 148), (131, 148), (138, 152), (142, 159), (139, 158), (130, 154), (124, 153), (118, 150), (101, 147)],
        'bottom_lip': [(142, 159), (134, 168), (126, 170), (120, 168), (113, 166), (105, 159), (99, 146), (101, 147), (116, 159), (122, 161), (128, 162), (139, 158)]} 
        '''
        
        '''
        chin [0, 16]
        left_eyebrow [17, 21]
        right_eyebrow [22, 26]
        nose_bridge [27, 30]
        nose_tip [31, 35]
        left_eye [36, 41]
        right_eye [42, 47]
        top_lip [48, 59]，
        bottom_lip [60, 67]
        '''
        
        eye_center, angle = align_face(landmarks=face_landmarks_dict)
        rotated_landmarks = rotate_landmarks(landmarks=face_landmarks_dict, eye_center=eye_center, angle=angle, row=768)
        # visualize_landmark(landmarks=rotated_landmarks)
        
        left, top, w = corp_face(landmarks=rotated_landmarks)
        transferred_landmarks = transfer_landmark(landmarks=rotated_landmarks, left=left, top=top, size=IMG_SIZE, landmark_size=w)
        img_landmark = visualize_landmark(landmarks=transferred_landmarks)
        im_gif.append(img_landmark)
        # plt.show()
        
        
        for idx, i in enumerate(transferred_landmarks['chin']):
            data_item['x{}'.format(idx)] = i[0]
            data_item['y{}'.format(idx)] = i[1]
        for idx, i in enumerate(transferred_landmarks['left_eyebrow']):
            data_item['x{}'.format(idx+17)] = i[0]
            data_item['y{}'.format(idx+17)] = i[1]
        for idx, i in enumerate(transferred_landmarks['right_eyebrow']):
            data_item['x{}'.format(idx+22)] = i[0]
            data_item['y{}'.format(idx+22)] = i[1]
        for idx, i in enumerate(transferred_landmarks['nose_bridge']):
            data_item['x{}'.format(idx+27)] = i[0]
            data_item['y{}'.format(idx+27)] = i[1]
        for idx, i in enumerate(transferred_landmarks['nose_tip']):
            data_item['x{}'.format(idx+31)] = i[0]
            data_item['y{}'.format(idx+31)] = i[1]
        for idx, i in enumerate(transferred_landmarks['left_eye']):
            data_item['x{}'.format(idx+36)] = i[0]
            data_item['y{}'.format(idx+36)] = i[1]
        for idx, i in enumerate(transferred_landmarks['right_eye']):
            data_item['x{}'.format(idx+42)] = i[0]
            data_item['y{}'.format(idx+42)] = i[1]
        for idx, i in enumerate(transferred_landmarks['top_lip']):
            data_item['x{}'.format(idx+48)] = i[0]
            data_item['y{}'.format(idx+48)] = i[1]
        for idx, i in enumerate(transferred_landmarks['bottom_lip']):
            data_item['x{}'.format(idx+60)] = i[0]
            data_item['y{}'.format(idx+60)] = i[1]
        
        face_landmarks_df.loc[df_index] = data_item
        
        if df_index % 1000 == 0:
            print(df_index)
        
    df = face_landmarks_df.to_csv('./out_test.csv', index=None)
    imageio.mimsave('./out_test.gif', im_gif, 'GIF', duration = 1/30)
    