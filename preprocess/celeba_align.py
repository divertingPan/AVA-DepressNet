# -*- coding: utf-8 -*-
import os
import dlib
import cv2
from pandas.core.frame import DataFrame
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def visualize_landmark(image_array, landmarks):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    origin_img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(origin_img)
    for facial_feature in landmarks.keys():
        draw.point(landmarks[facial_feature])
    plt.imshow(origin_img)
    plt.show()
    return origin_img

def get_face(fa, image):
    detector = dlib.get_frontal_face_detector()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = gray.shape[0] // 4
    
    rects = detector(gray, 2)
    face_aligned = []
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        if w > thresh:
            face_aligned = fa.align(image, gray, rect)
    return face_aligned


def extract_landmark(img, predictor):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     
    # get face
    hog_face_detector = dlib.get_frontal_face_detector()
    shape_predictor = predictor
    
    rects, scores, idx = hog_face_detector.run(img_rgb, 2, 0)
    faces = dlib.full_object_detections()
    for rect in rects:
        faces.append(shape_predictor(img_rgb, rect))
    
    for landmark in faces:
        landmark_list_x = []
        landmark_list_y = []
        for idx, point in enumerate(landmark.parts()):
            landmark_list_x.append(point.x)
            landmark_list_y.append(point.y)
    return landmark_list_x, landmark_list_y


##### celeba #####
img_path = './celeba/img_align_celeba'
subject_list = os.listdir(img_path)
subject_list = sorted(subject_list)

save_landmark_path = './celeba/landmark_aligned'
save_img_path = './celeba/img_cropped'

if not os.path.exists(save_landmark_path):
    os.makedirs(save_landmark_path)
if not os.path.exists(save_img_path):
    os.makedirs(save_img_path)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)

label = {'img': [],
         'landmark': []}
for i in subject_list:
    print('===== processing {} ====='.format(i))
    image_array = cv2.imread('{}/{}.jpg'.format(img_path, i.split('.')[0]))
    frame = get_face(fa, image_array)
    
    try:
        cv2.imwrite('{}/{}.jpg'.format(save_img_path, i.split('.')[0]), frame)
    except:
        print('pass')
        continue

    try:
        landmark_list_x, landmark_list_y = extract_landmark(frame, predictor)
    except:
        print('pass')
        continue
    c = {}
    for idx in range(68):
        c['x{}'.format(idx)] = landmark_list_x[idx]
        c['y{}'.format(idx)] = landmark_list_y[idx]
        
    face_landmarks_df = DataFrame([c])
    face_landmarks_df.to_csv('{}/{}.csv'.format(save_landmark_path, i.split('.')[0]), index=None)

    label['img'].append('{}/{}.jpg'.format(save_img_path.split('/')[-1], i.split('.')[0]))
    label['landmark'].append('{}/{}.csv'.format(save_landmark_path.split('/')[-1], i.split('.')[0]))
    
    
    # visualize
    # for df_index in range(face_landmarks_df.shape[0]):
    
    #     data_item = face_landmarks_df.loc[df_index]
    #     face_landmarks_dict = {'chin': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(17)],
    #                             'left_eyebrow': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(17, 22)],
    #                             'right_eyebrow': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(22, 27)],
    #                             'nose_bridge': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(27, 31)],
    #                             'nose_tip': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(31, 36)],
    #                             'left_eye': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(36, 42)],
    #                             'right_eye': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(42, 48)],
    #                             'top_lip': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(48, 60)],
    #                             'bottom_lip': [(data_item['x{}'.format(i)], data_item['y{}'.format(i)]) for i in range(60, 68)],
    #                           }
    # visualize_landmark(frame, face_landmarks_dict)
    
    
df = DataFrame(label)
df.to_csv('./celeba/label.csv', index=None)
    
    
    