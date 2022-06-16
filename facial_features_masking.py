from collections import OrderedDict
import numpy as np
import cv2
import argparse
import dlib
import imutils
import os
from pathlib import Path

def shape_to_numpy_array(shape, dtype="int"):
    coordinates = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y) 
        
        # 코 영역 좌표값 조정
        if i == 27:
            coordinates[i] = (shape.part(32).x, shape.part(44).y-10)
        elif i == 28:
            coordinates[i] = (shape.part(34).x, shape.part(44).y-10)
        elif i == 31:
            coordinates[i] = (shape.part(i).x-50, shape.part(i).y) 
        elif i == 35:
            coordinates[i] = (shape.part(i).x+50, shape.part(i).y) 

        # 눈 영역 좌표값 조정
        elif i == 36:
            coordinates[i] = (shape.part(i).x-25, shape.part(i).y)
        elif i == 37:
            coordinates[i] = (shape.part(i).x-25, shape.part(i).y-10)
        elif i == 38:
            coordinates[i] = (shape.part(i).x+15, shape.part(i).y-10)
        elif i == 39:
            coordinates[i] = (shape.part(i).x+15, shape.part(i).y)
        elif i == 40:
            coordinates[i] = (shape.part(i).x+15, shape.part(i).y+10)
        elif i == 41:
            coordinates[i] = (shape.part(i).x-25, shape.part(i).y+10)
            
        elif i == 42:
            coordinates[i] = (shape.part(i).x-15, shape.part(i).y)
        elif i == 43:
            coordinates[i] = (shape.part(i).x-15, shape.part(i).y-10)
        elif i == 44:
            coordinates[i] = (shape.part(i).x+25, shape.part(i).y-10)
        elif i == 45:
            coordinates[i] = (shape.part(i).x+25, shape.part(i).y)
        elif i == 46:
            coordinates[i] = (shape.part(i).x+25, shape.part(i).y+10)
        elif i == 47:
            coordinates[i] = (shape.part(i).x-15, shape.part(i).y+10)
            
        # 입 영역 좌표값 조정    
        elif i == 48:
            coordinates[i] = (shape.part(i).x-20, shape.part(i).y)
        elif i == 49:
            coordinates[i] = (shape.part(i).x-20, shape.part(i).y-10)
        elif i == 50:
            coordinates[i] = (shape.part(i).x, shape.part(i).y-20)
        elif i == 51:
            coordinates[i] = (shape.part(i).x, shape.part(i).y)
        elif i == 52:
            coordinates[i] = (shape.part(i).x, shape.part(i).y-20)
        elif i == 53:
            coordinates[i] = (shape.part(i).x+20, shape.part(i).y-10)
        elif i == 54:
            coordinates[i] = (shape.part(i).x+20, shape.part(i).y)
        elif i == 55:
            coordinates[i] = (shape.part(i).x+20, shape.part(i).y+5)
        elif i == 56:
            coordinates[i] = (shape.part(i).x+20, shape.part(i).y+5)
        elif i == 57:
            coordinates[i] = (shape.part(i).x, shape.part(i).y+10)
        elif i == 58:
            coordinates[i] = (shape.part(i).x-20, shape.part(i).y+5)
        elif i == 59:
            coordinates[i] = (shape.part(i).x-20, shape.part(i).y+5)
        elif i == 60:
            coordinates[i] = (shape.part(i).x-20, shape.part(i).y)

    return coordinates

def shape_to_numpy_array2(shape, dtype="int"):
    coordinates = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)  
        # 눈썹 영역 좌표값 설정
        if i == 1:
            coordinates[i] = (shape.part(17).x-15, shape.part(17).y-10)
        elif i == 2:
            coordinates[i] = (shape.part(17).x-15, shape.part(17).y+30)
        elif i == 3:
            coordinates[i] = (shape.part(18).x, shape.part(18).y-10)
        elif i == 4:
            coordinates[i] = (shape.part(18).x, shape.part(18).y+30)
        elif i == 5:
            coordinates[i] = (shape.part(19).x, shape.part(19).y-10)
        elif i == 6:
            coordinates[i] = (shape.part(19).x, shape.part(19).y+30)
        elif i == 7:
            coordinates[i] = (shape.part(20).x, shape.part(20).y-10)
        elif i == 8:
            coordinates[i] = (shape.part(20).x, shape.part(20).y+30)
        elif i == 9:
            coordinates[i] = (shape.part(21).x+10, shape.part(21).y-10)
        elif i == 10:
            coordinates[i] = (shape.part(21).x+10, shape.part(21).y+30)
            
        elif i == 11:
            coordinates[i] = (shape.part(22).x-10, shape.part(22).y-10)
        elif i == 12:
            coordinates[i] = (shape.part(22).x-10, shape.part(22).y+30)
        elif i == 13:
            coordinates[i] = (shape.part(23).x, shape.part(23).y-10)
        elif i == 14:
            coordinates[i] = (shape.part(23).x, shape.part(23).y+30)
        elif i == 15:
            coordinates[i] = (shape.part(24).x, shape.part(24).y-10)
        elif i == 16:
            coordinates[i] = (shape.part(24).x, shape.part(24).y+30)
        elif i == 17:
            coordinates[i] = (shape.part(25).x, shape.part(25).y-10)
        elif i == 18:
            coordinates[i] = (shape.part(25).x, shape.part(25).y+30)
        elif i == 19:
            coordinates[i] = (shape.part(26).x+15, shape.part(26).y-10)
        elif i == 20:
            coordinates[i] = (shape.part(26).x+15, shape.part(26).y+30)
        
    return coordinates
def visualize_eyes_landmarks(image, shape, colors_eyes=None, alpha=0.75):

    overlay_eyes = image.copy()
    output = np.zeros(overlay_eyes.shape, dtype = np.uint8)
    eyes_cordinates = {}
    FACIAL_LANDMARKS_EYES = OrderedDict([
        ("Right_Eye", (36, 42)),
        ("Left_Eye", (42, 48))
    ])
    if colors_eyes is None:
        colors_eyes = [(255, 255, 255), (255, 255, 255)]
        
    for (i, name) in enumerate(FACIAL_LANDMARKS_EYES.keys()):
        (j, k) = FACIAL_LANDMARKS_EYES[name]
        pts = shape[j:k]
        eyes_cordinates[name] = pts
        hull = cv2.convexHull(pts)
        cv2.drawContours(overlay_eyes, [hull], -1, colors_eyes[i], 1)
        ellipse = cv2.fitEllipse(pts)
        cv2.ellipse(overlay_eyes, ellipse, (255, 255, 255), 1)
        
        cv2.drawContours(output, [hull], -1, colors_eyes[i], -1)
        ellipse = cv2.fitEllipse(pts)
        cv2.ellipse(output, ellipse, (255, 255, 255), -1)
    return output

def visualize_mouth_landmarks(image, shape, colors_mouth=None, alpha=0.75):

    overlay_mouth = image.copy()
    output_mouth = np.zeros(overlay_mouth.shape, dtype = np.uint8)
    mouth_cordinates = {}
    FACIAL_LANDMARKS_MOUTH = OrderedDict([
        ("Mouth_Outline", (48, 61))
    ])
    if colors_mouth is None:
        colors_mouth = [(255, 255, 255)]
        
    for (i, name) in enumerate(FACIAL_LANDMARKS_MOUTH.keys()):
        (j, k) = FACIAL_LANDMARKS_MOUTH[name]
        pts = shape[j:k]
        mouth_cordinates[name] = pts
        hull = cv2.convexHull(pts)
        cv2.drawContours(overlay_mouth, [hull], -1, colors_mouth[i], 1)
        cv2.drawContours(output_mouth, [hull], -1, colors_mouth[i], -1)
    return output_mouth

def visualize_eyebrows_landmarks(image, shape, colors_eyebrows=None, alpha=0.75):

    overlay_eyebrows = image.copy()
    output_eyebrows = np.zeros(overlay_eyebrows.shape, dtype = np.uint8)
    eyebrows_cordinates = {}
    FACIAL_LANDMARKS_EYEBROWS = OrderedDict([
        ("Left_Eyebrow_1", (1, 5)),
        ("Left_Eyebrow_2", (3, 7)),
        ("Left_Eyebrow_3", (5, 9)),
        ("Left_Eyebrow_4", (7, 11)),
        ("Right_Eyebrow_1", (11, 15)),
        ("Right_Eyebrow_2", (13, 17)),
        ("Right_Eyebrow_3", (15, 19)),
        ("Right_Eyebrow_4", (17, 21)),
        
    ])
    if colors_eyebrows is None:
        colors_eyebrows = [(255, 255, 255), (255, 255, 255), (255, 255, 255),
                           (255, 255, 255), (255, 255, 255), (255, 255, 255),
                           (255, 255, 255), (255, 255, 255)]
        
    for (i, name) in enumerate(FACIAL_LANDMARKS_EYEBROWS.keys()):
        (j, k) = FACIAL_LANDMARKS_EYEBROWS[name]
        pts = shape[j:k]
        eyebrows_cordinates[name] = pts
        hull = cv2.convexHull(pts)
        cv2.drawContours(overlay_eyebrows, [hull], -1, colors_eyebrows[i], -1)
        cv2.drawContours(output_eyebrows, [hull], -1, colors_eyebrows[i], -1)
    return output_eyebrows

def visualize_nose_landmarks(image, shape, colors_nose=None, alpha=0.75):

    overlay_nose = image.copy()
    output_nose = np.zeros(overlay_nose.shape, dtype = np.uint8)
    nose_cordinates = {}
    FACIAL_LANDMARKS_NOSE = OrderedDict([
    ("Nose", (27, 36))
    ])
    if colors_nose is None:
        colors_nose = [(255, 255, 255)]
        
    for (i, name) in enumerate(FACIAL_LANDMARKS_NOSE.keys()):
        (j, k) = FACIAL_LANDMARKS_NOSE[name]
        pts = shape[j:k]
        nose_cordinates[name] = pts
        hull = cv2.convexHull(pts)
        cv2.drawContours(overlay_nose, [hull], -1, colors_nose[i], 1)
        cv2.drawContours(output_nose, [hull], -1, colors_nose[i], -1)
    return output_nose


def eyes_masking_call():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmark.dat')
    path = os.path.join('images', 'test1.jpg')
    files = os.listdir('images')
    image = cv2.imread(path)
    rects = detector(image, 1)
    
    for (i, rect) in enumerate(rects):
        shape = predictor(image, rect)
        shape = shape_to_numpy_array(shape)
        output = visualize_eyes_landmarks(image, shape)
        cv2.imwrite('masking_images/output5.jpg',output)


def mouth_masking_call():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmark.dat')
    path = os.path.join('images', 'test1.jpg')
    files = os.listdir('images')
    image = cv2.imread(path)
    rects = detector(image, 1)
    
    for (i, rect) in enumerate(rects):
        shape = predictor(image, rect)
        shape = shape_to_numpy_array(shape)
        output_mouth = visualize_mouth_landmarks(image, shape)
        cv2.imwrite('masking_images/output5.jpg',output_mouth)


def eyebrows_masking_call():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmark.dat')
    path = os.path.join('images', 'test2.jpg')
    files = os.listdir('images')
    image = cv2.imread(path)
    rects = detector(image, 1)
    
    for (i, rect) in enumerate(rects):
        shape = predictor(image, rect)
        shape = shape_to_numpy_array2(shape)
        output_eyebrows = visualize_eyebrows_landmarks(image, shape)
        cv2.imwrite('masking_images/output7.jpg',output_eyebrows)

def nose_masking_call():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmark.dat')
    path = os.path.join('images', 'test1.jpg')
    files = os.listdir('images')
    image = cv2.imread(path)
    rects = detector(image, 1)
    
    for (i, rect) in enumerate(rects):
        shape = predictor(image, rect)
        shape = shape_to_numpy_array(shape)
        output_nose = visualize_nose_landmarks(image, shape)
        cv2.imwrite('masking_images/nose_output.jpg',output_nose)



