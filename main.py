import cv2
import pathlib
import time
import matplotlib
import matplotlib.pyplot as plt
import torch
from PIL import Image
from resnet18 import ResNet18
from finetune_resnet18 import ResNet18Finetune
import torchvision.transforms as transforms
import dlib
import numpy as np
from webcam import Webcam
from activity import draw_point

cap = cv2.VideoCapture(0)
webcam = Webcam()
mask_annotation_points = webcam.load_landmark_mask('mask/annotation.csv')
mask = cv2.imread('mask/obama.jpg')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    bb = webcam.get_bb(frame)
    if bb.shape[0] != 0:
        face = webcam.get_face(bb, frame)
        landmarks_list = webcam.get_landmarks(face, bb)
        triangleList  = webcam.triangle_list(frame, landmarks_list)        
        frame = webcam.put_on_mask(triangleList, frame, mask, mask_annotation_points, landmarks_list)
    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release
cv2.destroyAllWindows()