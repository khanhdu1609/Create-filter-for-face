import cv2
import pathlib
import time
import torch
import torch
import numpy as np
import csv
from PIL import Image
from finetune_resnet18 import ResNet18Finetune
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class Webcam:
    def __init__(self):
        self.cap = cv2.VideoCapture()
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        self.face_model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/face_model.pt') 
        self.landmarks_model = ResNet18Finetune()
        checkpoint = torch.load("model/face_landmarks.pt", map_location='cpu')
        self.landmarks_model.load_state_dict(checkpoint)
    def draw_point(self, img, position):
        cv2.circle(img, position, 1, (0, 0, 255), cv2.FILLED, cv2.LINE_AA, 0)
    def load_landmark_mask(self, annotation_file):
        '''
        Load landmarks from annotation file
        '''
        with open(annotation_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            points = []
            for i, row in enumerate(csv_reader):
                try:
                    x, y = int(row[0]), int(row[1])
                    points.append((x, y))
                except ValueError:
                    continue
        return points

    def get_bb(self, original_img):
        '''
        Detect face from frame
        '''
        self.face_model.cuda()
        self.face_model.eval()
        with torch.no_grad():
            result = self.face_model(original_img)
        bb = result.xyxy[0].cpu().numpy().astype(int)
        return bb
    def get_face(self, bb, original_img):
        if bb.shape[0] != 0:
            face = original_img[bb[0][1]: bb[0][3],bb[0][0]:bb[0][2]]
        return face
    def get_landmarks(self, face, bb):
        face = Image.fromarray(face)
        face = face.convert('RGB')
        transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        face = transformer(face) 
        face = TF.normalize(face, [0.5], [0.5])
        face = face.unsqueeze(0)
        self.landmarks_model.eval()
        with torch.no_grad():
            prediction = (self.landmarks_model(face) + 0.5) * 224
            prediction = prediction.view(-1, 68, 2)

        prediction[0][:, 0] = prediction[0][:, 0] / (224/int(bb[0][2] - bb[0][0])) + bb[0][0]
        prediction[0][:, 1] = prediction[0][:, 1] / (224/int(bb[0][3] - bb[0][1])) + bb[0][1]
        landmark_list = []
        for point in prediction[0]:
            landmark_list.append((int(point[0]), int(point[1])))
        return landmark_list

    def triangle_list(self, original_img, landmarks_list):
        subdiv = cv2.Subdiv2D((0, 0, original_img.shape[1], original_img.shape[0]))
        for point in landmarks_list:
            subdiv.insert((int(point[0]), int(point[1])))
        triangleList = subdiv.getTriangleList()
        return triangleList
    def put_on_mask(self, triangleList, original_img, mask, mask_annotation_points,landmark_list):
        for triangle in triangleList:
        #Get mask's annotated point 
            tri1 = np.float32([mask_annotation_points[landmark_list.index((int(triangle[0]), int(triangle[1])))], mask_annotation_points[landmark_list.index((int(triangle[2]), int(triangle[3])))], mask_annotation_points[landmark_list.index((int(triangle[4]), int(triangle[5])))]])
            #Get image's triangle list
            tri2 = np.float32([[triangle[0], triangle[1]], [triangle[2], triangle[3]], [triangle[4], triangle[5]]])
            
            r1 = cv2.boundingRect(tri1)
            r2 = cv2.boundingRect(tri2)

            tri1Cropped = []
            tri2Cropped = []
            for i in range(0, 3):
                tri1Cropped.append(((tri1[i][0] - r1[0]), (tri1[i][1] - r1[1])))
                tri2Cropped.append(((tri2[i][0] - r2[0]), (tri2[i][1] - r2[1])))
            maskCropped = mask[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
            # print(maskCropped.shape)
            warpMat = cv2.getAffineTransform(np.float32(tri1Cropped), np.float32(tri2Cropped))
            imgCropped = cv2.warpAffine(maskCropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            temp = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
            cv2.fillConvexPoly(temp, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0)
            imgCropped = imgCropped * temp


            original_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = original_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - temp )
            original_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = original_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + imgCropped
        return original_img