from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from torchvision import transforms
from facenet_pytorch import MTCNN
import torch
import cv2
from PIL import Image
import numpy as np

from datetime import datetime
import sys
import os
import glob
import time
import joblib

from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from utils import *


class Recognition:
    def __init__(self, url=''):
        # self.AUG_IMG_DATA = 'Dataset\Augmented_data'
        # self.ORI_IMG_DATA = 'Dataset\Images_crop'
        # self.NEW_DATA = 'Dataset\Images_new'
        self.IMG_DATA = 'Dataset\Images'
        self.CHECKPOINT_PATH = 'Dataset\Checkpoint'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(
            classify=False,
            pretrained="casia-webface"
        ).to(self.device)
        self.model.eval()
        self.mtcnn = MTCNN(margin = 20, select_largest=True, keep_all=False, post_process=False, device = self.device)
        self.url = url

        model_path = os.path.join(self.CHECKPOINT_PATH, 'svm.pkl')
        label_path = os.path.join(self.CHECKPOINT_PATH, 'labels.pkl')
        self.svm = joblib.load(model_path)
        self.labels = joblib.load(label_path)
        self.in_encoder = Normalizer(norm='l2')
        
    def add_new(self,usr_name, count=50, leap=1):
        # usr_name = input("Input ur name: ")
        USR_PATH = os.path.join(self.IMG_DATA, usr_name)
        
        if self.url:
            cap = cv2.VideoCapture('http://10.61.24.167:4747/video')
        else:
            cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        while cap.isOpened() and count:
            isSuccess, frame = cap.read()
            if self.url:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if self.mtcnn(frame) is not None and leap%2:
                path = str(USR_PATH + f'/{count}.JPG')
                face_img = self.mtcnn(frame, save_path = path)
                count-=1
            leap+=1
            cv2.imshow('Face Capturing', frame)
            if cv2.waitKey(1)&0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # def augmentation_data(self):
    #     transform_data = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    #     transforms.ToPILImage(),
    #     ])
    #     for usr in os.listdir(self.IMG_DATA):
    #         usr_path = os.path.join(self.IMG_DATA, usr)
    #         aug_usr_path = os.path.join(self.AUG_IMG_DATA, usr)
    #         os.mkdir(aug_usr_path)
    #         for img in os.listdir(usr_path):
    #             img_path = os.path.join(usr_path, img)
    #             ori_img = Image.open(img_path)
    #             aug_img = transform_data(ori_img)
    #             ori_img.save(aug_usr_path + '/' + img)
    #             aug_img.save(aug_usr_path + '/_' + img)

    def svm_Classify(self):
        embeddings = []
        names = []
        for usr in os.listdir(self.IMG_DATA):
            for file in glob.glob(os.path.join(self.IMG_DATA, usr)+'/*.JPG'):
                try:
                    img = Image.open(file)
                except:
                    continue
                with torch.no_grad():
                    embed = self.model(trans(img).unsqueeze(0).to(self.device))

                embeddings.append(np.array(embed.cpu()[0])) # 1 cai list n cai [1,512]
                names.append(usr)
        
        X_train, X_test, y_train, y_test = train_test_split(embeddings, names, test_size=.1, random_state=42)

        # 
        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(X_train)
        testX = in_encoder.transform(X_test)
        out_encoder = LabelEncoder()
        out_encoder.fit(y_train)
        trainy = out_encoder.transform(y_train)
        testy = out_encoder.transform(y_test)
        model = SVC(kernel='linear', probability=True)
        model.fit(trainX, trainy)

        # predict
        print(trainX)
        yhat_train = model.predict(trainX)
        yhat_test = model.predict(testX)
        # score
        score_train = accuracy_score(trainy, yhat_train)
        score_test = accuracy_score(testy, yhat_test)
        # summarize
        print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
        model_path = os.path.join(self.CHECKPOINT_PATH, 'svm.pkl')
        label_path = os.path.join(self.CHECKPOINT_PATH, 'labels.pkl')
        joblib.dump(model, model_path)
        joblib.dump(out_encoder, label_path)

        model_path = os.path.join(self.CHECKPOINT_PATH, 'svm.pkl')
        label_path = os.path.join(self.CHECKPOINT_PATH, 'labels.pkl')
        self.svm = joblib.load(model_path)
        self.labels = joblib.load(label_path)

    def face_reg(self, frame):
        boxes, prob = self.mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                bbox = list(map(int,box.tolist()))
                face = extract_face(bbox, frame)
                embed = self.model(trans(face).unsqueeze(0).to(self.device))
                embed = embed.cpu().detach().numpy()
                embed = self.in_encoder.transform(embed)
                y_pred_encoded = self.svm.predict(embed)
                y_pred = self.labels.inverse_transform(y_pred_encoded)
                
                frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                frame = cv2.putText(frame, str(y_pred[0]), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 1, cv2.LINE_8)

            return frame, y_pred[0], face
        return frame

# if __name__ == '__main__':
#     reg = Recognition()
#     for i in os.listdir('Nguyen Duc Huy'):
#         path = os.path.join('Nguyen Duc Huy', i)
#         image = cv2.imread(path)
#         try:
#             frame,y_pred = reg.face_reg(image)
#             print(y_pred)
#             print(y_pred)
#         except:
#             print('Error:', path)
#     image = cv2.imread('Ngoc-Trinh-2.JPG')
#     frame,y_pred,y_pred_encoded = reg.face_reg(image)
#     print(y_pred)
#     print(y_pred_encoded)
    # reg.add_new()
    # reg.augmentation_data()
    # reg.svm_Classify()