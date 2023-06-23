from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from torchvision import transforms
from facenet_pytorch import MTCNN
import torch
import cv2
from PIL import Image
import numpy as np

from datetime import datetime
import os
import glob
import time
import joblib
import shutil

from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from utils import *

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Your device is:', device)
 

class Recognition:
    def __init__(self, AUG_IMG_DATA, ORI_IMG_DATA, NEW_DATA, CHECKPOINT_PATH, device, url=''):
        self.AUG_IMG_DATA = AUG_IMG_DATA
        self.ORI_IMG_DATA = ORI_IMG_DATA
        self.NEW_DATA = NEW_DATA
        self.CHECKPOINT_PATH = CHECKPOINT_PATH
        self.device = device
        self.model = InceptionResnetV1(
            classify=False,
            pretrained="casia-webface"
        ).to(device)
        self.model.eval()
        self.mtcnn = MTCNN(margin = 20, select_largest=True, keep_all=False, post_process=False, device = self.device)
        self.url = url

    def add_new(self, count=50, leap=1):
        usr_name = input("Input ur name: ")
        USR_PATH = os.path.join(self.NEW_DATA, usr_name)
        
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

    def augmentation_data(self):
        transform_data = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToPILImage(),
        ])
        for usr in os.listdir(self.NEW_DATA):
            usr_path = os.path.join(self.NEW_DATA, usr)
            aug_usr_path = os.path.join(self.AUG_IMG_DATA, usr)
            os.mkdir(aug_usr_path)
            for img in os.listdir(usr_path):
                img_path = os.path.join(usr_path, img)
                ori_img = Image.open(img_path)
                aug_img = transform_data(ori_img)
                ori_img.save(aug_usr_path + '/' + img)
                aug_img.save(aug_usr_path + '/_' + img)

    def svm_Classify(self):
        embeddings = []
        names = []
        for usr in os.listdir(self.AUG_IMG_DATA):
            for file in glob.glob(os.path.join(self.AUG_IMG_DATA, usr)+'/*.JPG'):
                try:
                    img = Image.open(file)
                except:
                    continue
                with torch.no_grad():
                    embed = self.model(trans(img).unsqueeze(0).to(device))

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
        model = SVC(kernel='linear')
        model.fit(trainX, trainy)

        # predict
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

    def face_reg(self):
        model_path = os.path.join(self.CHECKPOINT_PATH, 'svm.pkl')
        label_path = os.path.join(self.CHECKPOINT_PATH, 'labels.pkl')
        svm = joblib.load(model_path)
        labels = joblib.load(label_path)
        in_encoder = Normalizer(norm='l2')
        if self.url:
            cap = cv2.VideoCapture(self.url)
        else:
            cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,960)
        while cap.isOpened():
            isSuccess, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if self.url:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if isSuccess:
                boxes, prob = self.mtcnn.detect(frame)
                try:
                    print(len(boxes))
                except:
                    print(0)
                if boxes is not None:
                    # box = boxes[0]
                    for box in boxes:
                        bbox = list(map(int,box.tolist()))
                        face = extract_face(bbox, frame)
                        embed = self.model(trans(face).unsqueeze(0).to(device))
                        embed = embed.cpu().detach().numpy()
                        embed = in_encoder.transform(embed)
                        y_pred_encoded = svm.predict(embed)

                        y_pred = labels.inverse_transform(y_pred_encoded)
                        
                        frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                        frame = cv2.putText(frame, str(y_pred), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 1, cv2.LINE_8)

            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1)&0xFF == 27:
                break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    AUG_IMG_DATA = 'Dataset/Augmented_data'
    ORI_IMG_DATA = 'DatasetImages_crop'
    NEW_DATA = 'DatasetImages_new'
    CHECKPOINT_PATH = 'Dataset/Checkpoint'

    reg = Recognition(AUG_IMG_DATA, ORI_IMG_DATA, NEW_DATA, CHECKPOINT_PATH, device)
    


    reg.face_reg()
    # reg.add_new()
    # reg.augmentation_data()
    # reg.svm_Classify()

