from facenet_pytorch import fixed_image_standardization
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms

import torch
import numpy as np
import pandas as pd
from unidecode import unidecode
import cv2
from PIL import Image

import pickle


def cosine_simi(detected_emb, embeds_list):
    simi_scores = []
    for emb in embeds_list:
        simi = cosine_similarity(detected_emb, emb)
        simi_scores.append(simi)
    return simi_scores

def most_similarity(embed_vecs, vec, labels):
  embed_vecs = torch.stack(embed_vecs).cpu().numpy()
  vec = vec.detach().numpy()
  sim = cosine_simi(vec, embed_vecs)
  sim = np.squeeze(sim, axis = 1)
  argmax = np.argsort(sim)[::-1][:1]
  label = [labels[idx] for idx in argmax][0]
  max_sim = sim[argmax]
  return float(max_sim), label

def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            # fixed_image_standardization
        ])
    return transform(img)

def extract_face(box, img, margin=20):
    face_size = 160
    img_size = (1280, 960)
    margin = [
        margin * (box[2] - box[0]) / (face_size - margin),
        margin * (box[3] - box[1]) / (face_size - margin),
    ] 
    box = [ #box[0] và box[1] là tọa độ của điểm góc trên cùng trái
        int(max(box[0] - margin[0] / 2, 0)), #nếu thêm vào margin bị ra khỏi rìa ảnh => đưa về điểm 0
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img_size[0])), #nếu thêm vào margin bị ra khỏi rìa ảnh => đưa về tọa độ của ảnh gốc
        int(min(box[3] + margin[1] / 2, img_size[1])),
    ] #tạo margin mới bao quanh box cũ
    img = img[box[1]:box[3], box[0]:box[2]]
    face = cv2.resize(img,(face_size, face_size), interpolation=cv2.INTER_AREA)
    face = Image.fromarray(face)
    return face

def take_info(df, name):
    df['Name_nosight'] = df['Name'].apply(lambda x: unidecode(x))
    res = df[df['Name_nosight'] == name]
    return res['roll_no'], res['Name'], res['Class'], res['Status'], res['Time'], res['Note']

# def attendance(df, date, name):
    