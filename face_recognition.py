import cv2
import numpy as np
import os
from cv2 import face
# 載入人臉追蹤模型
detector = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')  
# 啟用訓練人臉模型方法
recog = cv2.face.LBPHFaceRecognizer_create()      
faces = []   
ids = []     

folder_path ="./Datahouse/FaceData"
all_folder = os.listdir(folder_path)

folder_names = [item for item in all_folder if os.path.isdir(os.path.join(folder_path, item))]
#讀取全部資料夾名稱
for folder_name in folder_names:
    files = os.listdir(os.path.join(folder_path, folder_name))
    #抓取資料夾內全部照片
    for  file in files:
        print(os.path.join(folder_path,folder_name, file))
        img = cv2.imread(os.path.join(folder_path,folder_name, file))           
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_np = np.array(gray,'uint8')               
        face = detector.detectMultiScale(gray)        
        for(x,y,w,h) in face:
            faces.append(img_np[y:y+h,x:x+w])         
            ids.append(folder_name)                             

ids = np.array(ids)
print('training...')                              
recog.train(faces, ids.astype(int))                 
recog.save('./Datahouse/FaceData/face.yml')
print('ok!')