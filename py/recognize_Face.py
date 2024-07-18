import json
import cv2

with open("./Datahouse/FaceData/List.Json", "r", encoding="utf-8") as f:
    data = json.load(f)

name = {key: value["姓名"] for key, value in data.items()}
cascade_path = "./xml/haarcascade_frontalface_default.xml"
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./Datahouse/FaceData/face.yml")
face_cascade = cv2.CascadeClassifier(cascade_path)
#連續幾張圖片為同一人，顯示判定成功
face_trigger_threshold = 15

def face_recognition(frame) :
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    now_people = ""
    Face_count = 0
    img = frame
    txt1=""
    img = cv2.resize(img, (frameWidth, frameHeight))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        if Face_count == face_trigger_threshold:
            now_people=name[str(idnum)]
            txt1=str(idnum)
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            idnum, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            # print(f"{confidence:.2f}")
            # print(Face_count)
            if confidence < 130:
                now_people=name[str(idnum)]
                txt1=str(idnum)
                if now_people == name[str(idnum)]:
                    Face_count += 1
                elif now_people != name[str(idnum)]:
                    Face_count = 0
            else:
                Face_count = 0
    return now_people,txt1