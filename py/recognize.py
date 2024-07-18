
from PIL import Image

import pyttsx3
import math
import json
import cv2

# # initialize
with open(r"C:\Users\chunc\Desktop\python\Flask\Datahouse\FaceData/List.Json", "r", encoding="utf-8") as f:
    data = json.load(f)

name = {key: value["姓名"] for key, value in data.items()}
cascade_path = "./xml/haarcascade_frontalface_default.xml"
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face.yml")

face_cascade = cv2.CascadeClassifier(cascade_path)

BODY_PARTS = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "RHip": 8,
    "RKnee": 9,
    "RAnkle": 10,
    "LHip": 11,
    "LKnee": 12,
    "LAnkle": 13,
    "REye": 14,
    "LEye": 15,
    "REar": 16,
    "LEar": 17,
    "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"],
    ["Neck", "LShoulder"],
    ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"],
    ["LShoulder", "LElbow"],
    ["LElbow", "LWrist"],
    ["Neck", "RHip"],
    ["RHip", "RKnee"],
    ["RKnee", "RAnkle"],
    ["Neck", "LHip"],
    ["LHip", "LKnee"],
    ["LKnee", "LAnkle"],
    ["Neck", "Nose"],
    ["Nose", "REye"],
    ["REye", "REar"],
    ["Nose", "LEye"],
    ["LEye", "LEar"]
]

net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# args
IN_WIDTH = 368
IN_HEIGHT = 368
THR = .2


# 提示模組參數
engine = pyttsx3.init()
rate = engine.getProperty("rate")
engine.setProperty("rate", rate-50)

# 計數器
face_trigger_threshold = 10
pose_trigger_threshold = 30

def face_recognition(frame: Image) -> str:
    face_State=False
    """Generates name for face recognition

    Args:
        frame (Image): camera capture
        text (str): name of the face recognition

    Returns:
        str: Name of recognition
    """
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    now_people = ""
    Face_count = 0
    img = frame
    text = ""

    img = cv2.resize(img, (frameWidth, frameHeight))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        if Face_count == face_trigger_threshold:
            text = name[str(idnum)]
            engine.say(text + "你好，請到指定位子")
            engine.runAndWait()
            print("人臉鎖定為:"+text)
            face_State=True
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            idnum, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 60:
                text = name[str(idnum)] + f"{confidence:.2f}"
                Face_count += 1
                if now_people == "":
                    now_people = name[str(idnum)]
                    # print("現在偵測人員為:"+name[str(idnum)])
                elif now_people != name[str(idnum)]:
                    now_people = name[str(idnum)]
                    # print("現在偵測人員為:"+name[str(idnum)])
            else:
                text = "???" + f"{confidence:.2f}"
                Face_count = 0

    return text,face_State



def pose_recognizer(frame: Image, text: str = "") -> Image:
    """pose recognition

    Args:
        frame (Image): camera capture
        text (str): name

    Returns:
        Image: frame after recognition (putting text)
    """
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    Pose_count = 0

    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (IN_WIDTH, IN_HEIGHT),
                                       (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]
    assert (len(BODY_PARTS) == out.shape[1])
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > THR else None)

    RShoulder_index = BODY_PARTS["RShoulder"]
    LShoulder_index = BODY_PARTS["LShoulder"]
    hip_index = BODY_PARTS["RHip"]
    knee_index = BODY_PARTS["RKnee"]
    ankle_index = BODY_PARTS["RAnkle"]

    if pose_trigger_threshold == Pose_count:
        cv2.putText(frame, text + "_" + pose, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        if pose == "front view-Sitting" or pose == "Side view-Sitting":
            pose = "坐下"
        else:
            pose = "站立"
        engine.say("要開始測試30秒坐站")
        engine.say("接下來會倒數三秒")
        engine.say("三二一請開始")
        engine.runAndWait()
    else:
        if points[RShoulder_index] and points[LShoulder_index]:
            RShoulder_x, RShoulder_y = points[RShoulder_index]
            LShoulder_x, LShoulder_y = points[LShoulder_index]
            distance_RShoulder_to_LShoulder = math.sqrt(
                (RShoulder_x - LShoulder_x)**2 + (RShoulder_y - LShoulder_y)**2)
            if points[hip_index] and points[knee_index] and points[ankle_index]:
                hip_x, hip_y = points[hip_index]
                knee_x, knee_y = points[knee_index]
                ankle_x, ankle_y = points[ankle_index]

                if distance_RShoulder_to_LShoulder > 50:
                    distance_hip_to_knee = math.sqrt(
                        (knee_x - hip_x)**2 + (knee_y - hip_y)**2)
                    distance_knee_to_ankle = math.sqrt(
                        (knee_x - ankle_x)**2 + (knee_y - ankle_y)**2)

                    if distance_hip_to_knee*1.2 > distance_knee_to_ankle:
                        pose = "front view-Standing"
                        Pose_count = Pose_count+1
                        cv2.putText(
                            frame, text+pose, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        if pose != "front view-Standing":
                            Pose_count = 0
                    else:
                        pose = "front view-Sitting"
                        Pose_count = Pose_count+1
                        cv2.putText(
                            frame, text+pose, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        if pose != "front view-Sitting":
                            Pose_count = 0

                else:
                    vec_ab = [hip_x - knee_x, hip_y - knee_y]
                    vec_ac = [ankle_x - knee_x, ankle_y - knee_y]
                    dot_product = vec_ab[0] * \
                        vec_ac[0] + vec_ab[1] * vec_ac[1]
                    mag_ab = math.sqrt(vec_ab[0] ** 2 + vec_ab[1] ** 2)
                    mag_ac = math.sqrt(vec_ac[0] ** 2 + vec_ac[1] ** 2)
                    cos_theta = dot_product / (mag_ab * mag_ac)
                    theta_rad = math.acos(cos_theta)
                    theta_deg = math.degrees(theta_rad)
                    if theta_deg > 120:
                        pose = "Side view-Standing"
                        Pose_count = Pose_count+1
                        cv2.putText(
                            frame, text+pose, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        if pose != "Side view-Standing":
                            Pose_count = 0
                    else:
                        pose = "Side view-Sitting"
                        Pose_count = Pose_count+1
                        cv2.putText(
                            frame, text+pose, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        if pose != "Side view-Sitting":
                            Pose_count = 0
            else:
                if distance_RShoulder_to_LShoulder > 80:
                    pose = "front view"
                    cv2.putText(frame, text + "_" + pose, (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    if pose != "front view":
                        # print(pose)
                        Pose_count = 0
                else:
                    pose = "side view"
                    cv2.putText(frame, text + "_" + pose, (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    if pose != "side view":
                        # print(pose)
                        Pose_count = 0
        else:
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


if name == "__main__":
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    while True:
        recognized_name = face_recognition(frame=frame)
        image = pose_recognizer(frame=frame, text=recognized_name)
        cv2.imwrite("./asd.png", image)

# while True:
#     _, frame = cap.read()
#     img = cv2.resize(frame, (368, 368))
#     print(img)
    # cv2.imshow("OpenPose using OpenCV", img)
