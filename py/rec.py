import json
import cv2
import math

#人臉模組
with open("./Datahouse/FaceData/List.Json", "r", encoding="utf-8") as f:
    data = json.load(f)
name = {key: value["姓名"] for key, value in data.items()}
cascade_path = "./xml/haarcascade_frontalface_default.xml"
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./Datahouse/FaceData/face.yml")
face_cascade = cv2.CascadeClassifier(cascade_path)
#姿勢模組
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
net = cv2.dnn.readNetFromTensorflow("./Datahouse/FaceData/graph_opt.pb")
IN_WIDTH = 368
IN_HEIGHT = 368
THR = .2

#連續幾張圖片為同一人，顯示判定成功 目前每0.5秒傳第一張照片
face_trigger_threshold = 50
pose_trigger_threshold = 50



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
    confidence=0
    for (x, y, w, h) in faces:
        if Face_count == face_trigger_threshold:
            now_people=name[str(idnum)]
            txt1=str(idnum)
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            idnum, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            print(f"{confidence:.2f}")
            # print(Face_count)
            if confidence <130:
                now_people=name[str(idnum)]
                txt1=str(idnum)
                if now_people == name[str(idnum)]:
                    Face_count += 1
                elif now_people != name[str(idnum)]:
                    Face_count = 0
            else:
                Face_count = 0
    return now_people,txt1,confidence


def pose_recognition(frame,Pose_count1):
    pose=""
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    Pose_count = Pose_count1

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
    try:
        if points[RShoulder_index] and points[LShoulder_index]:
            RShoulder_x, RShoulder_y = points[RShoulder_index]
            LShoulder_x, LShoulder_y = points[LShoulder_index]

            distance_RShoulder_to_LShoulder = math.sqrt((RShoulder_x - LShoulder_x)**2 + (RShoulder_y - LShoulder_y)**2)

            if points[hip_index] and points[knee_index] and points[ankle_index]:
                hip_x, hip_y = points[hip_index]
                knee_x, knee_y = points[knee_index]
                ankle_x, ankle_y = points[ankle_index]

                distance_hip_to_knee = math.sqrt((knee_x - hip_x)**2 + (knee_y - hip_y)**2)
                distance_knee_to_ankle = math.sqrt((knee_x - ankle_x)**2 + (knee_y - ankle_y)**2)

                vec_ab = [hip_x - knee_x, hip_y - knee_y]
                vec_ac = [ankle_x - knee_x, ankle_y - knee_y]
                dot_product = vec_ab[0] * \
                    vec_ac[0] + vec_ab[1] * vec_ac[1]
                mag_ab = math.sqrt(vec_ab[0] ** 2 + vec_ab[1] ** 2)
                mag_ac = math.sqrt(vec_ac[0] ** 2 + vec_ac[1] ** 2)
                cos_theta = dot_product / (mag_ab * mag_ac)
                theta_rad = math.acos(cos_theta)
                theta_deg = math.degrees(theta_rad)
                #判斷正面 側面
                # if distance_RShoulder_to_LShoulder > 30:
                if distance_hip_to_knee*1.2 > distance_knee_to_ankle:
                    pose = "front view-Standing"
                    # print(pose)
                    # Pose_count = Pose_count+1
                    if pose != "front view-Standing":
                        Pose_count = 0
                else:
                    pose = "front view-Sitting"
                    Pose_count = Pose_count+1
                    if pose != "front view-Sitting":
                        Pose_count = 0
                # else:
                #     if theta_deg > 120:
                #         pose = "Side view-Standing"
                #         # Pose_count = Pose_count+1
                #         if pose != "Side view-Standing":
                #             Pose_count = 0
                #     else:
                #         pose = "Side view-Sitting"
                #         # Pose_count = Pose_count+1
                #         if pose != "Side view-Sitting":
                #             Pose_count = 0
        else:
            if distance_RShoulder_to_LShoulder > 30:
                pose = "front view"
                # print(pose)
                if pose != "front view":
                    Pose_count = 0
            else:
                pose = "side view"
                # print(pose)
                if pose != "side view":
                    Pose_count = 0
    except :
        pass

    return pose,Pose_count



