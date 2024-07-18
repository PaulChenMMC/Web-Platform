
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import pandas as pd
from py.func import Create_Folder_With_SID,Data_Calibration
from py.keypointRCNN import Run_Keypiont_RCNN
from py.rec import face_recognition,pose_recognition
from py.pose_analysis import get_Pose
from py.Draw import Drawing
import subprocess
import cv2
import numpy as np
import socketio
from flask_sslify import SSLify
from flask_socketio import SocketIO,emit
from base64 import b64decode
from PIL import Image
import io

app = Flask(__name__)
sslify = SSLify(app)
app.config["SECRET_KEY"] = "!@#$%^&*()"

app.config["Face"]=""
app.config["Pose"]=""
app.config["Pose_Count"]=0
app.config["Start"]=True
app.config["SID"]=""

socketio = SocketIO(app, debug=True, cors_allowed_origins="*")


# 判斷是否接收到資料
@socketio.on("testConnection")
def test_connection(data: dict):
    print(data)

#重製資料
@socketio.on("Reset")    
def test_connection(data: str):
    app.config["Face"]=data["Face"]
    app.config["Pose"]=data["Pose"]
    app.config["Pose_Count"]=data["Pose_Count"]
    app.config["Start"]=data["Start"]
    app.config["SID"]=data["SID"]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cluster_visualization')
def cluster_visualization():
    if os.path.exists('./static/1ST.png'):
        os.remove('./static/1ST.png')
    return render_template('cluster_visualization.html')

#按下開始錄製 回傳影像分析
@socketio.on("image")
def image(data: str):
    if type(data) != str:
        return
    #將字串轉為圖片
    image_data = b64decode(data.replace("data:image/jpeg;base64,", ""))
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    cv2_image = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    #進行臉部識別
    if app.config["Start"]: 
        if app.config["Face"]=="":
            app.config["Face"],app.config["SID"],AA=face_recognition(cv2_image)
            if app.config["Face"]:
                print(AA)
                print("臉部辨識成功 : "+app.config["Face"] )
                socketio.emit('get_SID', {'SID': app.config["SID"]})
                socketio.emit('play_sound', {'sound': '/static/'+app.config["SID"]+'.wav'})
        if app.config["Face"] and app.config["Pose_Count"]<10:
            app.config["Pose"],app.config["Pose_Count"]=pose_recognition(cv2_image,app.config["Pose_Count"])
        if app.config["Face"] and app.config["Pose_Count"]>=10:
            print("姿勢辨識成功")
            socketio.emit('play_Pose', {'sound': '/static/Pose.wav'})
            app.config["Start"]=False
            print("開始錄影")
    return "Response(200)"

# 開始辨識
@app.route('/start_recording', methods=['POST'])
def start_recording():
    return 'Recording started'

# 停止錄影並上傳影片&後續分析
@socketio.on("Finish")
@app.route('/upload_video', methods=['POST'])
def stop_recording():
    # # #判斷是否有對應SID，沒有則產生對應資料夾
    SID = request.form.get('SID')
    Record_type = request.form.get('Record_type')
    file_path = os.path.join("./Datahouse/Keypiont", str(Record_type))
    RawData_Path,Time=Create_Folder_With_SID(file_path,SID)
    print(RawData_Path)
    print(file_path)
    print(SID)
    print("產生資料夾")
    # print(Folder_SID)
    # # 獲取影片
    video = request.files['video']
    if video :
        filename = secure_filename(video.filename)
        video.save(os.path.join(RawData_Path, filename))
        print("上傳影片")
        #進行轉檔webm->mp4
        input_file = os.path.join(RawData_Path, filename)
        output_file =os.path.join(RawData_Path, 'recorded_video.mp4')
        command = ["./ffmpeg-2023-06-27-git-9b6d191a66-essentials_build/bin/ffmpeg.exe", '-i', input_file, '-r', '60', output_file]
        subprocess.run(command, check=True)
        print("完成轉檔")
        # try:
        #執行RCNN  

        Run_Keypiont_RCNN(output_file,RawData_Path)
        print("完成RCNN")
        #執行校正，默認Video_SID1
        Data_Calibration_Path=Data_Calibration(1,RawData_Path)
        print("完成校正")
        #抓取每次坐站 座標 填入Result
        get_Pose(Data_Calibration_Path,RawData_Path,SID,Time,file_path)
        print("填入資料")
        # except:
        #     print("校正 執行失敗")
        return'OK'
    return 'Fail'

#獲取資料夾檔案數量
@app.route('/get_folders', methods=['POST'])
def get_folders():
    record_type = request.form.get('Record_type')
    folder_path = os.path.join('./Datahouse/Keypiont', record_type)
    folders = os.listdir(folder_path)
    return jsonify(folders)

#執行查詢
@app.route('/process_data', methods=['POST'])
def process_data():
    data = request.get_json()
    record_type = data.get('Record_type')
    select_id = data.get('Select_ID')
    start_date = data.get('start_date')

    state=Drawing(select_id,record_type,start_date)  
    if state:
        print("產生PNG")
        imgName='Merge.png'
    else:
        print("產生PNG 執行失敗")
        imgName='Nan.png'

    return jsonify({'new_image_filename': imgName})

if __name__ == '__main__':
    socketio.run(app, host="192.168.18.7", port="8051", debug=True,certfile='cert.pem', keyfile='key.pem',server_side=True)
