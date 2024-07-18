import cv2
import os

#資料夾路徑
folder_path =os.getcwd()
#輸入影片路徑
video_path = os.path.join(folder_path, '1.mp4')
#輸出
output_folder = os.path.join(folder_path, 'Video_Split')
#讀取影片
cap = cv2.VideoCapture(video_path)
print(video_path)
frame_count = 0
image_count = 1
#多少Frame 擷取照片
frame_ShotCut=5
#讀取影片
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    target_frame = int(frame_ShotCut * image_count)
    if frame_count == target_frame:
        image_count += 1
        output_path = os.path.join(output_folder, f"{image_count}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"Saved image {image_count}")

cap.release()
