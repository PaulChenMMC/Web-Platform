from PIL import Image
import os
import cv2

# 畫 bbox 的 thickness 與 text 的 size
rect_th= 1 #3
text_th=1
text_size=1

# skeleton drawing parameters
circle_r =1   #2
line_th = 2  #6

# get color for different SID
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

# draw skeleton
# 畫線 0 (nose)-3(LEar), 0-4, ...
# For joint indices: https://learnopencv.com/human-pose-estimation-using-keypoint-rcnn-in-pytorch/?ck_subscriber_id=297191382 
PtPairLst = [[0,3], [0, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [11, 13], [13, 15], [11, 12], [12, 14], [14, 16]]

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_skeleton(frame, keyPts, color, radius, thickness):
    # keyPts = the 17 key points of this person (x1, y1, visiablity1), ...
    # frame = the image to draw the skeleton
    for i in range(17):
        x = int(keyPts[i][0])
        y = int(keyPts[i][1])
        cv2.circle(frame, (x, y), radius=radius, color=color, thickness=thickness) 

    # 多畫 2 個點: 5 (LShoulder)-6 (RS) 中間,  11(LHip)-12(RH) 中間
    x5_6, y5_6, visiablity1  = (keyPts[5] + keyPts[6])/2
    x11_12, y11_12, visiablity2  = (keyPts[11] + keyPts[12])/2
    x5_6, y5_6, x11_12, y11_12 = int(x5_6), int(y5_6), int(x11_12), int(y11_12)
    cv2.circle(frame, (x5_6, y5_6), radius=radius, color=color, thickness=thickness) 
    cv2.circle(frame, (x11_12, y11_12), radius=radius, color=color, thickness=thickness) 

    # 畫線 LShoulder-RS 5-6, Left Arm: 5-7-9,  RArm 6-8-10
    # LHip-RHip 11-12, Left Leg: 11-13-15, RLeg: 12-14-16
    # Body 0- (5-6中間)-(11-12 中間) 
    for pointPair in PtPairLst:
        ptIdx1, ptIdx2 = pointPair
        x0, y0 = int(keyPts[ptIdx1][0]), int(keyPts[ptIdx1][1])
        x1, y1 = int(keyPts[ptIdx2][0]), int(keyPts[ptIdx2][1])
        cv2.line(frame, (x0, y0), (x1,y1), color=color, thickness=thickness) 
    x0, y0 = int(keyPts[0][0]), int(keyPts[0][1])
    cv2.line(frame, (x0, y0), (x5_6, y5_6), color=color, thickness=thickness)
    cv2.line(frame, (x5_6, y5_6), (x11_12, y11_12), color=color, thickness=thickness)


def save_image (frame_count, frame, boxes, idx, msg, __width, __height, result_dir):
    # 每次新增或移除 Subject ID 時, 不再詢問使用者,  只是記錄一張 image, 讓使用者可以人工追蹤比對數據檔
    # boxes - the recognized bbox 
    # idx - bbox to be added or removed
    # subjectID - subject ID of this bbox
    # result_dir is global variables
    x1,y1,x2,y2 = boxes[idx]
    x1,y1,x2,y2 = int(x1*__width), int(y1*__height), int(x2*__width), int(y2*__height)
    cv2.rectangle(frame,(x1, y1), (x2,y2), color=(255, 255,255), thickness=rect_th)
    
    #在 img 底部黑底白字顯示 msg
    t_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
    cv2.rectangle(frame,(x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), (0, 0, 0),-1)
    cv2.putText(frame, msg, (x1, y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
   
    fname = "frame_" + str(frame_count)+ ".jpg"
    im = Image.fromarray(frame)
    #產出檔案
    im.save(os.path.join(result_dir, fname))


def Draw_tracking_results_to_frame(active_tracklets, frame, __width, __height):
    for i in range(len(active_tracklets)):
        x1,y1,x2,y2 = active_tracklets[i]['boxes'][0]
        x1,y1,x2,y2 = int(x1*__width), int(y1*__height), int(x2*__width), int(y2*__height)
        sid=active_tracklets[i]['sid']
        color=compute_color_for_labels(sid+2)
        t_size = cv2.getTextSize(str(sid), cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(frame,(x1, y1), (x2,y2), color=color, thickness=rect_th)
        cv2.rectangle(frame,(x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), color,-1)
        cv2.putText(frame,str(sid),(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255],2) 
        draw_skeleton(frame,active_tracklets[i]['skeleton'][0], color, radius=circle_r, thickness=line_th)
    return frame