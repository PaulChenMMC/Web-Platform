import numpy as np
from scipy.optimize import linear_sum_assignment
import os
import datetime
import imageio
import pandas as pd

import py.My_Drawing_functions as DF

#路徑
class My_Simple_SubjectTracking():
    def  __init__(self,folder_path):
        super(My_Simple_SubjectTracking, self).__init__()
 
        self.redundant_box_pixels = 5 
        self.score_threshold_to_AddNewSubject = 0.98 #starting from 2nd frame, we will use a stricted threshold to add subject

        self.frame = None   #顯示 key pt tracking 結果之 current frame
        self.frame_count = 1 
        self.subjectID = 0  #畫在 img 上之 subject ID 號碼

        #用來做 subject tracking 之 list
        self.active_tracklets = []  #目前追蹤中
        self.finished_tracklets = [] #完成追蹤
        self.SubjectTrackingResults = []
        self.JointDetectionResults = []

        # subject tracking 用到的 data structure
        self.prev_indices=[]
        self.boxes_indices=[]
        self.boxes=[]
        self.bboxScore=[]
        self.skeleton=[]

        # 紀錄 key pt 辨識結果的 video + csv files
        current_time = datetime.datetime.now()

        self.result_dir = folder_path
        os.makedirs(self.result_dir, exist_ok=True)
        self.Writer = imageio.get_writer(os.path.join(self.result_dir, "Result.mp4"), fps=30)
        self.logf = open(os.path.join(self.result_dir, "Log.csv"), 'w')
        self.logf.write("Frame, No. of subjects recognized in this frame, No. of subjects tracked, SID, Bboxes in prev frame, Matched bbox indices in this frame \n")

    # 主程式呼叫, 此 func 再呼叫 Track_subjects
    def Call_Track_subjects(self, BBox_tracking_mode, __width, __height):
        if(self.frame_count == 1): #first frame
            self.prev_indices = list(range(len(self.boxes)))  #0, 1, 2, ..., len(bbox with score > threshold)
            self.boxes_indices = [] #mask in current frame
            self.prev_boxes = self.boxes
        else:
            self.logMsg1() #記錄 subject recognition 結果到 log.csv
            self.Track_subjects(BBox_tracking_mode, __width, __height)
            #self.frame = DF.Draw_tracking_results_to_frame(self.active_tracklets, self.frame, __width, __height) # show prev frame tracking results on current frame
            self.logMsg2() #記錄 subject tracking 結果到 log.csv


    def logMsg1(self):
        #Frame,  No. of subjects recognized in this frame,  No. of subjects tracked
        self.logf.write(str(self.frame_count)+"," + str(len(self.boxes)) + ", " + str(len(self.active_tracklets)) + ",")
        #SID
        logMsg = " "
        for tracklet in self.active_tracklets:
            logMsg = logMsg + str(tracklet['sid']) + " "
        logMsg = logMsg + ","
        self.logf.write(logMsg)


    def logMsg2(self):
        # Bboxes in prev frame, Matched bbox indices in this frame 
        logMsg = ' '.join(str(e) for e in self.prev_indices)+ ", " + ' '.join(str(e) for e in self.boxes_indices) + "\n"
        self.logf.write(logMsg)


    # 主程式呼叫 Call_Track_subjects, Call_Track_subjects 呼叫 Track_subjects, 
    # Track_subjects 呼叫 Ask_user_to_match, My_bbox_match
    def Track_subjects(self, BBox_tracking_mode, __width, __height):
        #The 2nd method to match the bbox indices in this frame vs prev frame 
        #The 2nd tracking method linear_sum_assignment(cost) is adopted from: https://sparrow.dev/simple-pytorch-object-tracking/
        #find matched bboxes in prev frame and current frame
        
        cost=np.linalg.norm(self.prev_boxes[:, None] - self.boxes[None], axis=-1)
        self.prev_indices, self.boxes_indices = linear_sum_assignment(cost)

        # for each prev idx, find the best match
        matched_prev_indices = []
        matched_indices = []
        for prev_idx, box_idx in zip(self.prev_indices, self.boxes_indices):
            matched_prev_indices.append(prev_idx)
            box_idx2 = self.My_bbox_match(prev_idx, __width, __height)
            if(box_idx == box_idx2):
                matched_indices.append(box_idx)
            else: #2個 match方式結果不一致
                img_clone = self.frame.copy()
                
                if(BBox_tracking_mode == 1):
                    answer = self.Ask_user_to_match(img_clone, self.prev_boxes[prev_idx], self.boxes[box_idx], self.boxes[box_idx2], __width, __height)
                else:
                    answer = BBox_tracking_mode -1  # (BBox_tracking_mode=2 linear_sum_assignment, 3 My simple criteris)

                if(answer == 1):
                    matched_indices.append(box_idx)
                elif(answer == 2):
                    matched_indices.append(box_idx2)
                else: 
                    matched_prev_indices.pop() #do not add this prev_idx 
        self.prev_indices = matched_prev_indices
        self.boxes_indices = matched_indices


    ## 這一個 func 目前沒有用, 我先暫時不 maintain
    def Ask_user_to_match(self, box1, box2, box3, __width, __height):
        #call from main tracking loop, Ask_user_for_match(prev_boxes[prev_idx], boxes[box_idx], boxes[box_idx2])
        #box1: the bbox in prev. frame, draw in WHITE 
        #box2: the bbox in current frame that matches the prev box by criteria 1 (my criteria)
        #box3: the bbox in current frame that matches the prev box by criteria 2 
        #display.clear_output(wait=True)
        colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0)]
        for idx, box in enumerate([box1, box2, box3]):
            x1,y1,x2,y2 = box
            x1,y1,x2,y2 = int(x1*__width), int(y1*__height), int(x2*__width), int(y2*__height)
            cv2.rectangle(self.frame,(x1, y1), (x2,y2), color=colors[idx], thickness=rect_th*3)
        #plt.figure(figsize=(12, 6))
        #plt.imshow(self.frame) 
        #plt.show()
        print("Should the WHITE be tracked as RED(1, by linear_sum_assignment) or GREEN(2, by my criteria), or None(3, will be removed)?")
        choice = input("Enter 1~3")
        return int(choice)

    # 主程式呼叫 Call_Track_subjects, Call_Track_subjects 呼叫 Track_subjects, 
    # Track_subjects 呼叫 Ask_user_to_match, My_bbox_match
    def My_bbox_match(self, prev_idx, __width, __height):
    #my simple rule to match bbox by their corner points
        minValue = 9999
        idx = -99
        x0, y0, x1, y1 = self.prev_boxes[prev_idx]
        x0, y0, x1, y1  = int(x0*__width), int(y0*__height), int(x1*__width), int(y1*__height)
        for i in range(len(self.boxes)):
            self.bbox = self.boxes[i]
            minX, minY, maxX, maxY = self.bbox
            minX,minY,maxX,maxY = int(minX*__width), int(minY*__height), int(maxX*__width), int(maxY*__height)
            diff=abs(minX-x0)+abs(minY-y0)+abs(maxX-x1)+abs(maxY-y1)
            if(diff<minValue):
                idx = i
                minValue = diff
        return idx


    # 主程式呼叫此 func, 此 func 再呼叫 Redundant_bbox, Update_tracking_data_list
    def Maintain_tracking_lists(self, __width, __height):
        # Mactch bboxes in current frame with prev frame and update active tracklets
        for prev_idx, idx in zip(self.prev_indices, self.boxes_indices):
            self.active_tracklets[prev_idx]["boxes"] = [np.round(self.boxes[idx], 3).tolist()]
            self.active_tracklets[prev_idx]["skeleton"] = [np.round(self.skeleton[idx], 3)]

        # record lost tracklets
        lost_indices = set(range(len(self.active_tracklets))) - set(self.prev_indices)
        for lost_idx in sorted(lost_indices, reverse=True):
            msg = "Frame "+ str(self.frame_count) + " Remove subject " + str(self.active_tracklets[lost_idx]["sid"])
            DF.save_image (self.frame_count, self.frame, self.prev_boxes, lost_idx, msg, __width, __height, self.result_dir)
            self.finished_tracklets.append(self.active_tracklets.pop(lost_idx))

        # Activate new tracklets
        new_indices = set(range(len(self.boxes))) - set(self.boxes_indices)
        for new_idx in new_indices:
        #if we found new subjects in current frame
            if(self.frame_count ==1):
                self.subjectID +=1
                self.active_tracklets.append({"sid": self.subjectID, 
                                        "boxes": [np.round(self.boxes[new_idx],3).tolist()], 
                                        "skeleton": [np.round(self.skeleton[new_idx],3)]})
            #if this frame is not the first frame
            #we check whether the bbox is overlapping with existing bbox, if so we do not add
            elif(self.Redundant_bbox(new_idx, __width, __height)==False): 
                if(self.bboxScore[new_idx]>self.score_threshold_to_AddNewSubject):
                    self.subjectID +=1
                    msg =  "Frame "+ str(self.frame_count) + " Add new subject " + str(self.subjectID)
                    DF.save_image (self.frame_count, self.frame, self.prev_boxes, new_idx, msg, __width, __height, self.result_dir)
                    self.active_tracklets.append({"sid": self.subjectID, 
                                            "boxes": [np.round(self.boxes[new_idx],3).tolist()], 
                                            "skeleton": [np.round(self.skeleton[new_idx],3)]}) #dictionary is passed as a reference
        self.Update_tracking_data_list(__width, __height)
        self.frame = DF.Draw_tracking_results_to_frame(self.active_tracklets, self.frame, __width, __height)
        self.Writer.append_data(self.frame)

    # 主程式呼叫 Maintain_tracking_lists, 此 func 再呼叫 Redundant_bbox, Update_tracking_data_list
    def Redundant_bbox(self, new_idx, __width, __height):
        #排除跟目前 bbox 重疊的新 bbox
        threshold = self.redundant_box_pixels
        x1,y1,x2,y2 = self.boxes[new_idx]
        x1,y1,x2,y2 = int(x1*__width), int(y1*__height), int(x2*__width), int(y2*__height)
        for i in range(len(self.active_tracklets)):
            xx1,yy1,xx2,yy2 = self.active_tracklets[i]['boxes'][0]
            xx1,yy1,xx2,yy2 = int(xx1*__width), int(yy1*__height), int(xx2*__width), int(yy2*__height)
            if( (abs(x1-xx1)<threshold or x1>xx1) and  (abs(x2-xx2)<threshold or x2<xx2) and 
                (abs(y1-yy1)<threshold or y1>yy1) and  (abs(y2-yy2)<threshold or y2<yy2) ): 
                print("New box is inside an existed box")
                return True
            elif( (abs(x1-xx1)<threshold or x1<xx1) and  (abs(x2-xx2)<threshold or x2>xx2) and 
                (abs(y1-yy1)<threshold or y1<yy1) and  (abs(y2-yy2)<threshold or y2>yy2) ): 
                print("New box includes an existed box")
                return True
        return False
    
    # 主程式呼叫 Maintain_tracking_lists, 此 func 再呼叫 Redundant_bbox, Update_tracking_data_list
    def Update_tracking_data_list(self, __width, __height):
        for i in range(len(self.active_tracklets)):
            x1,y1,x2,y2 = self.active_tracklets[i]['boxes'][0]
            x1,y1,x2,y2 = int(x1*__width), int(y1*__height), int(x2*__width), int(y2*__height)
            sid=self.active_tracklets[i]['sid']
            self.SubjectTrackingResults.append((self.frame_count, sid, x1,y1, x2, y2)) 
            self.JointDetectionResults.append((self.frame_count,sid, self.active_tracklets[i]['skeleton'][0])) 

    # 主程式呼叫此 function  
    def Predict_next(self):
        self.prev_boxes = np.array([tracklet["boxes"][-1] for tracklet in self.active_tracklets])
        self.prev_skeleton = np.array([tracklet["skeleton"][-1] for tracklet in self.active_tracklets])


    def Save_Subject_tracking_data(self, __height):
        save_format = '{frame},{id},{x1},{y1},{x2},{y2},{xc},{yc},{xc1},{yc1}\n'
        fname = "SubjectBboxData.csv"
        with open(os.path.join(self.result_dir, fname), 'w') as f:
            f.write("frame,ID,x1,y1,x2,y2,xc,yc,xc1,yc1\n")
            for frame_id, sid, x1, y1, x2, y2 in self.SubjectTrackingResults:
                xc, yc = (x1+x2)/2, (y1+y2)/2
                yc1 = __height - yc
                line = save_format.format(frame=frame_id, id=sid, x1=x1, y1=y1,x2=x2, y2=y2, xc=xc, yc=yc, xc1=xc, yc1=yc1)
                f.write(line)


    def Save_Joints(self):
        # save joint data to csv file
        # header line for joint data csv
        columnLst = ["frameNo", "sid"]
        for i in range(1, 18):
            xs = "x" + str(i)
            ys = "y" + str(i)
            columnLst = columnLst + [xs, ys]

        lst = []
        for frame_id, sid, skeletonarray in self.JointDetectionResults:
            elt = [frame_id, sid]
            for x, y, visiability in skeletonarray:
                elt = elt + [x,y]
            lst.append(elt)
        df = pd.DataFrame(lst, columns = columnLst)
        fname = "JointData.csv"
        #產出檔案
        df.to_csv(os.path.join(self.result_dir, fname), index = False)