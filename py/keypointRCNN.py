import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
import numpy as np
from PIL import Image
import pandas as pd
from scipy.optimize import linear_sum_assignment
import os
import cv2
import imageio
import matplotlib.pyplot as plt
import os
import pandas as pd
import py.My_Simple_Subject_Tracking as MST
import py.My_Drawing_functions as DF

def Run_Keypiont_RCNN(VideoName,folder_path):
    #3
    # get video file informaiton
    vid = imageio.get_reader(VideoName, 'ffmpeg')
    cap = cv2.VideoCapture(VideoName)
    total_frames = int(cap.get(7))
    fps = cap.get(cv2.CAP_PROP_FPS)
    __width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    __height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('No. of frames = ', total_frames, ", w =", __width, ", h =", __height, ", fps=", fps)
    #4
    # take a look at the input video
    # MaxFrame = 10
    # frame_count = 1
    # try:
    #     while(frame_count <= MaxFrame):
    #         # display.clear_output(wait=True)
    #         plt.title(str(frame_count)+'/'+str(total_frames))
    #         frame = vid.get_data(frame_count)  # Capture frame-by-frame
    #         frame_count += 1
    #         plt.imshow(frame)
    #         plt.pause(0.1)
    # except:
    #     print("Read video error!")
    #7
    class NeuralNetworkThread():
        def __init__(self,):
            super(NeuralNetworkThread, self).__init__()
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("cuda")
            else:
                self.device = torch.device("cpu")
                print("cup")
            self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
            self.model.eval().to(self.device)

            #subject recognition 參數
            self.class_index = 1  # 1 = people
            self.score_threshold = 0.8  # threshold to recognize subject, if too restricted, we will keep loosing subjects
            self.tracker = MST.My_Simple_SubjectTracking(folder_path) # 用來做 subject tracking
            self.BBox_tracking_mode = 3  # 1 Interaction mode, 2 linear_sum_assignment(cost) to match, 3 My simple criteris to match

        def Recognize_subjects(self, frame, __width, __height):
            transform = transforms.Compose([transforms.ToTensor()]) # Defing PyTorch Transform
            img = transform(frame).to(self.device) # Apply the transform to the image
            result = self.model([img])[0] # Pass the image to the model
            bboxLabels = list(result['labels'].cpu().numpy())
            bboxScore = list(result['scores'].cpu().detach().numpy())
            mask=[]
            for i in range(len(bboxLabels)): #find bbox with score > threshold
                if(bboxLabels[i]==self.class_index and bboxScore[i]>=self.score_threshold):
                    mask.append(i)
            self.tracker.boxes = result["boxes"][mask].data.cpu().numpy()/np.array([__width, __height, __width, __height])
            self.tracker.bboxScore = list(result['scores'][mask].cpu().detach().numpy())
            self.tracker.skeleton = result['keypoints'][mask].cpu().detach().numpy() #keypoints
    #8
    nn=NeuralNetworkThread()
   #9
    while(nn.tracker.frame_count < 10):  #測試 10 frames, 正式跑要改回 total_frames
        # display.clear_output(wait=True)
        # plt.title(str(nn.tracker.frame_count)+'/'+str(total_frames))
        camera_frame = vid.get_data(nn.tracker.frame_count)  # Capture frame-by-frame
        
        nn.Recognize_subjects(camera_frame, __width, __height)
        
        nn.tracker.frame = camera_frame
        
        nn.tracker.Call_Track_subjects(nn.BBox_tracking_mode, __width, __height)

        if(nn.tracker.frame_count >1):
            print("No. of subjects recognized in this frame =", len(nn.tracker.boxes))
            print("No. of subjects tracked = ", len(nn.tracker.active_tracklets), ", SID = ", [tracklet['sid'] for tracklet in nn.tracker.active_tracklets])
            print("Bboxes in PREV frame:              ", nn.tracker.prev_indices)
            print("Matched bbox indices in THIS frame:", nn.tracker.boxes_indices)

        nn.tracker.Maintain_tracking_lists(__width, __height)

        plt.imshow(nn.tracker.frame)                     
        # plt.pause(0.0001)

        nn.tracker.Predict_next()
        nn.tracker.frame_count += 1
    
    nn.tracker.Writer.close()
    nn.tracker.logf.close()
    #10
    save_format = '{frame},{id},{x1},{y1},{x2},{y2},{xc},{yc},{xc1},{yc1}\n'
    fname = "SubjectBboxData.csv"
    with open(os.path.join(nn.tracker.result_dir, fname), 'w') as f:
        f.write("frame,ID,x1,y1,x2,y2,xc,yc,xc1,yc1\n")
        for frame_id, sid, x1, y1, x2, y2 in nn.tracker.SubjectTrackingResults:
            xc, yc = (x1+x2)/2, (y1+y2)/2
            yc1 = __height - yc
            line = save_format.format(frame=frame_id, id=sid, x1=x1, y1=y1,x2=x2, y2=y2, xc=xc, yc=yc, xc1=xc, yc1=yc1)
            f.write(line)
    #11
    # save joint data to csv file
    # header line for joint data csv
    columnLst = ["frameNo", "sid"]
    for i in range(1, 18):
        xs = "x" + str(i)
        ys = "y" + str(i)
        columnLst = columnLst + [xs, ys]

    lst = []
    for frame_id, sid, skeletonarray in nn.tracker.JointDetectionResults:
        elt = [frame_id, sid]
        for x, y, visiability in skeletonarray:
            elt = elt + [x,y]
        lst.append(elt)
    df = pd.DataFrame(lst, columns = columnLst)
    fname = "JointData.csv"
    df.to_csv(os.path.join(nn.tracker.result_dir, fname), index = False)