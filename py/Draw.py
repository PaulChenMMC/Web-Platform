import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
import numpy as np
from datetime import datetime

def get_max_min_xy(times_row,selected_columns,max_x,min_x,max_y,min_y):
    for index, row in times_row.iterrows():
        for col in selected_columns:
            x_y_values = row[col][1:-1].split(',')
            x = float(x_y_values[0])
            y = float(x_y_values[1])
            max_x = max(max_x, x)
            min_x = min(min_x, x)
            max_y = max(max_y, y)
            min_y = min(min_y, y)
            # print(f'{col}: X={x}, Y={y}')
    return max_x,min_x,max_y,min_y

def get_xy_abs(max_x,min_x,max_y,min_y):
    offset_x = abs(min_x)
    offset_y = abs(min_y)
    max_x += offset_x
    min_x += offset_x
    max_y += offset_y
    min_y += offset_y
    return max_x,min_x,max_y,min_y,offset_x,offset_y

def creat_png(min_times_row,max_x,max_y,offset_x,offset_y,imgname,no):
    #繪製五張圖片
    plt.close('all')

    for i in range(1, 6):
        rShoulder_data = min_times_row['RShoulder'+ str(i) +'(X,Y)'].iloc[0]
        x_y_values = rShoulder_data[1:-1].split(',')
        rShoulder_x=float(x_y_values[0])
        rShoulder_y=float(x_y_values[1])
        rHip_data = min_times_row['RHip'+ str(i) +'(X,Y)'].iloc[0]
        x_y_values = rHip_data[1:-1].split(',')
        rHip_x=float(x_y_values[0])
        rHip_y=float(x_y_values[1])
        plt.scatter(rShoulder_x+offset_x+20, rShoulder_y+offset_y+20, color='red', marker='o',s=150)
        plt.scatter(rHip_x+offset_x+20, rHip_y+offset_y+20, color='black', marker='o',s=150) 
        plt.xlim(0, max_x+30)
        plt.ylim(0, max_y+30)
        plt.axis('off')  # 关闭坐标轴
        plt.grid(False)  # 关闭网格线
        plt.box(False)
        plt.savefig('./static/'+str(i)+'.png', bbox_inches='tight')
        plt.savefig('./static/'+no+'_'+str(i)+'.png', bbox_inches='tight')
        plt.clf()
    #圖片合併
    fig, axs = plt.subplots(1, 5, figsize=(15, 4), gridspec_kw={'wspace': 0, 'hspace': 0})
    fig.tight_layout()
    # 为每个子图添加图片
    for i in range(1, 6):
        img = Image.open('./static/'+str(i) + '.png')
        axs[i-1].imshow(img)
        axs[i-1].axis('off')  # 关闭坐标轴
    # 调整子图之间的间距
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    # 保存组合后的图片
    save_path ='./static/'+imgname
    plt.savefig(save_path)
    # plt.savefig('./static/'+no+'test')
    plt.clf()
    #移除圖片
    for i in range(1, 6):
        os.remove('./static/'+str(i) + '.png')
    # 读取图像
    image = cv2.imread(save_path)
    # 將圖片轉換為 HSV 色彩空間
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 定義紅色的 HSV 範圍
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    # 定義黑色的 HSV 範圍
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])
    # 在 HSV 圖像中找到紅色區域的遮罩
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    # 在 HSV 圖像中找到黑色區域的遮罩
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    # 使用形態學操作進行過濾和增強
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    # 找到紅色區域和黑色區域的連通區域
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找到紅色點和黑色點的中心
    red_centers = [cv2.minEnclosingCircle(contour)[0] for contour in red_contours]
    black_centers = [cv2.minEnclosingCircle(contour)[0] for contour in black_contours]
    # 將中心點用線段連接起來
    for center1 in red_centers:
        for center2 in red_centers:
            if center1 != center2 and np.linalg.norm(np.array(center1) - np.array(center2)) < 500:
                cv2.line(image, (int(center1[0]), int(center1[1])), (int(center2[0]), int(center2[1])), (0, 0, 255), 2)
    for center1 in black_centers:
        for center2 in black_centers:
            if center1 != center2 and np.linalg.norm(np.array(center1) - np.array(center2)) < 500:
                cv2.line(image, (int(center1[0]), int(center1[1])), (int(center2[0]), int(center2[1])), (0, 0, 0), 2)
    border_color = (0, 0, 0)  # 在这里，使用BGR颜色格式，这里表示绿色
    border_width = 3  # 边框宽度
    image_with_border = cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), border_color, border_width)

    cv2.imwrite(save_path, image_with_border)

def MergePNG(n):
    plt.close('all')
    image1 = mpimg.imread('./static/1ST.png')
    image2 = mpimg.imread('./static/mid.png')
    image3 = mpimg.imread('./static/Last.png')
    
    title_text = "Total Times : "+str(n)
    fig, ax = plt.subplots(figsize=(15, 16))
    # fig.tight_layout()
    ax.axis('off')
    ax.imshow(np.vstack((image1, image2, image3)))
    plt.figtext(0.05, 0.9, title_text, fontsize=16, ha='left')
    save_path ='./static/Merge.png'
    plt.savefig(save_path)
    
    
    
def Drawing(select_id,record_type,date):
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        date = date_obj.strftime("%Y%m%d")
        file_path='./Datahouse/Keypiont/'
        RawData_Path=os.path.join(file_path,record_type,str(select_id),str(select_id)+'_Result.csv')
        max_x = float('-inf')
        min_x = float('inf')
        max_y = float('-inf')
        min_y = float('inf')
        df=pd.read_csv(RawData_Path)
        selected_columns = df.columns[5:15]
        filtered_data = df[df['Date'].str.startswith(date)]
        #抓取第一筆&最後一筆 最大值
        min_times_row = filtered_data[filtered_data['Times'] == filtered_data['Times'].min()]
        max_x,min_x,max_y,min_y=get_max_min_xy(min_times_row,selected_columns,max_x,min_x,max_y,min_y)

        n = len(filtered_data['Times'])
        Mid_times_row = filtered_data[filtered_data['Times'] == round(n/2,0)]
        max_x,min_x,max_y,min_y=get_max_min_xy(Mid_times_row,selected_columns,max_x,min_x,max_y,min_y)

        last_times_row = filtered_data[filtered_data['Times'] == filtered_data['Times'].max()]
        max_x,min_x,max_y,min_y=get_max_min_xy(last_times_row,selected_columns,max_x,min_x,max_y,min_y)
        
        max_x,min_x,max_y,min_y,offset_x,offset_y=get_xy_abs(max_x,min_x,max_y,min_y)

        # #產生圖片
        creat_png(min_times_row,max_x,max_y,offset_x,offset_y,'1ST.png','1')
        creat_png(Mid_times_row,max_x,max_y,offset_x,offset_y,'mid.png','2')
        creat_png(last_times_row,max_x,max_y,offset_x,offset_y,'Last.png','3')
        
        MergePNG(n)
            
        state=True
    except:
        state=False
    return state

