import os
import pandas as pd
import numpy as np
import math
class Analysis:
    def __init__(self, Data_arr, StandPercentage=0.85, SitPercentage=1.8, StandK=0.5, SitK=0.8):
        self.Data_arr = list(Data_arr)
        self.StandPercentage = StandPercentage
        self.SitPercentage = SitPercentage
        self.StandK = StandK
        self.SitK = SitK
        #數值的最大值
        self.Max = Data_arr.max()
        #數值的最小值
        self.Min = Data_arr.min()
        #坐下-起立的完整次數
        self.TotalTimes = 1
        #目前狀態
        self.TypeStatue = None
        #起始狀態
        self.StartTypeStatue = None
        # 用dict來紀錄計算後的數值
        self.Sp_Value_dict = {}
        self.Sp_Item_dict = {}
        self.Sp_Cal_dict = {}

        # 一開始判斷是坐還是站
        if self.Data_arr[0] <= self.Min * self.SitPercentage:
            self.TypeStatue = "Sit"
            self.StartTypeStatue = "Sit"
        else:
            self.TypeStatue = "Stand"
            self.StartTypeStatue = "Stand"

    def record_kv(self, dict_name, dict_data):
        totaltimes = list(dict_data.keys())[0]
        if dict_name.get(totaltimes):
            dict_name[totaltimes].update(dict_data[totaltimes])
        else:
            dict_name.update(dict_data)

    def gen_cal(self):
        temp_list = []
        calculate_list = []
        #把影格dict的key值存成list
        for i in self.Sp_Item_dict.keys():  # 1~13
            temp_list += self.Sp_Item_dict[i].values()
        #把影格一一列出來計算，計算後存為list
        for item, value in enumerate(temp_list[:-2]):
            calculate_list.append((temp_list[item + 1] - value) / 60)
        # 算好資料，再重新組成dict，
        #資料格式{ totaltimes : {
        #                       各個點位的名稱:計算後的資料
        #                      }
        #        }
        # totaltimes當作
        for i in self.Sp_Item_dict.keys():  # 1~13
            self.Sp_Cal_dict.update({i: {}})
            for key in self.Sp_Item_dict[i].keys():
                if len(calculate_list) > 0:
                    self.Sp_Cal_dict[i].update({key: calculate_list[0]})
                    calculate_list.pop(0)

    #將資料進行整合存為XLSX
    def combine_data(self, filename,Sp_Value_dict,Sp_Item_dict,Sp_Cal_dict):
        #將dict讀入dataframe
        df1 = pd.DataFrame(Sp_Value_dict)
        df2 = pd.DataFrame(Sp_Item_dict)
        df3 = pd.DataFrame(Sp_Cal_dict)
        #把3個dataframe合併
        # df = pd.concat([df1, df2, df3])
        df = pd.concat([ df2 ])
        #回傳存為XLSX
        df_transposed = df.transpose()
        df_transposed.to_csv(filename, index_label='No')

    def analysis_data(self):
        A = 0
        for i in range(len(self.Data_arr) - 1):
            if i < A:
                continue
            if self.TypeStatue == "Sit":
                # 為了找到最小值*1.8後的臨界值，然後開始判斷是否坐下
                if self.Min<0 :
                    if self.Data_arr[i] > self.Min+abs(self.Min * self.SitPercentage):
                        continue
                    A, self.TypeStatue = self.type_sit(i)
                    if self.StartTypeStatue == "Stand":
                        self.TotalTimes += 1
                else:
                    if self.Data_arr[i] > self.Min * self.SitPercentage:
                        continue
                    A, self.TypeStatue = self.type_sit(i)
                    if self.StartTypeStatue == "Stand":
                        self.TotalTimes += 1 
            elif self.TypeStatue == "Stand":
                if self.Data_arr[i] < self.Max * self.StandPercentage:
                    continue
                # 要調整
                A, self.TypeStatue = self.type_stand(i)
                if self.StartTypeStatue == "Sit":
                    self.TotalTimes += 1
            # print(self.TotalTimes, self.TypeStatue)
        #產生影格進行計算後的資料
        self.gen_cal()

    def type_stand(self, i):
        # print('type_stand')
        A = i + 5
        #計算坐下曲線中的L到H間有幾個影格
        Data_C = 0 + 5
        for a in range(A, len(self.Data_arr) - 1):
            if self.Data_arr[a] < self.Max * self.StandPercentage:
                break
            Data_C += 1
        #站起來時曲線向上，第一個碰到站起來參數的影格值(基準值*0.85)
        L = i
        #站起來時曲線經過高點向下後，第一個碰到站起來參數的影格值(基準值*0.85)
        H = i + Data_C
        Data_C = Data_C / 2
        AlreadStand = self.Data_arr[i]
        A = i
        for ii in range(L, i + int(Data_C * self.StandK) + 1):
            if AlreadStand < self.Data_arr[ii]:
                AlreadStand = self.Data_arr[ii]
                A = ii
        # Prepare to sit
        # print('Prepare to sit:', A + 1, AlreadStand)
        self.record_kv(self.Sp_Value_dict, {self.TotalTimes: {'Prep-to-sit': AlreadStand}})
        self.record_kv(self.Sp_Item_dict, {self.TotalTimes: {'Prep-to-sit': A + 1}})
        AlreadStand = self.Data_arr[H]
        for ii in range(H, H - int(Data_C * self.StandK) - 1, -1):
            if AlreadStand < self.Data_arr[ii]:
                AlreadStand = self.Data_arr[ii]
                A = ii
        # Stand to sit
        # print('Stand to sit:', A + 1, AlreadStand)
        self.record_kv(self.Sp_Value_dict, {self.TotalTimes: {'Std-to-sit': AlreadStand}})
        self.record_kv(self.Sp_Item_dict, {self.TotalTimes: {'Std-to-sit': A + 1}})
        return A, "Sit"

    def type_sit(self, i):
        A = i + 5
        #計算坐下曲線中的L到H間有幾個影格
        Data_C = 0 + 5
        for a in range(A, len(self.Data_arr) - 1):
            if self.Min<0 :
                if self.Data_arr[a] > self.Min+abs(self.Min * self.SitPercentage):
                    break
                Data_C += 1
            else:
                if self.Data_arr[a] > self.Min * self.SitPercentage:
                    break
                Data_C += 1
        #正要坐下時曲線向下，第一個碰到坐下參數的影格值(基準值*1.8)
        L = i
        #從坐下狀態要站起來曲線向上後，第一個碰到坐下參數的影格值(基準值*1.8)
        H = i + Data_C
        Data_C = Data_C / 2
        # Data_C =坐下數據的項次
        AlreadSit = self.Data_arr[i]
        A = i
        for ii in range(L, i + int(Data_C * self.SitK) + 1):
            if AlreadSit > self.Data_arr[ii]:
                AlreadSit = self.Data_arr[ii]
                A = ii
        # print('Prepare for next stand:', A + 1, AlreadSit)
        self.record_kv(self.Sp_Value_dict, {self.TotalTimes: {'Prep-to-std': AlreadSit}})
        self.record_kv(self.Sp_Item_dict, {self.TotalTimes: {'Prep-to-std': A + 1}})
        AlreadSit = self.Data_arr[H]
        for ii in range(H, int(H - Data_C * self.SitK) + 1, -1):
            if AlreadSit > self.Data_arr[ii]:
                AlreadSit = self.Data_arr[ii]
                A = ii
        # print('Sit-to-stand:', A + 1, AlreadSit)
        self.record_kv(self.Sp_Value_dict, {self.TotalTimes: {'Sit-to-std': AlreadSit}})
        self.record_kv(self.Sp_Item_dict, {self.TotalTimes: {'Sit-to-std': A + 1}})

        return A, "Stand"



def get_Pose(Data_Calibration_Path,RawData_Path,SID,Time,file_path):
    save_path = os.path.join(RawData_Path,'process.csv')
    #判斷每次動作Frame
    df = pd.read_csv(Data_Calibration_Path)
    Data_Arr = df['y7']
    analysis = Analysis(Data_arr=Data_Arr, StandPercentage=0.8, SitPercentage=1.8, StandK=0.7, SitK=0.8)
    analysis.analysis_data()
    analysis.combine_data(save_path, analysis.Sp_Value_dict, analysis.Sp_Item_dict, analysis.Sp_Cal_dict)


    #將資料填入Result
    df_result = pd.read_csv(os.path.join(file_path,str(SID),str(SID)+'_Result.csv'))
    #讀取校正後檔案
    df = pd.read_csv(os.path.join(RawData_Path,'JointData_Data_Calibration.csv'))

    #抓取對應坐標
    with open(save_path, 'r') as file:
        next(file)
        for line in file:
            data = line.strip().split(',')
            RShouledr=[]
            RHip=[]
            if data[3]:
                numbers = [math.ceil(num) for num in np.linspace(float(data[1]), float(data[3]), 5)]
                for Num in numbers:
                    Rshoulderx = df.loc[df['frameNo'] == Num, 'x7'].values[0]
                    Rshouldery = df.loc[df['frameNo'] == Num, 'y7'].values[0]
                    RHipx = df.loc[df['frameNo'] == Num, 'x13'].values[0]
                    RHipy = df.loc[df['frameNo'] == Num, 'y13'].values[0]
                    Rshoulderx_rounded = round(Rshoulderx, 3)
                    Rshouldery_rounded = round(Rshouldery, 3)
                    RHipx_rounded = round(RHipx, 3)
                    RHip_rounded = round(RHipy, 3)

                    RShouledr.append((Rshoulderx_rounded,Rshouldery_rounded))
                    RHip.append((RHipx_rounded,RHip_rounded))

                df_result = df_result._append({
                    'Date': Time,
                    'SID': SID,
                    'Times': data[0],
                    'Sit-frame': data[1],
                    'Stand-frame': data[3],
                    'RShoulder1(X,Y)': RShouledr[0],
                    'RShoulder2(X,Y)': RShouledr[1],
                    'RShoulder3(X,Y)': RShouledr[2],
                    'RShoulder4(X,Y)': RShouledr[3],
                    'RShoulder5(X,Y)': RShouledr[4],
                    'RHip1(X,Y)': RHip[0],
                    'RHip2(X,Y)': RHip[1],
                    'RHip3(X,Y)': RHip[2],
                    'RHip4(X,Y)': RHip[3],
                    'RHip5(X,Y)': RHip[4]
                    }, ignore_index=True)
            # else:
            #         df_result = df_result._append({
            #         'Date': Time,
            #         'SID': SID,
            #         'Times': data[0],
            #         'Sit-frame': data[1],
            #         'Stand-frame': data[3]
            #         }, ignore_index=True)
    df_result.to_csv(os.path.join(file_path,str(SID),str(SID)+'_Result.csv'), index=False)
