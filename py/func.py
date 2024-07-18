import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.signal import convolve
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
from datetime import  datetime,timedelta
import pyttsx3

#產出音檔
def create_viod(str):
    # 提示模組參數
    engine = pyttsx3.init()
    rate = engine.getProperty("rate")
    engine.setProperty("rate", rate-50)
    string=str+"，你好請到指定位子"
    engine.say(str+"，你好請到指定位子")
    engine.save_to_file(string, './static/Face.wav')
    engine.runAndWait()
    return

#建立資料夾
# def Create_Folder_Without_SID(folder_path):
#     max_number = None
#     max_file = None
#     for folder_name in os.listdir(folder_path):
#         folder_full_path = os.path.join(folder_path, folder_name)
#         if os.path.isdir(folder_full_path):
#                     # 檢查是否是具有數字資料夾名稱
#                     try:
#                         number = int(folder_name)  # 假設資料夾名稱是數字
#                         if max_number is None or number > max_number:
#                             max_number = number
#                             max_folder = folder_name
#                     except ValueError:
#                         pass  # 如果資料夾名稱無法轉換成數字，則跳過

#     if max_number is not None:
#         max_number = max_number + 1
#     else:
#         max_number = 1

#     #產生資料夾
#     Create_Folder_Path=os.path.join(folder_path, str(max_number))
#     os.makedirs(Create_Folder_Path)
#     #產生CSV
#     output_file = os.path.join(Create_Folder_Path, str(max_number)+'_MSE_Result.csv')
#     data = {'Date': [],
#             'SID': [],
#             'Times': [],
#             'RShoulder1(X,Y)': [],
#             'RShoulder2(X,Y)': [],
#             'RShoulder3(X,Y)': [],
#             'RShoulder4(X,Y)': [],
#             'RShoulder5(X,Y)': [],
#             'RHip1(X,Y)': [],
#             'RHip2(X,Y)': [],
#             'RHip3(X,Y)': [],
#             'RHip4(X,Y)': [],
#             'RHip5(X,Y)': []
#             }
#     df = pd.DataFrame(data)
#     df.to_csv(output_file, index=False)

#     #產生存放原始資料資料夾
#     RawData_Path=os.path.join(Create_Folder_Path, "Raw Data")
#     os.makedirs(RawData_Path)

#     #產生對應時間的資料夾
#     now = datetime.now()
#     formatted_datetime = now.strftime("%Y%m%d_%H%M%S")
#     Create_Folder_Path=os.path.join(RawData_Path, formatted_datetime)
#     os.makedirs(Create_Folder_Path)

#     return Create_Folder_Path,max_number,formatted_datetime,output_file

def Create_Folder_With_SID(folder_path,SID):
    folder_Exist=None
    folder_Name = os.path.join(folder_path, SID)
    for folder_name in os.listdir(folder_path):
         if os.path.isdir(folder_Name):
                try:
                    folder_Exist=True
                except ValueError:      
                    pass
    #如果 folder_Exist 不存在則產生資料夾
    if folder_Exist ==None :
        os.makedirs(folder_Name)
        os.makedirs(os.path.join(folder_Name, "Raw Data"))
         #產生CSV
        output_file = os.path.join(folder_Name, SID+'_Result.csv')
        data = {
            'Date': [],
            'SID': [],
            'Times': [],
            'Sit-frame': [],
            'Stand-frame': [],
            'RShoulder1(X,Y)': [],
            'RShoulder2(X,Y)': [],
            'RShoulder3(X,Y)': [],
            'RShoulder4(X,Y)': [],
            'RShoulder5(X,Y)': [],
            'RHip1(X,Y)': [],
            'RHip2(X,Y)': [],
            'RHip3(X,Y)': [],
            'RHip4(X,Y)': [],
            'RHip5(X,Y)': []
        }
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
    #產生對應時間的資料夾    
    now = datetime.now()
    # output_file=os.path.join(folder_Name, SID+'_MSE_Result.csv')
    formatted_datetime = now.strftime("%Y%m%d_%H%M%S")
    RawData_Path=os.path.join(folder_Name, "Raw Data")
    Create_Folder_Path=os.path.join(RawData_Path, formatted_datetime)
    os.makedirs(Create_Folder_Path)
    return Create_Folder_Path,formatted_datetime

def Data_Calibration(SID,RawData_Path):
    dfA = pd.read_csv(os.path.join(RawData_Path, 'jointdata.csv'))
    dfB = pd.read_csv(os.path.join(RawData_Path, 'subjectbboxdata.csv'))
    #抓取 sub YX XC
    # 分離csv
    df_filteredB = dfB.loc[dfB['ID'] == 1]
    XC = df_filteredB.loc[df_filteredB.index[0], 'xc']
    YC = df_filteredB.loc[df_filteredB.index[0], 'yc']

    # 分離csv
    df_filtered = dfA.loc[dfA['sid'] == SID]
    # 為了避免產生SettingWithCopyWarning，要將資料先進行copy再進行操作
    df_filter_copy = df_filtered.copy()
    # 把每一列的標題取出為list，再一一代入進行運算
    for i in list(df_filtered.keys())[2:]:
        # 進行資料XC，YC運算
        if 'x' in i:
            df_filter_copy[i] = df_filtered[i] - XC
        if 'y' in i:
            df_filter_copy[i] = (df_filtered[i] - YC)*-1
    #存回RawData
    DF_Path=os.path.join(RawData_Path, 'JointData_Data_Calibration.csv')
    df_filter_copy.to_csv(DF_Path, index=False)
    return DF_Path

def keypointData(Main_Path,cluster_Path,start_date,end_date):
    df = pd.read_csv(os.path.join(Main_Path, 'BenchmarkDB_cwt.csv'))
    df_filtered = df
    df_filtered.to_csv(os.path.join(Main_Path, 'BenchmarkDB_cwt.csv'), index=False)
    kpts = ['x7', 'y7', 'x13', 'y13'] 
    pts_per_second = 60  #許主任收30秒坐站影片每秒有60幀
    fmin = 0.1
    fmax = 1   #社區老人做坐到站每秒極限是2.5次,
    fstep = 0.01
    min_no_features = 5000 #用來記錄30s STS 時間最短的 sujbect, 以便大家 features 數一致 
    outputLst = []
    #讀取路徑內檔案名稱
    file_names = os.listdir(cluster_Path)
    for file_name in file_names:
        file_path = os.path.join(cluster_Path, file_name)
        file_name = os.path.basename(file_path)
        date_str = file_name[:8]
        SD=convert_date_format(start_date)
        ED=convert_date_format(end_date)
        #檔案日期在區間內進行計算
        if SD <= date_str <= ED :
            df=pd.read_csv(file_path)
            subject_features = [file_name, 'Current']
            for pt_str in kpts:
                df1=df[[pt_str]] 
                time_series = np.array(df1.values)
                time_series_length = time_series.shape[0]
                ts = time_series.reshape(time_series_length,)
                spec = tfa_morlet(ts, pts_per_second, fmin, fmax, fstep)
                spec_reverse = np.flip(spec, axis=0)

                #從 wavelet transform 矩陣中 sample 特徵值
                NoRows = spec_reverse.shape[0]
                NoColumns = spec_reverse.shape[1]
                rowIdx = 0
                while(rowIdx < NoRows):
                    colIdx = 0
                    while(colIdx < NoColumns):
                        subject_features.append(spec_reverse[rowIdx][colIdx])
                        colIdx += pts_per_second  #time sampling - every second
                    rowIdx += 10  #freq sampling - every 10
            if(len(subject_features)< min_no_features):
                min_no_features = len(subject_features)
            outputLst.append(subject_features) 

            line = outputLst[0] #take the first line 
            num_rows,num_columns =df_filtered.shape
            if num_columns <= min_no_features :
                line = line[:num_columns]
            else:
                df_filtered = df_filtered.iloc[:,:min_no_features]

            new_data = dict(zip(df_filtered.columns, line))
            df_filtered = pd.concat([df_filtered, pd.DataFrame([new_data])], ignore_index=True)
    df_filtered.to_csv(os.path.join(Main_Path, 'out_cwt.csv'), index=False)

    return None

def convert_date_format(date_str):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    new_date_str = date_obj.strftime("%Y%m%d")
    return new_date_str

def tfa_morlet(td, fs, fmin, fmax, fstep):
    TFmap = np.array([])
    for fc in np.arange(fmin, fmax+fstep, fstep):
        MW = MorletWavelet(fc/fs)
        cr = convolve(td, MW, mode='same')
        TFmap = np.vstack([TFmap, abs(cr)]) if TFmap.size else abs(cr)
    return TFmap

def MorletWavelet(fc):
    F_RATIO = 7
    Zalpha2 = 3.3
    
    sigma_f = fc / F_RATIO
    sigma_t = 1 / (2 * np.pi * sigma_f)
    A = 1 / np.sqrt(sigma_t * np.sqrt(np.pi))
    max_t = np.ceil(Zalpha2 * sigma_t)
    
    t = np.arange(-max_t, max_t + 1)
    
    v1 = 1 / (-2 * sigma_t**2)
    v2 = 2j * np.pi * fc
    MW = A * np.exp(t * (t * v1 + v2))
    
    return MW

# def t_SNE(Main_Path,perplexity_input):
#     df = pd.read_csv(os.path.join(Main_Path, 'out_cwt.csv'))
#     dfX = df.drop(df.columns[0:2], axis=1)
#     numpyX = dfX.values
#     dfY = df[' class']
#     numpyY = dfY.values
#     tsne = TSNE(perplexity=perplexity_input, n_components=2, init='pca', n_iter=500)
#     x1 = tsne.fit_transform(numpyX, numpyY)

#     #讀取__之間的資料
#     pattern = r"_(.*?)_"
#     file_names = df.iloc[:, 0:1]
#     extracted_file_names = []
#     for file_name in file_names:
#         try:
#             match = re.search(pattern, file_name[0])
#             if file_name[1]=="Current" or  file_name[1]=="Upload":
#                 extracted_file_names.append(file_name[0])  
#             if match:
#                 extracted_string = match.group(1)
#                 extracted_file_names.append(extracted_string)
#             else:
#                 extracted_file_names.append(file_name[0])  
#         except:
#             extracted_file_names.append(file_name[0])
#     # 產生Tsne_data
#     new_df = pd.DataFrame({
#         "FileName": df.iloc[:, 0],
#         "Class": df.iloc[:, 1],  
#         "X": x1[:, 0],  
#         "Y": x1[:, 1]  
#     })
#     # new_df.to_csv(os.path.join(Main_Path, 'Tsne_data.csv'), index=False)
#     return x1,numpyY,new_df

# def list_png_files(directory_path):
#     png_files = []
#     for file in os.listdir(directory_path):
#         if file.lower().endswith(".png"):
#             png_files.append(file)
#     return png_files

def MSE(Record_type,Data_Calibration_Path,CSV_File_Path,Folder_SID,Time):
    #讀取校正好的檔案
    df = pd.read_csv(Data_Calibration_Path)
    #判斷錄製動作，採取對應分析KeyPiont
    if Record_type=='STS':
        MSE_df=df['x7']
    elif Record_type=='30s STS':
        MSE_df=df['x7']
    elif Record_type=='Standing on one foot':
        MSE_df=df['x7']
    elif Record_type=='Standing front and back':
        MSE_df=df['x7']

    time_series = np.array(MSE_df.values)
    time_series_length = time_series.shape[0]
    ts = time_series.reshape(time_series_length,)
    lst1 = []
    maxScale = 6
    r_ratio = 0.15
    for scale_factor in range (1, maxScale+1):#粗粒化1-6
        ts_i = coarse_grain(ts, scale_factor)
        se1 = sample_entropy1(ts_i, 1, r_ratio) 
        lst1.append(se1)

    new_lst1 = []
    for idx in range(1): #0, 1, 2
        tmp = []
        for elt in lst1:
            tmp.append(elt[idx])
        new_lst1.append(tmp)

    print(new_lst1)

    MSE_CSV_File = pd.read_csv(CSV_File_Path)

    new_data = pd.DataFrame([[Time, Folder_SID] + new_lst1[0]], columns=['Date', 'SID', '1', '2', '3', '4', '5', '6'])
    combined_data = pd.concat([MSE_CSV_File, new_data], ignore_index=True)
    print(combined_data)
    combined_data.to_csv(CSV_File_Path, index=False)


    return new_lst1

def coarse_grain(ts, scale):  #比較容易了解版本
    seg = int(np.floor(len(ts)/scale))
    ts1 = np.zeros(seg) #new time series
    for i in range(seg):
        head = i*scale 
        tail = head+scale-1
        seg = ts[head:tail+1]
        ts1[i] = np.mean(seg)
    return ts1

def sample_entropy1(ts, Mdim, r_ratio):
    n = len(ts)
    r = r_ratio*np.std(ts)
    SE=np.zeros(Mdim) # Mdim 個 SE
    count_m = np.zeros(Mdim+1) # 計算 Midm SE 時需要用到 Midm+1, 因此要多一個

    for i in range(n-Mdim):  # index (min,max)=(0,n-1), i+Mdim=n-1 => i=n-Mdim-1
        for j in range(i+1, n-Mdim): # j+Mdim 要 <= n-1, 所以 range到 n-Mdim
            m=0  # 0~Mdim 
            while(m<=Mdim and abs(ts[i+m]-ts[j+m]) <= r):
                count_m[m] += 1
                m = m+1
                
    for m in range(Mdim):
        if(count_m[m] ==0 or count_m[m+1] ==0):
            #SE[m] = -np.log(1/((n-m)*(n-m-1))) # a large number
            SE[m] = 0
        else:
            SE[m] = -np.log(count_m[m+1]/count_m[m]) 
    return SE

def Drawing(record_type,BenchmarkDB_Path,SID_Path,Png_Path,start_date,end_date):

    #判斷動作，讀取對應對照資料
    if record_type=='STS':
        SID='STS'
    elif record_type=='30s STS':
        SID='30s STS'
    elif record_type=='Standing on one foot':
        SID='Standing on one foot'
    elif record_type=='Standing front and back':
        SID='Standing front and bac'

    #清除原有資料
    plt.clf()
    #X座標
    x = [1, 2, 3, 4, 5, 6]

    #讀取 BenchmarkDB &對應SID資料
    BenchmarkDB_Data=pd.read_csv(BenchmarkDB_Path)
    BenchmarkDB_Mean = BenchmarkDB_Data[(BenchmarkDB_Data['Clinical'] == SID) & (BenchmarkDB_Data['Data_Type'] == 'Mean')]
    BenchmarkDB_Mean.drop(['Clinical', 'Data_Type'], axis=1, inplace=True)
    BenchmarkDB_SD = BenchmarkDB_Data[(BenchmarkDB_Data['Clinical'] == SID) & (BenchmarkDB_Data['Data_Type'] == 'SD')]
    BenchmarkDB_SD.drop(['Clinical', 'Data_Type'], axis=1, inplace=True)
    #平均值
    MT_Data_Mean=BenchmarkDB_Mean[BenchmarkDB_Mean['SID'] == 'MT']
    MT_list_Mean = MT_Data_Mean.values.tolist()[0][1:]
    Co_Data_Mean=BenchmarkDB_Mean[BenchmarkDB_Mean['SID'] == 'Co']
    Co_list_Mean = Co_Data_Mean.values.tolist()[0][1:]
    DVR_Data_Mean=BenchmarkDB_Mean[BenchmarkDB_Mean['SID'] == 'DVR']
    DVR_list_Mean = DVR_Data_Mean.values.tolist()[0][1:]
    #標準差
    MT_Data_SD=BenchmarkDB_SD[BenchmarkDB_SD['SID'] == 'MT']
    MT_list_SD = MT_Data_SD.values.tolist()[0][1:]
    Co_Data_SD=BenchmarkDB_SD[BenchmarkDB_SD['SID'] == 'Co']
    Co_list_SD = Co_Data_SD.values.tolist()[0][1:]
    DVR_Data_SD=BenchmarkDB_SD[BenchmarkDB_SD['SID'] == 'DVR']
    DVR_list_SD = DVR_Data_SD.values.tolist()[0][1:]

    #資料讀取
    Csv_Data=pd.read_csv(SID_Path)
    #日期格式轉換
    Csv_Data['Date'] = pd.to_datetime(Csv_Data['Date'], format='%Y%m%d_%H%M%S')
    #篩選日期
    Original_End_Date = datetime.strptime(end_date, '%Y-%m-%d')
    End_Date_Add_Day = Original_End_Date + timedelta(days=1)
    Csv_Filtered_Data = Csv_Data[(Csv_Data['Date'] >= start_date) & (Csv_Data['Date'] <=End_Date_Add_Day)]
    #刪除DATE & SID
    Csv_Filtered_Data.drop(['Date', 'SID'], axis=1, inplace=True)
    print(Csv_Filtered_Data)
    #計算Csv平均值&標準差，如果沒有資料則跳過
    if Csv_Filtered_Data.empty:
        print('日期區間沒有資料')
    else:
        df = pd.DataFrame(Csv_Filtered_Data)
        #計算全部資料
        Csv_Mean = []
        Csv_SD = []
        for column in df.columns:
            mean = df[column].mean()
            std = df[column].std()
            Csv_Mean.append(mean)
            Csv_SD.append(std)
        plt.plot(x, Csv_Mean, marker='o', linestyle='-', label='Current', color='red')
        plt.errorbar(x, Csv_Mean, yerr=Csv_Mean, linestyle='None', marker='None', capsize=5, ecolor='red')


    #建立折線圖
    plt.plot(x, MT_list_Mean, marker='o', linestyle='-', label='MT', color='black')
    plt.plot(x, Co_list_Mean, marker='o', linestyle='-', label='Co', color='blue')
    plt.plot(x, DVR_list_Mean, marker='o', linestyle='-', label='DVR', color='green')
    #建立標準差
    plt.errorbar(x, MT_list_Mean, yerr=MT_list_SD, linestyle='None', marker='None', capsize=5, ecolor='black')
    plt.errorbar(x, Co_list_Mean, yerr=Co_list_SD, linestyle='None', marker='None', capsize=5, ecolor='blue')
    plt.errorbar(x, DVR_list_Mean, yerr=DVR_list_SD, linestyle='None', marker='None', capsize=5, ecolor='green')
    # 隱藏座標
    plt.xticks([])
    plt.yticks([])
    # 圖例移動至左邊
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1))

    timestamp = int(time.time())
    filename = f'Drawing_{timestamp}.png'

    #儲存PNG
    plt.savefig(os.path.join(Png_Path,'static',filename))

    #設定PNG路徑
    new_image_filename = os.path.join(Png_Path,'static',filename)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(root_dir, 'static')
    new_image_relative_path = os.path.relpath(new_image_filename, start=static_dir)
    return new_image_relative_path  

def Remove_Rng(Png_Path):
    files = os.listdir(Png_Path)
    File_Prefix = "Drawing_"
    for file in files:
        if file.startswith(File_Prefix) and file.endswith(".png"):
            # 使用 os.path.join 組合完整的檔案路徑
            file_to_delete = os.path.join(Png_Path, file)
            try:
                os.remove(file_to_delete)
                print(f"{file_to_delete} 已刪除")
            except Exception as e:
                print(f"刪除 {file_to_delete} 時出現錯誤: {e}")
    return
