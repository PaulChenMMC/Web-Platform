import os
import pandas as pd


#獲取指定資料夾內檔案(MOV MP4)
def get_VideoNames(folder_path):
    mp4_files = []
    for file in os.listdir(os.path.join(folder_path, "Input")):
        if file.endswith(".mp4") or file.endswith(".MOV"):
            mp4_files.append(file)
    return mp4_files

#建立資料夾
def create_Folder(folder_path,filename):
    filename = os.path.splitext(filename)[0]
    create_folder_path = os.path.join(folder_path, "Output",filename)
    if not os.path.exists(create_folder_path):
        os.makedirs(create_folder_path)
    else:
        return False,'Na'
    return True,create_folder_path

#資料校正
def Data_Calibration(RawData_Path):
    SID_data=[]
    df=pd.read_csv(os.path.join(RawData_Path, "SubjectBboxData.csv"))
    SID_data=df['ID'].unique()

    for SID in SID_data:
        dfA = pd.read_csv(os.path.join(RawData_Path, 'jointdata.csv'))
        dfB = pd.read_csv(os.path.join(RawData_Path, 'subjectbboxdata.csv'))
        #抓取 sub YX XC
        # 分離csv
        df_filteredB = dfB.loc[dfB['ID'] == SID]
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
        #DF_Path=os.path.join(RawData_Path,'SID_'+SID+'_Data_Calibration.csv')
        df_filter_copy.to_csv(os.path.join(RawData_Path,'SID_'+str(SID)+'_Data_Calibration.csv'), index=False)
    return 