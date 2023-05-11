# %%
# In[package]:
import streamlit as st
import pandas as pd
from pandas import DataFrame as df
import numpy as np
import matplotlib.pyplot as plt
import re
import json
from requests_html import HTMLSession
from requests.auth import HTTPBasicAuth
import warnings
warnings.filterwarnings('ignore')

# %%
# In[爬虫]:
session = HTMLSession()

def LoadData(url):
    r = session.get(url, auth=HTTPBasicAuth('admin', 'admin12345'))
    
    odata = r.html.text
    data = re.findall('\{(.*?)\}', re.findall('\[(.*?)\]', odata)[0])
    
    dict_list = []
    for d in data:
        dict_obj = json.loads("{" + d + "}")
        dict_list.append(dict_obj)

    result = {}
    for d in dict_list:
        for key, value in d.items():
            name = key.split(":")[0].strip('"')
            if name not in result:
                result[name] = []
            result[name].append(value)
    
    df = pd.DataFrame(result)
    df = df.drop(df.columns[0], axis=1)
    
    return (df)


def DataPrep(url_car, url_data, datatype):
    df_car = LoadData(url_car)
    df_car['Date'] = pd.to_datetime(df_car['Date']).dt.date
    df_car = df_car[df_car['DataType'] == datatype]
    df_car = df_car.drop(['DataType', 'Mna1Type'], axis=1)
    
    df_data = LoadData(url_data)
    
    df_merged = pd.concat([df_car.reset_index(drop=True), df_data.reset_index(drop=True)], axis=1)
    
    return df_merged


def DataPrep_Mna1(url_car, url_data, man1type):
    df_car = LoadData(url_car)
    df_car['Date'] = pd.to_datetime(df_car['Date']).dt.date
    df_car = df_car[df_car['Mna1Type'] == man1type]
    df_car = df_car.drop(['DataType', 'Mna1Type'], axis=1)
    
    df_data = LoadData(url_data)
    
    df_merged = pd.concat([df_car.reset_index(drop=True), df_data.reset_index(drop=True)], axis=1)
    
    return df_merged

# car 车
url_car = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/Cars'

# '''
# 检具数据
# '''
# # RoofSurface 车顶型面
# url_RoofSurface = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RoofSurface_Sunroofs'
# df_RoofSurface = DataPrep(url_car, url_RoofSurface, 'RoofSurface')

# # WindshieldFrameFace 前后风窗框型面
# # FrontWindshield 前风窗
# url_FrontWindshield = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/FrontWindshields'
# df_FrontWindshield = DataPrep(url_car, url_FrontWindshield, 'WindshieldFrameFace')
# # RearWindshield 后风窗
# url_RearWindshield = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RearWindshields'
# df_RearWindshield = DataPrep(url_car, url_RearWindshield, 'WindshieldFrameFace')
# # FrameFaceBackCover 框型面后盖
# url_FrameFaceBackCover = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/FrameFaceBackCovers'
# df_FrameFaceBackCover = DataPrep(url_car, url_FrameFaceBackCover, 'WindshieldFrameFace')

# # RoofLaserWeldingDrop 车顶激光焊落差
# # RoofLaserWeldingDrop_Left 左侧落差
# url_RoofLaserWeldingDrop_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RoofLaserWeldingDrop_Lefts'
# df_RoofLaserWeldingDrop_Left = DataPrep(url_car, url_RoofLaserWeldingDrop_Left, 'RoofLaserWeldingDrop')
# # RoofLaserWeldingDrop_Right 右侧落差
# url_RoofLaserWeldingDrop_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RoofLaserWeldingDrop_Rights'
# df_RoofLaserWeldingDrop_Right = DataPrep(url_car, url_RoofLaserWeldingDrop_Right, 'RoofLaserWeldingDrop')


# '''
# 匹配数据
# '''
# # ZP5
# # ZP5_FrontCover_Left 前盖左侧
# url_ZP5_FrontCover_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontCover_Lefts'
# df_ZP5_FrontCover_Left = DataPrep(url_car, url_ZP5_FrontCover_Left, 'ZP5')
# # ZP5_FrontCover_Right 前盖右侧
# url_ZP5_FrontCover_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontCover_Rights'
# df_ZP5_FrontCover_Right = DataPrep(url_car, url_ZP5_FrontCover_Right, 'ZP5')
# # ZP5_BackCover_Left 后盖左侧
# url_ZP5_BackCover_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_BackCover_Lefts'
# df_ZP5_BackCover_Left = DataPrep(url_car, url_ZP5_BackCover_Left, 'ZP5')
# # ZP5_BackCover_Right 后盖右侧
# url_ZP5_BackCover_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_BackCover_Rights'
# df_ZP5_BackCover_Right = DataPrep(url_car, url_ZP5_BackCover_Right, 'ZP5')
# # ZP5_FrontRearDoor_Left 前后门左侧
# url_ZP5_FrontRearDoor_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontRearDoor_Lefts'
# df_ZP5_FrontRearDoor_Left = DataPrep(url_car, url_ZP5_FrontRearDoor_Left, 'ZP5')
# # ZP5_FrontRearDoor_Right 前后门右侧
# url_ZP5_FrontRearDoor_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontRearDoor_Rights'
# df_ZP5_FrontRearDoor_Right = DataPrep(url_car, url_ZP5_FrontRearDoor_Right, 'ZP5')
# # ZP5_FrontRearDoor_Flatness_Left 前后门平整度左侧
# url_ZP5_FrontRearDoor_Flatness_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontRearDoor_Flatness_Lefts'
# df_ZP5_FrontRearDoor_Flatness_Left = DataPrep(url_car, url_ZP5_FrontRearDoor_Flatness_Left, 'ZP5')
# # ZP5_FrontRearDoor_Flatness_Right 前后门平整度右侧
# url_ZP5_FrontRearDoor_Flatness_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontRearDoor_Flatness_Rights'
# df_ZP5_FrontRearDoor_Flatness_Right = DataPrep(url_car, url_ZP5_FrontRearDoor_Flatness_Right, 'ZP5')

# # ZP8
# # ZP8_FrontCover_Left 前盖左侧
# url_ZP8_FrontCover_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontCover_Lefts'
# df_ZP8_FrontCover_Left = DataPrep(url_car, url_ZP8_FrontCover_Left, 'ZP8')
# # ZP8_FrontCover_Right 前盖右侧
# url_ZP8_FrontCover_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontCover_Rights'
# df_ZP8_FrontCover_Right = DataPrep(url_car, url_ZP8_FrontCover_Right, 'ZP8')
# # ZP8_BackCover_Left 后盖左侧
# url_ZP8_BackCover_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_BackCover_Lefts'
# df_ZP8_BackCover_Left = DataPrep(url_car, url_ZP8_BackCover_Left, 'ZP8')
# # ZP8_BackCover_Right 后盖右侧
# url_ZP8_BackCover_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_BackCover_Rights'
# df_ZP8_BackCover_Right = DataPrep(url_car, url_ZP8_BackCover_Right, 'ZP8')
# # ZP8_FrontRearDoor_Left 前后门左侧
# url_ZP8_FrontRearDoor_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontRearDoor_Lefts'
# df_ZP8_FrontRearDoor_Left = DataPrep(url_car, url_ZP8_FrontRearDoor_Left, 'ZP8')
# # ZP8_FrontRearDoor_Right 前后门右侧
# url_ZP8_FrontRearDoor_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontRearDoor_Rights'
# df_ZP8_FrontRearDoor_Right = DataPrep(url_car, url_ZP8_FrontRearDoor_Right, 'ZP8')
# # ZP8_FrontRearDoor_Flatness_Left 前后门平整度左侧
# url_ZP8_FrontRearDoor_Flatness_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontRearDoor_Flatness_Lefts'
# df_ZP8_FrontRearDoor_Flatness_Left = DataPrep(url_car, url_ZP8_FrontRearDoor_Flatness_Left, 'ZP8')
# # ZP8_FrontRearDoor_Flatness_Right 前后门平整度右侧
# url_ZP8_FrontRearDoor_Flatness_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontRearDoor_Flatness_Rights'
# df_ZP8_FrontRearDoor_Flatness_Right = DataPrep(url_car, url_ZP8_FrontRearDoor_Flatness_Right, 'ZP8')


# '''
# 扭矩数据
# '''
# # Mna1
# # LNF_1
# url_LNF_1 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LNF_1s'
# df_LNF_1 = DataPrep_Mna1(url_car, url_LNF_1, 'Mna1_7830')
# # LNF_2
# url_LNF_2 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LNF_2s'
# df_LNF_2 = DataPrep_Mna1(url_car, url_LNF_2, 'Mna1_7830')

# # LNF2_1
# url_LNF2_1 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LNF2_1s'
# df_LNF2_1 = DataPrep_Mna1(url_car, url_LNF2_1, 'Mna1_7920')
# # LNF2_2
# url_LNF2_2 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LNF2_2s'
# df_LNF2_2 = DataPrep_Mna1(url_car, url_LNF2_2, 'Mna1_7920')

# # BA7
# url_BA7 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/BA7s'
# df_BA7 = DataPrep_Mna1(url_car, url_BA7, 'Mna1_BA7')

# # LeftWingPanel
# url_LeftWingPanel = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LeftWingPanels'
# df_LeftWingPanel = DataPrep_Mna1(url_car, url_LeftWingPanel, 'Mna1_LeftWingPanel')

# # RightWingPanel
# url_RightWingPanel = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RightWingPanels'
# df_RightWingPanel = DataPrep_Mna1(url_car, url_RightWingPanel, 'Mna1_RightWingPanel')

# %%
# In[控制图部分]:
import os
from fitter import Fitter
import math as m

def toChar(self, toSmallChar=False, toBigChar=True):
    #把数字转换成相应的字符,0-->'A' 26-->'AA'
    init_number = 0
    increment = 0
    res_char = ''
    if not toSmallChar and not toBigChar:
        return '请指定类型'
    else:
        if toSmallChar:
            init_number = self
            increment = ord('a')
        else:
            init_number = self
            increment = ord('A')
    
    shang = init_number
    
    if shang>=26:
        while shang>=26 :
            shang,yu = divmod(shang, 26)
            char = chr(yu + increment)
            res_char = char + res_char
        res_char = chr(shang + increment - 1) + res_char
    else:
        res_char = chr(shang + increment)
    return res_char

def EWMA_norm(data,dist):
    #data type is dataframe with the first column'date' and second'data'
    #重命名，方便后续处理
    data.columns = ['date','data']
    groupMean = data.groupby(by='date').mean()
    groupStd = data.groupby(by='date').std()
    groupNum = data.groupby(by='date').count()
    date = groupMean.index
    groupMean.index = range(1,len(groupMean)+1)
    groupStd.index = range(1,len(groupStd)+1)
    groupNum.index = range(1,len(groupNum)+1)
    
    #估计整体均值和方差
    mean = (groupMean*groupNum).sum()/groupNum.sum()
    std = m.sqrt(((groupStd**2)*(groupNum-1)).sum()/(groupNum.sum()-groupNum.count()))
    #设定权重和k值，详见论文
    r = 0.2
    k = 2.962
    #计算各个子组的关键参数
    EWMA = pd.DataFrame(data=None, columns = ['dot','LCL','CL','UCL'])
    for i in range(1, len(groupMean)+1):                
        if i == 1:
            sigmaz = (r*std)/(m.sqrt(groupNum.loc[i]))
            row = pd.concat([r*groupMean.loc[i] + (1-r)*mean ,mean - k * sigmaz, mean, mean + k * sigmaz],axis=1)
            row = row.values[0]
            EWMA.loc[i] = row
        else:
            s = 0
            for j in range(1,i+1):
                s = s + (1-r)**(2*(i-j))/groupNum.loc[j]
            sigmaz = std*r*m.sqrt(s)
            row = pd.concat([r*groupMean.loc[i] + (1-r)*EWMA.loc[i-1, 'dot'],mean - k*sigmaz, mean, mean+k*sigmaz],axis=1)
            row = row.values[0]
            EWMA.loc[i] = row
    
    #标出出界点
    #EWMA_outer = EWMA.loc[(EWMA['dot']<=EWMA['LCL']) | (EWMA['dot']>=EWMA['UCL']), ['dot']]
    #plt.scatter(EWMA_outer.index,EWMA_outer['dot'],marker='o', color='red')
    flag = (EWMA['dot']<=EWMA['LCL']) | (EWMA['dot']>=EWMA['UCL'])
    #绘图环节
    plt.figure(figsize=(15,7.5))
    plt.plot(range(1, len(EWMA['dot'])+1), EWMA['dot'], linestyle='-', color = 'black')
    plt.scatter(EWMA.loc[flag,'dot'].index, EWMA.loc[flag,'dot'], marker ='o', color = 'red')
    plt.scatter(EWMA.loc[~flag,'dot'].index, EWMA.loc[~flag,'dot'], marker ='o', color = 'black')
    plt.step(x=range(1, len(EWMA['dot'])+1), y=EWMA['LCL'], color='red', linestyle='dashed')
    plt.step(x=range(1, len(EWMA['dot'])+1), y=EWMA['UCL'], color='red', linestyle='dashed')
    plt.step(x=range(1, len(EWMA['dot'])+1), y=EWMA['CL'], color='blue', linestyle='dashed')
    plt.xticks(range(1, len(EWMA['dot'])+1, 5), date[::5], color='black', rotation=90)
    if dist == 'lognorm':
        plt.title('EWMA_lognorm chart')
    else:
        plt.title('EWMA_norm chart')
    plt.xlabel('Date')
    plt.ylabel('EWMA value')
    st.pyplot(plt.gcf())
    errordate = date[flag].tolist()
    errordate = list(map(lambda x:str(x), errordate))
    st.write("问题日期")
    st.write(errordate)
    return

def EWMA_lognorm(data,datatype):
    data.columns = ['date','data']
    data.loc[:,'data'] = data['data'].apply(lambda x:m.log(x))
    EWMA_norm(data,datatype)
    return

def t_expon(data,loc,theta):
    #变换参数beta
    beta = 3.6
    #变换
    data.columns = ['date','data']
    data.loc[:,'data'] = data['data'].apply(lambda x:(x-loc)**(1/beta))
    thetaS = theta**(1/beta)
    #计算关键参数
    import math as m##可能要改
    muS = thetaS*(m.gamma(1+1/beta))
    stdS = thetaS*(m.sqrt(m.gamma(1+2/beta)-m.gamma(1+1/beta)**2))
    #设定参数，引用自论文,r0=370,n=3
    k1 = 2.86733
    k2 = 1.80537
    n = 3
    #计算子组情况
    groupMean = data.groupby(by='date').mean()
    groupNum = data.groupby(by='date').count()
    date = groupMean.index
    groupMean.index = range(1,len(groupMean)+1)
    groupNum.index = range(1,len(groupNum)+1)
    t = pd.DataFrame(data=None, columns = ['dot','LCL1','LCL2','CL','UCL2','UCL1'])
    for i in range(1,len(groupMean)+1):
        row = [groupMean.loc[i].values[0],muS - k1*stdS/(m.sqrt(groupNum.loc[i])),                muS - k2*stdS/(m.sqrt(groupNum.loc[i])),muS,muS + k2*stdS/(m.sqrt(groupNum.loc[i])),muS + k1*stdS/(m.sqrt(groupNum.loc[i]))]
        t.loc[i] = row
    #判异准则 
    flag = np.zeros(len(groupMean),dtype = int)#出界:1，连续n点:2，正常:0
    outer = ((t['dot']<=t['LCL1']).values | (t['dot']>=t['UCL1']).values)
    decide = ((t['dot']<=t['LCL2']).values | (t['dot']>=t['UCL2']).values)
    #准则1
    for i in range(n,len(groupMean)+1):
        j = i-1#index in the list
        temp = np.zeros(len(groupMean),dtype = int)#一个暂时的boolarray
        if sum(decide[j-n+1:j+1])>=2: #超过2个在decide领域
            temp[j-n+1:j+1] = decide[j-n+1:j+1]
            flag[temp] = 2

    #准则2
    flag[outer] = 1
    
    #标出出界点
    #绘图环节
    plt.figure(figsize=(15,7.5))
    plt.plot(range(1, len(t['dot'])+1), t['dot'], linestyle='-', color = 'black')
    plt.scatter(t.loc[flag == 0,'dot'].index, t.loc[flag == 0,'dot'], marker ='o', color = 'black')
    plt.scatter(t.loc[flag == 1,'dot'].index, t.loc[flag == 1,'dot'], marker ='o', color = 'red')
    plt.scatter(t.loc[flag == 2,'dot'].index, t.loc[flag == 2,'dot'], marker ='o', color = 'green')
    plt.step(x=range(1, len(t['dot'])+1), y=t['LCL1'], color='red', linestyle='dashed')
    plt.step(x=range(1, len(t['dot'])+1), y=t['LCL2'], color='green', linestyle='dashed')
    plt.step(x=range(1, len(t['dot'])+1), y=t['UCL1'], color='red', linestyle='dashed')
    plt.step(x=range(1, len(t['dot'])+1), y=t['UCL2'], color='green', linestyle='dashed')
    plt.step(x=range(1, len(t['dot'])+1), y=t['CL'], color='blue', linestyle='dashed')
    plt.xticks(range(1, len(t['dot'])+1, 5), date[::5], color='black', rotation=90)
    plt.title('t_expon chart')
    plt.xlabel('Date')
    plt.ylabel('dot value')
    st.pyplot(plt.gcf())
    errordate = date[flag != 0].tolist()
    errordate = list(map(lambda x:str(x), errordate))
    st.write('有问题的日期是:')
    st.write(errordate)
    return

def t_gamma(data,shape,loc,scale):
    #对shape加以限制
    if shape>25:
        print('拟合结果为gamma为偶然因素，请查看其是否为正态分布')
        EWMA_norm(data,'norm')
        return
    #变换参数beta
    beta = 3
    #变换
    data.columns = ['date','data']
    data.loc[:,'data'] = data['data'].apply(lambda x:(x-loc)**(1/beta))
    #计算关键参数
    import math as m##可能要改
    muS = (scale**(1/beta))*(m.gamma(shape+1/beta))/(m.gamma(shape))
    stdS = m.sqrt((scale**(2/beta))*(m.gamma(shape+2/beta))/(m.gamma(shape))-(muS**2))
    print(muS)
    print(stdS)
    #设定参数，引用自论文,r0=370,n=3
#     a = 2         a = 5         a =10         a = 20
#     k1 = 3.480068 k1 = 4.587742 k1 = 5.79097  k1 = 7.35734
#     k2 = 2.982044 k2 = 4.293158 k2 = 5.372559 k2 = 6.89495
    n = 3
    k1sample =np.array([3.480068,4.587742,5.79097,7.35734])
    k2sample =np.array([2.982044,4.293158,5.372559,6.89495])
    a = np.array([2,5,10,20])
    
    from scipy.interpolate import interp1d
    
    func_k1 = interp1d(a,k1sample,kind='cubic')
    func_k2 = interp1d(a,k2sample,kind='cubic')
    k1 = func_k1(shape)
    k2 = func_k2(shape)
    
    #计算子组情况
    groupMean = data.groupby(by='date').mean()
    groupNum = data.groupby(by='date').count()
    date = groupMean.index
    groupMean.index = range(1,len(groupMean)+1)
    groupNum.index = range(1,len(groupNum)+1)
    t = pd.DataFrame(data=None, columns = ['dot','LCL1','LCL2','CL','UCL2','UCL1'])
    for i in range(1,len(groupMean)+1):
        row = [groupMean.loc[i].values[0],muS - k1*stdS/(m.sqrt(groupNum.loc[i])), muS - k2*stdS/(m.sqrt(groupNum.loc[i])),muS,muS + k2*stdS/(m.sqrt(groupNum.loc[i])),muS + k1*stdS/(m.sqrt(groupNum.loc[i]))]
        t.loc[i] = row
    print(t)
    #判异准则 
    flag = np.zeros(len(groupMean),dtype = int)#出界:1，连续n点:2，正常:0
    outer = ((t['dot']<=t['LCL1']).values | (t['dot']>=t['UCL1']).values)
    decide = ((t['dot']<=t['LCL2']).values | (t['dot']>=t['UCL2']).values)
    #准则1
    for i in range(n,len(groupMean)+1):
        j = i-1#index in the list
        temp = np.zeros(len(groupMean),dtype = int)#一个暂时的boolarray
        if sum(decide[j-n+1:j+1])>=2: #超过2个在decide领域
            temp[j-n+1:j+1] = decide[j-n+1:j+1]
            flag[temp] = 2

    #准则2
    flag[outer] = 1
    
    #标出出界点
    #绘图环节
    plt.figure(figsize=(15,7.5))
    plt.plot(range(1, len(t['dot'])+1), t['dot'], linestyle='-', color = 'black')
    plt.scatter(t.loc[flag == 0,'dot'].index, t.loc[flag == 0,'dot'], marker ='o', color = 'black')
    plt.scatter(t.loc[flag == 1,'dot'].index, t.loc[flag == 1,'dot'], marker ='o', color = 'red')
    plt.scatter(t.loc[flag == 2,'dot'].index, t.loc[flag == 2,'dot'], marker ='o', color = 'green')
    plt.step(x=range(1, len(t['dot'])+1), y=t['LCL1'], color='red', linestyle='dashed')
    plt.step(x=range(1, len(t['dot'])+1), y=t['LCL2'], color='green', linestyle='dashed')
    plt.step(x=range(1, len(t['dot'])+1), y=t['UCL1'], color='red', linestyle='dashed')
    plt.step(x=range(1, len(t['dot'])+1), y=t['UCL2'], color='green', linestyle='dashed')
    plt.step(x=range(1, len(t['dot'])+1), y=t['CL'], color='blue', linestyle='dashed')
    plt.xticks(range(1, len(t['dot'])+1, 5), date[::5], color='black', rotation=90)
    plt.title('t_gamma chart')
    plt.xlabel('Date')
    plt.ylabel('dot value')
    st.pyplot(plt.gcf())
    errordate = date[flag != 0].tolist()
    errordate = list(map(lambda x:str(x), errordate))
    st.write('有问题的日期是：')
    st.write(errordate)
    return

def EWMA_rayleigh(data,loc,scale):
    #重命名，方便后续处理
    data.columns = ['date','data']
    #估计子组信息
    groupNum = data.groupby(by='date').count()#获取各个子组内数据个数
    date = groupNum.index
    groupNum.index = range(1,len(groupNum)+1)
    data.loc[:,'data'] = data['data'].apply(lambda x:(x-loc)**2)#先去除loc影响，再求平方和
    groupSquareSum = data.groupby(by='date').sum()
    groupVr = groupSquareSum
    groupVr.loc[:,'data'] = groupSquareSum.values/(2*groupNum.values)
    groupVr.index = range(1,len(groupNum)+1)
    
    #构建控制图
    #Vri~Gamma(ni,scale^2/ni)左边shape右边parameter
    #估计整体均值和方差
    mean = scale**2
    std = scale**2
    #设定权重和k值，详见论文
    r = 0.2
    k = 2.962
    #计算各个子组的关键参数
    EWMA = pd.DataFrame(data=None, columns = ['dot','LCL','CL','UCL'])
    for i in range(1, len(groupVr)+1):                
        if i == 1:
            sigmaz = (r*std)/(m.sqrt(groupNum.loc[i]))
            row = [r*groupVr.loc[i].values + (1-r)*mean ,mean - k * sigmaz, mean, mean + k * sigmaz]
            EWMA.loc[i] = row
        else:
            s = 0
            for j in range(1,i+1):
                s = s + (1-r)**(2*(i-j))/groupNum.loc[j]
            sigmaz = std*r*m.sqrt(s)
            row = [r*groupVr.loc[i].values + (1-r)*EWMA.loc[i-1, 'dot'],mean - k*sigmaz, mean, mean+k*sigmaz]
            EWMA.loc[i] = row
    
    #标出出界点
    flag = (EWMA['dot']<=EWMA['LCL']) | (EWMA['dot']>=EWMA['UCL'])
    #绘图环节
    plt.figure(figsize=(15,7.5))
    plt.plot(range(1, len(EWMA['dot'])+1), EWMA['dot'], linestyle='-', color = 'black')
    plt.scatter(EWMA.loc[flag,'dot'].index, EWMA.loc[flag,'dot'], marker ='o', color = 'red')
    plt.scatter(EWMA.loc[~flag,'dot'].index, EWMA.loc[~flag,'dot'], marker ='o', color = 'black')
    plt.step(x=range(1, len(EWMA['dot'])+1), y=EWMA['LCL'], color='red', linestyle='dashed')
    plt.step(x=range(1, len(EWMA['dot'])+1), y=EWMA['UCL'], color='red', linestyle='dashed')
    plt.step(x=range(1, len(EWMA['dot'])+1), y=EWMA['CL'], color='blue', linestyle='dashed')
    plt.xticks(range(1, len(EWMA['dot'])+1, 5), date[::5], color='black', rotation=90)
    plt.title('EWMA_rayleigh chart')
    plt.xlabel('Date')
    plt.ylabel('EWMA value')
    st.pyplot(plt.gcf())
    errordate = date[flag].tolist()
    errordate = list(map(lambda x:str(x), errordate))
    st.write('有问题的日期是：')
    st.write(errordate)
    return

def shewhart_rayleigh(data,loc,scale):
    #重命名，方便后续处理
    data.columns = ['date','data']
    #估计子组信息
    groupNum = data.groupby(by='date').count()#获取各个子组内数据个数
    date = groupNum.index
    groupNum.index = range(1,len(groupNum)+1)
    data.loc[:,'data'] = data['data'].apply(lambda x:(x-loc)**2)#先去除loc影响，再求平方和
    groupSquareSum = data.groupby(by='date').sum()
    groupVr = groupSquareSum
    groupVr.loc[:,'data'] = groupSquareSum.values/(2*groupNum.values)
    
    #构建控制图
    #Vri~Gamma(ni,scale^2/ni)左边shape右边parameter
    alpha = 0.0027
    CL = scale**2
    from scipy.stats import gamma
    UCL = gamma.isf(alpha/2,groupNum.values,loc=0,scale=1)*(scale**2)/groupNum.values
    LCL = gamma.isf(1-alpha/2,groupNum.values,loc=0,scale=1)*(scale**2)/groupNum.values
    dot = groupVr.values
    CL = np.full_like(dot, CL)
    #标出出界点
    flag = ((dot<=LCL) | (dot>=UCL))[:,0]
    #绘图环节
    point = np.array(range(1, len(dot)+1))
    plt.figure(figsize=(15,7.5))
    plt.plot(range(1, len(dot)+1), dot, linestyle='-', color = 'black')
    plt.scatter(point[flag], dot[flag], marker ='o', color = 'red')
    plt.scatter(point[~flag], dot[~flag], marker ='o', color = 'black')
    plt.step(x=range(1, len(dot)+1), y=LCL, color='red', linestyle='dashed')
    plt.step(x=range(1, len(dot)+1), y=UCL, color='red', linestyle='dashed')
    plt.step(x=range(1, len(dot)+1), y=CL, color='blue', linestyle='dashed')
    plt.xticks(range(1, len(dot)+1, 5), date[::5], color='black', rotation=90)
    plt.title('shewhart_rayleigh chart')
    plt.xlabel('Date')
    plt.ylabel('Vr value')
    st.pyplot(plt.gcf())
    errordate = date[flag].tolist()
    errordate = list(map(lambda x:str(x), errordate))
    st.write('有问题的日期是：')
    st.write(errordate)
    return

def Lapage_chart(data,sample,table):
    from scipy.stats import rankdata
    m_sample = len(sample)
    #重命名，方便后续处理
    data.columns = ['date','data']
    record = []
    #计算每一组的统计量
    for date,group in data.groupby(by='date'):
        test = group['data'].values#得到这一组的测试样本
        n = len(test)
        N = (m_sample+n)
        #step 1-3
        full = np.concatenate([test,sample])
        T1 = np.sum(rankdata(full)[0:len(test)])
        T2 = np.sum(abs(rankdata(full)[0:len(test)] - 1/2*(N+1)))
        muT1 = 1/2*n*(N+1)
        varT1 = 1/12*m_sample*n*(N+1)
        if (N % 2) == 0:
            muT2 = n*N/4
            varT2 = (1/48)*(m_sample*n)*((N**2)-4)/(N-1)
        else:
            muT2 = (n*((N**2)-1))/(4*N)
            varT2 = (1/48)*((m_sample*n*(N+1)*((N**2)+3))/(N**2))
        #step 4-5
        s1_2 = ((T1-muT1)/m.sqrt(varT1))**2
        s2_2 = ((T2-muT2)/m.sqrt(varT2))**2
        s_2 = s1_2+s2_2
        #查找对应的上H
        H = table.loc[n,'H']
        H1 = table.loc[n,'H1']
        H2 = table.loc[n,'H2']
        #record
        record.append([H,H1,H2,s1_2,s2_2,s_2])
    record = np.array(record)
    #标出出界点
    groupN = np.array(range(1, len(record)+1))
    H = record[:,0]
    H1 = record[:,1]
    H2 = record[:,2]
    s1_2 = record[:,3]
    s2_2 = record[:,4]
    s_2 = record[:,5]
    flag = (s_2 >= H)
    flag1 = (s1_2 >= H1) & (s2_2 < H2) & flag 
    flag2 = (s2_2 >= H2) & (s1_2 < H1) & flag
    flag12 = (s1_2 >= H1) & (s2_2 >= H2) & flag
    flagother = (s1_2 < H1) & (s2_2 < H2) & flag
    
    groupNum = data.groupby(by='date').count()#获取各个子组内数据个数
    date = groupNum.index#获取日期
    #画图
    plt.figure(figsize=(15,7.5))
    plt.plot(groupN, s_2, linestyle='-', color = 'black')
    plt.scatter(groupN[flag12], s_2[flag12], marker ='o', color = 'red', label = 'loc&scale changes')
    plt.scatter(groupN[flag1], s_2[flag1], marker ='o', color = 'blue', label = 'loc changes')
    plt.scatter(groupN[flag2], s_2[flag2], marker ='o', color = 'green', label = 'scale changes')
    plt.scatter(groupN[flagother], s_2[flagother], marker ='o', color = 'yellow', label = 'unknown changes')
    plt.legend(loc = 1)
    plt.scatter(groupN[~flag], s_2[~flag], marker ='o', color = 'black')
    plt.step(x=groupN, y=np.zeros_like(H), color='red', linestyle='dashed')
    plt.step(x=groupN, y=H, color='red', linestyle='dashed')
    plt.xticks(range(1, len(record)+1, 5), date[::5], color='black', rotation=90)
    plt.title('Lapage Chart')
    plt.xlabel('Date')
    plt.ylabel('s^2')
    st.pyplot(plt.gcf())
    errordate = date[flag].tolist()
    errordate = list(map(lambda x:str(x), errordate))
    st.write('有问题的日期是：')
    st.write(errordate)
    return

from fitter import get_common_distributions
dist_common = get_common_distributions()

def print_SPC(data,left,right):
    col = data.iloc[: ,1]
    dataList = col.tolist()
    #可指定拟合池子
    dist = dist_common
    binNum = round(m.sqrt(len(dataList)))
    
    #拟合结果并输出
    f = Fitter(dataList, xmin=left, xmax=right, bins=binNum, distributions=dist_common, timeout=100, density=True)
    f.fit(progress=False, n_jobs=-1, max_workers=-1)
    best = f.get_best(method='bic')
    st.write("最佳方法和参数：")
    st.write(best)
    result = f.summary(Nbest=5, lw=2, plot=True, method='bic') # 返回最好的Nbest个分布及误差，并绘制数据分布和Nbest分布
    st.write("最好的五结果和拟合准则：")
    st.write(result)
    st.write("拟合直方图：")
    st.pyplot(plt.gcf())
    
    #对该列的变量进行绘图
    (dist, para), = best.items()
    st.write("控制图：")
    dataSPC = data
    table = pd.DataFrame(data=[[5.65403938293457,2.88461538461538,2.76940801134349],
                       [10.8827877044677,5.66037735849056,5.66666666666666],
                       [15.46875,7.05333333333333,6.55006374795848],
                       [18.75,8.25709090909091,7.34282967032967],
                       [18.75,9.09342857142857,7.6542743913946],
                       [18.75,9.44280701754386,7.89246487867177],
                       [22.5,9.99014778325123,8.02405540508607],
                       [18.75,10.1089830508474,7.82528571428571],
                       [18.75,10.2684444444444,8.0423013139431],
                       [18.75,10.5820327868852,7.93999555061179]],
                 columns = ['H','H1','H2'],
                 index=[1,2,3,4,5,6,7,8,9,10])
    if dist == 'norm':
        EWMA_norm(dataSPC,dist)
    elif dist == 'lognorm':
        EWMA_lognorm(dataSPC,dist)
    elif dist == 'expon':
        loc1 = para.get('loc')
        scale1 = para.get('scale')
        t_expon(dataSPC,loc1,scale1)
    elif dist == 'gamma':
        a1 = para.get('a')
        loc1 = para.get('loc')
        scale1 = para.get('scale')
        t_gamma(dataSPC,a1,loc1,scale1)
    elif dist == 'rayleigh':
        loc1 = para.get('loc')
        scale1 = para.get('scale')
        EWMA_rayleigh(dataSPC,loc1,scale1)
    elif dist == 'cauchy':
        loc1 = para.get('loc')
        scale1 = para.get('scale')
        from scipy.stats import cauchy
        random = cauchy.rvs(loc=loc1, scale=scale1,size=50,random_state = 42)
        Lapage_chart(dataSPC, random, table)
    elif dist == 'chi2':
        loc1 = para.get('loc')
        scale1 = para.get('scale')
        df1 = para.get('df')
        from scipy.stats import chi2
        random = chi2.rvs(df=df1,loc=loc1,scale=scale1,size=50,random_state=42)
        Lapage_chart(dataSPC, random, table)
    elif dist == 'exponpow':
        loc1 = para.get('loc')
        scale1 = para.get('scale')
        b1 = para.get('b')
        from scipy.stats import exponpow
        random = exponpow.rvs(b=b1,loc=loc1,scale=scale1,size=50,random_state=42)
        Lapage_chart(dataSPC, random, table)
    elif dist == 'powerlaw':
        loc1 = para.get('loc')
        scale1 = para.get('scale')
        a1 = para.get('a')
        from scipy.stats import powerlaw
        random = powerlaw.rvs(a=a1,loc=loc1,scale=scale1,size=50,random_state=42)
        Lapage_chart(dataSPC, random, table)
    elif dist == 'uniform':
        loc1 = para.get('loc')
        scale1 = para.get('scale')
        from scipy.stats import uniform
        random = uniform.rvs(loc=loc1,scale=scale1,size=50,random_state=42)
        Lapage_chart(dataSPC, random, table)
    return

# %%
# In[能力指数部分]
import numpy as np
import math
def boxcox_change(data,boxcox):
    data1=data.iloc[:, 1] ##数据
    quantity=len(data1)

    for i in range(0,quantity):##bc变换
        data.iloc[i,1]=(pow(data1[i],boxcox)-1)/boxcox

    st.write("转换方法：Box-Cox变换")
    st.write("最佳参数λ：",boxcox)
    return data

def johnson_change(data, Jtype, J1, J2, J3, J4):
    data1 = data.iloc[:, 1]  ##数据
    quantity = len(data1)
    if Jtype == 1:##有界变换
        for i in range(0, quantity):
            data[i:1]=J1+J2*(np.log((data1[i]-J3)/(J4-data1[i])))
        st.write("转换方法：Johnson变换，有界变换")
    elif Jtype == 2:
        for i in range(0, quantity):
            data[i:1] = J1 + J2 * (math.asinh(data1[i] - J3) / J4)
        st.write("转换方法：Johnson变换，无界变换")
    st.write("最佳参数：形状参数：",J1,J2,"，""位置参数：",J3,"尺度参数：",J4)
    return data

def short_norm_Cp(data, up, low,cplong,corrtype):
    middle = data.iloc[:, 1]
    mean = np.mean(middle)  ##标准化
    standard = np.std(middle, ddof=1)
    quantity = len(data)
    for i in range(1,quantity):
        data.iloc[i, 1] = (middle[i] - mean) / standard
    data1 = data.iloc[:, 1]
    group=int((quantity-100)/10)
    date = data.iloc[:, 0]  ##日期
    cpkx = np.zeros(group) ##储存cpk
    y = np.zeros(group)##储存cp
    long_cp= np.zeros(group)
    numberforCp = np.zeros(group)  ##储存横轴
    for i in range (0,group):
        start=i*10+1
        end=start+100
        all=0
        for j in range (start,end):
             dt=data1[j+1]-data1[j]
             dt2=pow(dt,2)
             all+=dt2
        sigma1 = all /100 / 2
        sigma = pow(sigma1, 0.5)
        mean = np.mean(data1[start:end])
        Cp1 = (up - mean) / (3 * sigma)
        Cp2 = (mean - low) / (3 * sigma)
        Cp=(up - low) / (6 * sigma)
        Cpk=min(Cp1,Cp2)
        y[i] = Cp
        cpkx[i]=Cpk
        numberforCp[i] = i
    for i in range (0,10):
        around=round(i*quantity/10)+1
        date[i]=date[around]
    for i in range(0,group):
        long_cp[i]=cplong

    st.markdown(">**测点长短期能力指数折线图**")
    plt.figure(figsize=(10, 7))  # 设置绘图大小为20*15
    plt.xlabel('日期')  # 设置x、y轴标签
    plt.ylabel('过程能力指数Cp')  # 设置y轴刻度范围为0~11
    plt.plot(numberforCp,y, color='m', label='短期Cp')
    plt.plot(numberforCp,cpkx, color='g', label='短期Cpk')
    plt.xticks(ticks=range(0, round(group / 10) * 10, round(group / 10)), labels=date[0:10], rotation=45)
    if corrtype == 1:
        plt.plot(numberforCp, long_cp, color='b', label='长期单变量能力指数')
    elif corrtype == 2:
        plt.plot(numberforCp, long_cp, color='b', label='长期多元能力指数')
    plt.legend(loc="upper left")

    st.pyplot(plt.gcf())
    numberfordata = np.zeros(quantity)
    for i in range(0, quantity):
        numberfordata[i] = i

    st.markdown(
        '''
        - 注：短期能力指数由对应日期开始的100个观测值数据计算。
        - 图中若出现零值，则为短期内数据值一致，标准差为0，无法计算能力指数。
        ''')

    st.markdown(">**测点观测值折线图**")
    st.write("可将数据与边界值情况与上图对照查看过程能力。")
    plt.figure(figsize=(10, 7))  # 设置绘图大小为20*15
    plt.xlabel('日期')  # 设置x、y轴标签
    plt.ylabel('数据')  # 设置y轴刻度范围为0~11
    plt.plot(numberfordata[1:quantity], data1[1:quantity], color='m', label='测点数据')
    upline = np.array([up] * quantity)
    lowline = np.array([low] * quantity)
    plt.plot(numberfordata, upline, color='b', label='上限')
    plt.plot(numberfordata, lowline, color='g', label='下限')
    plt.xticks(ticks=range(0, round(quantity / 10) * 10, round(quantity / 10)), labels=date[0:10], rotation=45)

    plt.legend(loc="upper left")
    st.pyplot(plt.gcf())
    return

def unnorm_Cp(data,up,low):
    # 读取数据
    data1 = data.iloc[:, 1]  ##数据
    date = data.iloc[:, 0]  ##日期
    quantity=len(data1)
    group=int((quantity-100)/10)

    for i in range (0,quantity):##转为可作用于坐标轴格式
        date[i]=str(date[i])

    y=np.zeros(group)##储存cp
    numberforCp=np.zeros(group)##储存横轴

    long_cp=np.zeros(group)##长期cp
    for j in range (0,group):##短期cp计算

        start=j*10##起始点

        end=start+100##跨度100

        mean = np.mean(data1[start:end])
        p=0
        for i in range(start,end):
            k=mean-data1[i]
            if k>0:
                p+=1
            else:
                p+=0
        Px=p/100
        all=0
        for p in range(start, end):
            dt = data1[p+1] - data1[p]
            dt2 = pow(dt, 2)
            all+=dt2
        sigma1 =all/(100*2)
        sigma = pow(sigma1, 0.5)
        mid1=1-2*Px
        mid2=abs(mid1)
        mid3=mid2+1
        Wx=pow(mid3,0.5)
        Cp=(up-low)/(6*sigma*Wx)

        y[j]=Cp
        numberforCp[j]=j
    mean = np.mean(data1)
    sigma = np.std(data1, ddof=1)
    p=0
    for u in range(0,quantity):##长期cp计算
        k = mean - data1[u]
        if k > 0:
            p += 1
        else:
            p += 0

    Px = p / quantity
    mid1 = 1 - 2 * Px
    mid2 = abs(mid1)
    mid3 = mid2 + 1
    Wx = pow(mid3, 0.5)
    Cp=(up-low)/(6*sigma*Wx)
    for i in range(0,group):
        long_cp[i]=Cp
    for i in range (0,10):
        around=round(i*quantity/10)
        date[i]=date[around]

    st.write("转换方法：非正态方法")
    st.markdown(">**测点长短期能力指数折线图**")
    plt.figure(figsize=(10,7))  # 设置绘图大小为20*15
    plt.xlabel('日期')  # 设置x、y轴标签
    plt.ylabel('过程能力指数Cp') # 设置y轴刻度范围为0~11
    plt.plot(numberforCp,y,color='m',label='短期能力指数')

    plt.xticks(ticks=range(0,round(group/10)*10,round(group/10)), labels=date[0:10], rotation=45)
    plt.plot(numberforCp, long_cp, color='b', label='长期能力指数')
    plt.legend(loc="upper left")

    st.pyplot(plt.gcf())
    numberfordata = np.zeros(quantity)
    for i in range(0,quantity):
        numberfordata[i]=i
    st.markdown(
        '''
        - 注：短期能力指数由对应日期开始的100个观测值数据计算。
        - 图中若出现零值，则为短期内数据值一致，标准差为0，无法计算能力指数。
        ''')
    st.markdown(">**测点观测值折线图**")
    st.write("可将数据与边界值情况与上图对照查看过程能力。")
    plt.figure(figsize=(10, 7))  # 设置绘图大小为20*15
    plt.xlabel('日期')  # 设置x、y轴标签
    plt.ylabel('数据')  # 设置y轴刻度范围为0~11
    plt.plot(numberfordata, data1, color='m', label='测点数据')
    upline=np.array([up]*quantity)
    lowline=np.array([low]*quantity)
    plt.plot(numberfordata, upline, color='b', label='上限')
    plt.plot(numberfordata, lowline, color='g', label='下限')
    plt.xticks(ticks=range(0, round(quantity/10)*10, round(quantity/10)), labels=date[0:10], rotation=45)

    plt.legend(loc="upper left")
    st.pyplot(plt.gcf())


    return

def print_Cp(data,up,low,type,boxcox,Jtype,J1,J2,J3,J4,cplong,corrtype):
    if type==1:
        short_norm_Cp(boxcox_change(data,boxcox),up,low,cplong,corrtype)
    elif type==2:
        short_norm_Cp(johnson_change(data,Jtype,J1,J2,J3,J4),up,low,cplong,corrtype)
    elif type==3:
        short_norm_Cp(data,up,low,cplong,corrtype)
        st.write("转换类型：原始数据符合正态分布。")
    elif type==4:
        unnorm_Cp(data,up,low)

    return
# %%
# In[质量预测部分]
# import library
import seaborn as sns
from scipy import stats
import ruptures as rpt
from matplotlib.ticker import MultipleLocator
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from pandas.plotting import autocorrelation_plot
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing,ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.rc('font', family='SimHei', weight='bold')
plt.rcParams['axes.unicode_minus'] = False 


# multiple change point detection
def MultipleChangePointDetection(series, min_size):
    signal = [[x] for x in series]
    
    algo = rpt.Pelt(model='rbf', min_size=min_size, jump=5).fit(np.array(signal))
    result = algo.predict(pen=m.log(len(series), m.e))

    rpt.display(np.array(signal), result, figsize=(13, 2), dpi=100)
    # plt.title(colname, fontproperties='SimHei', fontsize=14)
    st.pyplot(plt.gcf())

    return result


# Box-Ljung test
def LB(series, freq):
    try:
        pvalue = acorr_ljungbox(series, lags=3*freq)['lb_pvalue']
        if (max(pvalue) < 0.05):
            return ('非白噪声')
        else:
            return ('白噪声')
    except:
        return ('白噪声')
    

# ADF test
def ADF(series):
    pvalue = adfuller(series)[1]
    if (pvalue > 0.05):
        return ('非平稳')
    else:
        return ('平稳')


# difference method
def DifferenceMethod(series):
    d = 0
    while (True):
        series = series.diff(1).dropna()
        d += 1
        if (ADF(series) == '平稳'):
            break
    return series, d


# ARIMA model
def ARIMA_Model(series):
    d = 0
    if (ADF(series) == '非平稳'):
        series, d = DifferenceMethod(series)

    trend_evaluate = sm.tsa.arma_order_select_ic(series, ic=['bic'], max_ar=5, max_ma=5)
    p = trend_evaluate.bic_min_order[0]
    q = trend_evaluate.bic_min_order[1]
    # model = ARIMA(series, (p,d,q)).fit()
    # model.summary2()
    # forcast = model.forecast(4)[1]
    return p, d, q


def ARIMA_Forcasting(series, p, d, q):
    series.index = np.array(range(len(series)))
    if (d > 0):
        diff = []
        for i in range(d):
            diff.append(series[len(series)-1])
            series = series.diff(1).dropna()
    model = ARIMA(endog=series, order=(p,d,q)).fit()
    # model.summary2()
    if (d == 0):
        forecast = model.forecast(steps=1).iloc[0]
    else:
        forecast = model.forecast(steps=1).iloc[0] + sum(diff)
    return round(forecast, 2)


# triple exponential smoothing model
def TripleExponentialSmoothing_Forcasting(series, freq):
    model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=freq).fit()
    return round(model.forecast(1), 2)


def print_Forcast(df, freq):
    startdate = []
    enddate = []
    mean_data = []
    var_data = []
    segments = []
    tmp = 0
    result = MultipleChangePointDetection(df.iloc[:, 1], 3*freq)
    for i in range(len(result)):
        startdate.append(df['Date'][tmp])
        enddate.append(df['Date'][result[i]-1])
        mean_data.append(np.mean(df.iloc[tmp:result[i], 1]))
        var_data.append(np.var(df.iloc[tmp:result[i], 1]))
        segment = f"片段{i+1}"
        segments.append(segment)
        tmp = result[i]
    df_result = pd.DataFrame({'片段': segments,
                            '开始日期': startdate,
                            '结束日期': enddate,
                            '均值': mean_data,
                            '方差': var_data})
    st.table(df_result)
    
    if (len(result) > 2):
        signal = df.iloc[result[-2]:result[-1], 1]
    else:
        signal = df.iloc[:, 1]
    
    if (LB(signal, freq) == '白噪声'):
        st.write('数据为白噪声序列，无法预测')
    else:
        try:
            p, d, q = ARIMA_Model(signal)
            pred = ARIMA_Forcasting(signal, p, d, q)
            st.write(f"预测模型: ARIMA({p},{d},{q})")
            st.write(f"下一时间点质量数据预测值: {pred}")
        except ValueError:
            st.write('预测模型: 三次指数平滑模型')
            pred = TripleExponentialSmoothing_Forcasting(signal, freq)
            st.write(f"下一时间点质量数据预测值: {float(pred)}")
    return


# %%
# In[绘图部分]:
# data = pd.read_excel("C:\\Users\\41952\\Desktop\\streamlit\\pythonProject5\\data.xlsx")
st.set_page_config(page_title="可视化页面", page_icon=":bar_chart:", layout="centered")
st.sidebar.header('可视化页面')
st.sidebar.button("刷新")
st.title(":bar_chart: 可视化页面")
st.markdown("##")
mode = st.sidebar.selectbox('选择类型',['过程能力指数','控制图','质量预测']) #三种类型
# 能力指数页面
if mode == '过程能力指数':
    st.subheader('过程能力指数')
    guage, match, twist = st.tabs(["检具数据", "匹配数据", "扭矩数据"])  ##标签页
    with guage:
        Sheet = st.selectbox('选择测量表', ['车顶型面', '前风窗', '后风窗', '框型面后盖', '左侧车顶激光焊落差', '右侧车顶激光焊落差'])

        if Sheet == '车顶型面':
            url_RoofSurface = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RoofSurface_Sunroofs'
            df_RoofSurface = DataPrep(url_car, url_RoofSurface, 'RoofSurface')
            data = df_RoofSurface

            pool = data.columns.tolist()
            Variable = st.selectbox('选择测点', pool[2:len(pool) - 1])
            dataUsed = data.loc[:, ['Date', Variable]]
            print_Cp(dataUsed,0.5,0,4,0,0,0,0,0,0,0,0) ## data,上限，下限，转换类型，bc,sbsu,J1234，cplong,corr

        elif Sheet == '前风窗':
            url_FrontWindshield = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/FrontWindshields'
            df_FrontWindshield = DataPrep(url_car, url_FrontWindshield, 'WindshieldFrameFace')
            data = df_FrontWindshield

            pool = data.columns.tolist()
            Variable = st.selectbox('选择测点', pool[2:len(pool)])
            dataUsed = data.loc[:, ['Date', Variable]]
            print_Cp(dataUsed, 3.8,2.2,4,0,0,0,0,0,0,0,0)
        elif Sheet == '后风窗':
            url_RearWindshield = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RearWindshields'
            df_RearWindshield = DataPrep(url_car, url_RearWindshield, 'WindshieldFrameFace')
            data = df_RearWindshield

            pool = data.columns.tolist()
            pool = pool[2:len(pool)]
            Variable = st.selectbox('选择测点', pool)
            lLimit = [2.2, 2.2, 2.7, 2.7, 2.7, 2.7, 2.2, 2.2]
            rLimit = [3.8, 3.8, 4.3, 4.3, 4.3, 4.3, 3.8, 3.8]
            idx = pool.index(Variable)
            dataUsed = data.loc[:, ['Date', Variable]]
            print_Cp(dataUsed,rLimit[idx],lLimit[idx],4,0,0,0,0,0,0,0,0)
        elif Sheet == '框型面后盖':
            url_FrameFaceBackCover = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/FrameFaceBackCovers'
            df_FrameFaceBackCover = DataPrep(url_car, url_FrameFaceBackCover, 'WindshieldFrameFace')
            data = df_FrameFaceBackCover

            pool = data.columns.tolist()
            Variable = st.selectbox('选择测点', pool[2:len(pool)])
            dataUsed = data.loc[:, ['Date', Variable]]
            print_Cp(dataUsed,5.5,3.5,4,0,0,0,0,0,0,0,0)
        elif Sheet == '左侧车顶激光焊落差':
            url_RoofLaserWeldingDrop_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RoofLaserWeldingDrop_Lefts'
            df_RoofLaserWeldingDrop_Left = DataPrep(url_car, url_RoofLaserWeldingDrop_Left, 'RoofLaserWeldingDrop')
            data = df_RoofLaserWeldingDrop_Left

            pool = data.columns.tolist()
            pool = pool[2:len(pool)]
            Variable = st.selectbox('选择测点', pool)
            lLimit = [2.8, 3, 2.9, 3, 2.8, 3, 3, 3, 2.8, 0]
            rLimit = [4.8, 5, 4.9, 5, 4.8, 5, 5, 5, 4.8, 1]
            idx = pool.index(Variable)
            dataUsed = data.loc[:, ['Date', Variable]]
            print_Cp(dataUsed,  rLimit[idx],lLimit[idx],4,0,0,0,0,0,0,0,0)
        elif Sheet == '右侧车顶激光焊落差':
            url_RoofLaserWeldingDrop_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RoofLaserWeldingDrop_Rights'
            df_RoofLaserWeldingDrop_Right = DataPrep(url_car, url_RoofLaserWeldingDrop_Right, 'RoofLaserWeldingDrop')
            data = df_RoofLaserWeldingDrop_Right

            pool = data.columns.tolist()
            pool = pool[2:len(pool)]
            Variable = st.selectbox('选择测点', pool)
            lLimit = [2.8, 3, 2.9, 3, 2.8, 3, 3, 3, 2.8, 0]
            rLimit = [4.8, 5, 4.9, 5, 4.8, 5, 5, 5, 4.8, 1]
            idx = pool.index(Variable)
            dataUsed = data.loc[:, ['Date', Variable]]
            print_Cp(dataUsed,  rLimit[idx],lLimit[idx],4,0,0,0,0,0,0,0,0)

    with match:
        carType = st.selectbox('选择车型', ['ZP5', 'ZP8'])
        Sheet = st.selectbox('选择测量表', ['前盖左侧', '前盖右侧', '后盖左侧', '后盖右侧', '前后门左侧', '前后门右侧', '前后门平整度左侧', '前后门平整度右侧'])
        if carType == 'ZP5':
            if Sheet == '前盖左侧':
                url_ZP5_FrontCover_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontCover_Lefts'
                df_ZP5_FrontCover_Left = DataPrep(url_car, url_ZP5_FrontCover_Left, 'ZP5')
                data = df_ZP5_FrontCover_Left

                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点', pool)
                lLimit = [2.5, 2.5, 2.5, 2, 3.4, 1, 0.1, 1.2, 1.7, 0.7, -1.5, -0.4]
                rLimit = [3.5, 3.5, 3.5, 3, 4.4, 3, 1.1, 2.2, 2.7, 1.7, -1.0, 0.6]
                idx = pool.index(Variable)
                dataUsed = data.loc[:, ['Date', Variable]]
                print_Cp(dataUsed,  rLimit[idx],lLimit[idx],4,0,0,0,0,0,0,0,0)
            elif Sheet == '前盖右侧':
                url_ZP5_FrontCover_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontCover_Rights'
                df_ZP5_FrontCover_Right = DataPrep(url_car, url_ZP5_FrontCover_Right, 'ZP5')
                data = df_ZP5_FrontCover_Right

                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点', pool)
                lLimit = [2, 2.5, 2.5, 2, 3.4, 1, 0.1, 1.2, 1.7, 0.7, -1.5, -0.4]
                rLimit = [3, 3.5, 3.5, 3, 4.4, 3, 1.1, 2.2, 2.7, 1.7, -1.0, 0.6]
                idx = pool.index(Variable)
                dataUsed = data.loc[:, ['Date', Variable]]
                print_Cp(dataUsed,  rLimit[idx],lLimit[idx],4,0,0,0,0,0,0,0,0)
            elif Sheet == '后盖左侧':
                url_ZP5_BackCover_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_BackCover_Lefts'
                df_ZP5_BackCover_Left = DataPrep(url_car, url_ZP5_BackCover_Left, 'ZP5')
                data = df_ZP5_BackCover_Left

                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点', pool)
                lLimit = [2.2, 3, 3.4, 3.6, 3.8, 4.1, -2.0, -2.5, -1.9, -1, -2, -1.75]
                rLimit = [3.2, 4, 4.4, 4.6, 4.8, 5.1, -1.5, -1.5, -0.9, 0, -1, -1.25]
                idx = pool.index(Variable)
                dataUsed = data.loc[:, ['Date', Variable]]
                print_Cp(dataUsed,  rLimit[idx],lLimit[idx],4,0,0,0,0,0,0,0,0)
            elif Sheet == '后盖右侧':
                url_ZP5_BackCover_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_BackCover_Rights'
                df_ZP5_BackCover_Right = DataPrep(url_car, url_ZP5_BackCover_Right, 'ZP5')
                data = df_ZP5_BackCover_Right

                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点', pool)
                lLimit = [2.2, 3, 3.4, 3.6, 3.8, 4.1, -1.75, -2.25, -1.9, -1, -2, -2]
                rLimit = [3.3, 4, 4.4, 4.6, 4.8, 5.1, -1.25, -1.25, -0.9, 0, -1, -1.5]
                idx = pool.index(Variable)
                dataUsed = data.loc[:, ['Date', Variable]]
                print_Cp(dataUsed,  rLimit[idx],lLimit[idx],4,0,0,0,0,0,0,0,0)
            elif Sheet == '前后门左侧':
                url_ZP5_FrontRearDoor_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontRearDoor_Lefts'
                df_ZP5_FrontRearDoor_Left = DataPrep(url_car, url_ZP5_FrontRearDoor_Left, 'ZP5')
                data = df_ZP5_FrontRearDoor_Left

                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点', pool)
                lLimit = [3, 3, 3, 4.2, 5.6, 5.6, 3.7, 3.7, 3.7, 5.6, 5.6, 3.3, 3.3, 3.3, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5,
                          9.5, 9.5]
                rLimit = [4, 4, 4, 5.8, 7.6, 7.6, 4.7, 4.7, 4.7, 7.6, 7.6, 4.3, 4.3, 4.3, 10.5, 10.5, 10.5, 10.5, 10.5,
                          11.5, 11.5, 11.5]
                idx = pool.index(Variable)
                dataUsed = data.loc[:, ['Date', Variable]]
                print_Cp(dataUsed,  rLimit[idx],lLimit[idx],4,0,0,0,0,0,0,0,0)
            elif Sheet == '前后门右侧':
                url_ZP5_FrontRearDoor_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontRearDoor_Rights'
                df_ZP5_FrontRearDoor_Right = DataPrep(url_car, url_ZP5_FrontRearDoor_Right, 'ZP5')
                data = df_ZP5_FrontRearDoor_Right

                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点', pool)
                lLimit = [3, 3, 3, 4.2, 5.6, 5.6, 3.7, 3.7, 3.7, 5.6, 5.6, 3.3, 3.3, 3.3, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5,
                          9.5, 9.5]
                rLimit = [4, 4, 4, 5.8, 7.6, 7.6, 4.7, 4.7, 4.7, 7.6, 7.6, 4.3, 4.3, 4.3, 10.5, 10.5, 10.5, 10.5, 10.5,
                          11.5, 11.5, 11.5]
                idx = pool.index(Variable)
                dataUsed = data.loc[:, ['Date', Variable]]
                print_Cp(dataUsed,  rLimit[idx],lLimit[idx],4,0,0,0,0,0,0,0,0)
            elif Sheet == '前后门平整度左侧':
                url_ZP5_FrontRearDoor_Flatness_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontRearDoor_Flatness_Lefts'
                df_ZP5_FrontRearDoor_Flatness_Left = DataPrep(url_car, url_ZP5_FrontRearDoor_Flatness_Left, 'ZP5')
                data = df_ZP5_FrontRearDoor_Flatness_Left

                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点', pool)
                lLimit = [1.3, 0, 0, 0, 0, 0, 0, 0, 0]
                rLimit = [1.8, 1, 1, 1, 1, 1, 1, 1, 1]
                idx = pool.index(Variable)
                dataUsed = data.loc[:, ['Date', Variable]]
                print_Cp(dataUsed,  rLimit[idx],lLimit[idx],4,0,0,0,0,0,0,0,0)
            elif Sheet == '前后门平整度右侧':
                url_ZP5_FrontRearDoor_Flatness_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontRearDoor_Flatness_Rights'
                df_ZP5_FrontRearDoor_Flatness_Right = DataPrep(url_car, url_ZP5_FrontRearDoor_Flatness_Right, 'ZP5')
                data = df_ZP5_FrontRearDoor_Flatness_Right

                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点', pool)
                lLimit = [1.3, 0, 0, 0, 0, 0, 0, 0, 0]
                rLimit = [1.8, 1, 1, 1, 1, 1, 1, 1, 1]
                idx = pool.index(Variable)
                dataUsed = data.loc[:, ['Date', Variable]]
                print_Cp(dataUsed,  rLimit[idx],lLimit[idx],4,0,0,0,0,0,0,0,0)
        else:
            if Sheet == '前盖左侧':
                url_ZP8_FrontCover_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontCover_Lefts'
                df_ZP8_FrontCover_Left = DataPrep(url_car, url_ZP8_FrontCover_Left, 'ZP8')
                data = df_ZP8_FrontCover_Left

                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点', pool)
                lLimit = [2.5, 2.5, 2.5, 2.5, 2.5, 3.4, 1, 0.1, -0.4, 0.25, 0.7, -0.4, -0.6]
                rLimit = [3.5, 3.5, 3.5, 3.5, 3.5, 4.4, 3, 1.1, 0.6, 1.25, 1.7, 0.6, 0.4]
                idx = pool.index(Variable)
                dataUsed = data.loc[:, ['Date', Variable]]
                print_Cp(dataUsed,  rLimit[idx],lLimit[idx],4,0,0,0,0,0,0,0,0)
            elif Sheet == '前盖右侧':
                url_ZP8_FrontCover_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontCover_Rights'
                df_ZP8_FrontCover_Right = DataPrep(url_car, url_ZP8_FrontCover_Right, 'ZP8')
                data = df_ZP8_FrontCover_Right

                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点', pool)
                lLimit = [2.5, 2.5, 2.5, 2.5, 2.5, 3.4, 1, 0.1, -0.4, 0.25, 0.7, -0.4, -0.6]
                rLimit = [3.5, 3.5, 3.5, 3.5, 3.5, 4.4, 3, 1.1, 0.6, 1.25, 1.7, 0.6, 0.4]
                idx = pool.index(Variable)
                dataUsed = data.loc[:, ['Date', Variable]]
                print_Cp(dataUsed,  rLimit[idx],lLimit[idx],4,0,0,0,0,0,0,0,0)
            elif Sheet == '后盖左侧':
                url_ZP8_BackCover_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_BackCover_Lefts'
                df_ZP8_BackCover_Left = DataPrep(url_car, url_ZP8_BackCover_Left, 'ZP8')
                data = df_ZP8_BackCover_Left

                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点', pool)
                lLimit = [3, 3, 3, 3, 3.1, 3, -0.3, 0.1, -0.2, -0.3, -0.2, -0.1]
                rLimit = [4, 4, 4, 4, 4.1, 4, 0.7, 1.1, 0.8, 0.7, 0.8, 0.9]
                idx = pool.index(Variable)
                dataUsed = data.loc[:, ['Date', Variable]]
                print_Cp(dataUsed,  rLimit[idx],lLimit[idx],4,0,0,0,0,0,0,0,0)
            elif Sheet == '后盖右侧':
                url_ZP8_BackCover_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_BackCover_Rights'
                df_ZP8_BackCover_Right = DataPrep(url_car, url_ZP8_BackCover_Right, 'ZP8')
                data = df_ZP8_BackCover_Right

                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点', pool)
                lLimit = [3, 3, 3, 3, 3.1, 3, -0.3, 0.1, -0.2, -0.3, -0.2, -0.1]
                rLimit = [4, 4, 4, 4, 4.1, 4, 0.7, 1.1, 0.8, 0.7, 0.8, 0.9]
                idx = pool.index(Variable)
                dataUsed = data.loc[:, ['Date', Variable]]
                print_Cp(dataUsed,  rLimit[idx],lLimit[idx],4,0,0,0,0,0,0,0,0)
            elif Sheet == '前后门左侧':
                url_ZP8_FrontRearDoor_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontRearDoor_Lefts'
                df_ZP8_FrontRearDoor_Left = DataPrep(url_car, url_ZP8_FrontRearDoor_Left, 'ZP8')
                data = df_ZP8_FrontRearDoor_Left

                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点', pool)
                lLimit = [3.2, 3, 3, 4.2, 5.6, 5.6, 3.7, 3.7, 3.7, 5.6, 5.6, 3.3, 3.3, 3.3, 3.1, 3.1, 3.1]
                rLimit = [4.2, 4, 4, 5.8, 7.6, 7.6, 4.7, 4.7, 4.7, 7.6, 7.6, 4.3, 4.3, 4.3, 5.1, 5.1, 5.1]
                idx = pool.index(Variable)
                dataUsed = data.loc[:, ['Date', Variable]]
                print_Cp(dataUsed,  rLimit[idx],lLimit[idx],4,0,0,0,0,0,0,0,0)
            elif Sheet == '前后门右侧':
                url_ZP8_FrontRearDoor_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontRearDoor_Rights'
                df_ZP8_FrontRearDoor_Right = DataPrep(url_car, url_ZP8_FrontRearDoor_Right, 'ZP8')
                data = df_ZP8_FrontRearDoor_Right

                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点', pool)
                lLimit = [3.2, 3, 3, 4.2, 5.6, 5.6, 3.7, 3.7, 3.7, 5.6, 5.6, 3.3, 3.3, 3.3, 3.1, 3.1, 3.1]
                rLimit = [4.2, 4, 4, 5.8, 7.6, 7.6, 4.7, 4.7, 4.7, 7.6, 7.6, 4.3, 4.3, 4.3, 5.1, 5.1, 5.1]
                idx = pool.index(Variable)
                dataUsed = data.loc[:, ['Date', Variable]]
                print_Cp(dataUsed,  rLimit[idx],lLimit[idx],4,0,0,0,0,0,0,0,0)
            elif Sheet == '前后门平整度左侧':
                url_ZP8_FrontRearDoor_Flatness_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontRearDoor_Flatness_Lefts'
                df_ZP8_FrontRearDoor_Flatness_Left = DataPrep(url_car, url_ZP8_FrontRearDoor_Flatness_Left, 'ZP8')
                data = df_ZP8_FrontRearDoor_Flatness_Left

                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点', pool)
                lLimit = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                rLimit = [1.6, 1, 1, 1, 1, 1, 1, 1, 1]
                idx = pool.index(Variable)
                dataUsed = data.loc[:, ['Date', Variable]]
                print_Cp(dataUsed,  rLimit[idx],lLimit[idx],4,0,0,0,0,0,0,0,0)
            elif Sheet == '前后门平整度右侧':
                url_ZP8_FrontRearDoor_Flatness_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontRearDoor_Flatness_Rights'
                df_ZP8_FrontRearDoor_Flatness_Right = DataPrep(url_car, url_ZP8_FrontRearDoor_Flatness_Right, 'ZP8')
                data = df_ZP8_FrontRearDoor_Flatness_Right

                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点', pool)
                lLimit = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                rLimit = [1.6, 1, 1, 1, 1, 1, 1, 1, 1]
                idx = pool.index(Variable)
                dataUsed = data.loc[:, ['Date', Variable]]
                print_Cp(dataUsed,  rLimit[idx],lLimit[idx],4,0,0,0,0,0,0,0,0)

    with twist:
        Sheet = st.selectbox('选择测量表', ['LNF_1', 'LNF_2', 'LNF2_1', 'LNF2_2', 'BA7', '左翼子板', '右翼子板'])

        if Sheet == 'LNF_1':
            url_LNF_1 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LNF_1s'
            df_LNF_1 = DataPrep_Mna1(url_car, url_LNF_1, 'Mna1_7830')
            data = df_LNF_1

            pool = data.columns.tolist()
            pool = pool[2:len(pool)]
            Variable = st.selectbox('选择测点', pool)
            lLimit = [-3.585,-3.651,-9.010,-7.471,-6.427,-8.154,-4.365,-3.587,-5.404,-7.603,40,-4.498 ]
            rLimit = [4.265,4.057,5.820,5.707,5.271,5.005,4.510,4.449,5.769,4.720,69,3.750 ]
            Cplong = [1.194, 1.194, 1.194, 1.194, 1.950, 1.304, 1.248,1.248, 1.248, 1.248,0,1.304]
            paraboxcox = [0,0,-1.332,-1.256,-2.462,-2.236,0,0,0,-2.085,0,0]
            J1 = [0.157, -0.0934,0,0,0,0,-0.1780,0.4700,0,0,0,-0.8690]
            J2 = [1.417, 1.630, 0,0,0,0,2.907,1.840,0,0,0,2.072]
            J3 = [50.85, 49.76,0,0,0,0, 50.01,50.60,0,0,0, 49.95]
            J4 = [1.896, 2.339,0,0,0,0,5.463,2.813,0,0,0, 3.341]
            idx = pool.index(Variable)
            dataUsed = data.loc[:, ['Date', Variable]]
            if idx in [2,3,5,9]:##正态相关BC
              print_Cp(dataUsed,  rLimit[idx],lLimit[idx],1,paraboxcox[idx],0,0,0,0,0,Cplong[idx],2)
            elif idx in [4]:##正态独立BC
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 1, paraboxcox[idx], 0, 0, 0, 0, 0, Cplong[idx], 1)
            elif idx in [0,1,6,7,11]:##正态相关JC-SU
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 2, 0, 2, J1[idx], J2[idx], J3[idx], J4[idx], Cplong[idx], 2)
            elif idx in [8]:##正态相关原始
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 3, 0, 0, 0,0,0,0, Cplong[idx],2)
            elif idx in [10]:  ##非正态
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 4, 0, 0, 0, 0, 0, 0, 0,0)
        elif Sheet == 'LNF_2':
            url_LNF_2 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LNF_2s'
            df_LNF_2 = DataPrep_Mna1(url_car, url_LNF_2, 'Mna1_7830')
            data = df_LNF_2

            pool = data.columns.tolist()
            pool = pool[2:len(pool)]
            Variable = st.selectbox('选择测点', pool)
            lLimit = [-9.589,-10.835,-10.094,-9.756,-5.382,-8.871,-5.254,-4.180,-5.486,-4.730,-4.683,-4.770]
            rLimit = [8.169,9.865,3.679,3.357,13.098,3.425,2.805,2.900,7.667,12.259,3.265,2.874]
            Cplong = [1.910,1.910,1.358,1.358,1.792,2.049,1.358,1.358,1.358,1.358,1.403,1.403]
            paraboxcox = [0,0,-2.839,-3.593,0,-7.286,0,0,3.794,5.477,0,0,4.774,0]
            J1 = [0,0,0,0,0.3290,0,-1.224,-0.5804,0,0,-1.624,-1.707,0,0]
            J2 = [0,0,0,0,1.293,0,1.898,1.674,0,0,1.592, 1.432,0,0]
            J3 = [0,0,0,0,42.19,0,49.95,50.40,0,0, 45.28,45.29,0,0]
            J4 = [0,0,0,0,52.69,0,2.395,2.339,0,0, 1.540,1.416,0,0]
            idx = pool.index(Variable)
            dataUsed = data.loc[:, ['Date', Variable]]
            if idx in [2,3,8,9]:  ##正态相关BC
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 1, paraboxcox[idx], 0, 0, 0, 0, 0, Cplong[idx], 2)
            elif idx in [5]:  ##正态独立BC
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 1, paraboxcox[idx], 0, 0, 0, 0, 0, Cplong[idx], 1)
            elif idx in [4]:  ##正态独立JC-SB
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 2, 0, 1, J1[idx], J2[idx], J3[idx], J4[idx], Cplong[idx],1)
            elif idx in [6,7,10,11]:  ##正态相关JC-SU
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 2, 0, 2, J1[idx], J2[idx], J3[idx], J4[idx], Cplong[idx],2)
            elif idx in [0,1]:  ##正态相关原始
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 3, 0, 0, 0, 0, 0, 0, Cplong[idx], 2)
            elif idx in [12]:
                st.write("转换方法：Boxcox")
                st.write("最佳参数λ：",paraboxcox[idx])
                st.write(">上下限数据缺失，请联系管理员设置数据后查看。")
            elif idx in [13]:
                st.write("转换方法：非正态方法")
                st.write(">上下限数据缺失，请联系管理员设置数据后查看。")
        elif Sheet == 'LNF2_1':
            url_LNF2_1 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LNF2_1s'
            df_LNF2_1 = DataPrep_Mna1(url_car, url_LNF2_1, 'Mna1_7920')
            data = df_LNF2_1

            pool = data.columns.tolist()
            pool = pool[2:len(pool)]
            Variable = st.selectbox('选择测点', pool)
            rLimit = [24,3.836,5.322,3.019,5.057,3.518,4.857,2.824,24,3.505,24,4.378]
            lLimit = [16,-3.269,-3.595,-3.389,-2.953,-5.697,-4.472,-3.963,16,-3.896,16,-3.332]
            Cplong = [0,1.310,1.194,1.194,1.310,1.310,1.194,1.194,0,1.397,0,1.397]
            paraboxcox = [0,5.075,3.166,0,0,0,0,0,0,0,0,0]
            J1 = [0,0,0,-0.1694,-2.671,0,0,-0.8255,0,0,0,0.2926]
            J2 = [0,0,0,1.809,2.127,0,0,2.247,0,0,0,0.9663]
            J3 = [0,0,0,20.02,12.14,0,0,19.64,0,0,0,18.36]
            J4 = [0,0,0,1.351,23.75,0,0,1.614,0,0,0,23.34]
            idx = pool.index(Variable)
            dataUsed = data.loc[:, ['Date', Variable]]
            if idx in [1,2]:  ##正态相关BC
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 1, paraboxcox[idx], 0, 0, 0, 0, 0, Cplong[idx], 2)
            elif idx in [5,6,9]:  ##正态相关原始
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 3, 0, 0, 0, 0, 0, 0, Cplong[idx], 2)
            elif idx in [4,11]:  ##正态相关JC-SB
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 2, 0, 1, J1[idx], J2[idx], J3[idx], J4[idx], Cplong[idx],2)
            elif idx in [3,7]:  ##正态相关JC-SU
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 2, 0, 2, J1[idx], J2[idx], J3[idx], J4[idx], Cplong[idx],2)
            elif idx in [0,8,10]:  ##非正态
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 4, 0, 0, 0, 0, 0, 0,0,0)
        elif Sheet == 'LNF2_2':
            url_LNF2_2 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LNF2_2s'
            df_LNF2_2 = DataPrep_Mna1(url_car, url_LNF2_2, 'Mna1_7920')
            data = df_LNF2_2

            pool = data.columns.tolist()
            pool = pool[2:len(pool)]
            Variable = st.selectbox('选择测点', pool)
            rLimit = [6.596,6.018,4.686,7.953]
            lLimit = [-4.527,-9.991,-4.149,-7.467]
            Cplong = [1.420,1.420,1.420,1.420]
            paraboxcox = [0,-3.367,0,0]
            J1 = [0.7964,0,-0.07666,-0.1386]
            J2 = [2.950,0, 1.850,1.998]
            J3 = [8.861,0,8.634,7.745]
            J4 = [0.5721,0,0.3403,9.360]
            idx = pool.index(Variable)
            dataUsed = data.loc[:, ['Date', Variable]]
            if idx in [1]:  ##正态相关BC
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 1, paraboxcox[idx], 0, 0, 0, 0, 0, Cplong[idx], 2)
            elif idx in [3]:  ##正态相关JC-SB
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 2, 0, 1, J1[idx], J2[idx], J3[idx], J4[idx], Cplong[idx],2)
            elif idx in [0, 2]:  ##正态相关JC-SU
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 2, 0, 2, J1[idx], J2[idx], J3[idx], J4[idx], Cplong[idx],2)
        elif Sheet == 'BA7':
            url_BA7 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/BA7s'
            df_BA7 = DataPrep_Mna1(url_car, url_BA7, 'Mna1_BA7')
            data = df_BA7

            pool = data.columns.tolist()
            pool = pool[2:len(pool)]
            Variable = st.selectbox('选择测点', pool)
            rLimit = [4.428,4.174,4.106,4.124,4.343,4.103,3.272,30,4.030,3.168,8.908,4.204 ]
            lLimit = [-4.533,-4.635,-4.144,-4.276,-5.203,-4.907,-7.764,20,-3.566,-4.278,-7.669,-5.066]
            Cplong = [1.404,1.404,1.337,1.337,1.336,1.336,1.336,0,1.099,1.099,1.454,1,454]
            paraboxcox= [0,0,0,0,0,0,-3.292,0,0,-0.6533,0,0]
            J1 = [0,0,0,0,0,0,0,0,0,0,0.2014,0]
            J2 = [0,0,0,0,0,0,0,0,0,0, 1.232, 0]
            J3 = [0,0,0,0,0,0,0,0,0,0,17.80,0]
            J4 = [0,0,0,0,0,0,0,0,0,0, 22.85,0]
            idx = pool.index(Variable)
            dataUsed = data.loc[:, ['Date', Variable]]
            if idx in [6,9]:  ##正态相关BC
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 1, paraboxcox[idx], 0, 0, 0, 0, 0, Cplong[idx], 2)
            elif idx in [0,1,2,3,4,5,8,11]:  ##正态相关原始
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 3, 0, 0, 0, 0, 0, 0, Cplong[idx], 2)
            elif idx in [10]:  ##正态相关JC-SB
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 2, 0, 1, J1[idx], J2[idx], J3[idx], J4[idx], Cplong[idx],2)
            elif idx in [7]:  ##非正态
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 4, 0, 0, 0, 0, 0, 0, 0, 0)
        elif Sheet == '左翼子板':
            url_LeftWingPanel = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LeftWingPanels'
            df_LeftWingPanel = DataPrep_Mna1(url_car, url_LeftWingPanel, 'Mna1_LeftWingPanel')
            data = df_LeftWingPanel

            pool = data.columns.tolist()
            pool = pool[2:len(pool)]
            Variable = st.selectbox('选择测点', pool)
            rLimit = [3.935, 5.414, 9.6, 4.500, 6.090, 6.750, 9.6, 6.706, 6.550, 5.580, -5.621, 2.517, 3.204, 2.798,
                      3.249, 4.793]
            lLimit = [-5.461, -4.261, 6.4, -4.006, -4.680, -4.380, 6.4, -5.354, -5.150, -3.750, 3.260, -7.705, -6.203,
                      -3.497, -5.854, -4.016]
            Cplong = [1.342, 1.342, 0, 1.342, 1.342, 1.342, 0, 1.457, 1.457, 1.457, 1.133, 1.133, 1.133, 1.133, 1.133,
                      1.339]
            paraboxcox = [0, 3.869, 0, 3.707, 0, 0, 0, 0, 0, 0, 0, -9.465, 0, 0, -4.548, 4.298]
            J1 = [0, 0, 0, 0, -0.7710, -0.3361, 0, 0.7491, 0.5641, 1.099, -2.590, 0, -3.659, -1.094, 0, 0]
            J2 = [0, 0, 0, 0, 1.190, 1.209, 0, 1.408, 1.064, 1.293, 1.853, 0, 2.157, 1.397, 0, 0]
            J3 = [0, 0, 0, 0, 7.243, 7.371, 0, 7.079, 7.131, 6.909, 6.958, 0, 6.705, 7.214, 0, 0]
            J4 = [0, 0, 0, 0, 9.327, 9.383, 0, 9.138, 9.051, 9.831, 0.2256, 0, 0.2687, 0.2867, 0, 0]
            idx = pool.index(Variable)
            dataUsed = data.loc[:, ['Date', Variable]]
            if idx in [1,3,11,14]:  ##正态相关BC
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 1, paraboxcox[idx], 0, 0, 0, 0, 0, Cplong[idx], 2)
            elif idx in [15]:##正态独立BC
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 1, paraboxcox[idx], 0, 0, 0, 0, 0, Cplong[idx], 1)
            elif idx in [0]:  ##正态相关原始
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 3, 0, 0, 0, 0, 0, 0, Cplong[idx], 2)
            elif idx in [4,5,7,8,9]:  ##正态相关JC-SB
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 2, 0, 1, J1[idx], J2[idx], J3[idx], J4[idx], Cplong[idx],2)
            elif idx in [10,12,13]:  ##正态相关JC-SU
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 2, 0, 2, J1[idx], J2[idx], J3[idx], J4[idx], Cplong[idx],2)
            elif idx in [2,6]:  ##非正态
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 4, 0, 0, 0, 0, 0, 0, 0, 0)
        elif Sheet == '右翼子板':
            url_RightWingPanel = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RightWingPanels'
            df_RightWingPanel = DataPrep_Mna1(url_car, url_RightWingPanel, 'Mna1_RightWingPanel')
            data = df_RightWingPanel

            pool = data.columns.tolist()
            pool = pool[2:len(pool)]
            Variable = st.selectbox('选择测点', pool)
            rLimit = [4.181,4.256,3.661,5.028,4.162,3.607,5.961,9.6,9.6,9.6,9.6,9.6,9.6,6.730,9.6,5.560]
            lLimit = [-5.166,-5.317,-4.782,-3.557,-3.284,-3.733,-3.849,6.4,6.4,6.4,6.4,6.4,6.4,-4.310,6.4,-4.057]
            Cplong = [1.289,1.289,1.289,1.289,1.289,1.289,1.280,0,0,0,0,0,0,1.450,0,1.352]
            paraboxcox = [0,0,0,4.030,5.084,4.348,0,0,0,0,0,0,0,0,0,5.268]
            J1 = [0,0,0,0,0,0,0.4058,0,0,0,0,0,0,1.700,0,0]
            J2 = [0,0,0,0,0,0,1.104,0,0,0,0,0,0,1.358,0,0]
            J3 = [0,0,0,0,0,0,7.152,0,0,0,0,0,0,6.964,0,0]
            J4 = [0,0,0,0,0,0,9.452,0,0,0,0,0,0,9.768,0,0]
            idx = pool.index(Variable)
            dataUsed = data.loc[:, ['Date', Variable]]
            if idx in [3,4,5]:  ##正态相关BC
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 1, paraboxcox[idx], 0, 0, 0, 0, 0, Cplong[idx], 2)
            elif idx in [15]:  ##正态独立BC
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 1, paraboxcox[idx], 0, 0, 0, 0, 0, Cplong[idx], 1)
            elif idx in [0,1,2]:  ##正态相关原始
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 3, 0, 0, 0, 0, 0, 0, Cplong[idx], 2)
            elif idx in [6,13]:  ##正态独立JC-SB
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 2, 0, 1, J1[idx], J2[idx], J3[idx], J4[idx], Cplong[idx],1)
            elif idx in [7,8,9,10,11,12,14]:  ##非正态
                print_Cp(dataUsed, rLimit[idx], lLimit[idx], 4, 0, 0, 0, 0, 0, 0, 0, 0)




# 控制图页面
if mode == '控制图':
    st.subheader('控制图')
    guage, match, twist = st.tabs(["检具数据", "匹配数据","扭矩数据"])  ##标签页
    with guage:
        Sheet = st.selectbox('选择测量表', ['车顶型面','前风窗','后风窗','框型面后盖','左侧车顶激光焊落差','右侧车顶激光焊落差'])

        if Sheet == '车顶型面':
            url_RoofSurface = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RoofSurface_Sunroofs'
            df_RoofSurface = DataPrep(url_car, url_RoofSurface, 'RoofSurface')
            data = df_RoofSurface
            
            pool = data.columns.tolist()
            Variable = st.selectbox('选择测点',pool[2:len(pool)-1])
            dataUsed = data.loc[:,['Date',Variable]]
            print_SPC(dataUsed,0,0.5)
        elif Sheet == '前风窗':
            url_FrontWindshield = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/FrontWindshields'
            df_FrontWindshield = DataPrep(url_car, url_FrontWindshield, 'WindshieldFrameFace')
            data = df_FrontWindshield
            
            pool = data.columns.tolist()
            Variable = st.selectbox('选择测点',pool[2:len(pool)])
            dataUsed = data.loc[:,['Date',Variable]]
            print_SPC(dataUsed,2.2,3.8)
        elif Sheet == '后风窗':
            url_RearWindshield = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RearWindshields'
            df_RearWindshield = DataPrep(url_car, url_RearWindshield, 'WindshieldFrameFace')
            data = df_RearWindshield
            
            pool = data.columns.tolist()
            pool = pool[2:len(pool)]
            Variable = st.selectbox('选择测点',pool)
            lLimit = [2.2,2.2,2.7,2.7,2.7,2.7,2.2,2.2]
            rLimit = [3.8,3.8,4.3,4.3,4.3,4.3,3.8,3.8]
            idx = pool.index(Variable)
            dataUsed = data.loc[:,['Date',Variable]]
            print_SPC(dataUsed,lLimit[idx],rLimit[idx])
        elif Sheet == '框型面后盖':
            url_FrameFaceBackCover = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/FrameFaceBackCovers'
            df_FrameFaceBackCover = DataPrep(url_car, url_FrameFaceBackCover, 'WindshieldFrameFace')
            data = df_FrameFaceBackCover
            
            pool = data.columns.tolist()
            Variable = st.selectbox('选择测点',pool[2:len(pool)])
            dataUsed = data.loc[:,['Date',Variable]]
            print_SPC(dataUsed,3.5,5.5)
        elif Sheet == '左侧车顶激光焊落差':
            url_RoofLaserWeldingDrop_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RoofLaserWeldingDrop_Lefts'
            df_RoofLaserWeldingDrop_Left = DataPrep(url_car, url_RoofLaserWeldingDrop_Left, 'RoofLaserWeldingDrop')
            data = df_RoofLaserWeldingDrop_Left
            
            pool = data.columns.tolist()
            pool = pool[2:len(pool)]
            Variable = st.selectbox('选择测点',pool)
            lLimit = [2.8,3,2.9,3,2.8,3,3,3,2.8,0]
            rLimit = [4.8,5,4.9,5,4.8,5,5,5,4.8,1]
            idx = pool.index(Variable)
            dataUsed = data.loc[:,['Date',Variable]]
            print_SPC(dataUsed,lLimit[idx],rLimit[idx])
        elif Sheet == '右侧车顶激光焊落差':
            url_RoofLaserWeldingDrop_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RoofLaserWeldingDrop_Rights'
            df_RoofLaserWeldingDrop_Right = DataPrep(url_car, url_RoofLaserWeldingDrop_Right, 'RoofLaserWeldingDrop')
            data = df_RoofLaserWeldingDrop_Right
            
            pool = data.columns.tolist()
            pool = pool[2:len(pool)]
            Variable = st.selectbox('选择测点',pool)
            lLimit = [2.8,3,2.9,3,2.8,3,3,3,2.8,0]
            rLimit = [4.8,5,4.9,5,4.8,5,5,5,4.8,1]
            idx = pool.index(Variable)
            dataUsed = data.loc[:,['Date',Variable]]
            print_SPC(dataUsed,lLimit[idx],rLimit[idx])

    with match:
        carType = st.selectbox('选择车型', ['ZP5','ZP8'])
        Sheet = st.selectbox('选择测量表',['前盖左侧','前盖右侧','后盖左侧','后盖右侧','前后门左侧','前后门右侧','前后门平整度左侧','前后门平整度右侧'])
        if carType == 'ZP5':
            if Sheet == '前盖左侧':
                url_ZP5_FrontCover_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontCover_Lefts'
                df_ZP5_FrontCover_Left = DataPrep(url_car, url_ZP5_FrontCover_Left, 'ZP5')
                data = df_ZP5_FrontCover_Left
                
                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点',pool)
                lLimit = [2.5,2.5,2.5,2,3.4,1,0.1,1.2,1.7,0.7,-1.5,-0.4]
                rLimit = [3.5,3.5,3.5,3,4.4,3,1.1,2.2,2.7,1.7,-1.0,0.6]
                idx = pool.index(Variable)
                dataUsed = data.loc[:,['Date',Variable]]
                print_SPC(dataUsed,lLimit[idx],rLimit[idx])
            elif Sheet == '前盖右侧':
                url_ZP5_FrontCover_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontCover_Rights'
                df_ZP5_FrontCover_Right = DataPrep(url_car, url_ZP5_FrontCover_Right, 'ZP5')
                data = df_ZP5_FrontCover_Right
                
                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点',pool)
                lLimit = [2,2.5,2.5,2,3.4,1,0.1,1.2,1.7,0.7,-1.5,-0.4]
                rLimit = [3,3.5,3.5,3,4.4,3,1.1,2.2,2.7,1.7,-1.0,0.6]
                idx = pool.index(Variable)
                dataUsed = data.loc[:,['Date',Variable]]
                print_SPC(dataUsed,lLimit[idx],rLimit[idx])
            elif Sheet == '后盖左侧':
                url_ZP5_BackCover_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_BackCover_Lefts'
                df_ZP5_BackCover_Left = DataPrep(url_car, url_ZP5_BackCover_Left, 'ZP5')
                data = df_ZP5_BackCover_Left
                
                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点',pool)
                lLimit = [2.2,3,3.4,3.6,3.8,4.1,-2.0,-2.5,-1.9,-1,-2,-1.75]
                rLimit = [3.2,4,4.4,4.6,4.8,5.1,-1.5,-1.5,-0.9,0,-1,-1.25]
                idx = pool.index(Variable)
                dataUsed = data.loc[:,['Date',Variable]]
                print_SPC(dataUsed,lLimit[idx],rLimit[idx])
            elif Sheet == '后盖右侧':
                url_ZP5_BackCover_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_BackCover_Rights'
                df_ZP5_BackCover_Right = DataPrep(url_car, url_ZP5_BackCover_Right, 'ZP5')
                data = df_ZP5_BackCover_Right
                
                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点',pool)
                lLimit = [2.2,3,3.4,3.6,3.8,4.1,-1.75,-2.25,-1.9,-1,-2,-2]
                rLimit = [3.3,4,4.4,4.6,4.8,5.1,-1.25,-1.25,-0.9,0,-1,-1.5]
                idx = pool.index(Variable)
                dataUsed = data.loc[:,['Date',Variable]]
                print_SPC(dataUsed,lLimit[idx],rLimit[idx])
            elif Sheet == '前后门左侧':
                url_ZP5_FrontRearDoor_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontRearDoor_Lefts'
                df_ZP5_FrontRearDoor_Left = DataPrep(url_car, url_ZP5_FrontRearDoor_Left, 'ZP5')
                data = df_ZP5_FrontRearDoor_Left
                
                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点',pool)
                lLimit = [3,3,3,4.2,5.6,5.6,3.7,3.7,3.7,5.6,5.6,3.3,3.3,3.3,9.5,9.5,9.5,9.5,9.5,9.5,9.5,9.5]
                rLimit = [4,4,4,5.8,7.6,7.6,4.7,4.7,4.7,7.6,7.6,4.3,4.3,4.3,10.5,10.5,10.5,10.5,10.5,11.5,11.5,11.5]
                idx = pool.index(Variable)
                dataUsed = data.loc[:,['Date',Variable]]
                print_SPC(dataUsed,lLimit[idx],rLimit[idx])
            elif Sheet == '前后门右侧':
                url_ZP5_FrontRearDoor_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontRearDoor_Rights'
                df_ZP5_FrontRearDoor_Right = DataPrep(url_car, url_ZP5_FrontRearDoor_Right, 'ZP5')
                data = df_ZP5_FrontRearDoor_Right
                
                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点',pool)
                lLimit = [3,3,3,4.2,5.6,5.6,3.7,3.7,3.7,5.6,5.6,3.3,3.3,3.3,9.5,9.5,9.5,9.5,9.5,9.5,9.5,9.5]
                rLimit = [4,4,4,5.8,7.6,7.6,4.7,4.7,4.7,7.6,7.6,4.3,4.3,4.3,10.5,10.5,10.5,10.5,10.5,11.5,11.5,11.5]
                idx = pool.index(Variable)
                dataUsed = data.loc[:,['Date',Variable]]
                print_SPC(dataUsed,lLimit[idx],rLimit[idx])
            elif Sheet == '前后门平整度左侧':
                url_ZP5_FrontRearDoor_Flatness_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontRearDoor_Flatness_Lefts'
                df_ZP5_FrontRearDoor_Flatness_Left = DataPrep(url_car, url_ZP5_FrontRearDoor_Flatness_Left, 'ZP5')
                data = df_ZP5_FrontRearDoor_Flatness_Left
                
                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点',pool)
                lLimit = [1.3,0,0,0,0,0,0,0,0]
                rLimit = [1.8,1,1,1,1,1,1,1,1]
                idx = pool.index(Variable)
                dataUsed = data.loc[:,['Date',Variable]]
                print_SPC(dataUsed,lLimit[idx],rLimit[idx])
            elif Sheet == '前后门平整度右侧':
                url_ZP5_FrontRearDoor_Flatness_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontRearDoor_Flatness_Rights'
                df_ZP5_FrontRearDoor_Flatness_Right = DataPrep(url_car, url_ZP5_FrontRearDoor_Flatness_Right, 'ZP5')
                data = df_ZP5_FrontRearDoor_Flatness_Right
                
                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点',pool)
                lLimit = [1.3,0,0,0,0,0,0,0,0]
                rLimit = [1.8,1,1,1,1,1,1,1,1]
                idx = pool.index(Variable)
                dataUsed = data.loc[:,['Date',Variable]]
                print_SPC(dataUsed,lLimit[idx],rLimit[idx])
        else:
            if Sheet == '前盖左侧':
                url_ZP8_FrontCover_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontCover_Lefts'
                df_ZP8_FrontCover_Left = DataPrep(url_car, url_ZP8_FrontCover_Left, 'ZP8')
                data = df_ZP8_FrontCover_Left
                
                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点',pool)
                lLimit = [2.5,2.5,2.5,2.5,2.5,3.4,1,0.1,-0.4,0.25,0.7,-0.4,-0.6]
                rLimit = [3.5,3.5,3.5,3.5,3.5,4.4,3,1.1,0.6,1.25,1.7,0.6,0.4]
                idx = pool.index(Variable)
                dataUsed = data.loc[:,['Date',Variable]]
                print_SPC(dataUsed,lLimit[idx],rLimit[idx])
            elif Sheet == '前盖右侧':
                url_ZP8_FrontCover_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontCover_Rights'
                df_ZP8_FrontCover_Right = DataPrep(url_car, url_ZP8_FrontCover_Right, 'ZP8')
                data = df_ZP8_FrontCover_Right
                
                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点',pool)
                lLimit = [2.5,2.5,2.5,2.5,2.5,3.4,1,0.1,-0.4,0.25,0.7,-0.4,-0.6]
                rLimit = [3.5,3.5,3.5,3.5,3.5,4.4,3,1.1,0.6,1.25,1.7,0.6,0.4]
                idx = pool.index(Variable)
                dataUsed = data.loc[:,['Date',Variable]]
                print_SPC(dataUsed,lLimit[idx],rLimit[idx])
            elif Sheet == '后盖左侧':
                url_ZP8_BackCover_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_BackCover_Lefts'
                df_ZP8_BackCover_Left = DataPrep(url_car, url_ZP8_BackCover_Left, 'ZP8')
                data = df_ZP8_BackCover_Left
                
                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点',pool)
                lLimit = [3,3,3,3,3.1,3,-0.3,0.1,-0.2,-0.3,-0.2,-0.1]
                rLimit = [4,4,4,4,4.1,4,0.7,1.1,0.8,0.7,0.8,0.9]
                idx = pool.index(Variable)
                dataUsed = data.loc[:,['Date',Variable]]
                print_SPC(dataUsed,lLimit[idx],rLimit[idx])
            elif Sheet == '后盖右侧':
                url_ZP8_BackCover_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_BackCover_Rights'
                df_ZP8_BackCover_Right = DataPrep(url_car, url_ZP8_BackCover_Right, 'ZP8')
                data = df_ZP8_BackCover_Right
                
                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点',pool)
                lLimit = [3,3,3,3,3.1,3,-0.3,0.1,-0.2,-0.3,-0.2,-0.1]
                rLimit = [4,4,4,4,4.1,4,0.7,1.1,0.8,0.7,0.8,0.9]
                idx = pool.index(Variable)
                dataUsed = data.loc[:,['Date',Variable]]
                print_SPC(dataUsed,lLimit[idx],rLimit[idx])
            elif Sheet == '前后门左侧':
                url_ZP8_FrontRearDoor_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontRearDoor_Lefts'
                df_ZP8_FrontRearDoor_Left = DataPrep(url_car, url_ZP8_FrontRearDoor_Left, 'ZP8')
                data = df_ZP8_FrontRearDoor_Left
                
                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点',pool)
                lLimit = [3.2,3,3,4.2,5.6,5.6,3.7,3.7,3.7,5.6,5.6,3.3,3.3,3.3,3.1,3.1,3.1]
                rLimit = [4.2,4,4,5.8,7.6,7.6,4.7,4.7,4.7,7.6,7.6,4.3,4.3,4.3,5.1,5.1,5.1]
                idx = pool.index(Variable)
                dataUsed = data.loc[:,['Date',Variable]]
                print_SPC(dataUsed,lLimit[idx],rLimit[idx])
            elif Sheet == '前后门右侧':
                url_ZP8_FrontRearDoor_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontRearDoor_Rights'
                df_ZP8_FrontRearDoor_Right = DataPrep(url_car, url_ZP8_FrontRearDoor_Right, 'ZP8')
                data = df_ZP8_FrontRearDoor_Right
                
                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点',pool)
                lLimit = [3.2,3,3,4.2,5.6,5.6,3.7,3.7,3.7,5.6,5.6,3.3,3.3,3.3,3.1,3.1,3.1]
                rLimit = [4.2,4,4,5.8,7.6,7.6,4.7,4.7,4.7,7.6,7.6,4.3,4.3,4.3,5.1,5.1,5.1]
                idx = pool.index(Variable)
                dataUsed = data.loc[:,['Date',Variable]]
                print_SPC(dataUsed,lLimit[idx],rLimit[idx])
            elif Sheet == '前后门平整度左侧':
                url_ZP8_FrontRearDoor_Flatness_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontRearDoor_Flatness_Lefts'
                df_ZP8_FrontRearDoor_Flatness_Left = DataPrep(url_car, url_ZP8_FrontRearDoor_Flatness_Left, 'ZP8')
                data = df_ZP8_FrontRearDoor_Flatness_Left
                
                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点',pool)
                lLimit = [0,0,0,0,0,0,0,0,0]
                rLimit = [1.6,1,1,1,1,1,1,1,1]
                idx = pool.index(Variable)
                dataUsed = data.loc[:,['Date',Variable]]
                print_SPC(dataUsed,lLimit[idx],rLimit[idx])
            elif Sheet == '前后门平整度右侧':
                url_ZP8_FrontRearDoor_Flatness_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontRearDoor_Flatness_Rights'
                df_ZP8_FrontRearDoor_Flatness_Right = DataPrep(url_car, url_ZP8_FrontRearDoor_Flatness_Right, 'ZP8')
                data = df_ZP8_FrontRearDoor_Flatness_Right
                
                pool = data.columns.tolist()
                pool = pool[2:len(pool)]
                Variable = st.selectbox('选择测点',pool)
                lLimit = [0,0,0,0,0,0,0,0,0]
                rLimit = [1.6,1,1,1,1,1,1,1,1]
                idx = pool.index(Variable)
                dataUsed = data.loc[:,['Date',Variable]]
                print_SPC(dataUsed,lLimit[idx],rLimit[idx])
    
    with twist:
        Sheet = st.selectbox('选择测量表', ['LNF_1', 'LNF_2', 'LNF2_1','LNF2_2','BA7','左翼子板','右翼子板'])

        if Sheet == 'LNF_1':
            url_LNF_1 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LNF_1s'
            df_LNF_1 = DataPrep_Mna1(url_car, url_LNF_1, 'Mna1_7830')
            data = df_LNF_1
            
            pool = data.columns.tolist()
            pool = pool[2:len(pool)]
            Variable = st.selectbox('选择测点',pool)
            lLimit = [38,40,38,39,40,40,39,38,38,39,40,40]
            rLimit = [67,64,65,65,66,68,63,63,62,63,69,66]
            idx = pool.index(Variable)
            dataUsed = data.loc[:,['Date',Variable]]
            print_SPC(dataUsed,lLimit[idx],rLimit[idx])
        elif Sheet == 'LNF_2':
            url_LNF_2 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LNF_2s'
            df_LNF_2 = DataPrep_Mna1(url_car, url_LNF_2, 'Mna1_7830')
            data = df_LNF_2
            
            pool = data.columns.tolist()
            pool = pool[2:len(pool)]
            Variable = st.selectbox('选择测点',pool)
            lLimit = [39,39,39,40,40,40,40,40,39,39,40,40,33.5,33.5]
            rLimit = [61,62,62,61,67,63,60,60,62,63,62,62,55,55]
            idx = pool.index(Variable)
            dataUsed = data.loc[:,['Date',Variable]]
            print_SPC(dataUsed,lLimit[idx],rLimit[idx])
        elif Sheet == 'LNF2_1':
            url_LNF2_1 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LNF2_1s'
            df_LNF2_1 = DataPrep_Mna1(url_car, url_LNF2_1, 'Mna1_7920')
            data = df_LNF2_1
            
            pool = data.columns.tolist()
            pool = pool[2:len(pool)]
            Variable = st.selectbox('选择测点',pool)
            lLimit = [16,16,16,16,16,16,16,16,16,16,16,16,7.2,7.2,7.2,7.2]
            rLimit = [24,24,24,24,24,24,24,24,24,24,24,24,10.8,10.8,10.8,10.8]
            idx = pool.index(Variable)
            dataUsed = data.loc[:,['Date',Variable]]
            print_SPC(dataUsed,lLimit[idx],rLimit[idx])
        elif Sheet == 'LNF2_2':
            url_LNF2_2 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LNF2_2s'
            df_LNF2_2 = DataPrep_Mna1(url_car, url_LNF2_2, 'Mna1_7920')
            data = df_LNF2_2
            
            pool = data.columns.tolist()
            pool = pool[2:len(pool)]
            Variable = st.selectbox('选择测点',pool)
            lLimit = [7.2,7.2,7.2,7.2]
            rLimit = [10.8,10.8,10.8,10.8]
            idx = pool.index(Variable)
            dataUsed = data.loc[:,['Date',Variable]]
            print_SPC(dataUsed,lLimit[idx],rLimit[idx])
        elif Sheet == 'BA7':
            url_BA7 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/BA7s'
            df_BA7 = DataPrep_Mna1(url_car, url_BA7, 'Mna1_BA7')
            data = df_BA7
            
            pool = data.columns.tolist()
            pool = pool[2:len(pool)]
            Variable = st.selectbox('选择测点',pool)
            lLimit = [20,20,20,20,20,20,20,20,9.6,9.6,16,16]
            rLimit = [30,30,30,30,30,30,30,30,14.4,14.4,24,24]
            idx = pool.index(Variable)
            dataUsed = data.loc[:,['Date',Variable]]
            print_SPC(dataUsed,lLimit[idx],rLimit[idx])
        elif Sheet == '左翼子板':
            url_LeftWingPanel = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LeftWingPanels'
            df_LeftWingPanel = DataPrep_Mna1(url_car, url_LeftWingPanel, 'Mna1_LeftWingPanel')
            data = df_LeftWingPanel
            
            pool = data.columns.tolist()
            pool = pool[2:len(pool)]
            Variable = st.selectbox('选择测点',pool)
            lLimit = [6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,3.6]
            rLimit = [9.6,9.6,9.6,9.6,9.6,9.6,9.6,9.6,9.6,9.6,9.6,9.6,9.6,9.6,9.6,5.4]
            idx = pool.index(Variable)
            dataUsed = data.loc[:,['Date',Variable]]
            print_SPC(dataUsed,lLimit[idx],rLimit[idx])
        elif Sheet == '右翼子板':
            url_RightWingPanel = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RightWingPanels'
            df_RightWingPanel = DataPrep_Mna1(url_car, url_RightWingPanel, 'Mna1_RightWingPanel')
            data = df_RightWingPanel
            
            pool = data.columns.tolist()
            pool = pool[2:len(pool)]
            Variable = st.selectbox('选择测点',pool)
            lLimit = [6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,3.6]
            rLimit = [9.6,9.6,9.6,9.6,9.6,9.6,9.6,9.6,9.6,9.6,9.6,9.6,9.6,9.6,9.6,5.4]
            idx = pool.index(Variable)
            dataUsed = data.loc[:,['Date',Variable]]
            print_SPC(dataUsed,lLimit[idx],rLimit[idx])



# 质量预测页面
if mode == '质量预测':
    st.subheader('质量预测')
    guage, match, twist = st.tabs(["检具数据", "匹配数据", "扭矩数据"])  # 标签页
    with guage:
        Sheet = st.selectbox('选择测量表', ['车顶型面','前风窗','后风窗','框型面后盖','左侧车顶激光焊落差','右侧车顶激光焊落差'])

        if Sheet == '车顶型面':
            url_RoofSurface = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RoofSurface_Sunroofs'
            df_RoofSurface = DataPrep(url_car, url_RoofSurface, 'RoofSurface')
            data = df_RoofSurface
            pool = data.columns.tolist()
            Variable = st.selectbox('选择测点',pool[2:len(pool)-1])
            dataUsed = data.loc[:,['Date',Variable]]
            print_Forcast(dataUsed,8)
        elif Sheet == '前风窗':
            url_FrontWindshield = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/FrontWindshields'
            df_FrontWindshield = DataPrep(url_car, url_FrontWindshield, 'WindshieldFrameFace')
            data = df_FrontWindshield
            pool = data.columns.tolist()
            Variable = st.selectbox('选择测点',pool[2:len(pool)])
            dataUsed = data.loc[:,['Date',Variable]]
            print_Forcast(dataUsed,2)
        elif Sheet == '后风窗':
            url_RearWindshield = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RearWindshields'
            df_RearWindshield = DataPrep(url_car, url_RearWindshield, 'WindshieldFrameFace')
            data = df_RearWindshield
            pool = data.columns.tolist()
            Variable = st.selectbox('选择测点',pool[2:len(pool)])
            dataUsed = data.loc[:,['Date',Variable]]
            print_Forcast(dataUsed,2)
        elif Sheet == '框型面后盖':
            url_FrameFaceBackCover = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/FrameFaceBackCovers'
            df_FrameFaceBackCover = DataPrep(url_car, url_FrameFaceBackCover, 'WindshieldFrameFace')
            data = df_FrameFaceBackCover
            pool = data.columns.tolist()
            Variable = st.selectbox('选择测点',pool[2:len(pool)])
            dataUsed = data.loc[:,['Date',Variable]]
            print_Forcast(dataUsed,2)
        elif Sheet == '左侧车顶激光焊落差':
            url_RoofLaserWeldingDrop_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RoofLaserWeldingDrop_Lefts'
            df_RoofLaserWeldingDrop_Left = DataPrep(url_car, url_RoofLaserWeldingDrop_Left, 'RoofLaserWeldingDrop')
            data = df_RoofLaserWeldingDrop_Left
            pool = data.columns.tolist()
            Variable = st.selectbox('选择测点',pool[2:len(pool)])
            dataUsed = data.loc[:,['Date',Variable]]
            print_Forcast(dataUsed,8)
        elif Sheet == '右侧车顶激光焊落差':
            url_RoofLaserWeldingDrop_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RoofLaserWeldingDrop_Rights'
            df_RoofLaserWeldingDrop_Right = DataPrep(url_car, url_RoofLaserWeldingDrop_Right, 'RoofLaserWeldingDrop')
            data = df_RoofLaserWeldingDrop_Right
            pool = data.columns.tolist()
            Variable = st.selectbox('选择测点',pool[2:len(pool)])
            dataUsed = data.loc[:,['Date',Variable]]
            print_Forcast(dataUsed,8)

    with match:
        carType = st.selectbox('选择车型', ['ZP5','ZP8'])
        Sheet = st.selectbox('选择测量表',['前盖左侧','前盖右侧','后盖左侧','后盖右侧','前后门左侧','前后门右侧','前后门平整度左侧','前后门平整度右侧'])
        if carType == 'ZP5':
            if Sheet == '前盖左侧':
                url_ZP5_FrontCover_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontCover_Lefts'
                df_ZP5_FrontCover_Left = DataPrep(url_car, url_ZP5_FrontCover_Left, 'ZP5')
                data = df_ZP5_FrontCover_Left
                pool = data.columns.tolist()
                Variable = st.selectbox('选择测点',pool[2:len(pool)])
                dataUsed = data.loc[:,['Date',Variable]]
                print_Forcast(dataUsed,4)
            elif Sheet == '前盖右侧':
                url_ZP5_FrontCover_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontCover_Rights'
                df_ZP5_FrontCover_Right = DataPrep(url_car, url_ZP5_FrontCover_Right, 'ZP5')
                data = df_ZP5_FrontCover_Right
                pool = data.columns.tolist()
                Variable = st.selectbox('选择测点',pool[2:len(pool)])
                dataUsed = data.loc[:,['Date',Variable]]
                print_Forcast(dataUsed,4)
            elif Sheet == '后盖左侧':
                url_ZP5_BackCover_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_BackCover_Lefts'
                df_ZP5_BackCover_Left = DataPrep(url_car, url_ZP5_BackCover_Left, 'ZP5')
                data = df_ZP5_BackCover_Left
                pool = data.columns.tolist()
                Variable = st.selectbox('选择测点',pool[2:len(pool)])
                dataUsed = data.loc[:,['Date',Variable]]
                print_Forcast(dataUsed,4)
            elif Sheet == '后盖右侧':
                url_ZP5_BackCover_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_BackCover_Rights'
                df_ZP5_BackCover_Right = DataPrep(url_car, url_ZP5_BackCover_Right, 'ZP5')
                data = df_ZP5_BackCover_Right
                pool = data.columns.tolist()
                Variable = st.selectbox('选择测点',pool[2:len(pool)])
                dataUsed = data.loc[:,['Date',Variable]]
                print_Forcast(dataUsed,4)
            elif Sheet == '前后门左侧':
                url_ZP5_FrontRearDoor_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontRearDoor_Lefts'
                df_ZP5_FrontRearDoor_Left = DataPrep(url_car, url_ZP5_FrontRearDoor_Left, 'ZP5')
                data = df_ZP5_FrontRearDoor_Left
                pool = data.columns.tolist()
                Variable = st.selectbox('选择测点',pool[2:len(pool)])
                dataUsed = data.loc[:,['Date',Variable]]
                print_Forcast(dataUsed,4)
            elif Sheet == '前后门右侧':
                url_ZP5_FrontRearDoor_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontRearDoor_Rights'
                df_ZP5_FrontRearDoor_Right = DataPrep(url_car, url_ZP5_FrontRearDoor_Right, 'ZP5')
                data = df_ZP5_FrontRearDoor_Right
                pool = data.columns.tolist()
                Variable = st.selectbox('选择测点',pool[2:len(pool)])
                dataUsed = data.loc[:,['Date',Variable]]
                print_Forcast(dataUsed,4)
            elif Sheet == '前后门平整度左侧':
                url_ZP5_FrontRearDoor_Flatness_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontRearDoor_Flatness_Lefts'
                df_ZP5_FrontRearDoor_Flatness_Left = DataPrep(url_car, url_ZP5_FrontRearDoor_Flatness_Left, 'ZP5')
                data = df_ZP5_FrontRearDoor_Flatness_Left
                pool = data.columns.tolist()
                Variable = st.selectbox('选择测点',pool[2:len(pool)])
                dataUsed = data.loc[:,['Date',Variable]]
                print_Forcast(dataUsed,4)
            elif Sheet == '前后门平整度右侧':
                url_ZP5_FrontRearDoor_Flatness_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontRearDoor_Flatness_Rights'
                df_ZP5_FrontRearDoor_Flatness_Right = DataPrep(url_car, url_ZP5_FrontRearDoor_Flatness_Right, 'ZP5')
                data = df_ZP5_FrontRearDoor_Flatness_Right
                pool = data.columns.tolist()
                Variable = st.selectbox('选择测点',pool[2:len(pool)])
                dataUsed = data.loc[:,['Date',Variable]]
                print_Forcast(dataUsed,4)
        else:
            if Sheet == '前盖左侧':
                url_ZP8_FrontCover_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontCover_Lefts'
                df_ZP8_FrontCover_Left = DataPrep(url_car, url_ZP8_FrontCover_Left, 'ZP8')
                data = df_ZP8_FrontCover_Left
                pool = data.columns.tolist()
                Variable = st.selectbox('选择测点',pool[2:len(pool)])
                dataUsed = data.loc[:,['Date',Variable]]
                print_Forcast(dataUsed,8)
            elif Sheet == '前盖右侧':
                url_ZP8_FrontCover_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontCover_Rights'
                df_ZP8_FrontCover_Right = DataPrep(url_car, url_ZP8_FrontCover_Right, 'ZP8')
                data = df_ZP8_FrontCover_Right
                pool = data.columns.tolist()
                Variable = st.selectbox('选择测点',pool[2:len(pool)])
                dataUsed = data.loc[:,['Date',Variable]]
                print_Forcast(dataUsed,8)
            elif Sheet == '后盖左侧':
                url_ZP8_BackCover_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_BackCover_Lefts'
                df_ZP8_BackCover_Left = DataPrep(url_car, url_ZP8_BackCover_Left, 'ZP8')
                data = df_ZP8_BackCover_Left
                pool = data.columns.tolist()
                Variable = st.selectbox('选择测点',pool[2:len(pool)])
                dataUsed = data.loc[:,['Date',Variable]]
                print_Forcast(dataUsed,8)
            elif Sheet == '后盖右侧':
                url_ZP8_BackCover_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_BackCover_Rights'
                df_ZP8_BackCover_Right = DataPrep(url_car, url_ZP8_BackCover_Right, 'ZP8')
                data = df_ZP8_BackCover_Right
                pool = data.columns.tolist()
                Variable = st.selectbox('选择测点',pool[2:len(pool)])
                dataUsed = data.loc[:,['Date',Variable]]
                print_Forcast(dataUsed,8)
            elif Sheet == '前后门左侧':
                url_ZP8_FrontRearDoor_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontRearDoor_Lefts'
                df_ZP8_FrontRearDoor_Left = DataPrep(url_car, url_ZP8_FrontRearDoor_Left, 'ZP8')
                data = df_ZP8_FrontRearDoor_Left
                pool = data.columns.tolist()
                Variable = st.selectbox('选择测点',pool[2:len(pool)])
                dataUsed = data.loc[:,['Date',Variable]]
                print_Forcast(dataUsed,8)
            elif Sheet == '前后门右侧':
                url_ZP8_FrontRearDoor_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontRearDoor_Rights'
                df_ZP8_FrontRearDoor_Right = DataPrep(url_car, url_ZP8_FrontRearDoor_Right, 'ZP8')
                data = df_ZP8_FrontRearDoor_Right
                pool = data.columns.tolist()
                Variable = st.selectbox('选择测点',pool[2:len(pool)])
                dataUsed = data.loc[:,['Date',Variable]]
                print_Forcast(dataUsed,8)
            elif Sheet == '前后门平整度左侧':
                url_ZP8_FrontRearDoor_Flatness_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontRearDoor_Flatness_Lefts'
                df_ZP8_FrontRearDoor_Flatness_Left = DataPrep(url_car, url_ZP8_FrontRearDoor_Flatness_Left, 'ZP8')
                data = df_ZP8_FrontRearDoor_Flatness_Left
                pool = data.columns.tolist()
                Variable = st.selectbox('选择测点',pool[2:len(pool)])
                dataUsed = data.loc[:,['Date',Variable]]
                print_Forcast(dataUsed,8)
            elif Sheet == '前后门平整度右侧':
                url_ZP8_FrontRearDoor_Flatness_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontRearDoor_Flatness_Rights'
                df_ZP8_FrontRearDoor_Flatness_Right = DataPrep(url_car, url_ZP8_FrontRearDoor_Flatness_Right, 'ZP8')
                data = df_ZP8_FrontRearDoor_Flatness_Right
                pool = data.columns.tolist()
                Variable = st.selectbox('选择测点',pool[2:len(pool)])
                dataUsed = data.loc[:,['Date',Variable]]
                print_Forcast(dataUsed,8)
    
    with twist:
        Sheet = st.selectbox('选择测量表', ['LNF_1', 'LNF_2', 'LNF2_1','LNF2_2','BA7','左翼子板','右翼子板'])

        if Sheet == 'LNF_1':
            url_LNF_1 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LNF_1s'
            df_LNF_1 = DataPrep_Mna1(url_car, url_LNF_1, 'Mna1_7830')
            data = df_LNF_1
            pool = data.columns.tolist()
            Variable = st.selectbox('选择测点',pool[2:len(pool)])
            dataUsed = data.loc[:,['Date',Variable]]
            print_Forcast(dataUsed,2)
        elif Sheet == 'LNF_2':
            url_LNF_2 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LNF_2s'
            df_LNF_2 = DataPrep_Mna1(url_car, url_LNF_2, 'Mna1_7830')
            data = df_LNF_2
            pool = data.columns.tolist()
            Variable = st.selectbox('选择测点',pool[2:len(pool)])
            dataUsed = data.loc[:,['Date',Variable]]
            print_Forcast(dataUsed,2)
        elif Sheet == 'LNF2_1':
            url_LNF2_1 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LNF2_1s'
            df_LNF2_1 = DataPrep_Mna1(url_car, url_LNF2_1, 'Mna1_7920')
            data = df_LNF2_1
            pool = data.columns.tolist()
            Variable = st.selectbox('选择测点',pool[2:len(pool)])
            dataUsed = data.loc[:,['Date',Variable]]
            print_Forcast(dataUsed,2)
        elif Sheet == 'LNF2_2':
            url_LNF2_2 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LNF2_2s'
            df_LNF2_2 = DataPrep_Mna1(url_car, url_LNF2_2, 'Mna1_7920')
            data = df_LNF2_2
            pool = data.columns.tolist()
            Variable = st.selectbox('选择测点',pool[2:len(pool)])
            dataUsed = data.loc[:,['Date',Variable]]
            print_Forcast(dataUsed,2)
        elif Sheet == 'BA7':
            url_BA7 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/BA7s'
            df_BA7 = DataPrep_Mna1(url_car, url_BA7, 'Mna1_BA7')
            data = df_BA7
            pool = data.columns.tolist()
            Variable = st.selectbox('选择测点',pool[2:len(pool)])
            dataUsed = data.loc[:,['Date',Variable]]
            print_Forcast(dataUsed,2)
        elif Sheet == '左翼子板':
            url_LeftWingPanel = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LeftWingPanels'
            df_LeftWingPanel = DataPrep_Mna1(url_car, url_LeftWingPanel, 'Mna1_LeftWingPanel')
            data = df_LeftWingPanel
            pool = data.columns.tolist()
            Variable = st.selectbox('选择测点',pool[2:len(pool)])
            dataUsed = data.loc[:,['Date',Variable]]
            print_Forcast(dataUsed,2)
        elif Sheet == '右翼子板':
            url_RightWingPanel = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RightWingPanels'
            df_RightWingPanel = DataPrep_Mna1(url_car, url_RightWingPanel, 'Mna1_RightWingPanel')
            data = df_RightWingPanel
            pool = data.columns.tolist()
            Variable = st.selectbox('选择测点',pool[2:len(pool)])
            dataUsed = data.loc[:,['Date',Variable]]
            print_Forcast(dataUsed,2)


# 隐藏streamlit默认格式信息
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            <![]()yle>
            """

st.markdown(hide_st_style, unsafe_allow_html=True)


# %%
