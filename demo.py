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
from statsmodels.tsa.arima_model import ARIMA
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
    plt.ylabel('质量数据', fontproperties='SimHei', fontsize=12)
    # plt.title(colname, fontproperties='SimHei', fontsize=14)
    st.pyplot(plt.gcf())

    return result


# Box-Ljung test
def LB(series, freq):
    pvalue = acorr_ljungbox(series, lags=3*freq)[1]
    if (max(pvalue) < 0.05):
        return ('非白噪声')
    else:
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
    model = ARIMA(series, (p,d,q)).fit()
    # model.summary2()
    if (d == 0):
        forcast = model.forecast()[0][0]
    else:
        forcast = model.forecast()[0][0] + sum(diff)
    return round(forcast, 1)


# triple exponential smoothing model
def TripleExponentialSmoothing_Forcasting(series, freq):
    model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=freq).fit()
    return round(model.forecast(1), 1)


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
#     captabzp5,captabzp8 = st.tabs(["ZP5四门匹配","ZP8四门匹配"])##标签页
#     with captabzp5:
#         capoptionzp5 = st.selectbox('选择测点', ['ZP5测点1能力指数','ZP5测点2能力指数','ZP5测点3能力指数'])
#         if capoptionzp5 == 'ZP5测点1能力指数':
#             target = data.iloc[:, 6]
#         elif capoptionzp5 == 'ZP5测点2能力指数':
#             target = data.iloc[:, 7]
#         elif capoptionzp5 == 'ZP5测点3能力指数':
#             target = data.iloc[:, 8]

#         st.line_chart(target)##绘图

#     with captabzp8:
#         capoptionzp8 = st.selectbox('选择测点', ['ZP8测点1能力指数', 'ZP8测点2能力指数', 'ZP5测点8能力指数'])

#         if capoptionzp8 == 'ZP5测点8能力指数':
#             target = data.iloc[:, 1]
#         elif capoptionzp8 == 'ZP5测点8能力指数':
#             target = data.iloc[:, 2]
#         elif capoptionzp8 == 'ZP5测点8能力指数':
#             target = data.iloc[:, 3]

#         st.line_chart(target)  ##绘图


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
