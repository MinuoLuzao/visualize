#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import json
import pandas as pd
from requests_html import HTMLSession
from requests.auth import HTTPBasicAuth
session = HTMLSession()


# In[2]:


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


# In[3]:


# car 车
url_car = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/Cars'

'''
检具数据
'''
# RoofSurface 车顶型面
url_RoofSurface = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RoofSurface_Sunroofs'
df_RoofSurface = DataPrep(url_car, url_RoofSurface, 'RoofSurface')

# WindshieldFrameFace 前后风窗框型面
# FrontWindshield 前风窗
url_FrontWindshield = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/FrontWindshields'
df_FrontWindshield = DataPrep(url_car, url_FrontWindshield, 'WindshieldFrameFace')
# RearWindshield 后风窗
url_RearWindshield = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RearWindshields'
df_RearWindshield = DataPrep(url_car, url_RearWindshield, 'WindshieldFrameFace')
# FrameFaceBackCover 框型面后盖
url_FrameFaceBackCover = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/FrameFaceBackCovers'
df_FrameFaceBackCover = DataPrep(url_car, url_FrameFaceBackCover, 'WindshieldFrameFace')

# RoofLaserWeldingDrop 车顶激光焊落差
# RoofLaserWeldingDrop_Left 左侧落差
url_RoofLaserWeldingDrop_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RoofLaserWeldingDrop_Lefts'
df_RoofLaserWeldingDrop_Left = DataPrep(url_car, url_RoofLaserWeldingDrop_Left, 'RoofLaserWeldingDrop')
# RoofLaserWeldingDrop_Right 右侧落差
url_RoofLaserWeldingDrop_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RoofLaserWeldingDrop_Rights'
df_RoofLaserWeldingDrop_Right = DataPrep(url_car, url_RoofLaserWeldingDrop_Right, 'RoofLaserWeldingDrop')


'''
匹配数据
'''
# ZP5
# ZP5_FrontCover_Left 前盖左侧
url_ZP5_FrontCover_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontCover_Lefts'
df_ZP5_FrontCover_Left = DataPrep(url_car, url_ZP5_FrontCover_Left, 'ZP5')
# ZP5_FrontCover_Right 前盖右侧
url_ZP5_FrontCover_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontCover_Rights'
df_ZP5_FrontCover_Right = DataPrep(url_car, url_ZP5_FrontCover_Right, 'ZP5')
# ZP5_BackCover_Left 后盖左侧
url_ZP5_BackCover_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_BackCover_Lefts'
df_ZP5_BackCover_Left = DataPrep(url_car, url_ZP5_BackCover_Left, 'ZP5')
# ZP5_BackCover_Right 后盖右侧
url_ZP5_BackCover_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_BackCover_Rights'
df_ZP5_BackCover_Right = DataPrep(url_car, url_ZP5_BackCover_Right, 'ZP5')
# ZP5_FrontRearDoor_Left 前后门左侧
url_ZP5_FrontRearDoor_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontRearDoor_Lefts'
df_ZP5_FrontRearDoor_Left = DataPrep(url_car, url_ZP5_FrontRearDoor_Left, 'ZP5')
# ZP5_FrontRearDoor_Right 前后门右侧
url_ZP5_FrontRearDoor_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontRearDoor_Rights'
df_ZP5_FrontRearDoor_Right = DataPrep(url_car, url_ZP5_FrontRearDoor_Right, 'ZP5')
# ZP5_FrontRearDoor_Flatness_Left 前后门平整度左侧
url_ZP5_FrontRearDoor_Flatness_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontRearDoor_Flatness_Lefts'
df_ZP5_FrontRearDoor_Flatness_Left = DataPrep(url_car, url_ZP5_FrontRearDoor_Flatness_Left, 'ZP5')
# ZP5_FrontRearDoor_Flatness_Right 前后门平整度右侧
url_ZP5_FrontRearDoor_Flatness_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP5_FrontRearDoor_Flatness_Rights'
df_ZP5_FrontRearDoor_Flatness_Right = DataPrep(url_car, url_ZP5_FrontRearDoor_Flatness_Right, 'ZP5')

# ZP8
# ZP8_FrontCover_Left 前盖左侧
url_ZP8_FrontCover_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontCover_Lefts'
df_ZP8_FrontCover_Left = DataPrep(url_car, url_ZP8_FrontCover_Left, 'ZP8')
# ZP8_FrontCover_Right 前盖右侧
url_ZP8_FrontCover_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontCover_Rights'
df_ZP8_FrontCover_Right = DataPrep(url_car, url_ZP8_FrontCover_Right, 'ZP8')
# ZP8_BackCover_Left 后盖左侧
url_ZP8_BackCover_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_BackCover_Lefts'
df_ZP8_BackCover_Left = DataPrep(url_car, url_ZP8_BackCover_Left, 'ZP8')
# ZP8_BackCover_Right 后盖右侧
url_ZP8_BackCover_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_BackCover_Rights'
df_ZP8_BackCover_Right = DataPrep(url_car, url_ZP8_BackCover_Right, 'ZP8')
# ZP8_FrontRearDoor_Left 前后门左侧
url_ZP8_FrontRearDoor_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontRearDoor_Lefts'
df_ZP8_FrontRearDoor_Left = DataPrep(url_car, url_ZP8_FrontRearDoor_Left, 'ZP8')
# ZP8_FrontRearDoor_Right 前后门右侧
url_ZP8_FrontRearDoor_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontRearDoor_Rights'
df_ZP8_FrontRearDoor_Right = DataPrep(url_car, url_ZP8_FrontRearDoor_Right, 'ZP8')
# ZP8_FrontRearDoor_Flatness_Left 前后门平整度左侧
url_ZP8_FrontRearDoor_Flatness_Left = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontRearDoor_Flatness_Lefts'
df_ZP8_FrontRearDoor_Flatness_Left = DataPrep(url_car, url_ZP8_FrontRearDoor_Flatness_Left, 'ZP8')
# ZP8_FrontRearDoor_Flatness_Right 前后门平整度右侧
url_ZP8_FrontRearDoor_Flatness_Right = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/ZP8_FrontRearDoor_Flatness_Rights'
df_ZP8_FrontRearDoor_Flatness_Right = DataPrep(url_car, url_ZP8_FrontRearDoor_Flatness_Right, 'ZP8')


'''
扭矩数据
'''
# Mna1
# LNF_1
url_LNF_1 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LNF_1s'
df_LNF_1 = DataPrep_Mna1(url_car, url_LNF_1, 'Mna1_7830')
# LNF_2
url_LNF_2 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LNF_2s'
df_LNF_2 = DataPrep_Mna1(url_car, url_LNF_2, 'Mna1_7830')

# LNF2_1
url_LNF2_1 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LNF2_1s'
df_LNF2_1 = DataPrep_Mna1(url_car, url_LNF2_1, 'Mna1_7920')
# LNF2_2
url_LNF2_2 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LNF2_2s'
df_LNF2_2 = DataPrep_Mna1(url_car, url_LNF2_2, 'Mna1_7920')

# BA7
url_BA7 = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/BA7s'
df_BA7 = DataPrep_Mna1(url_car, url_BA7, 'Mna1_BA7')

# LeftWingPanel
url_LeftWingPanel = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/LeftWingPanels'
df_LeftWingPanel = DataPrep_Mna1(url_car, url_LeftWingPanel, 'Mna1_LeftWingPanel')

# RightWingPanel
url_RightWingPanel = 'https://qrkapp-sandbox.mxapps.io/odata/QRKCarODataService/v1/RightWingPanels'
df_RightWingPanel = DataPrep_Mna1(url_car, url_RightWingPanel, 'Mna1_RightWingPanel')


# In[4]:


# In[ ]:




