#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from fitter import Fitter
import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt


# In[2]:


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


# ## 画图

# In[3]:


#norm EWMA控制图
def EWMA_norm(data,dirName,repName):
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
    if repName.split('_')[-1] == 'lognorm':
        plt.title('EWMA_lognorm chart')
    else:
        plt.title('EWMA_norm chart')
    plt.xlabel('Date')
    plt.ylabel('EWMA value')
    print('有问题的日期是',date[flag].tolist())
    plt.savefig('.\\' + dirName + '\\' + repName + '.png')
    plt.show()
    return


# In[4]:


#lognorm EWMA控制图
def EWMA_lognorm(data,dirName,repName):
    data.columns = ['date','data']
    data.loc[:,'data'] = data['data'].apply(lambda x:m.log(x))
    EWMA_norm(data,dirName,repName+'_lognorm')
    return


# In[5]:


#expon t控制图
def t_expon(data,loc,theta,dirName,repName):
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
    print('有问题的日期是',date[flag != 0].tolist())
    plt.savefig('.\\' + dirName + '\\' + repName + '.png')
    plt.show()
    return


# In[6]:


#gamma t控制图
def t_gamma(data,shape,loc,scale,dirName,repName):
    #对shape加以限制
    if shape>25:
        print('拟合结果为gamma为偶然因素，请查看其是否为正态分布')
        EWMA_norm(data,dirName,repName)
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
        row = [groupMean.loc[i].values[0],muS - k1*stdS/(m.sqrt(groupNum.loc[i])),                muS - k2*stdS/(m.sqrt(groupNum.loc[i])),muS,muS + k2*stdS/(m.sqrt(groupNum.loc[i])),muS + k1*stdS/(m.sqrt(groupNum.loc[i]))]
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
    print('有问题的日期是',date[flag != 0].tolist())
    plt.savefig('.\\' + dirName + '\\' + repName + '.png')
    plt.show()
    return


# In[7]:


#rayleigh REWMA控制图
def EWMA_rayleigh(data,loc,scale,dirName,repName):
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
    print('有问题的日期是',date[flag].tolist())
    plt.savefig('.\\' + dirName + '\\' + repName + '.png')
    plt.show()
    return


# In[8]:


def shewhart_rayleigh(data,loc,scale,dirName,repName):
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
    print('有问题的日期是',date[flag].tolist())
    plt.savefig('.\\' + dirName + '\\' + repName + '.png')
    plt.show()
    return


# In[9]:


def Lapage_chart(data,sample,table,dirName,repName):
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
    plt.xticks(range(1, len(record), 5), date[::5], color='black', rotation=90)
    plt.title('Lapage Chart')
    plt.xlabel('Date')
    plt.ylabel('s^2')
    print('有问题的日期是',date[flag].tolist())
    plt.savefig('.\\' + dirName + '\\' + repName + '.png')
    plt.show()
    return


# In[10]:


def Lapage_simulation(m_sample, n_max=5, simu_time = 1000, alpha = 0.0027, ARL0 = 370):
    # m_sample is the sample size, alpha =0.0027, ARL = 370
    #target:H,H1,H2
    #获得样例数据N(0,1)
    from scipy.stats import norm
    from scipy.stats import rankdata
    sample = norm.rvs(size = m_sample,random_state = 42)
    recordH = []
    #对每一个n寻找H,H1,H2
    for n in range(1,n_max+1):
        UCL = 30
        LCL = 0
        #H = n #init_value
        while True:
            print(UCL)
            print(LCL)
            H = (UCL+LCL)/2#init_value
            records = []#只要H不对，重新开始运行，records就清0
            #每一个针对H的simulation运行到越界为止
            ARL = []
            for i in range(1,simu_time+1):
                count = 0
                s_2 = -1
                while s_2<H:
                    count = count+1
                    test = norm.rvs(size = n)
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
                    #record所有simulation中的s1_2,s2_2
                    if s_2<H:
                        records.append([s1_2,s2_2])
                    if count>ARL0*1.01:
                        break
                ARL.append(count)
            #所有simulation的ARL均值
            ARL = np.mean(ARL)
            print(ARL)
            if (UCL - LCL)<=0.0001:
                print(n)
                print('非正常结束')
                break
            if ARL>ARL0*1.01:
                #H = H - 0.01
                UCL = H
            elif ARL<ARL0*0.99:
                #H = H + 0.01
                LCL = H
            else:
                break
        #此时选择到了正确的H或者选不到正确的H了
        p = 1 - m.sqrt(1-alpha)#P(s1_2>H1)以及P(s2_2>H2)
        records = np.array(records)
        s1_2_sorted = sorted(records[:,0],reverse = True)
        s2_2_sorted = sorted(records[:,1],reverse = True)
        H1 = s1_2_sorted[int(len(s1_2_sorted)*p)]
        H2 = s2_2_sorted[int(len(s2_2_sorted)*p)]
        recordH.append([H,H1,H2])
    table = pd.DataFrame(recordH,index = range(1,n_max+1),columns=['H','H1','H2'])
    return table


# ## 脚本

# In[11]:


#获得所有支持分布
from fitter import get_common_distributions
dist_common = get_common_distributions()
print(dist_common)


# In[12]:


def fit_data(excel_name, name, col_from, col_end, limitTable):
        data = pd.read_excel(excel_name, sheet_name=name)
        record = []
        for i in range(col_from, col_end): #对单个sheet寻找每一列
            col = data.iloc[: ,i]          #读取每一列数据转成List
            dataList = col.tolist()
            repName = name + '_column' + toChar(i) + '_FitResult'
            #可指定拟合池子
            dist = dist_common
            binNum = round(m.sqrt(len(dataList)))
            
            #拟合结果并输出
            f = Fitter(dataList, xmin=limitTable[i-col_from][0], xmax=limitTable[i-col_from][1], bins=binNum, distributions=dist_common, timeout=100, density=True)
            f.fit(progress=False, n_jobs=-1, max_workers=-1)
            print('\033[0;34m'+repName+'\033[0m')
            best = f.get_best(method='bic')
            print("\033[0;31mBestMethod&Parameters\033[0m")
            print(best)
            result = f.summary(Nbest=5, lw=2, plot=True, method='bic') # 返回最好的Nbest个分布及误差，并绘制数据分布和Nbest分布
            print("\033[0;31mBest5Method&Criteria\033[0m")
            print(result)
            print()
            
            dirName = excel_name.split('.')[0] + '-' + name
            
            #记录所有变量的分布和参数
            (dist, para), = best.items()
            column = toChar(i)
            record = record + [[column, dist, str(para)]]
            
            #存储图片
            if not os.path.exists('.\\' + dirName):
                os.mkdir('.\\' + dirName)
            plt.savefig('.\\' + dirName + '\\' + repName + '.png')
            plt.show()
            
            #对该列的变量进行绘图
            #已经有dirName了
            repName_SPC = 'column'+toChar(i)+'_SPC'
            print('\033[0;34m'+dirName+'column'+toChar(i)+'SPC\033[0m')
            dataSPC = data.iloc[:,[1,i]]
            if dist == 'norm':
                EWMA_norm(dataSPC,dirName,repName_SPC)
            elif dist == 'lognorm':
                EWMA_lognorm(dataSPC,dirName,repName_SPC)
            elif dist == 'expon':
                loc1 = para.get('loc')
                scale1 = para.get('scale')
                t_expon(dataSPC,loc1,scale1,dirName,repName_SPC)
            elif dist == 'gamma':
                a1 = para.get('a')
                loc1 = para.get('loc')
                scale1 = para.get('scale')
                t_gamma(dataSPC,a1,loc1,scale1,dirName,repName_SPC)
            elif dist == 'rayleigh':
                loc1 = para.get('loc')
                scale1 = para.get('scale')
                EWMA_rayleigh(dataSPC,loc1,scale1,dirName,repName_SPC)
            elif dist == 'cauchy':
                loc1 = para.get('loc')
                scale1 = para.get('scale')
                from scipy.stats import cauchy
                random = cauchy.rvs(loc=loc1, scale=scale1,size=50,random_state = 42)
                if os.path.isfile('table.csv'):
                    table = pd.read_csv('table.csv',index_col = 0)
                else:
                    table = Lapage_simulation(50)
                Lapage_chart(dataSPC, random, table,dirName,repName_SPC)
            elif dist == 'chi2':
                loc1 = para.get('loc')
                scale1 = para.get('scale')
                df1 = para.get('df')
                from scipy.stats import chi2
                random = chi2.rvs(df=df1,loc=loc1,scale=scale1,size=50,random_state=42)
                if os.path.isfile('table.csv'):
                    table = pd.read_csv('table.csv',index_col = 0)
                else:
                    table = Lapage_simulation(50)
                Lapage_chart(dataSPC, random, table,dirName,repName_SPC)
            elif dist == 'exponpow':
                loc1 = para.get('loc')
                scale1 = para.get('scale')
                b1 = para.get('b')
                from scipy.stats import exponpow
                random = exponpow.rvs(b=b1,loc=loc1,scale=scale1,size=50,random_state=42)
                if os.path.isfile('table.csv'):
                    table = pd.read_csv('table.csv',index_col = 0)
                else:
                    table = Lapage_simulation(50)
                Lapage_chart(dataSPC, random, table,dirName,repName_SPC)
            elif dist == 'powerlaw':
                loc1 = para.get('loc')
                scale1 = para.get('scale')
                a1 = para.get('a')
                from scipy.stats import powerlaw
                random = powerlaw.rvs(a=a1,loc=loc1,scale=scale1,size=50,random_state=42)
                if os.path.isfile('table.csv'):
                    table = pd.read_csv('table.csv',index_col = 0)
                else:
                    table = Lapage_simulation(50)
                Lapage_chart(dataSPC, random, table,dirName,repName_SPC)
            elif dist == 'uniform':
                loc1 = para.get('loc')
                scale1 = para.get('scale')
                from scipy.stats import uniform
                random = uniform.rvs(loc=loc1,scale=scale1,size=50,random_state=42)
                if os.path.isfile('table.csv'):
                    table = pd.read_csv('table.csv',index_col = 0)
                else:
                    table = Lapage_simulation(50)
                Lapage_chart(dataSPC, random, table,dirName,repName_SPC)
        #将sheet所有拟合信息输出到一个csv文件
        record = pd.DataFrame(record)
        record.to_csv('.\\' + dirName + '\\' + dirName + '.csv', header = ['column','dist','Paras'], index=False)
        return record


# ### PFA2B_Mna1 sheet=7830

# In[13]:


#整张sheet的测点上下界
Limit =         [[38.0,67.0],
                [40.0,64.0],
                [38.0,65.0],
                [39.0,65.0],
                [40.0,66.0],
                [40.0,68.0],
                [39.0,63.0],
                [38.0,63.0],
                [38.0,62.0],
                [39.0,63.0],
                [40.0,69.0],
                [40.0,66.0],
                [39.0,61.0],
                [39.0,62.0],
                [39.0,62.0],
                [40.0,61.0],
                [40.0,67.0],
                [40.0,63.0],
                [40.0,60.0],
                [40.0,60.0],
                [39.0,62.0],
                [39.0,63.0],
                [40.0,62.0],
                [40.0,62.0],
                [33.5,55.0],
                [33.5,55.0]]


# In[14]:


#PFA2B_Mna1 sheet=7830
sheet7830 = fit_data('PFA2B_Mna1.xlsx','7830',4,30,Limit)


# ### PFA2B_Mna1 sheet=7920

# In[15]:


Limit = [[16.0,24.0],
        [16.0,24.0],
        [16.0,24.0],
        [16.0,24.0],
        [16.0,24.0],
        [16.0,24.0],
        [16.0,24.0],
        [16.0,24.0],
        [16.0,24.0],
        [16.0,24.0],
        [16.0,24.0],
        [16.0,24.0],
        [7.2,10.8],
        [7.2,10.8],
        [7.2,10.8],
        [7.2,10.8]]


# In[16]:


#PFA2B_Mna1 sheet=7920
sheet7920 = fit_data('PFA2B_Mna1.xlsx','7920',4,20,Limit)


# ### PFA2B_Mna1 sheet=BA7

# In[17]:


Limit = [[20,30],
        [20,30],
        [20,30],
        [20,30],
        [20,30],
        [20,30],
        [20,30],
        [20,30],
        [9.6,14.4],
        [9.6,14.4],
        [16,24],
        [16,24]]


# In[18]:


#PFA2B_Mna1 sheet=BA7
BA7 = fit_data('PFA2B_Mna1.xlsx','BA7',4,16,Limit)


# In[19]:


data = pd.read_excel('PFA2B_Mna1.xlsx', 'BA7')


# ### PFA2B_Mna1 sheet=左翼子板

# In[20]:


Limit = [[6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [3.6,5.4]]


# In[ ]:


#PFA2B_Mna1 sheet=左翼子板
LW = fit_data('PFA2B_Mna1.xlsx','左翼子板',4,20,Limit)


# ### PFA2B_Mna1 sheet=右翼子板

# In[ ]:


Limit = [[6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [6.4,9.6],
        [3.6,5.4]]


# In[ ]:


#PFA2B_Mna1 sheet=右翼子板
RW = fit_data('PFA2B_Mna1.xlsx','右翼子板',4,20,Limit)


# In[ ]:




