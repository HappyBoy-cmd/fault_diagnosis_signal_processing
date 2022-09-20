import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def get_time_domain_features(data):
    '''data为一维振动信号'''
    x_rms = 0
    absXbar = 0
    x_r = 0
    S = 0
    K = 0
    k = 0
    x_rms = 0
    fea = []
    len_ = len(data)
    x_std = data.std()
    ##----有量纲指标---##
    x_mean = data.mean(axis=0)  # 1.均值
    x_var = data.var(axis=0)  # 2.方差
    x_max = data.max(axis=0)  # 3.最大值
    x_min = data.min(axis=0)  # 4.最小值
    for i in range(len_):
        x_rms += data[i] ** 2
        absXbar += abs(data[i])
        x_r += math.sqrt(abs(data[i]))
        S += (data[i] - x_mean) ** 3
        K += (data[i] - x_mean) ** 4
    x_r = (x_r / len_) ** 2  # 5.方根幅值
    x_absXbar = absXbar / len_  # 6.绝对平均幅值
    alpha = S / ((len_ - 1) *x_std ** 4)  # 7.偏度
    beta = K / ((len_ - 1) * x_std ** 4)  # 8.峭度
    x_rms = math.sqrt(x_rms / len_)  # 9.均方根
    x_p = max(abs(x_max), abs(x_min)) # 10峰值
    x_pp = x_max - x_min # 11.峰峰值
    
    ##----无量纲指标---##
    W_f = x_rms / x_absXbar  # 12.波形指标
    C_f = x_p / x_rms  # 13.峰值指标
    I_f = x_p / x_absXbar  # 14.脉冲指标
    L_f = x_p / x_r  # 15.裕度指标
    K_f = beta / x_rms ** 4 # 16. 峭度指标
    P_f = alpha / x_rms ** 3 # 16. 偏（斜）度指标
    
#     time_fea_2_name_dict = {'x_mean': '均值', 'x_var': '方差', x_max: '最大值', x_min: '最小值', x_r'方根幅值', 'x_absXbar': '绝对平均幅值', 'alpha': '偏度', 'beta': '峭度', 'x_rms': '均方根', 'x_p': '峰值', 'x_pp': '峰峰值', 'W_f': '波形系数', 'C_f': '峰值指标', 'I_f': '脉冲指标', 'L_f': '裕度指标', 'K_f': '峭度指标', 'P_f': '偏斜度指标'}

    time_fea_list = [x_mean,x_var, x_max, x_min, x_r, x_absXbar, alpha, beta,x_rms, x_p, x_pp, W_f, C_f, I_f, L_f, K_f, P_f]
#     for time_fea in time_fea_list:
#         print(time_fea_2_name_dict(time_fea))
    return time_fea_list

''' ============== 特征提取的类 =====================
时域特征 ：11类
频域特征 : 13类
总共提取特征 ： 24类

参考文献 英文文献 016_C_(Q1 时域和频域共24种特征参数 )  Fault diagnosis of rotating machinery based on multiple ANFIS combination with GAs

'''
def get_freq_fea(data, fs):
    """
    提取频域特征 13类
    :param signal_:
    :return:
    """
    signal_ = np.array(data)
    Fs = fs
    L = len(signal_)
    PL = abs(np.fft.fft(signal_ / L))[: int(L / 2)]
    PL[0] = 0
    f = np.fft.fftfreq(L, 1 / Fs)[: int(L / 2)]
    x = f
    y = PL
    K = len(y)
    # print("signal_.shape:",signal_.shape)
    # print("PL.shape:", PL.shape)
    # print("L:", L)
    # print("K:", K)
    # print("x:",x)
    # print("y:",y)

    f_1 = np.mean(y)  #P1

    f_2 = np.var(y)  #P2

    f_3 = (np.sum((y - f_1)**3))/(K * ((np.sqrt(f_2))**3))  #P3

    f_4 = (np.sum((y - f_1)**4))/(K * ((f_2)**2)) #P4

    f_5 = (np.sum(x * y))/(np.sum(y))  #P5

    f_6 = np.sqrt((np.mean(((x- f_5)**2)*(y)))) #P6

    f_7 = np.sqrt((np.sum((x**2)*y))/(np.sum(y))) #P7

    f_8 = np.sqrt((np.sum((x**4)*y))/(np.sum((x**2)*y))) #P8

    f_9 = np.sqrt((np.sum((x**2)*y))/((np.sum(y))*(np.sum((x**4)*y)))) #P9

    f_10 = f_6/f_5 #P10

    f_11 = (np.sum(((x - f_5)**3)*y))/(K * (f_6**3)) #P11

    f_12 = (np.sum(((x - f_5)**4)*y))/(K * (f_6**4)) #P12

    f_fea = [f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9, f_10, f_11, f_12]

    return f_fea, y
