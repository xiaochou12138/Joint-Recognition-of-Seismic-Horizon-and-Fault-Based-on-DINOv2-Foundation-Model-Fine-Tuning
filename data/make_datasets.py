import numpy as np
import pandas as pd
import os
import segyio
import matplotlib
matplotlib.use('TKAgg')  # 切换到非交互式后端
import matplotlib.pyplot as plt
from tqdm import tqdm

# # 添加高斯噪声（根据信噪比）
# def add_gaussian_noise_by_snr(data, snr_db):
#     """
#     根据信噪比（SNR）向数据中添加高斯噪声
#     :param data: 原始数据
#     :param snr_db: 目标信噪比（单位：分贝）
#     :return: 加噪后的数据
#     """
#     # 计算信号功率
#     signal_power = np.mean(data ** 2)
#
#     # 将信噪比从分贝转换为线性比例
#     snr_linear = 10 ** (snr_db / 10)
#
#     # 计算噪声功率
#     noise_power = signal_power / snr_linear
#
#     # 计算噪声的标准差
#     noise_std = np.sqrt(noise_power)
#
#     # 生成噪声并添加到数据中
#     noise = np.random.normal(0, noise_std, data.shape)
#     noisy_data = data + noise
#
#     return noisy_data
#
#
#
# # 地震数据
# # 定义要遍历的目录
# directory_path = 'Horizon_and_fault/valid/seis'
#
# # 使用os.walk()递归遍历目录
# for root, dirs, files in os.walk(directory_path):
#     # root: 当前目录路径
#     # dirs: 当前目录下的子目录列表
#     # files: 当前目录下的文件列表
#
#     # 遍历文件
#     for file in files:
#         noisy_data = np.zeros(shape=(224, 224)).astype(np.float32)
#         file_path = os.path.join(root, file)  # 获取文件的完整路径
#         data = np.fromfile(file_path, np.float32).reshape(224, 224)
#
#         for i in range(len(data)):
#             data1 = data[i]
#             # 添加噪声
#             target_snr_db = 5  # 目标信噪比（单位：分贝）
#             noisy_data[i,:] = add_gaussian_noise_by_snr(data1, target_snr_db)
#
#         noisy_data.tofile('Horizon_and_fault/valid/seis_noise/' + file)  # 纯二进制流，需记录 dtype 和 shape 才能读取
#         print("successfully saved!")
#
#
# data = np.fromfile('Horizon_and_fault/valid/seis_noise/0.dat', np.float32).reshape(224, 224)
# # 可视化结果
# plt.figure(figsize=(10, 6))
# plt.imshow(data.T, cmap="RdBu")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()

'''
处理F3数据集
'''
# f3_seismic = segyio.cube("F3_Horizon_and_fault/gxl.sgy")
# f3_fault = segyio.cube("F3_Horizon_and_fault/fpx.sgy")
#
# # 层位数据导入
# f3_horizon1_file = "F3_Horizon_and_fault/gxl1_Horizon1_inline.txt"
# f3_horizon2_file = "F3_Horizon_and_fault/gxl1_Horizon2_inline.txt"
# f3_horizon3_file = "F3_Horizon_and_fault/gxl1_Horizon3_inline.txt"
#
# # 初始化第三列数据列表
# f3_horizon1 = []
# f3_horizon2 = []
# f3_horizon3 = []
#
# # 打开文件并逐行读取
# with open(f3_horizon1_file, 'r', encoding='utf-8') as file:
#     # 跳过前5行
#     for _ in range(6):
#         next(file)
#     for line in file:
#         # 去掉多余的空格并分割
#         values = line.strip().split()
#         # 提取第三列（索引为2）
#         f3_horizon1.append(float(values[5]))
#
# # 打开文件并逐行读取
# with open(f3_horizon2_file, 'r', encoding='utf-8') as file:
#     # 跳过前5行
#     for _ in range(6):
#         next(file)
#     for line in file:
#         # 去掉多余的空格并分割
#         values = line.strip().split()
#         # 提取第三列（索引为2）
#         f3_horizon2.append(float(values[5]))
#
# # 打开文件并逐行读取
# with open(f3_horizon3_file, 'r', encoding='utf-8') as file:
#     # 跳过前5行
#     for _ in range(6):
#         next(file)
#     for line in file:
#         # 去掉多余的空格并分割
#         values = line.strip().split()
#         # 提取第三列（索引为2）
#         f3_horizon3.append(float(values[5]))
#
# f3_horizon1 = np.array(f3_horizon1).reshape(511,384)
# f3_horizon2 = np.array(f3_horizon2).reshape(511,384)
# f3_horizon3 = np.array(f3_horizon3).reshape(511,384)
#
# # 制作层位标签  三维体形式
# f3_horizon = np.zeros(shape=(512,384,128)).astype(np.float32())
#
# for i in range(len(f3_horizon1)):
#     for j in range(len(f3_horizon1[i])):
#         time1 = f3_horizon1[i][j]
#         time2 = f3_horizon2[i][j]
#         time3 = f3_horizon3[i][j]
#         f3_horizon[i,j,int(time1)] = float(time1/128)
#         f3_horizon[i,j,int(time2)] = float(time2/128)
#         f3_horizon[i,j,int(time3)] = float(time3/128)
#
# print(f3_horizon.shape)
#
# # 导出训练集
# for i in tqdm(range(50)):
#     seismic = f3_seismic[i]
#     horizon = f3_horizon[i]
#     fault = f3_fault[i]
#     fault = np.where((fault<=0.25), 0, 1)
#     seismic.tofile('F3_Horizon_and_fault/train/seis/' + str(i) + '.dat')  # 纯二进制流，需记录 dtype 和 shape 才能读取
#     horizon.tofile("F3_Horizon_and_fault/train/horizon/" + str(i) + ".dat")  # 纯二进制流，需记录 dtype 和 shape 才能读取
#     fault.tofile('F3_Horizon_and_fault/train/fault/' + str(i) + '.dat')  # 纯二进制流，需记录 dtype 和 shape 才能读取
#
# # 导出预测
# for i in tqdm(range(0, 462)):
#     seismic = f3_seismic[i+50]
#     horizon = f3_horizon[i+50]
#     fault = f3_fault[i+50]
#     fault = np.where((fault<=0.25), 0, 1)
#     seismic.tofile('F3_Horizon_and_fault/valid/seis/' + str(i) + '.dat')  # 纯二进制流，需记录 dtype 和 shape 才能读取
#     horizon.tofile('F3_Horizon_and_fault/valid/horizon/' + str(i) + '.dat')  # 纯二进制流，需记录 dtype 和 shape 才能读取
#     fault.tofile('F3_Horizon_and_fault/valid/fault/' + str(i) + '.dat')  # 纯二进制流，需记录 dtype 和 shape 才能读取
#
#
# data = np.fromfile('F3_Horizon_and_fault/train/horizon/0.dat', np.float32).reshape(384, 128)
# # 可视化结果
# plt.figure(figsize=(10, 6))
# plt.imshow(data.T, cmap="seismic")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()



'''
处理F3数据集  换标签
'''

f3_seismic = segyio.cube("F3_dataset/gxl.sgy")
f3_label = np.fromfile('F3_dataset/label.dat', np.float64).reshape(512, 384, 128)

# 导出训练集
for i in tqdm(range(50)):
    seismic = f3_seismic[i]
    label = f3_label[i]
    seismic.tofile('F3_dataset/train/seis/' + str(i) + '.dat')  # 纯二进制流，需记录 dtype 和 shape 才能读取
    label.tofile('F3_dataset/train/label/' + str(i) + '.dat')  # 纯二进制流，需记录 dtype 和 shape 才能读取

# 导出预测
for i in tqdm(range(0, 462)):
    seismic = f3_seismic[i+50]
    label = f3_label[i+50]
    seismic.tofile('F3_dataset/valid/seis/' + str(i) + '.dat')  # 纯二进制流，需记录 dtype 和 shape 才能读取
    label.tofile('F3_dataset/valid/label/' + str(i) + '.dat')  # 纯二进制流，需记录 dtype 和 shape 才能读取


data = np.fromfile('F3_dataset/train/label/0.dat', np.float64).reshape(384, 128)
# 可视化结果
plt.figure(figsize=(10, 6))
plt.imshow(data.T, cmap="seismic")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()