#####训练和预测
import scipy.io as io
import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm#Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。
from models.Net import UNet
import matplotlib
matplotlib.use('TKAgg')  # 切换到非交互式后端
import matplotlib.pyplot as plt
import segyio
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'      #使用优化器、线性增加学习率lr
import cv2

def myNormalization(data):        #将数据进行归一化操作
    max_val = np.max(data)
    min_val = np.min(data)
    return (data[:,:,:] - min_val) / (max_val - min_val)       #data为3维的数据

def back_normalized(data,data_origin):       #将数据反归一化
    max_val = np.max(data_origin)
    min_val = np.min(data_origin)
    return (data[:,:])  *  (max_val - min_val) + min_val

lines=4       #线
cmp=208         #道
point=208     #采样点


######读取数据
dPath = 'data/img/train/data/'
dfile_list = os.listdir(dPath)
data = []
for i in range(len(dfile_list)):
    dfile_list1 = dfile_list[i]
    d = cv2.cvtColor(cv2.imread(dPath+dfile_list1), cv2.COLOR_BGR2RGB)[:208,:208,:]
    data.append(d)
data = np.array(data).reshape(lines, 3, cmp, point)
data=np.nan_to_num(data,nan=0)    #将inputdatas_np里面的nan变为0
print(data.shape)
data=myNormalization(data)   #将标签归一化
data = data.astype(np.float32)
print(data.shape)


tPath = 'data/img/train/label/'
lfile_list = os.listdir(tPath)
label = []
for i in range(len(lfile_list)):
    lfile_list1 = lfile_list[i]
    l = cv2.cvtColor(cv2.imread(tPath+lfile_list1), cv2.COLOR_BGR2RGB)[:208,:208,:]
    label.append(l)
label = np.array(label).reshape(lines, 3, cmp, point)
label=np.nan_to_num(label,nan=0)    #将inputdatas_np里面的nan变为0
label=myNormalization(label)   #将标签归一化
label = label.astype(np.float32)
print(label.shape)


#加噪
# inputdatas_np_noisy=np.zeros(shape=(lines,cmp,point))
# for i in range(lines):
#     for j in range(cmp):
#         noise_factor=0.15
#         inputdatas_np_noisy[i,j,:]=inputdatas_np[i,j,:]+noise_factor*np.random.normal(loc=0.0,scale=0.1,size=inputdatas_np[i,j,:].shape)
#         inputdatas_np_noisy[i,j,:]=np.clip(inputdatas_np_noisy[i,j,:],0.,1)
#         inputdatas_np_noisy[i,j,:]=inputdatas_np_noisy[i,j,:].astype(np.float32)
# plt.imshow(inputdatas_np_noisy[0].T,cmap="seismic")
# plt.show()

# Set up the dataloader
class SeismicLoader(Dataset):
    def __init__(self, x_train, y_train):
        self.input = x_train
        self.target = y_train

    def __getitem__(self, index):
        return self.input[index], self.target[index]

    def __len__(self):
        return self.input.shape[0]


# #######组建初始训练集
# X_train=[]  #训练集
# Y_train=[]  #训练集
# x=np.linspace(0,49,50)  ##选取多少线
# x=x.astype(np.int32)  ##转为int类型
# for i in range(len(x)):
#     a=x[i]
#     X_train1=data[a, :, :]
#     Y_train1=label[a, :, :]
#     X_train.append(X_train1)
#     Y_train.append(Y_train1)
# X_train=np.array(X_train).reshape(50,cmp,point)  ###转为ndarray格式
# Y_train=np.array(Y_train).reshape(50,cmp,point)  ###转为ndarray格式
# X_train=X_train.astype(np.float32)      #####初始学习器需要的地震数据
# Y_train=Y_train.astype(np.float32)             ####初始学习器需要的波阻抗


# #进行网络学习训练
dataset1 = SeismicLoader(data, label)
###训练的数据集
train_loader1 = DataLoader(dataset=dataset1,
                              batch_size=4,
                              shuffle=True)

# '''
# Train
# '''

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#torch.device代表将torch.Tensor分配到的设备的对象。torch.device包含一个设备类型（‘cpu’或‘cuda’）和可选的设备序号
# print(device)
# model1=UNet(3,3)
# model1.train()
# model1.to(device)
# trainloss1 = []  # 创建存放训练损失值的列表
# valloss1 = []  # 创建存放验证损失的列表
# epochs=2000
# criterion1=nn.MSELoss()#损失函数定义为均方差函数
# optimizer1 = optim.Adam( model1.parameters(),lr = 0.0001)#优化器采用Adam
#
# for epoch in tqdm(range(epochs)):
#     running_loss1 = 0.0  # 设定初始训练损失值为0
#     valling_oss1 = 0.0  # 设定初始验证损失值为0
#     for x, y in train_loader1:
#         optimizer1.zero_grad()#清空过往梯度
#         x_train = x[:,:,:,:]#第一条线3通道，66个采样点，17道
#
#         # x1 = np.expand_dims(x_train,axis=1)
#         # x1 = x_train[np.newaxis,:]      ##增加一维，第一维
#         x1 = torch.tensor(x_train)
#         x1 = x1.to(device)
#         pred = model1(x1)
#
#
#         y_train = y[:,:,:,:]
#         # y_train =np.expand_dims(y_train,axis=1)
#         # y_train=y_train[np.newaxis,:]
#         if np.isnan(y_train).any():
#             continue
#         y_train = torch.tensor(y_train)
#
#         y_train = y_train.to(device)
#         loss1 = criterion1(pred, y_train)      ##计算损失值
#         loss1.backward()#反向传播，计算当前梯度；
#         optimizer1.step()#根据梯度更新网络参数
#         # optimizer.zero_grad()
#         running_loss1 += loss1.item()
#             # print(loss.item())
#     if epoch % 5 == 0:
#         print("当前训练次数：{}/{}，损失值为：{}".format(epoch, epochs, running_loss1))
#     if epoch % 50 == 0:
#         torch.save(model1.state_dict(), 'checkpoint_Unet_img/unet_img_epo{}.pt'.format(epoch))
#     trainloss1.append(running_loss1)
#
# print('Finished Training')
# plt.plot(trainloss1,'r',label='train_loss')#绘制训练损失值图像
# plt.plot(valloss1,'g',label='val_loss')##绘制验证损失值图像
# plt.legend()#增加图例
# plt.xlabel('Epochs')#x轴方向为训练轮数
# plt.ylabel('Loss')#y轴方向为损失值
# plt.show()#显示图像
# torch.save(model1.state_dict(), 'checkpoint_Unet_img/unet_img_epo2000.pt')

'''
Predict
'''

######读取预测数据
dPath_pre = 'data/img/train/data/'
dfile_list_pre = os.listdir(dPath_pre)
data_pre = []
for i in range(len(dfile_list_pre)):
    dfile_list1_pre = dfile_list_pre[i]
    d = cv2.cvtColor(cv2.imread(dPath_pre+dfile_list1_pre), cv2.COLOR_BGR2RGB)[:208,:208,:]
    data_pre.append(d)
data_pre = np.array(data_pre).reshape(lines, 3, cmp, point)
data_pre=np.nan_to_num(data_pre,nan=0)    #将inputdatas_np里面的nan变为0
print(data_pre.shape)
data_pre=myNormalization(data_pre)   #将标签归一化
data_pre = data_pre.astype(np.float32)
print(data_pre.shape)


# ## 加噪
# data_pre_noisy=np.zeros(shape=(lines,cmp,point)).astype(np.float32)
# for i in range(lines):
#     for j in range(cmp):
#         noise_factor=0.30
#         data_pre_noisy[i,j,:]=data_pre[i,j,:]+noise_factor*np.random.normal(loc=0.0,scale=0.1,size=data_pre[i,j,:].shape)
#         data_pre_noisy[i,j,:]=np.clip(data_pre_noisy[i,j,:],0.,1)
#         data_pre_noisy[i,j,:]=data_pre_noisy[i,j,:].astype(np.float32)


tPath_pre = 'data/img/train/label/'
lfile_list_pre = os.listdir(tPath_pre)
label_pre = []
for i in range(len(lfile_list_pre)):
    lfile_list1_pre = lfile_list_pre[i]
    l = cv2.cvtColor(cv2.imread(tPath_pre+lfile_list1_pre), cv2.COLOR_BGR2RGB)[:208,:208,:]
    label_pre.append(l)
label_pre = np.array(label_pre).reshape(lines, 3, cmp, point)
label_pre=np.nan_to_num(label_pre,nan=0)    #将inputdatas_np里面的nan变为0
label_pre = myNormalization(label_pre)
print(label_pre.shape)


## 加噪
label_pre_noisy=np.zeros(shape=(lines, 3, cmp,point)).astype(np.float32)
for i in range(lines):
    for j in range(3):
        noise_factor=0.70
        label_pre_noisy[i,j,:,:]=label_pre[i,j,:,:]+noise_factor*np.random.normal(loc=0.0,scale=0.1,size=data_pre[i,j,:,:].shape)
        label_pre_noisy[i,j,:,:]=np.clip(label_pre_noisy[i,j,:,:],0.,1)
        label_pre_noisy[i,j,:,:]=label_pre_noisy[i,j,:,:].astype(np.float32)

label_pre = label_pre_noisy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#torch.device代表将torch.Tensor分配到的设备的对象。torch.device包含一个设备类型（‘cpu’或‘cuda’）和可选的设备序号
new_m=UNet(3,3)
new_m.train()
new_m.to(device)#将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。

criterion=nn.MSELoss()#损失函数定义为均方差函数
optimizer = optim.Adam( new_m.parameters(),lr = 0.001)#优化器采用Adam

EndLine=1
for k in range(1,EndLine+1):
    result=[]     #预测的波阻抗
    result_final = []
    new_m.load_state_dict(torch.load("checkpoint_Unet_img/unet_img_epo2000.pt"))   ##T1-pre-T5-pre

    for i in range(len(data_pre)):
        optimizer.zero_grad()    #梯度清零
        x_test = data_pre[i:i+1,:,:,:]  # 第4条线3通道，480个采样点，16道

        x = torch.tensor(x_test)

        x = x.to(device)
        outputs=new_m(x)
        outputs = outputs.to(device)
        outputs = outputs.to("cpu")
        outputs=outputs.detach().numpy()   ##保持网络中的一些参数，并不让梯度对主网络的梯度造成影响
        outputs=outputs.astype(np.float32)

        result.append(outputs)

    result=np.array(result)
    print(result.shape)

    print('####')
    result=result.flatten()
    result=result.reshape(lines,3,cmp,point)

    ## 阈值分割
    # result = np.where((result<=0.5), 0, result)
    # result = np.where((result <= 0.25), 0, 1)
    #
    # paint
    plt.imshow(label_pre[0][0,:,:].T,cmap="seismic")
    plt.show()

    label_pre.tofile('tezhengtu/framework/img.dat')