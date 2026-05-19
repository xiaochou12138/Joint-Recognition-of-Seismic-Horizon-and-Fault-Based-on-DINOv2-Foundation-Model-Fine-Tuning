import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('TKAgg')  # 切换到非交互式后端
import matplotlib.pyplot as plt
from my_cmap import get_cmap_xmw

# 计算横向梯度
def compute_horizontal_gradient_robust(A):
    A = np.array(A)
    A = np.nan_to_num(A)  # 将NaN替换为0
    gradient = np.roll(A, -1, axis=1) - A
    gradient[:, -1] = 0
    return gradient

# 计算梯度结构张量
def compute_structure_tensor(data):
    """
    计算地震数据在高度维度的横向梯度结构张量

    参数:
        data: 输入地震数据，形状为 (batch_size, channels, height, width)

    返回:
        结构张量，形状为 (batch_size, 3, height, width)
        三个通道分别对应: [gx², gx·gy, gy²]
    """
    if data.ndim != 4:
        raise ValueError("输入数据必须是4维张量 (batch, channels, height, width)")

    # 1. 计算高度和宽度方向的梯度
    # 使用Sobel滤波器增强梯度计算
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=data.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32, device=data.device).view(1, 1, 3, 3)

    # 扩展滤波器到所有通道
    sobel_x = sobel_x.repeat(data.size(1), 1, 1, 1)
    sobel_y = sobel_y.repeat(data.size(1), 1, 1, 1)

    # 计算梯度 (使用分组卷积保持通道独立性)
    gx = F.conv2d(data, sobel_x, padding=1, groups=data.size(1))
    gy = F.conv2d(data, sobel_y, padding=1, groups=data.size(1))

    # 2. 计算结构张量分量
    gx2 = gx.pow(2)  # ∂x²
    gy2 = gy.pow(2)  # ∂y²
    gx_gy = gx * gy  # ∂x·∂y

    # 3. 跨通道聚合 (对三个地震通道求和)
    J_xx = gx2.sum(dim=1, keepdim=True)  # 保持维度
    J_yy = gy2.sum(dim=1, keepdim=True)
    J_xy = gx_gy.sum(dim=1, keepdim=True)

    # 4. 组合结构张量
    structure_tensor = torch.cat([J_xx, J_xy, J_yy], dim=1)

    return structure_tensor[:,:,:,:]

def compute_second_derivatives(u):
    """
    计算输入数组的二阶导数 uxx 和 uyy

    参数:
        u: 输入数组，形状为 (batch, channels, height, width)

    返回:
        uxx: 二阶导数 (x方向)
        uyy: 二阶导数 (y方向)
    """
    if u.ndim != 4:
        raise ValueError("输入数组必须是4维张量 (batch, channels, height, width)")

    # 定义二阶导数卷积核
    kernel_xx = torch.tensor([[0, 0, 0],
                              [1, -2, 1],
                              [0, 0, 0]], dtype=torch.float32, device=u.device)

    kernel_yy = torch.tensor([[0, 1, 0],
                              [0, -2, 0],
                              [0, 1, 0]], dtype=torch.float32, device=u.device)

    # 调整核形状以适应卷积操作 (out_channels, in_channels, H, W)
    kernel_xx = kernel_xx.view(1, 1, 3, 3)
    kernel_yy = kernel_yy.view(1, 1, 3, 3)

    # 计算二阶导数
    uxx = F.conv2d(u, kernel_xx, padding=1)
    uyy = F.conv2d(u, kernel_yy, padding=1)

    return uxx, uyy

# 计算地震数据梯度
dPath = 'data/Horizon_and_fault/train/seis/0.dat'
data = np.fromfile(dPath, np.float32).reshape(224, 224)
data = (data - data.min()) / (data.max() - data.min())
data = data.reshape(1,1,224,224)
data = torch.tensor(data).float()

grad_x = compute_structure_tensor(data)

# 计算层位体
dPath1 = 'data/Horizon_and_fault/train/horizon/0.dat'
horizon = np.fromfile(dPath1, np.float32).reshape(224, 224)
horizon = horizon.reshape(1,1,224,224)
horizon = torch.tensor(horizon).float()

# 处理地震层位预测结果，将其转为类似RGT形式
# 找到层位点的位置
pre_horizon = horizon
RGT_arr = torch.zeros(size=(pre_horizon.shape[0], pre_horizon.shape[1], pre_horizon.shape[2], pre_horizon.shape[3]))
# 保留 >0.5 的值，其余置 0
pre_horizon = torch.where(pre_horizon > torch.tensor(0.04), pre_horizon, torch.tensor(0.0))
for i in range(len(pre_horizon)):
    pre_horizon1 = pre_horizon[i][0]
    for j in range(len(pre_horizon1)):
        pre_horizon2 = pre_horizon1[j]
        # 找到单道数据层位点
        idx = torch.where(pre_horizon2 != 0)
        arr = 1
        for k in range(len(idx[0])):
            idx1 = idx[0][k]
            if (k == 0):
                arr_value = torch.linspace(0.0001, arr, idx1)
                RGT_arr[i, 0, j, 0:idx1] = arr_value
            else:
                arr_value = torch.linspace(arr - 1, arr, idx1 - idx[0][k - 1])
                RGT_arr[i, 0, j, idx[0][k - 1]:idx1] = arr_value
            arr = arr + 1
RGT_arr = torch.where(RGT_arr == torch.tensor(0), torch.tensor(1), RGT_arr)

# 计算层位体的梯度
st_RGT = compute_structure_tensor(RGT_arr.to('cpu'))
st_RGT = (st_RGT - st_RGT.min()) / (st_RGT.max() - st_RGT.min()).float().to('cpu')

mycmap = get_cmap_xmw()
plt.imshow(st_RGT.cpu().numpy()[0,0,:128,:128].T, cmap='jet')
plt.show()

gradient_RGT = compute_horizontal_gradient_robust(RGT_arr.to('cpu')[0][0])
mycmap = get_cmap_xmw()
plt.imshow(gradient_RGT[:128,:128].T, cmap='jet')
plt.show()


# 计算地震层位填充体的二阶导数 uxx 和 uyy
# 二阶导有助于突出地震剖面中大尺度的不连续特征，由于uxx uyy聚焦于全局构造变化 低阶特征强化了局部反射细节，两则结合可以在噪声以及断层规模小的情况下，依旧可以识别到连续且合理的断层走向
RGT_xx, RGT_yy = compute_second_derivatives(RGT_arr)

# mycmap = get_cmap_xmw()
# plt.imshow(st_RGT.cpu().numpy()[0,0,:128,:128].T, cmap="jet")
# plt.show()