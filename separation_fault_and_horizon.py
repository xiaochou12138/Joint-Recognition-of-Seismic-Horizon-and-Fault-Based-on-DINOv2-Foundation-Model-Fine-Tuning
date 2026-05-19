import numpy as np
import matplotlib
matplotlib.use('TKAgg')  # 切换到非交互式后端
import matplotlib.pyplot as plt
from scipy.ndimage import sobel


# 1、分解标签 构造  --->  层位  +  断层
seis = np.fromfile("data/synthetic_data/valid/seis/0.dat", np.float32).reshape(224, 224)
label = np.fromfile("data/synthetic_data/valid/label/0.dat", np.float32).reshape(224, 224)

I = label

# 2. 傅里叶变换
F = np.fft.fft2(I)
F_shift = np.fft.fftshift(F)  # 将零频移到中心

# 3. 构造滤波器（理想低通）
h, w = I.shape
center_y, center_x = h // 2, w // 2

# 横向结构：垂直方向低通（保留水平变化）
mask_horizontal = np.ones((h, w))
mask_horizontal[center_y-30:center_y+30, :] = 0  # 垂直方向高频置零
F_horizontal = F_shift * mask_horizontal

# 纵向结构：水平方向低通（保留垂直变化）
mask_vertical = np.ones((h, w))
mask_vertical[:, center_x-50:center_x+50] = 0  # 水平方向高频置零
F_vertical = F_shift * mask_vertical

# 4. 逆傅里叶变换
I_horizontal = np.abs(np.fft.ifft2(np.fft.ifftshift(F_horizontal)))
I_vertical = np.abs(np.fft.ifft2(np.fft.ifftshift(F_vertical)))

# 5. 归一化
I_horizontal = (I_horizontal - I_horizontal.min())/(I_horizontal.max() - I_horizontal.min())
I_vertical = (I_vertical - I_vertical.min())/(I_vertical.max() - I_vertical.min())

# 6、给定阈值 分割
I_horizontal = np.where((I_horizontal > 0.73), 1, 0)
I_vertical = np.where((I_vertical > 0.56), 1, 0)
I_horizontal = I - I_vertical
I_horizontal = np.where((I_horizontal < 0), 0, I_horizontal)

# 7. 可视化
plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1)
plt.title("Seismic data")
plt.imshow(seis[:128,:128].T, cmap='seismic')
# plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Structure result")
plt.imshow(I[:128,:128].T, cmap='RdBu')
# plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Fault result")
plt.imshow(I_horizontal[:128,:128].T, cmap='RdBu')
# plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Horizon result")
plt.imshow(I_vertical[:128,:128].T, cmap='RdBu')
# plt.axis('off')

plt.tight_layout()
plt.show()