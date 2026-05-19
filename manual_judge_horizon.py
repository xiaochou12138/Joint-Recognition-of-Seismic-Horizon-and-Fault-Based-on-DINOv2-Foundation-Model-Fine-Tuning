import numpy as np
from scipy.signal import correlate
from dtaidistance import dtw
import matplotlib
matplotlib.use('TKAgg')  # 切换到非交互式后端
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. 读入深度学习网络预测的结果
# -------------------------------------------------
pred = np.fromfile('png/structure/F3_result_vit.dat', np.float64).reshape(100, 384,128)        # (T, X)
pred = pred[52]
# 可视化
plt.imshow(pred.T, cmap="seismic")
plt.show()
T, X = pred.shape


def fill_gap(signal, gap_start, gap_end, template_len=100):
    gap_len = gap_end - gap_start

    # Step 1: 创建前后模板
    before_template = signal[gap_start - template_len: gap_start]
    after_template = signal[gap_end: gap_end + template_len]

    # Step 2: 搜索匹配片段（以前模板为例）
    valid_region = signal[:gap_start - template_len]
    correlation = correlate(valid_region, before_template, mode='valid')
    best_match = np.argmax(correlation)

    # Step 3: 填充缺失段
    patch = signal[best_match + template_len: best_match + template_len + gap_len]
    filled_signal = signal.copy()
    filled_signal[gap_start:gap_end] = patch

    return filled_signal

def fill_gap_linear(img, left, right):
    """left, right 为缺失带左右列索引"""
    gap_w = right - left
    alpha = np.linspace(0, 1, gap_w)  # [1,gap,1]
    left_col  = img[left:left+1, :]
    right_col = img[right:right+1, :]
    for i in range(len(alpha)):
        fill = left_col * (1-alpha[i]) + right_col * alpha[i]
        img[left+i:left+i+1,:] = fill
    return img


result = fill_gap(pred, 287, 304,3)
result1 = result*0.1 + pred*0.7
# 可视化
plt.imshow(result.T, cmap="seismic")
plt.show()