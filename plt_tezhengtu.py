import numpy as np
import matplotlib
matplotlib.use('TKAgg')  # 切换到非交互式后端
import matplotlib.pyplot as plt
import torch
from skimage import io, transform
from models.adapter import dinov2_mla,dinov2_pup,dinov2_linear
from models.Dpt import dinov2_dpt

# 可视化功能
def show_embedding_slice(embedding, slice_index, ax):
    """
    显示 image_embeddings 的某个切片/特征图
    embedding: 图像嵌入张量 (B, C, H, W)
    slice_index: 要显示的切片索引
    """
    ax.imshow(embedding[0, slice_index], cmap='viridis')
    ax.axis('off')
    ax.set_title(f"Embedding Slice {slice_index}")
    plt.show()

def show_pca_rgb_embedding(embedding):
    from sklearn.decomposition import PCA
    # Flatten and reshape
    embedding_2d = np.transpose(embedding[0],axes=(1, 2, 0)).reshape(-1, embedding.shape[1])
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(embedding_2d)
    # Normalize to [0, 1]
    pca_norm = (pca_result - pca_result.min(axis=0)) / (pca_result.max(axis=0) - pca_result.min(axis=0))
    rgb = pca_norm.reshape(embedding.shape[2], embedding.shape[3], 3)
    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.axis('off')
    plt.title("PCA-based RGB Embedding")
    plt.show()


def get_feature_maps(model, input_data, layer_name):
    feature_maps = []

    def hook(module, input, output):
        feature_maps.append(output.detach().cpu().numpy())

    # 注册钩子函数
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook)
            break

    # 前向传播
    model(input_data,size=(1022, 2422))

    # 移除钩子
    handle.remove()

    return feature_maps[0]


# %% Main script
if __name__ == "__main__":
    # 设置路径
    input_image_path = "dino/2D_line.jpg"  # 输入图像路径

    # 设备设置
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 初始化 SAM 模型
    sam_model = dinov2_pup(3, pretrain='True', vit_type='large',frozen=False, finetune_method='unfrozen')
    # model_Path = ''  # 训练好的模型位置
    # sam_model.load_state_dict(torch.load(model_Path, map_location=device), strict=True)

    # 加载并预处理输入图像
    img_np = io.imread(input_image_path)
    if len(img_np.shape) == 2:  # 如果是灰度图，转换为 3 通道
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    elif img_np.shape[-1] == 4:  # 如果图像有 4 个通道（例如 RGBA）
        img_3c = img_np[..., :3]  # 移除 alpha 通道
    else:
        img_3c = img_np

    H, W, _ = img_3c.shape

    # 将图像调整为 1024x1024 以适应 SAM 输入
    img_1024 = transform.resize(
        img_3c, (1022, 2422), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.float32)

    # 归一化图像到 [0, 1]
    img_1024 = img_1024 / 255.0

    # 转换为张量并确保 3 通道
    if img_1024.shape[-1] == 4:
        img_1024 = img_1024[..., :3]
    elif len(img_1024.shape) == 2:
        img_1024 = np.stack([img_1024] * 3, axis=-1)

    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )

    # 加载模型权重并设置为评估模式
    sam_model = sam_model.to(device)
    sam_model.eval()

    # 执行推理并可视化 image_embeddings
    with torch.no_grad():
        feature_maps = get_feature_maps(sam_model, img_1024_tensor, 'decoder')

# 显示image-embedding切片
fig, ax = plt.subplots(1, 1, figsize=(6,6))
show_embedding_slice(feature_maps, 0, ax)

show_pca_rgb_embedding(feature_maps)