import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 假设原始图像数据（需要替换为真实的640x480图像）
original_image = cv2.imread("datasets/VOC_person/val/images/2007_002119.jpg")
original_image = cv2.resize(original_image, (640, 480))
# 生成模拟特征图数据 [1, 6, 128, 80, 60]
stage = "23"
# feature_tensor = np.load(f"runs/detect/exp59/2007_002142/stage{stage}_C3_features.npy")
feature_tensor = np.load(f"mask_s{stage}.npy")
# feature_tensor = np.load(f"absolute_maskd_S{stage}.npy")
print(feature_tensor.shape)
feature_tensor = torch.tensor(feature_tensor)  # [1, 6, 128, 80, 60]
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
# 转换为float32类型
# original_image = original_image.astype(np.float16) / 255.0  # [480, 640, 3]
# 转换为torch tensor
original_image = torch.tensor(original_image).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 480, 640]
# 预处理：合并128通道为单通道（取平均）
# merged_features = feature_tensor.squeeze(0).mean(dim=1)  # [6, 80, 60]
# merged_features = feature_tensor.mean(dim=0)  # [6, 80, 60]
merged_features = feature_tensor.squeeze(0) # for mask
# merged_features = feature_tensor

# 创建2x3子图画布
fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='white')


# 显示原始图像
original_image = original_image.squeeze().permute(1, 2, 0).numpy()  # [480, 640, 3]
# 对每个特征图进行可视化
prob_list_s17 = ["5.275e-3", "9.945e-1", "5.547e-6", "1.841e-4", "3.791e-6", "3.685e-6"]
prob_list_s20 = ["6.648e-5", "9.998e-1", "3.199e-5", "2.564e-5", "2.461e-5", "2.433e-5"]
prob_list_s23 = ["7.129e-5", "9.997e-1", "1.601e-5", "1.281e-4", "8.187e-6", "2.929e-5"]
for idx, ax in enumerate(axes.flat):
    if idx >= 6:
        break
    
    # 上采样到原始图像尺寸
    single_feature = merged_features[idx].unsqueeze(0).unsqueeze(0)  # [1, 1, 80, 60]
    upsampled = F.interpolate(single_feature, size=(480, 640), mode='bilinear', align_corners=False)
    feature_map = upsampled.squeeze().numpy()  # [480, 640]
    
    # 归一化到[0,1]范围
    normalized = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
    
    
    ax.imshow(original_image)
    
    # 叠加特征图热力图
    heatmap = ax.imshow(normalized, cmap='jet', alpha=0.4)
    
    # 添加颜色条
    # cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_ticks([])
    
    # 设置子图标题
    
    # ax.set_title(prob_list[idx], fontsize=8, y=-0.9)
    ax.axis('off')

# 调整布局并显示
plt.subplots_adjust(wspace=0.05, hspace=0.2)
# plt.show()
plt.savefig(f"feature_maps{stage}.png", dpi=300, bbox_inches='tight')

# fig, axes = plt.subplots(figsize=(18, 12), facecolor='white')
# # 显示原始图像
# original_image = original_image.squeeze().permute(1, 2, 0).numpy()  # [480, 640, 3]
# # 对每个特征图进行可视化
# # single_feature = merged_features[0].unsqueeze(0).unsqueeze(0)  # [1, 1, 80, 60]
# single_feature = merged_features.unsqueeze(0).unsqueeze(0)  # [1, 1, 80, 60]
# # 上采样到原始图像尺寸
# upsampled = F.interpolate(single_feature, size=(480, 640), mode='bilinear', align_corners=False, antialias=True)
# feature_map = upsampled.squeeze().numpy()  # [480, 640]
# # 设置阈值
# threshold = 0.6 * feature_map.max()
# # 将小于阈值的值设置为最小值
# feature_map[feature_map < threshold] = feature_map.min()
# # 归一化到[0,1]范围
# normalized = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
# # 叠加特征图热力图
# heatmap = axes.imshow(original_image)
# # 叠加特征图热力图
# heatmap = axes.imshow(normalized, cmap='jet', alpha=0.6, interpolation='antialiased')
# # 添加颜色条
# # cbar = fig.colorbar(heatmap, ax=axes, fraction=0.046, pad=0.04)
# # cbar.set_ticks([])
# # 设置子图标题
# # axes.set_title(f'Feature Map 1', fontsize=8, y=0.9)
# axes.axis('off')
# # 调整布局并显示
# plt.subplots_adjust(wspace=0.05, hspace=0.2)
# # plt.show()
# plt.savefig(f"feature_maps_sum_{stage}.png", dpi=300, bbox_inches='tight')
