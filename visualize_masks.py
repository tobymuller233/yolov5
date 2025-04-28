import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 假设原始图像数据（需要替换为真实的640x480图像）
original_image = cv2.imread("datasets/VOC_person/val/images/2007_002142.jpg")
original_image = cv2.resize(original_image, (640, 480))
# 生成模拟特征图数据 [1, 6, 128, 80, 60]
feature_tensor = np.load("mask_s20.npy")
# print(feature_tensor.shape)
feature_tensor = torch.tensor(feature_tensor)  # [1, 6, 128, 80, 60]
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
# 转换为float32类型
# original_image = original_image.astype(np.float16) / 255.0  # [480, 640, 3]
# 转换为torch tensor
original_image = torch.tensor(original_image).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 480, 640]
# 预处理：合并128通道为单通道（取平均）
# merged_features = feature_tensor.squeeze(0).mean(dim=1)  # [6, 80, 60]
# merged_features = feature_tensor.mean(dim=1)  # [6, 80, 60]
merged_features = feature_tensor.squeeze(0) # for mask

# 创建2x3子图画布
fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='white')


# 显示原始图像
original_image = original_image.squeeze().permute(1, 2, 0).numpy()  # [480, 640, 3]
# 对每个特征图进行可视化
prob_list_s17 = ["1.259e-3", "9.983e-1", "1.516e-6", "2.719e-4", "1.313e-6", "1.187e-6"]
prob_list_s20 = ["8.475e-8", "9.861e-1", "3.290e-3", "6.393e-4", "4.794e-4", "9.864e-4"]
prob_list_s23 = ["1.005e-2", "9.726e-1", "2.74e-3", "9.37e-3", "2.71e-3", "2.51e-3"]
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
plt.savefig("feature_maps.png", dpi=300, bbox_inches='tight')

# fig, axes = plt.subplots(figsize=(18, 12), facecolor='white')
# # 显示原始图像
# original_image = original_image.squeeze().permute(1, 2, 0).numpy()  # [480, 640, 3]
# # 对每个特征图进行可视化
# single_feature = merged_features[0].unsqueeze(0).unsqueeze(0)  # [1, 1, 80, 60]
# # 上采样到原始图像尺寸
# upsampled = F.interpolate(single_feature, size=(480, 640), mode='bilinear', align_corners=False)
# feature_map = upsampled.squeeze().numpy()  # [480, 640]
# # 归一化到[0,1]范围
# normalized = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
# # 叠加特征图热力图
# heatmap = axes.imshow(original_image)
# # 叠加特征图热力图
# heatmap = axes.imshow(normalized, cmap='jet', alpha=0.4)
# # 添加颜色条
# cbar = fig.colorbar(heatmap, ax=axes, fraction=0.046, pad=0.04)
# cbar.set_ticks([])
# # 设置子图标题
# axes.set_title(f'Feature Map 1', fontsize=8, y=0.9)
# axes.axis('off')
# # 调整布局并显示
# plt.subplots_adjust(wspace=0.05, hspace=0.2)
# # plt.show()
# plt.savefig("feature_maps_sum.png", dpi=300, bbox_inches='tight')


