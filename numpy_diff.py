import numpy as np
import matplotlib.pyplot as plt

def visualize_array_difference(arr1, arr2, diff_type='absolute'):
    """
    计算并可视化两个相同形状的二维 NumPy 数组的差值。

    Args:
        arr1 (np.ndarray): 第一个二维 NumPy 数组。
        arr2 (np.ndarray): 第二个二维 NumPy 数组。
        diff_type (str): 差值计算类型，'absolute' 表示绝对差 (|arr1 - arr2|)，
                         'signed' 表示带符号的差 (arr1 - arr2)。默认为 'absolute'。

    Raises:
        ValueError: 如果两个数组的形状不同。
        ValueError: 如果 diff_type 不是 'absolute' 或 'signed'。
    """
    # 检查数组形状是否相同
    if arr1.shape != arr2.shape:
        raise ValueError("两个数组的形状必须相同！ arr1.shape={}, arr2.shape={}".format(arr1.shape, arr2.shape))

    # 计算差值
    if diff_type == 'absolute':
        difference = np.abs(arr1 - arr2)
        title = 'Absolute Difference between Two Arrays'
        cmap = 'viridis' # 适合表示大小/幅度变化的颜色映射
        colorbar_label = 'Absolute Difference Value'
    elif diff_type == 'signed':
        difference = arr1 - arr2
        title = 'Signed Difference between Two Arrays (arr1 - arr2)'
        # 适合表示正负变化的颜色映射，中心为0（白色/灰色），两端是不同的颜色
        cmap = 'coolwarm'
        colorbar_label = 'Signed Difference Value (arr1 - arr2)'
    else:
        raise ValueError("diff_type 必须是 'absolute' 或 'signed'")

    # 创建可视化图
    plt.figure(figsize=(8, 6)) # 设置图的大小

    # 使用 imshow 绘制热力图
    # interpolation='nearest' 可以避免像素之间的平滑，更适合展示离散的数组值
    # aspect='auto' 可以让图像根据数据形状自适应长宽比
    imshow_plot = plt.imshow(difference, cmap=cmap, interpolation='nearest', aspect='auto')

    # 添加颜色条，显示颜色与数值的对应关系
    plt.colorbar(imshow_plot, label=colorbar_label)

    # 添加标题和轴标签
    plt.title(title)
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    # 保存图片  
    plt.savefig(f"{diff_type}_maskd_S17.png", dpi=300, bbox_inches='tight')
    # 可选：显示每个位置的具体数值（适用于小数组）
    # for i in range(difference.shape[0]):
    #     for j in range(difference.shape[1]):
    #         plt.text(j, i, f'{difference[i, j]:.2f}', ha='center', va='center', color='white' if difference[i, j] > difference.max()/2 else 'black', fontsize=8)


    # 确保显示所有刻度（对于较大的数组可能需要调整）
    # plt.xticks(np.arange(difference.shape[1]))
    # plt.yticks(np.arange(difference.shape[0]))


    # 显示图
    plt.tight_layout() # 调整布局，防止标签重叠
    plt.show()

# --- 示例使用 ---

# 创建两个形状相同的二维 NumPy 数组
array1 = np.load("runs/detect/exp41/2007_003020/stage17_C3_features.npy")
array2 = np.load("runs/detect/exp43/2007_003020/stage17_C3_features.npy")
array1 = np.mean(array1, axis=0)
array2 = np.mean(array2, axis=0)

# 可视化绝对差值
print("可视化绝对差值:")
visualize_array_difference(array1, array2, diff_type='absolute')

# 可视化带符号的差值 (array1 - array2)
print("\n可视化带符号的差值 (array1 - array2):")
visualize_array_difference(array1, array2, diff_type='signed')

# # 示例：两个形状不同的数组会引发错误
# array3 = np.array([[1, 2], [3, 4]])
# try:
#     visualize_array_difference(array1, array3)
# except ValueError as e:
#     print(f"\n捕获到预期的错误: {e}")
