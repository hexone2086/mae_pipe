import cv2
import numpy as np
import random
from loguru import logger

roi_last = (0, 0, 0, 0)
selected_indices_last = None

saved_patch_means = np.zeros((1, 0, 1, 3), dtype=np.float16)
saved_patch_vars = np.zeros((1, 0, 1, 3), dtype=np.float16)


# 读取图像并调整大小为 224x224
def load_and_resize_image(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (224, 224))
    return resized_image

# 将图像分割成 14x14 个 16x16 的 patch
def split_into_patches(image):
    patches = []
    for i in range(0, 224, 16):
        for j in range(0, 224, 16):
            patch = image[i:i+16, j:j+16]
            patches.append(patch)
    return patches

# 显示图像并让用户手动划定矩形
def select_roi(image):
    roi = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    return roi

# 根据 ROI 获取选区内和选区外的 patch 索引
def get_patch_indices_in_roi(roi):
    x1, y1, w, h = roi
    x2 = x1 + w
    y2 = y1 + h
    patch_indices_in_roi = []
    patch_indices_outside_roi = []
    

    for i in range(0, 224, 16):
        for j in range(0, 224, 16):
            if x1 <= j < x2 and y1 <= i < y2:
                patch_indices_in_roi.append((i // 16) * 14 + (j // 16))
            else:
                patch_indices_outside_roi.append((i // 16) * 14 + (j // 16))
    
    return patch_indices_in_roi, patch_indices_outside_roi

# 随机选择 patch 并生成新图像
def generate_rem_image(patches, selected_indices):
    """
    将保留的 patch 按照从上到下、从左到右的顺序拼合成一个长方形图块。

    :param patches: 所有 patch 的列表（14x14 个 16x16 的 patch）
    :param selected_indices: 保留的 patch 的索引列表
    :return: 拼合后的长方形图块 (h=16, w=16*remained_patch_num)
    """
    # 按照从上到下、从左到右的顺序对 selected_indices 进行排序
    selected_indices_sorted = sorted(selected_indices, key=lambda x: (x // 14, x % 14))
    
    # 计算新图像的宽度
    new_width = 16 * len(selected_indices_sorted)
    
    # 初始化新图像，高度为 16，宽度为 new_width
    new_image = np.zeros((16, new_width, 3), dtype=np.uint8)
    
    # 遍历排序后的 patch 索引
    for idx, patch_idx in enumerate(selected_indices_sorted):
        # 计算新图像中的列范围
        start_col = idx * 16
        end_col = start_col + 16
        
        # 将 patch 放入新图像的对应位置
        new_image[:, start_col:end_col] = patches[patch_idx]
    
    return new_image

def generate(roi, image, mode='auto-fixed'):

    global roi_last
    global selected_indices_last
    
    # 加载并调整图像大小
    image = cv2.resize(image, (224, 224))

    # 分割成 patch
    patches = split_into_patches(image)
    
    # 获取选区内和选区外的 patch 索引
    patch_indices_in_roi, patch_indices_outside_roi = get_patch_indices_in_roi(roi)
    
    # 随机选择 49 个选区内 patch，不足则从选区外补充
    if len(patch_indices_in_roi) < 49:
        selected_indices = patch_indices_in_roi + random.sample(patch_indices_outside_roi, 49 - len(patch_indices_in_roi))
    else:
        selected_indices = random.sample(patch_indices_in_roi, 49)
    
    # 如果选区发生变化，则重新生成掩码
    if roi != roi_last or mode == 'auto-random':
        mask_pos = np.ones(196, dtype=bool)
        mask_pos[selected_indices] = False

        roi_last = roi
        selected_indices_last = selected_indices

        logger.info(f"set roi {roi[0]}, {roi[1]}, {roi[2]}, {roi[3]}")
    else:
        selected_indices = selected_indices_last
        # 生成掩码位置
        mask_pos = np.ones(196, dtype=bool)
        mask_pos[selected_indices] = False

    rem_img = generate_rem_image(patches, selected_indices)

    return rem_img, mask_pos

def apply_mask_to_image(image, mask_pos):
    """
    根据 mask_pos 将输入的 224x224 图片中被掩蔽的 patch 部分填充为黑色。

    :param image: 输入的 224x224 图片
    :param mask_pos: 长度为 196 的布尔数组，表示哪些 patch 被掩蔽
    :return: 处理后的图片
    """
    # 确保图片大小为 224x224
    image = cv2.resize(image, (224, 224))
    
    # 创建一个副本，避免修改原始图片
    masked_image = image.copy()
    
    # 遍历所有 patch
    for i in range(0, 224, 16):
        for j in range(0, 224, 16):
            # 计算当前 patch 的索引
            patch_idx = (i // 16) * 14 + (j // 16)
            
            # 如果当前 patch 被掩蔽，则将其填充为黑色
            if mask_pos[patch_idx]:
                masked_image[i:i+16, j:j+16] = 0  # 填充为黑色
    
    return masked_image


def restore_image_from_patches(remained_patches_image, mask_pos, image_size=(224, 224)):
    """
    将保留的 patch 拼合成的长方形图块还原到 224x224 的图片中，被丢弃的 patch 位置留空。

    :param remained_patches_image: 保留的 patch 拼合成的长方形图块 (w=16*remained_patch_num, h=16)
    :param mask_pos: 长度为 196 的布尔数组，False 表示保留，True 表示丢弃
    :param image_size: 目标图片的大小 (默认 224x224)
    :return: 还原后的 224x224 图片
    """
    # 创建空白的目标图片
    restored_image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    
    # 获取保留的 patch 数量
    remained_patch_num = np.sum(~mask_pos)  # False 的数量表示保留的 patch
    
    # 检查输入的长方形图块宽度是否匹配
    expected_width = 16 * remained_patch_num
    if remained_patches_image.shape[1] != expected_width:
        raise ValueError(f"Input image width {remained_patches_image.shape[1]} does not match expected width {expected_width}")
    
    # 遍历 mask_pos，还原 patch 到目标图片
    patch_idx = 0  # 用于遍历 remained_patches_image 中的 patch
    for i in range(0, image_size[0], 16):
        for j in range(0, image_size[1], 16):
            # 计算当前 patch 的索引
            current_patch_idx = (i // 16) * 14 + (j // 16)
            
            # 如果当前 patch 被保留
            if not mask_pos[current_patch_idx]:
                # 从 remained_patches_image 中提取 patch
                patch = remained_patches_image[:, patch_idx * 16:(patch_idx + 1) * 16, :]
                # 将 patch 还原到目标图片的对应位置
                restored_image[i:i+16, j:j+16] = patch
                # 更新 patch_idx
                patch_idx += 1
    
    return restored_image

def calculate_patch_mean_variance_old(image, patch_size=(16, 16)):
    global saved_patch_means
    global saved_patch_vars

    """
    计算图像中每个 patch 的均值和方差，并调整输出形状为 (1, patch_num, 1, 3)。
    
    参数：
    - image: 输入的图像，类型为 NumPy 数组。
    - patch_size: 每个 patch 的大小，默认为 (16, 16)。
    
    返回：
    - patch_means: 每个 patch 的均值，形状为 (1, patch_num, 1, 3)。
    - patch_vars: 每个 patch 的方差，形状为 (1, patch_num, 1, 3)。
    """
    
    # 获取图像的高度、宽度和通道数
    h, w, c = image.shape

    # 将图像分割成 patch
    ph, pw = patch_size
    patches = image.reshape(h // ph, ph, w // pw, pw, c)
    patches = patches.transpose(0, 2, 1, 3, 4).reshape(-1, ph * pw, c)

    patches_reshaped = patches.reshape(1, -1, ph*pw, c)

    # 计算每个 patch 的均值和方差
    patches_reshaped = patches_reshaped / 255.0
    patch_means = np.mean(patches_reshaped, axis=-2, keepdims=True)
    patch_vars = np.sqrt(np.var(patches_reshaped, axis=-2, ddof=1, keepdims=True))

    patch_means = patch_means.astype(np.float16)
    patch_vars = patch_vars.astype(np.float16)
    
    # logger.info(f"patch_means shape: {patch_means.shape}")
    # logger.info(f"patch_vars shape: {patch_vars.shape}")

    # patch_means = patch_means.reshape(1, 196, 1, 3)
    # patch_vars = patch_vars.reshape(1, 196, 1, 3)

    saved_patch_means = np.concatenate((saved_patch_means, patch_means), axis=1)
    saved_patch_vars = np.concatenate((saved_patch_vars, patch_vars), axis=1)

    # logger.info(f"{saved_patch_means}")
    # logger.info(f"{saved_patch_vars}")

    return patch_means, patch_vars

def calculate_patch_mean_variance(image, patch_size=(16, 16)):
    global saved_patch_means
    global saved_patch_vars

    """
    计算图像中每个 patch 的均值和方差，并调整输出形状为 (1, patch_num, 1, 3)。
    
    参数：
    - image: 输入的图像，类型为 NumPy 数组。
    - patch_size: 每个 patch 的大小，默认为 (16, 16)。
    
    返回：
    - patch_means: 每个 patch 的均值，形状为 (1, patch_num, 1, 3)。
    - patch_vars: 每个 patch 的方差，形状为 (1, patch_num, 1, 3)。
    """
    
    # 获取图像的高度、宽度和通道数
    h, w, c = image.shape
    
    # 定义小 patch 和大 patch 的大小
    ph, pw = patch_size
    large_ph, large_pw = ph * 2, pw * 2
    
    # 检查输入图像尺寸是否符合要求
    if h % large_ph != 0 or w % large_pw != 0:
        # 对图像进行填充，使其高度和宽度是 large_ph 和 large_pw 的整数倍
        new_h = ((h + large_ph - 1) // large_ph) * large_ph
        new_w = ((w + large_pw - 1) // large_pw) * large_pw
        padded_image = np.zeros((new_h, new_w, c), dtype=image.dtype)
        padded_image[:h, :w, :] = image
        image = padded_image
        h, w = new_h, new_w
    
    # 将图像分割成大 patch (2 * patch_size x 2 * patch_size)
    large_patches = image.reshape(h // large_ph, large_ph, w // large_pw, large_pw, c)
    large_patches = large_patches.transpose(0, 2, 1, 3, 4).reshape(-1, large_ph * large_pw, c)
    
    # 计算每个大 patch 的均值和方差
    large_patches_reshaped = large_patches.reshape(1, -1, large_ph * large_pw, c)
    large_patches_reshaped = large_patches_reshaped / 255.0  # 归一化到 [0, 1]
    large_patch_means = np.mean(large_patches_reshaped, axis=-2, keepdims=True)
    large_patch_vars = np.sqrt(np.var(large_patches_reshaped, axis=-2, ddof=1, keepdims=True))
    
    # 显式地将每个大 patch 的统计量分配给其对应的小 patch
    num_large_patches_h = h // large_ph
    num_large_patches_w = w // large_pw
    num_small_patches_h = h // ph
    num_small_patches_w = w // pw
    
    # 初始化小 patch 的均值和方差数组
    patch_means = np.zeros((1, num_small_patches_h * num_small_patches_w, 1, c), dtype=np.float16)
    patch_vars = np.zeros((1, num_small_patches_h * num_small_patches_w, 1, c), dtype=np.float16)
    
    for i in range(num_large_patches_h):
        for j in range(num_large_patches_w):
            # 当前大 patch 的索引
            large_patch_idx = i * num_large_patches_w + j
            
            # 当前大 patch 对应的四个小 patch 的索引
            small_patch_indices = [
                (i * 2) * num_small_patches_w + (j * 2),       # 左上角
                (i * 2) * num_small_patches_w + (j * 2 + 1),   # 右上角
                (i * 2 + 1) * num_small_patches_w + (j * 2),   # 左下角
                (i * 2 + 1) * num_small_patches_w + (j * 2 + 1) # 右下角
            ]
            
            # 将当前大 patch 的均值和方差分配给四个小 patch
            for idx in small_patch_indices:
                patch_means[0, idx, 0, :] = large_patch_means[0, large_patch_idx, 0, :]
                patch_vars[0, idx, 0, :] = large_patch_vars[0, large_patch_idx, 0, :]
    
    # 转换为 float16 类型
    patch_means = patch_means.astype(np.float16)
    patch_vars = patch_vars.astype(np.float16)
    
    # 更新全局变量
    saved_patch_means = np.concatenate((saved_patch_means, patch_means), axis=1)
    saved_patch_vars = np.concatenate((saved_patch_vars, patch_vars), axis=1)
    
    # 打印日志信息
    # logger.info(f"patch_means shape: {patch_means.shape}")
    # logger.info(f"patch_vars shape: {patch_vars.shape}")
    # logger.info(f"{saved_patch_means}")
    # logger.info(f"{saved_patch_vars}")
    
    return patch_means, patch_vars

def visualize_patch_statistics(patch_means, patch_vars, patch_size, output_shape):
    """
    将 patch 的均值和方差可视化为灰度图像。
    
    参数：
    - patch_means: 每个 patch 的均值，形状为 (1, patch_num, 1, 3)。
    - patch_vars: 每个 patch 的方差，形状为 (1, patch_num, 1, 3)。
    - patch_size: 每个 patch 的大小，例如 (16, 16)。
    - output_shape: 输出图像的形状，例如 (224, 224)。
    
    返回：
    - mean_image: 均值可视化的灰度图像。
    - var_image: 方差可视化的灰度图像。
    """
    # 获取 patch 的数量和通道数
    _, patch_num, _, _ = patch_means.shape
    ph, pw = patch_size
    h, w = output_shape
    
    # 将均值和方差展平为单通道数据
    mean_values = np.mean(patch_means, axis=-1).squeeze()  # (patch_num,)
    var_values = np.mean(patch_vars, axis=-1).squeeze()    # (patch_num,)
    
    # 将均值和方差映射到 [0, 255] 范围
    mean_values = (mean_values * 255).astype(np.uint8)
    var_values = (var_values * 255).astype(np.uint8)
    
    # 将均值和方差重新排列为图像形状
    mean_image = mean_values.reshape(h // ph, w // pw).repeat(ph, axis=0).repeat(pw, axis=1)
    var_image = var_values.reshape(h // ph, w // pw).repeat(ph, axis=0).repeat(pw, axis=1)
    
    return mean_image, var_image

def reduce_to_large_patch(patch_stats, large_patch_shape=(1, 49, 1, 3)):
    """
    将均值或方差数组从 (1, 196, 1, 3) 还原为大 patch 的形状 (1, 49, 1, 3)。
    
    参数：
    - patch_stats: 均值或方差数组，形状为 (1, 196, 1, 3)。
    - large_patch_shape: 大 patch 的目标形状，默认为 (1, 49, 1, 3)。
    
    返回：
    - large_patch_stats: 大 patch 的均值或方差数组，形状为 (1, 49, 1, 3)。
    """
    # 检查输入形状是否正确
    if patch_stats.shape != (1, 196, 1, 3):
        raise ValueError(f"Expected input shape (1, 196, 1, 3), but got {patch_stats.shape}")
    
    # 初始化大 patch 数组
    large_patch_stats = np.zeros(large_patch_shape, dtype=patch_stats.dtype)
    
    # 遍历每个大 patch，并将其对应的四个小 patch 的统计量合并
    for i in range(7):  # 大 patch 的行数
        for j in range(7):  # 大 patch 的列数
            # 当前大 patch 的索引
            large_patch_idx = i * 7 + j
            
            # 对应的四个小 patch 的索引
            small_patch_indices = [
                (i * 2) * 14 + (j * 2),       # 左上角
                (i * 2) * 14 + (j * 2 + 1),   # 右上角
                (i * 2 + 1) * 14 + (j * 2),   # 左下角
                (i * 2 + 1) * 14 + (j * 2 + 1) # 右下角
            ]
            
            # 提取对应的小 patch 统计量并计算平均值
            small_patch_values = patch_stats[0, small_patch_indices, 0, :]
            large_patch_stats[0, large_patch_idx, 0, :] = np.mean(small_patch_values, axis=0)
    
    return large_patch_stats


def expand_to_small_patch(large_patch_stats, small_patch_shape=(1, 196, 1, 3)):
    """
    将大 patch 的均值或方差数组从 (1, 49, 1, 3) 扩展回小 patch 的形状 (1, 196, 1, 3)。
    
    参数：
    - large_patch_stats: 大 patch 的均值或方差数组，形状为 (1, 49, 1, 3)。
    - small_patch_shape: 小 patch 的目标形状，默认为 (1, 196, 1, 3)。
    
    返回：
    - small_patch_stats: 小 patch 的均值或方差数组，形状为 (1, 196, 1, 3)。
    """
    # 检查输入形状是否正确
    if large_patch_stats.shape != (1, 49, 1, 3):
        raise ValueError(f"Expected input shape (1, 49, 1, 3), but got {large_patch_stats.shape}")
    
    # 初始化小 patch 数组
    small_patch_stats = np.zeros(small_patch_shape, dtype=large_patch_stats.dtype)
    
    # 遍历每个大 patch，并将其统计量广播到对应的小 patch
    for i in range(7):  # 大 patch 的行数
        for j in range(7):  # 大 patch 的列数
            # 当前大 patch 的索引
            large_patch_idx = i * 7 + j
            
            # 当前大 patch 的统计量
            large_patch_value = large_patch_stats[0, large_patch_idx, 0, :]
            
            # 对应的四个小 patch 的索引
            small_patch_indices = [
                (i * 2) * 14 + (j * 2),       # 左上角
                (i * 2) * 14 + (j * 2 + 1),   # 右上角
                (i * 2 + 1) * 14 + (j * 2),   # 左下角
                (i * 2 + 1) * 14 + (j * 2 + 1) # 右下角
            ]
            
            # 将当前大 patch 的统计量分配给四个小 patch
            for idx in small_patch_indices:
                small_patch_stats[0, idx, 0, :] = large_patch_value
    
    return small_patch_stats

# 运行主函数
if __name__ == "__main__":
    image_path = "frame_2.jpg"  # 替换为你的图像路径
    # 加载并调整图像大小
    image = load_and_resize_image(image_path)
    
    # 分割成 patch
    patches = split_into_patches(image)
    
    # 显示图像并让用户选择 ROI
    roi = select_roi(image)
    
    # 获取选区内和选区外的 patch 索引
    patch_indices_in_roi, patch_indices_outside_roi = get_patch_indices_in_roi(roi)
    
    # 随机选择 49 个选区内 patch，不足则从选区外补充
    if len(patch_indices_in_roi) < 49:
        selected_indices = patch_indices_in_roi + random.sample(patch_indices_outside_roi, 49 - len(patch_indices_in_roi))
    else:
        selected_indices = random.sample(patch_indices_in_roi, 49)
    
    # 生成掩码位置
    mask_pos = np.ones(196, dtype=bool)
    mask_pos[selected_indices] = False

    rec_1 = generate_rem_image(patches, selected_indices)
    cv2.imwrite("rec_1.jpg", rec_1)