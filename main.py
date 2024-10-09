import numpy as np  # 导入numpy库，用于进行科学计算
import cv2  # 导入OpenCV库，用于图像处理
from sklearn.neighbors import KernelDensity  # 从sklearn库中导入KernelDensity，用于执行核密度估计
import os  # 导入os库，用于处理文件和目录
from tqdm import tqdm  # 从tqdm库中导入tqdm，用于显示进度条
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot，用于绘图
from skimage.transform import resize  # 从skimage库中导入resize，用于调整图像大小
from skimage import img_as_float  # 从skimage库中导入img_as_float，用于将图像转换为浮点数格式
#made by 东华大学 人工智能2201  代码苦手 郭睿知 刘涵帅 曹卢毅 马田泳

# 下面是定义的几个函数，每个函数都有其特定的功能

# 从图像文件夹中读取所有图像并提取颜色特征
# folder_path：图像文件夹路径
# downscale_factor：缩放因子，用于缩小图像尺寸以减少计算量
# sample_fraction：采样比例，决定了从图像中采样多少比例的像素用于特征提取
def load_images_and_extract_features_optimized(folder_path, downscale_factor=2, sample_fraction=0.1):
    # 获取文件夹中所有.jpg结尾的文件名
    image_filenames = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    all_features = []  # 用于存储所有图像的特征
    # 遍历所有图像文件名
    for image_filename in tqdm(image_filenames, desc="加载图像并提取特征"):
        file_path = os.path.join(folder_path, image_filename)  # 获取完整的文件路径
        image = cv2.imread(file_path)  # 使用cv2读取图像
        if image is None:  # 如果图像读取失败，则跳过
            print(f"警告：'{file_path}' 无法读取，跳过。")
            continue
        # 将图像缩放，并转换为浮点数格式
        image_resized = resize(image, (image.shape[0] // downscale_factor, image.shape[1] // downscale_factor), anti_aliasing=True).astype(np.float32)
        # 将BGR图像转换为RGB图像
        image_rgb = cv2.cvtColor((image_resized * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        # 将图像数据重塑并归一化
        features = image_rgb.reshape((-1, 3)) / 255.0
        # 从图像中随机采样一定比例的像素
        sampled_features = features[np.random.choice(features.shape[0], int(features.shape[0] * sample_fraction), replace=False), :]
        all_features.append(sampled_features)  # 将采样得到的特征添加到特征列表中
    return np.vstack(all_features)  # 将所有图像的特征堆叠成一个矩阵并返回

# 创建并训练KDE模型
# features：用于训练KDE模型的特征数据
# bandwidth：带宽参数，影响KDE模型的平滑程度
def train_kde(features, bandwidth=1.2):
    print("开始训练KDE模型...")
    kde = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth)  # 创建KDE模型
    kde.fit(features)  # 使用特征数据训练KDE模型
    print("KDE模型训练完成。")
    return kde  # 返回训练好的KDE模型

# 绘制3D颜色映射图，用于测试图像
# test_images_folder：测试图像文件夹路径
# kde：训练好的KDE模型
# downscale_factor：缩放因子
def plot_3d_kde_color_map_for_test_images(test_images_folder, kde, downscale_factor=1):
    # 遍历测试图像文件夹中的所有.jpg图像文件
    for filename in os.listdir(test_images_folder):
        if filename.endswith('.jpg'):
            test_image_path = os.path.join(test_images_folder, filename)  # 获取完整的文件路径
            image = cv2.imread(test_image_path)  # 读取图像
            if image is None:  # 如果无法加载图像，则跳过
                print(f"无法加载图像: {test_image_path}")
                continue
            # 将BGR图像转换为RGB颜色空间，并进行缩放
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = resize(image_rgb, (image_rgb.shape[0] // downscale_factor, image_rgb.shape[1] // downscale_factor), anti_aliasing=True).astype(np.float32) / 255.0
            # 将图像数据转换为一维数组，以便用于KDE评分
            pixel_features = image_resized.reshape(-1, 3)
            print(f"计算像素点的KDE评分: {filename}...")
            log_densities = kde.score_samples(pixel_features)  # 使用KDE模型评估图像中每个像素的密度
            densities = np.exp(log_densities)  # 将对数密度转换为密度值
            # 将密度数据转换回图像的形状，以便绘制
            densities_reshaped = densities.reshape(image_resized.shape[0], image_resized.shape[1])
            # 创建3D图形并绘制
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            x = np.linspace(0, image_resized.shape[1], image_resized.shape[1], endpoint=False)  # 创建x坐标网格
            y = np.linspace(0, image_resized.shape[0], image_resized.shape[0], endpoint=False)  # 创建y坐标网格
            X, Y = np.meshgrid(x, y)  # 生成网格数据
            ax.plot_surface(X, Y, densities_reshaped, cmap='viridis')  # 绘制表面图
            ax.set_xlabel('X坐标')  # 设置x轴标签
            ax.set_ylabel('Y坐标')  # 设置y轴标签
            ax.set_zlabel('密度')  # 设置z轴标签
            ax.set_box_aspect([1, 1, 0.15])  # 控制xyz三个方向的缩放比例
            ax.set_title(f'测试图像的3D KDE颜色映射图: {filename}')  # 设置图形标题
            plt.show()  # 显示图形

# 检测图像中的运动目标
# kde：训练好的KDE模型
# test_image_path：要检测的测试图像路径
def detect_motion_targets(kde, test_image_path):
    print("加载图像...")
    image = cv2.imread(test_image_path)  # 读取图像
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR图像转换为RGB颜色空间
    image_float = img_as_float(image_rgb)  # 将图像转换为浮点数格式
    features = image_float.reshape((-1, 3))  # 将图像数据转换为一维数组
    print("计算KDE得分...")
    log_densities = kde.score_samples(features)  # 使用KDE模型评估图像中每个像素的密度
    densities = np.exp(log_densities)  # 将对数密度转换为密度值
    print("选择阈值并生成运动掩码...")
    threshold = np.percentile(densities, 20)  # 设置密度的阈值
    motion_mask = densities < threshold  # 根据阈值生成运动掩码
    return motion_mask.reshape(image.shape[:2])  # 将掩码转换回图像的形状并返回

# 主函数，程序的入口点
def main():
    # 定义训练集和测试集文件夹的相对路径
    train_folder = './train'  # 训练集文件夹路径
    test_folder = './test'    # 测试集文件夹路径
    # 调用函数，加载训练集图像特征并训练KDE模型
    train_features = load_images_and_extract_features_optimized(train_folder)
    kde = train_kde(train_features)
    # 处理测试集中的图像，绘制3D彩色KDE图
    plot_3d_kde_color_map_for_test_images(test_folder, kde, downscale_factor=1)
    # 对测试集中的每张图像执行运动目标检测
    test_image_filenames = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]
    for image_filename in tqdm(test_image_filenames, desc="检测测试图像"):
        test_image_path = os.path.join(test_folder, image_filename)
        motion_mask = detect_motion_targets(kde, test_image_path)
        motion_mask_image = (motion_mask * 255).astype(np.uint8)  # 将掩码转换为图像
        plt.imshow(motion_mask, cmap='gray')  # 显示运动掩码图像
        plt.title(f'Motion Detection: {image_filename}')  # 设置图像标题
        plt.axis('off')  # 不显示坐标轴
        plt.show()  # 显示图形

# 如果这个脚本是直接运行的，而不是被导入，则执行main函数
if __name__ == '__main__':
    main()
