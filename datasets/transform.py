# @Time    : 2023/8/29 02:01
# @Author  : zhangchenming
import random
from typing import Dict, Tuple

import albumentations as A
import cv2
import numpy as np
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor(object):
    """将numpy数组转换为torch张量，支持BEV数据格式"""

    def __call__(self, sample):
        for k in sample.keys():
            if isinstance(sample[k], np.ndarray):
                if k == "image":
                    # 图像转换为CHW格式，保持RGB格式
                    if sample[k].shape[2] == 3:
                        img = sample[k]
                        # 转换为CHW格式
                        img = img.transpose(2, 0, 1)
                        sample[k] = torch.from_numpy(img.copy()).float() / 255.0
                elif k in ["camera_intrinsics", "ego_to_cam"]:
                    # 相机参数和变换矩阵直接转换为tensor
                    sample[k] = torch.from_numpy(sample[k].copy()).float()
                else:
                    # 其他数据转换为tensor
                    sample[k] = torch.from_numpy(sample[k].copy()).float()
        return sample


class Resize(object):
    """等比例缩放图像到指定尺寸，不足部分用pad补充，同时调整相机参数"""

    def __init__(
        self, size: Tuple[int, int], pad_value: int = 0, scale_factor: float = 1.0
    ):
        """
        Args:
            size: (height, width) 目标尺寸
            pad_value: 填充值，默认0（黑色）
        """
        self.size = size
        self.pad_value = pad_value
        self.scale_factor = scale_factor

    def __call__(self, sample: Dict) -> Dict:
        target_h, target_w = self.size
        img_h, img_w = sample["image"].shape[:2]

        # 计算缩放比例，保持宽高比
        scale_h = target_h / img_h
        scale_w = target_w / img_w
        assert scale_h == scale_w, print("scale_h not equals scale_w...\n")
        scale = scale_h

        # 缩放图像
        if scale != 1.0:
            sample["image"] = cv2.resize(
                sample["image"], (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
            sample["seg_gt_maps"] = np.transpose(sample["seg_gt_maps"], (1, 2, 0))
            tmp_seg_gt_maps = cv2.resize(
                sample["seg_gt_maps"],
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )
            tmp_seg_gt_maps = cv2.resize(tmp_seg_gt_maps, dsize=None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_NEAREST)
            sample["seg_gt_maps"] = (
                tmp_seg_gt_maps
                if sample["seg_gt_maps"].shape[-1] != 1
                else np.expand_dims(tmp_seg_gt_maps, axis=-1)
            )
            sample["seg_gt_maps"] = np.transpose(sample["seg_gt_maps"], (2, 0, 1))

        # 调整相机内参矩阵
        if (
            "camera_intrinsics" in sample
            and sample["with_camera_intrinsics"]
            and sample["camera_intrinsics"] is not None
        ):
            intrinsics = sample["camera_intrinsics"].copy()

            # 缩放调整：fx, fy需要乘以缩放比例
            intrinsics[0, 0] *= scale  # fx
            intrinsics[1, 1] *= scale  # fy

            # 平移调整：cx, cy需要加上padding偏移
            intrinsics[0, 2] = intrinsics[0, 2] * scale  # cx
            intrinsics[1, 2] = intrinsics[1, 2] * scale  # cy

            sample["camera_intrinsics"] = intrinsics

        # 注意：ego_to_cam不需要调整，因为它是相机到ego的变换，与图像尺寸无关

        return sample


class ColorEnhancement(object):
    """颜色增强类，专注于颜色相关的变换，不改变空间结构"""

    def __init__(
        self,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        saturation_limit: float = 0.3,
        hue_shift_limit: int = 20,
        prob: float = 0.8,
    ):
        """
        Args:
            brightness_limit: 亮度调整范围 [-brightness_limit, brightness_limit]
            contrast_limit: 对比度调整范围 [-contrast_limit, contrast_limit]
            saturation_limit: 饱和度调整范围 [-saturation_limit, saturation_limit]
            hue_shift_limit: 色调偏移范围 [-hue_shift_limit, hue_shift_limit]
            prob: 应用增强的概率
        """
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.saturation_limit = saturation_limit
        self.hue_shift_limit = hue_shift_limit
        self.prob = prob

        # 延迟初始化，避免多进程序列化问题
        self.transform = None

    def _ensure_transforms_initialized(self):
        """确保transforms已初始化"""
        if self.transform is None:
            self.transform = A.Compose(
                [
                    # 亮度和对比度调整
                    A.RandomBrightnessContrast(
                        brightness_limit=self.brightness_limit,
                        contrast_limit=self.contrast_limit,
                        p=0.8,
                    ),
                    # HSV色彩空间调整
                    A.HueSaturationValue(
                        hue_shift_limit=self.hue_shift_limit,
                        sat_shift_limit=int(self.saturation_limit * 100),
                        val_shift_limit=int(self.brightness_limit * 100),
                        p=0.7,
                    ),
                    # RGB通道独立调整
                    A.RGBShift(
                        r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5
                    ),
                    # 色彩抖动
                    A.ColorJitter(
                        brightness=self.brightness_limit,
                        contrast=self.contrast_limit,
                        saturation=self.saturation_limit,
                        hue=self.hue_shift_limit / 360.0,
                        p=0.6,
                    ),
                ]
            )

    def __call__(self, sample: Dict) -> Dict:
        """应用颜色增强"""
        if random.random() > self.prob:
            return sample

        self._ensure_transforms_initialized()

        image = sample["image"]

        # 应用颜色增强
        enhanced_image = self.transform(image=image)["image"]
        sample["image"] = enhanced_image

        return sample


class ContrastEnhancement(object):
    """对比度增强类，专注于对比度和亮度的精细调整"""

    def __init__(
        self,
        gamma_limit: Tuple[int, int] = (80, 120),
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        prob: float = 0.7,
    ):
        """
        Args:
            gamma_limit: Gamma校正范围
            clahe_clip_limit: CLAHE对比度限制增强的裁剪限制
            clahe_tile_grid_size: CLAHE网格大小
            prob: 应用增强的概率
        """
        self.gamma_limit = gamma_limit
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.prob = prob

        self.transform = None

    def _ensure_transforms_initialized(self):
        """确保transforms已初始化"""
        if self.transform is None:
            self.transform = A.Compose(
                [
                    # Gamma校正
                    A.RandomGamma(gamma_limit=self.gamma_limit, p=0.6),
                    # 对比度受限的自适应直方图均衡化
                    A.CLAHE(
                        clip_limit=self.clahe_clip_limit,
                        tile_grid_size=self.clahe_tile_grid_size,
                        p=0.4,
                    ),
                    # 直方图均衡化
                    A.Equalize(mode="cv", by_channels=True, p=0.3),
                    # 自动对比度
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1,
                        contrast_limit=0.3,
                        brightness_by_max=True,
                        p=0.5,
                    ),
                ]
            )

    def __call__(self, sample: Dict) -> Dict:
        """应用对比度增强"""
        if random.random() > self.prob:
            return sample

        self._ensure_transforms_initialized()

        image = sample["image"]

        # 应用对比度增强
        enhanced_image = self.transform(image=image)["image"]
        sample["image"] = enhanced_image

        return sample


class LightNoiseEnhancement(object):
    """轻量噪声增强类，添加轻微的噪声效果"""

    def __init__(
        self,
        gauss_noise_var_limit: Tuple[float, float] = (10.0, 50.0),
        multiplicative_noise_multiplier: Tuple[float, float] = (0.9, 1.1),
        iso_noise_color_shift: Tuple[float, float] = (0.01, 0.05),
        prob: float = 0.5,
    ):
        """
        Args:
            gauss_noise_var_limit: 高斯噪声方差范围
            multiplicative_noise_multiplier: 乘性噪声倍数范围
            iso_noise_color_shift: ISO噪声颜色偏移范围
            prob: 应用增强的概率
        """
        self.gauss_noise_var_limit = gauss_noise_var_limit
        self.multiplicative_noise_multiplier = multiplicative_noise_multiplier
        self.iso_noise_color_shift = iso_noise_color_shift
        self.prob = prob

        self.transform = None

    def _ensure_transforms_initialized(self):
        """确保transforms已初始化"""
        if self.transform is None:
            self.transform = A.Compose(
                [
                    # 高斯噪声
                    A.GaussNoise(
                        var_limit=self.gauss_noise_var_limit,
                        mean=0,
                        per_channel=True,
                        p=0.4,
                    ),
                    # 乘性噪声
                    A.MultiplicativeNoise(
                        multiplier=self.multiplicative_noise_multiplier,
                        per_channel=True,
                        p=0.3,
                    ),
                    # ISO噪声（模拟相机传感器噪声）
                    A.ISONoise(
                        color_shift=self.iso_noise_color_shift,
                        intensity=(0.1, 0.5),
                        p=0.3,
                    ),
                ]
            )

    def __call__(self, sample: Dict) -> Dict:
        """应用轻量噪声增强"""
        if random.random() > self.prob:
            return sample

        self._ensure_transforms_initialized()

        image = sample["image"]

        # 应用噪声增强
        enhanced_image = self.transform(image=image)["image"]
        sample["image"] = enhanced_image

        return sample


class LightingEnhancement(object):
    """光照增强类，模拟不同光照条件"""

    def __init__(
        self,
        shadow_prob: float = 0.3,
        sun_flare_prob: float = 0.2,
        fog_prob: float = 0.2,
        prob: float = 0.6,
    ):
        """
        Args:
            shadow_prob: 阴影效果概率
            sun_flare_prob: 阳光眩光效果概率
            fog_prob: 雾化效果概率
            prob: 应用增强的概率
        """
        self.shadow_prob = shadow_prob
        self.sun_flare_prob = sun_flare_prob
        self.fog_prob = fog_prob
        self.prob = prob

        self.transform = None

    def _ensure_transforms_initialized(self):
        """确保transforms已初始化"""
        if self.transform is None:
            self.transform = A.Compose(
                [
                    # 随机阴影
                    A.RandomShadow(
                        shadow_roi=(0, 0.5, 1, 1),
                        num_shadows_lower=1,
                        num_shadows_upper=2,
                        shadow_dimension=5,
                        p=self.shadow_prob,
                    ),
                    # 随机阳光眩光
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 0.5),
                        angle_lower=0,
                        angle_upper=1,
                        num_flare_circles_lower=4,
                        num_flare_circles_upper=8,
                        src_radius=300,
                        src_color=(255, 255, 255),
                        p=self.sun_flare_prob,
                    ),
                    # 随机雾化效果
                    A.RandomFog(
                        fog_coef_lower=0.1,
                        fog_coef_upper=0.3,
                        alpha_coef=0.08,
                        p=self.fog_prob,
                    ),
                ]
            )

    def __call__(self, sample: Dict) -> Dict:
        """应用光照增强"""
        if random.random() > self.prob:
            return sample

        self._ensure_transforms_initialized()

        image = sample["image"]

        # 应用光照增强
        enhanced_image = self.transform(image=image)["image"]
        sample["image"] = enhanced_image

        return sample


class BEVAugmentation(object):
    """
    BEV图像增强类，支持多种增强效果
    专门用于将高保真仿真图像转换为真实相机拍摄效果
    """

    def __init__(
        self,
        severity: float = 0.5,  # 增强强度 0-1
        color_prob: float = 0.8,  # 色彩增强概率
        noise_prob: float = 0.6,  # 噪声增强概率
        blur_prob: float = 0.4,  # 模糊增强概率
    ):
        self.severity = severity
        self.color_prob = color_prob
        self.noise_prob = noise_prob
        self.blur_prob = blur_prob

        # 延迟初始化transforms，避免序列化问题
        self.transform = None

    def _ensure_transforms_initialized(self):
        """确保transforms已初始化（延迟初始化避免多进程序列化问题）"""
        if self.transform is None:
            self.transform = self._create_augmentation_pipeline()

    def _create_augmentation_pipeline(self) -> A.Compose:
        """创建BEV图像增强pipeline"""
        return A.Compose(
            [
                # 色彩增强
                A.ColorJitter(
                    brightness=0.2 * self.severity,
                    contrast=0.2 * self.severity,
                    saturation=0.2 * self.severity,
                    hue=0.1 * self.severity,
                    p=self.color_prob,
                ),
                # 色彩平衡
                A.RandomBrightnessContrast(
                    brightness_limit=0.15 * self.severity,
                    contrast_limit=0.15 * self.severity,
                    p=self.color_prob,
                ),
                # Gamma调整
                A.RandomGamma(gamma_limit=(80, 120), p=0.6),
                # 高斯模糊
                A.GaussianBlur(blur_limit=(3, 7), p=self.blur_prob),
                # 运动模糊
                A.MotionBlur(blur_limit=(3, 7), p=self.blur_prob),
                # 高斯噪声
                A.GaussNoise(
                    var_limit=(0.02, 0.1 * self.severity),
                    per_channel=True,
                    p=self.noise_prob,
                ),
                # 色调调整
                A.HueSaturationValue(
                    hue_shift_limit=10 * self.severity,
                    sat_shift_limit=15 * self.severity,
                    val_shift_limit=10 * self.severity,
                    p=self.color_prob,
                ),
                # 色彩变换
                A.RGBShift(
                    r_shift_limit=15 * self.severity,
                    g_shift_limit=15 * self.severity,
                    b_shift_limit=15 * self.severity,
                    p=self.color_prob,
                ),
                # 直方图均衡化
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                # 随机阴影
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=3,
                    shadow_dimension=5,
                    p=0.3,
                ),
                # 随机阳光
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),
                    angle_lower=0,
                    angle_upper=1,
                    num_flare_circles_lower=6,
                    num_flare_circles_upper=10,
                    src_radius=400,
                    src_color=(255, 255, 255),
                    p=0.2,
                ),
            ]
        )

    def _add_film_grain(self, image: np.ndarray, grain_prob: float = 0.4) -> np.ndarray:
        """添加胶片颗粒效果"""
        if random.random() > grain_prob:
            return image

        # 生成颗粒噪声
        grain_strength = random.uniform(0.1, 0.3) * self.severity
        grain = np.random.normal(0, grain_strength * 20, image.shape).astype(np.float32)

        # 应用颗粒
        result = image.astype(np.float32) + grain
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def __call__(self, sample: Dict) -> Dict:
        """
        对BEV图像进行增强

        Args:
            sample: 包含 'image' 键的字典，输入假定为RGB格式

        Returns:
            增强后的sample，输出保持RGB格式
        """
        # 确保transforms已初始化
        self._ensure_transforms_initialized()

        image = sample["image"]

        # 应用增强
        transformed_image = self.transform(image=image)["image"]

        # 添加胶片颗粒效果
        transformed_image = self._add_film_grain(transformed_image)

        # 输出保持RGB格式
        sample["image"] = transformed_image

        return sample


class Normalize(object):
    """标准化图像，适用于BEV任务"""

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, sample):
        if "image" in sample:
            sample["image"] = (sample["image"] - self.mean) / self.std
        return sample


# 使用示例
if __name__ == "__main__":
    # 测试Resize类
    resize_transform = Resize(size=(352, 640), pad_value=0)

    # 创建测试图像
    test_img = np.random.randint(0, 255, (480, 720, 3), dtype=np.uint8)

    # 应用resize
    sample = {"image": test_img}
    result = resize_transform(sample)

    print(f"原始图像尺寸: {test_img.shape}")
    print(f"目标尺寸: (352, 640)")
    print(f"处理后尺寸: {result['image'].shape}")

    # 创建新的独立增强类
    color_enhancer = ColorEnhancement(
        brightness_limit=0.2,
        contrast_limit=0.2,
        saturation_limit=0.3,
        hue_shift_limit=20,
        prob=0.8,
    )

    contrast_enhancer = ContrastEnhancement(
        gamma_limit=(80, 120), clahe_clip_limit=2.0, prob=0.7
    )

    noise_enhancer = LightNoiseEnhancement(gauss_noise_var_limit=(10.0, 50.0), prob=0.5)

    lighting_enhancer = LightingEnhancement(
        shadow_prob=0.3, sun_flare_prob=0.2, fog_prob=0.2, prob=0.6
    )

    # 创建组合变换
    transform_pipeline = Compose(
        [
            resize_transform,
            color_enhancer,
            contrast_enhancer,
            noise_enhancer,
            lighting_enhancer,
            ToTensor(),
        ]
    )

    # 创建BEV增强器（原有的综合增强器）
    augmentor = BEVAugmentation(
        severity=0.6,  # 增强强度
        color_prob=0.8,  # 色彩增强概率
        noise_prob=0.6,  # 噪声增强概率
        blur_prob=0.4,  # 模糊增强概率
    )

    # 假设有测试图像
    test_img = cv2.imread("test_image.jpg")

    if test_img is not None:
        # 转换为RGB格式
        test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        # 测试独立增强类
        sample = {"image": test_img_rgb}

        # 应用颜色增强
        color_enhanced = color_enhancer(sample.copy())

        # 应用对比度增强
        contrast_enhanced = contrast_enhancer(sample.copy())

        # 应用噪声增强
        noise_enhanced = noise_enhancer(sample.copy())

        # 应用光照增强
        lighting_enhanced = lighting_enhancer(sample.copy())

        print("新增的独立增强类创建成功！")
        print("\n1. ColorEnhancement - 颜色增强类：")
        print("   - 亮度和对比度调整")
        print("   - HSV色彩空间调整")
        print("   - RGB通道独立调整")
        print("   - 色彩抖动")

        print("\n2. ContrastEnhancement - 对比度增强类：")
        print("   - Gamma校正")
        print("   - CLAHE对比度受限自适应直方图均衡化")
        print("   - 直方图均衡化")
        print("   - 自动对比度调整")

        print("\n3. LightNoiseEnhancement - 轻量噪声增强类：")
        print("   - 高斯噪声")
        print("   - 乘性噪声")
        print("   - ISO噪声（模拟相机传感器噪声）")

        print("\n4. LightingEnhancement - 光照增强类：")
        print("   - 随机阴影")
        print("   - 随机阳光眩光")
        print("   - 随机雾化效果")

        print("\n使用方法：")
        print("# 单独使用")
        print("color_enhancer = ColorEnhancement(brightness_limit=0.2, prob=0.8)")
        print("enhanced_sample = color_enhancer(sample)")
        print("\n# 组合使用")
        print(
            "pipeline = Compose([ColorEnhancement(), ContrastEnhancement(), ToTensor()])"
        )
        print("result = pipeline(sample)")

    else:
        print("测试图像路径不存在，请检查路径")
        print("\n新增的独立增强类已创建完成！")
        print(
            "包含：ColorEnhancement, ContrastEnhancement, LightNoiseEnhancement, LightingEnhancement"
        )
