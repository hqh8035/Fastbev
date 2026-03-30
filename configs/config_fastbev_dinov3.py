from yacs.config import CfgNode as CN

_C = CN()

X_RANGE = (-9.6, 9.6)
Z_RANGE = (0.0, 32.0)
Y_RANGE = (-3.0, 2.0)
PIXEL_PER_METER_H = 10
PIXEL_PER_METER_V = 2

# 模型配置
_C.MODEL = CN()
_C.MODEL.NAME = "FastBEV"
_C.MODEL.PRETRAINED_PATH = ""  # 预训练模型路径

# FastBEV特定配置
_C.MODEL.FASTBEV = CN()
_C.MODEL.FASTBEV.X_RANGE = X_RANGE                      # X范围：右侧-9.6~9.6米（相机坐标系X轴）
_C.MODEL.FASTBEV.Z_RANGE = Z_RANGE                      # Z范围：前方0~32米（相机坐标系Z轴）
_C.MODEL.FASTBEV.Y_RANGE = Y_RANGE                      # Y范围：高度-3~2米（相机坐标系Y轴）
_C.MODEL.FASTBEV.PIXEL_PER_METER_H = PIXEL_PER_METER_H  # BEV图像每米像素数
_C.MODEL.FASTBEV.PIXEL_PER_METER_V = PIXEL_PER_METER_V  # 竖直每米像素数
_C.MODEL.FASTBEV.DETECTION_CHANNELS = 128               # 检测特征通道数
_C.MODEL.FASTBEV.SEG_CLASSES = 3                        # 分割类别数
_C.MODEL.FASTBEV.DET_CLASSES = 4                        # 检测类别数
_C.MODEL.FASTBEV.BACKBONE = 'dinov3'                    # 主干网络
_C.MODEL.FASTBEV.NECK = 'dinov3'                        # 颈部网络

# 数据集配置
_C.DATASET = CN()

# 训练集配置
_C.DATASET.TRAIN = CN()
_C.DATASET.TRAIN.nuscenes = CN()
_C.DATASET.TRAIN.nuscenes.data_root = '/perception/users/cfx/WorkSpace_2025/nova_fastbev/fastbev'
_C.DATASET.TRAIN.nuscenes.annotations_dir = '/perception/users/cfx/WorkSpace_2025/nova_fastbev/fastbev/mini_labels/bbox3d_labels'
_C.DATASET.TRAIN.nuscenes.category_mapping = [
    ['car', 'truck', 'bus', 'trailer', 'construction_vehicle'],
    ['pedestrian'],
    ['motorcycle', 'bicycle'],
    ['barrier']
]
_C.DATASET.TRAIN.nuscenes.bev_width_m = X_RANGE[1] - X_RANGE[0]
_C.DATASET.TRAIN.nuscenes.bev_height_m = Z_RANGE[1] - Z_RANGE[0]
_C.DATASET.TRAIN.nuscenes.pixel_per_meter = PIXEL_PER_METER_H
_C.DATASET.TRAIN.nuscenes.weight = 1.0
# 确保颜色格式正确（BGR格式，因为OpenCV使用BGR）
_C.DATASET.TRAIN.nuscenes.seg_color_map = [
    [0, 0, 0],      # 0: 背景 (黑色)
    [0, 0, 255],    # 1: 类别1 (红色，BGR格式)
    [0, 255, 0],    # 2: 类别2 (绿色，BGR格式)
    [255, 0, 0]     # 3: 类别3 (蓝色，BGR格式)
]

# 验证集配置
_C.DATASET.VAL = CN()
_C.DATASET.VAL.nuscenes = CN()
_C.DATASET.VAL.nuscenes.data_root = '/perception/users/cfx/WorkSpace_2025/nova_fastbev/fastbev'
_C.DATASET.VAL.nuscenes.annotations_dir = '/perception/users/cfx/WorkSpace_2025/nova_fastbev/fastbev/mini_labels/bbox3d_labels'
_C.DATASET.VAL.nuscenes.category_mapping = [
    ['car', 'truck', 'bus', 'trailer', 'construction_vehicle'],
    ['pedestrian'],
    ['motorcycle', 'bicycle'],
    ['barrier']
]
_C.DATASET.VAL.nuscenes.bev_width_m = X_RANGE[1] - X_RANGE[0]
_C.DATASET.VAL.nuscenes.bev_height_m = Z_RANGE[1] - Z_RANGE[0]
_C.DATASET.VAL.nuscenes.pixel_per_meter = PIXEL_PER_METER_H
_C.DATASET.VAL.nuscenes.seg_color_map = [
    [0, 0, 0],      # 0: 背景 (黑色)
    [0, 0, 255],    # 1: 类别1 (红色，BGR格式)
    [0, 255, 0],    # 2: 类别2 (绿色，BGR格式)
    [255, 0, 0]     # 3: 类别3 (蓝色，BGR格式)
]

# 数据变换配置
_C.DATASET.TRANSFORM = CN()

# 训练配置
_C.DATASET.TRANSFORM.train = CN()
_C.DATASET.TRANSFORM.train.shape = (448, 800)  # (H, W)

# 数据增强配置
_C.DATASET.TRANSFORM.train.augmentation = CN()
_C.DATASET.TRANSFORM.train.augmentation.enabled = False  # 是否启用增强
_C.DATASET.TRANSFORM.train.augmentation.severity = 0.5  # 增强强度 (0-1)
_C.DATASET.TRANSFORM.train.augmentation.color_prob = 0.8  # 色彩增强概率
_C.DATASET.TRANSFORM.train.augmentation.noise_prob = 0.6  # 噪声增强概率
_C.DATASET.TRANSFORM.train.augmentation.blur_prob = 0.4  # 模糊增强概率

# 验证配置
_C.DATASET.TRANSFORM.val = CN()
_C.DATASET.TRANSFORM.val.shape = (448, 800)  # (H, W)

# 训练配置
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.EPOCHS = 100
_C.TRAIN.LR = 0.0001
_C.TRAIN.WEIGHT_DECAY = 0.00005
_C.TRAIN.CLIP_GRAD_NORM = 3.0
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.SAVE_FREQ = 5
_C.TRAIN.PRINT_FREQ = 50
_C.TRAIN.VAL_FREQ = 1
_C.TRAIN.LOSS_QUANTILE = 1.0

# 路径配置
_C.PATH = CN()
_C.PATH.CHECKPOINT_DIR = './checkpoints/fastbev'
_C.PATH.LOG_DIR = './outputs/train/logs'
_C.PATH.DATA_ROOT = './data'

def get_cfg_defaults():
    """获取默认配置的克隆
    
    Returns:
        CN: 配置节点的克隆
    """
    return _C.clone()

# 定义 cfg 变量，这是 YACS 所需要的
cfg = _C.clone()

if __name__ == "__main__":
    print(cfg)
