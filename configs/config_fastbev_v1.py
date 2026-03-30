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
_C.MODEL.FASTBEV.SEG_OUT_CHANNELS = 5                        # 分割类别数
_C.MODEL.FASTBEV.DET_OUT_CHANNELS = 36                        # 检测输出的channels数, center + obj_height + offset + lwh + yaw = 1 + 1 + 2 + 3 + 2 
_C.MODEL.FASTBEV.BACKBONE = ''                    # 主干网络
_C.MODEL.FASTBEV.NECK = ''                        # 颈部网络

# 数据集配置
_C.DATASET = CN()

# 训练集配置
_C.DATASET.TRAIN = CN()
_C.DATASET.TRAIN.nuscenes = CN()
_C.DATASET.TRAIN.nuscenes.annotations_file = 'whole_data_labels/nuscenes_bbox3d_labels/train_bbox3d_list.txt'
_C.DATASET.TRAIN.nuscenes.bbox_category_mapping = [
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
_C.DATASET.VAL.nuscenes.annotations_file = 'whole_data_labels/nuscenes_bbox3d_labels/train_bbox3d_list.txt'
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
# _C.DATASET.TRANSFORM.train.shape = (448, 800)  # (H, W)
_C.DATASET.TRANSFORM.train.shape = (800, 1280)  # (H, W)

# 数据增强配置
_C.DATASET.TRANSFORM.train.augmentation = CN()
_C.DATASET.TRANSFORM.train.augmentation.enabled = False  # 是否启用增强
_C.DATASET.TRANSFORM.train.augmentation.severity = 0.5  # 增强强度 (0-1)
_C.DATASET.TRANSFORM.train.augmentation.color_prob = 0.8  # 色彩增强概率
_C.DATASET.TRANSFORM.train.augmentation.noise_prob = 0.6  # 噪声增强概率
_C.DATASET.TRANSFORM.train.augmentation.blur_prob = 0.4  # 模糊增强概率

# 验证配置
_C.DATASET.TRANSFORM.val = CN()
_C.DATASET.TRANSFORM.val.shape = (800, 1280)  # (H, W)

# 训练配置
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.MAX_ITERS = 50000
_C.TRAIN.LR = 0.001
_C.TRAIN.WEIGHT_DECAY = 0.00005
_C.TRAIN.CLIP_GRAD_NORM = 5.0
# _C.TRAIN.WARMUP_ITERATION = 1000
_C.TRAIN.SAVE_FREQ = 2500
_C.TRAIN.PRINT_FREQ = 1
_C.TRAIN.VAL_FREQ = 10000000          # just set here, no validate for debug
_C.TRAIN.LOSS_QUANTILE = 1.0

# 路径配置
_C.PATH = CN()
_C.PATH.CHECKPOINT_DIR = './checkpoints/fastbev'
_C.PATH.LOG_DIR = './outputs/train/logs'


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
