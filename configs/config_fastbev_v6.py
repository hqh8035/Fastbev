from yacs.config import CfgNode as CN

_C = CN()

X_RANGE = (-16, 16)
Z_RANGE = (0.0, 32.0)
Y_RANGE = (-2.0, 3.0)
PIXEL_PER_METER_H = 10
PIXEL_PER_METER_V = 2       # 5

# 模型配置
_C.MODEL = CN()
_C.MODEL.NAME = "FastBEV"
_C.MODEL.PRETRAINED_PATH = ""  # 预训练模型路径

# FastBEV特定配置
_C.MODEL.FASTBEV = CN()
_C.MODEL.FASTBEV.DOWN_SCALE_FACTOR = 0.25                   # 输入的feature map进行下采样倍率
_C.MODEL.FASTBEV.X_RANGE = X_RANGE                      # X范围：右侧-16~16米（相机坐标系X轴）
_C.MODEL.FASTBEV.Z_RANGE = Z_RANGE                      # Z范围：前方0~32米（相机坐标系Z轴）
_C.MODEL.FASTBEV.Y_RANGE = Y_RANGE                      # Y范围：高度-3~2米（相机坐标系Y轴）
_C.MODEL.FASTBEV.PIXEL_PER_METER_H = PIXEL_PER_METER_H  # BEV图像每米像素数
_C.MODEL.FASTBEV.PIXEL_PER_METER_V = PIXEL_PER_METER_V  # 竖直每米像素数
_C.MODEL.FASTBEV.DETECTION_CHANNELS = 128               # 检测特征通道数
_C.MODEL.FASTBEV.DET3D_OUT_CHANNELS = 27                        # 检测输出的channels数, center + obj_height + offset + lwh + yaw = 1 + 1 + 2 + 3 + 2, 3个类别 
_C.MODEL.FASTBEV.SEG_OUT_CHANNELS = 1                        # 分割类别数
_C.MODEL.FASTBEV.BACKBONE = ''                    # 主干网络
_C.MODEL.FASTBEV.NECK = ''                        # 颈部网络
_C.MODEL.FASTBEV.generate_det_gt_format = 'standard'    # 用于计算loss 
_C.MODEL.FASTBEV.CATEGORY_NUM = 3       # vehicle/person/two-wheel


# 数据集配置
_C.DATASET = CN()

# 训练集配置
_C.DATASET.TRAIN = CN()
# 这个是waymo数据集的配置（全数据）
_C.DATASET.TRAIN.waymo = CN()
_C.DATASET.TRAIN.waymo.with_seg_lane_line_label = True
_C.DATASET.TRAIN.waymo.with_det3d_label = True
_C.DATASET.TRAIN.waymo.with_seg_label = True
_C.DATASET.TRAIN.waymo.generation_gt = 'standard'        # 用于生成gt, 和FASTBEV.generate_det_gt_format保持一致           # 'gaussian' for center map gaussian/ 'standard' for standard focal loss
_C.DATASET.TRAIN.waymo.annotations_file = 'waymo_train_image_list.txt'
_C.DATASET.TRAIN.waymo.bbox_category_mapping = [
    ['vehicle'],
    ['person'],
    ['cyclist'],
]
_C.DATASET.TRAIN.waymo.bev_width_m = X_RANGE[1] - X_RANGE[0]
_C.DATASET.TRAIN.waymo.bev_height_m = Z_RANGE[1] - Z_RANGE[0]
_C.DATASET.TRAIN.waymo.pixel_per_meter = PIXEL_PER_METER_H
_C.DATASET.TRAIN.waymo.weight = 1.0


# 这个是pedestrian数据集的配置
_C.DATASET.TRAIN.pedestrian = CN()
_C.DATASET.TRAIN.pedestrian.with_seg_lane_line_label = True
_C.DATASET.TRAIN.pedestrian.with_det3d_label = True
_C.DATASET.TRAIN.pedestrian.with_seg_label = True
_C.DATASET.TRAIN.pedestrian.generation_gt = 'standard'
_C.DATASET.TRAIN.pedestrian.annotations_file = 'pedestrian_train_image_list.txt'  # 修改为你的路径
_C.DATASET.TRAIN.pedestrian.bbox_category_mapping = [
    ['vehicle'],
    ['person'],
    ['cyclist'],
]
_C.DATASET.TRAIN.pedestrian.bev_width_m = X_RANGE[1] - X_RANGE[0]
_C.DATASET.TRAIN.pedestrian.bev_height_m = Z_RANGE[1] - Z_RANGE[0]
_C.DATASET.TRAIN.pedestrian.pixel_per_meter = PIXEL_PER_METER_H
_C.DATASET.TRAIN.pedestrian.weight = 1.83  # 权重 1.83，对应采样概率 25%


# 这个是corner数据集的配置
_C.DATASET.TRAIN.corner = CN()
_C.DATASET.TRAIN.corner.with_seg_lane_line_label = True
_C.DATASET.TRAIN.corner.with_det3d_label = True
_C.DATASET.TRAIN.corner.with_seg_label = True
_C.DATASET.TRAIN.corner.generation_gt = 'standard'
_C.DATASET.TRAIN.corner.annotations_file = 'corner_train_image_list.txt'  # 修改为你的路径
_C.DATASET.TRAIN.corner.bbox_category_mapping = [
    ['vehicle'],
    ['person'],
    ['cyclist'],
]
_C.DATASET.TRAIN.corner.bev_width_m = X_RANGE[1] - X_RANGE[0]
_C.DATASET.TRAIN.corner.bev_height_m = Z_RANGE[1] - Z_RANGE[0]
_C.DATASET.TRAIN.corner.pixel_per_meter = PIXEL_PER_METER_H
#_C.DATASET.TRAIN.corner.weight = 11.47  # 权重 11.47，对应采样概率 50%
_C.DATASET.TRAIN.corner.weight = 7.65  # 权重 7.65，对应采样概率 40%

# 验证集配置
_C.DATASET.VAL = CN()
_C.DATASET.VAL.waymo = CN()
_C.DATASET.VAL.waymo.with_det3d_label = True
_C.DATASET.VAL.waymo.with_seg_label = True
_C.DATASET.VAL.waymo.generation_gt = 'standard' 
# 验证集的图片列表文件，相对于项目根目录的路径，不会边训练边验证，所以此文件内容不重要，当前设置为训练集的图片列表文件
_C.DATASET.VAL.waymo.annotations_file = 'waymo_train_image_list.txt'
_C.DATASET.VAL.waymo.category_mapping = [
    ['vehicle'],
    ['person'],
    ['cyclist'],
]
_C.DATASET.VAL.waymo.bev_width_m = X_RANGE[1] - X_RANGE[0]
_C.DATASET.VAL.waymo.bev_height_m = Z_RANGE[1] - Z_RANGE[0]
_C.DATASET.VAL.waymo.pixel_per_meter = PIXEL_PER_METER_H


# 数据变换配置
_C.DATASET.TRANSFORM = CN()

# 训练配置
_C.DATASET.TRANSFORM.train = CN()
_C.DATASET.TRANSFORM.train.shape = (816, 960)  # (H, W)

# 数据增强配置
_C.DATASET.TRANSFORM.train.augmentation = CN()
_C.DATASET.TRANSFORM.train.augmentation.enabled = False  # 是否启用增强
_C.DATASET.TRANSFORM.train.augmentation.severity = 0.5  # 增强强度 (0-1)
_C.DATASET.TRANSFORM.train.augmentation.color_prob = 0.8  # 色彩增强概率
_C.DATASET.TRANSFORM.train.augmentation.noise_prob = 0.6  # 噪声增强概率
_C.DATASET.TRANSFORM.train.augmentation.blur_prob = 0.4  # 模糊增强概率

# 验证配置
_C.DATASET.TRANSFORM.val = CN()
_C.DATASET.TRANSFORM.val.shape = (816, 960)  # (H, W)

# 训练配置
_C.TRAIN = CN()
# 67 建议为 12，85 建议为 6
_C.TRAIN.BATCH_SIZE = 6
_C.TRAIN.NUM_WORKERS = 4
# 六分数据集 4w，waymo 数据集 8w
_C.TRAIN.MAX_ITERS = 48000
_C.TRAIN.LR = 3e-4          # 降低学习率：5e-4 → 3e-4，提升训练稳定性
_C.TRAIN.WEIGHT_DECAY = 0.0001  # 增加正则化：5e-5 → 1e-4，防止过拟合
_C.TRAIN.CLIP_GRAD_NORM = 2.0   # 降低梯度裁剪：5.0 → 2.0，减少梯度爆炸
_C.TRAIN.WARMUP_ITERS = 1000    # 添加warmup，前1000步线性增加学习率
_C.TRAIN.SAVE_FREQ = 2000
_C.TRAIN.PRINT_FREQ = 1
_C.TRAIN.VAL_FREQ = 10000000          # just set here, no validate for debug
_C.TRAIN.LOSS_QUANTILE = 1.0
_C.TRAIN.EMA_DECAY = 0.9999     # 添加EMA (Exponential Moving Average)，平滑模型参数

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
