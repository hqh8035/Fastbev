import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from glob import glob

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

sys.path.append(".")
from configs import get_cfg_defaults

try:
    from shapely.affinity import rotate
    from shapely.geometry import Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    print("Warning: shapely not installed. Please install with: pip install shapely")
    SHAPELY_AVAILABLE = False


class EvalResultSaver:
    """评估结果保存器，支持多种输出格式（打印、JSON、TXT、CSV）
    
    约定数据结构：
    - metadata: dict，包含评估配置信息，支持嵌套字典
    - overall: dict，包含总体指标 {metric_name: value, ...}
    - per_category: dict，{category_name: {metric_name: value, ...}, ...}
    类别会自动按key排序，确保输出顺序一致
    """
    
    def __init__(self, metadata, overall, per_category):
        """
        初始化结果保存器
        
        Args:
            metadata: 元数据字典，包含评估配置信息
            overall: 总体指标字典，{metric_name: value, ...}
            per_category: 每个类别的指标字典，{category_name: {metric_name: value, ...}, ...}
        """
        self.metadata = metadata or {}
        self.overall = overall or {}
        self.per_category = per_category or {}
    
    def _format_value(self, value):
        """格式化值用于显示"""
        if isinstance(value, float):
            return f'{value:.3f}'
        elif isinstance(value, dict):
            # 对于嵌套字典，返回字符串表示
            return str(value)
        else:
            return str(value)
    
    def _format_value_precise(self, value):
        """格式化值用于CSV（更高精度）"""
        if isinstance(value, float):
            return f'{value:.6f}'
        else:
            return str(value)
    
    def _flatten_dict(self, d, parent_key='', sep=' '):
        """展平嵌套字典"""
        items = []
        for k, v in sorted(d.items()):
            new_key = f'{parent_key}{sep}{k}' if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _print_metadata(self):
        """打印metadata信息"""
        flat_metadata = self._flatten_dict(self.metadata)
        for key, value in sorted(flat_metadata.items()):
            print(f'{key}: {self._format_value(value)}')
    
    def _print_metrics(self, metrics_dict, prefix=''):
        """打印指标字典"""
        metric_items = []
        for key, value in sorted(metrics_dict.items()):
            if isinstance(value, (int, float)):
                metric_items.append(f'{key}={self._format_value(value)}')
            else:
                metric_items.append(f'{key}={value}')
        if metric_items:
            print(f'{prefix}{", ".join(metric_items)}')
    
    def print(self):
        """打印结果到控制台"""
        # 打印metadata
        flat_metadata = self._flatten_dict(self.metadata)
        for key, value in sorted(flat_metadata.items()):
            print(f'{key}: {self._format_value(value)}')
        print()
        
        # 打印每个类别的指标（按类别名排序）
        print('=== Per-category results ===')
        for cat, metrics in sorted(self.per_category.items()):
            self._print_metrics(metrics, prefix=f'Category {cat}: ')
        print()
        
        # 打印总体指标
        print('=== Overall (micro) ===')
        self._print_metrics(self.overall)
    
    def save_json(self, output_dir):
        """保存JSON格式结果"""
        os.makedirs(output_dir, exist_ok=True)
        eval_results = {
            'metadata': self.metadata,
            'per_category': self.per_category,
            'overall': self.overall
        }
        json_path = os.path.join(output_dir, 'eval_results.json')
        with open(json_path, 'w') as f:
            json.dump(eval_results, f, indent=4)
        return json_path
    
    def save_txt(self, output_dir):
        """保存TXT格式结果"""
        os.makedirs(output_dir, exist_ok=True)
        txt_path = os.path.join(output_dir, 'eval_report.txt')
        with open(txt_path, 'w') as f:
            # 写入metadata
            flat_metadata = self._flatten_dict(self.metadata)
            for key, value in sorted(flat_metadata.items()):
                f.write(f'{key}: {self._format_value(value)}\n')
            f.write('\n')
            
            # 写入每个类别的指标（按类别名排序）
            f.write('=== Per-category results ===\n')
            for cat, metrics in sorted(self.per_category.items()):
                metric_items = []
                for metric_key, metric_value in sorted(metrics.items()):
                    if isinstance(metric_value, (int, float)):
                        metric_items.append(f'{metric_key}={self._format_value(metric_value)}')
                    else:
                        metric_items.append(f'{metric_key}={metric_value}')
                f.write(f'Category {cat}: {", ".join(metric_items)}\n')
            f.write('\n')
            
            # 写入总体指标
            f.write('=== Overall (micro) ===\n')
            metric_items = []
            for key, value in sorted(self.overall.items()):
                if isinstance(value, (int, float)):
                    metric_items.append(f'{key}={self._format_value(value)}')
                else:
                    metric_items.append(f'{key}={value}')
            f.write(f'{", ".join(metric_items)}\n')
        return txt_path
    
    def save_csv(self, output_dir):
        """保存CSV格式结果，方便对比不同版本的模型"""
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'eval_results.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入metadata（展平嵌套字典）
            writer.writerow(['Metadata', 'Value'])
            flat_metadata = self._flatten_dict(self.metadata)
            for key, value in sorted(flat_metadata.items()):
                writer.writerow([key, value])
            writer.writerow([])  # 空行分隔
            
            # 写入总体指标
            overall_keys = sorted(self.overall.keys())
            writer.writerow(['Overall (micro)'] + overall_keys)
            overall_values = [self._format_value_precise(self.overall[k]) if isinstance(self.overall[k], float) 
                             else self.overall[k] for k in overall_keys]
            writer.writerow(['Overall'] + overall_values)
            writer.writerow([])  # 空行分隔
            
            # 写入每个类别的指标（按类别名排序）
            # 收集所有类别中出现的所有指标名
            all_metric_keys = set()
            for metrics in self.per_category.values():
                all_metric_keys.update(metrics.keys())
            all_metric_keys = sorted(all_metric_keys)
            
            writer.writerow(['Category'] + all_metric_keys)
            for cat, metrics in sorted(self.per_category.items()):
                row = [cat]
                for metric_key in all_metric_keys:
                    value = metrics.get(metric_key, '')
                    if isinstance(value, float):
                        row.append(self._format_value_precise(value))
                    else:
                        row.append(value)
                writer.writerow(row)
        return csv_path
    
    def save_all(self, output_dir):
        """保存所有格式的结果"""
        saved_files = []
        if output_dir:
            saved_files.append(self.save_json(output_dir))
            saved_files.append(self.save_txt(output_dir))
            saved_files.append(self.save_csv(output_dir))
            print(f'\nResults saved to:')
            for file_path in saved_files:
                print(f'  - {file_path}')
        return saved_files


def camera_to_bev_pixel(x_cam, y_cam, z_cam, cam_range_x, cam_range_z, bev_resolution=10.0):
    """
    将camera坐标系转换为BEV像素坐标
    
    Args:
        x_cam, y_cam, z_cam: camera坐标系下的坐标 (X右Y下Z前)
        cam_range_x: BEV图像的X范围 [x_min, x_max] (对应camera的Z轴,前后)
        cam_range_z: BEV图像的Y范围 [y_min, y_max] (对应camera的X轴,左右)
        bev_resolution: BEV分辨率 (pixel/meter)
    
    Returns:
        bev_x, bev_y: BEV图像像素坐标
    """
    # Camera坐标系: X右, Y下, Z前
    # BEV图像: X轴表示左右(camera的X), Y轴表示前后(camera的Z), 原点在左上角
    
    # camera Z轴 -> BEV X轴 (前后方向)
    bev_x = x_cam*bev_resolution + (cam_range_x[1]-cam_range_x[0])*bev_resolution/2
    
    # camera X轴 -> BEV Y轴 (左右方向), 注意左右翻转
    bev_y = -1 *z_cam * bev_resolution + (cam_range_z[1]-cam_range_z[0]) * bev_resolution
    
    return bev_x, bev_y

def bev_pixel_to_camera(bev_x, bev_y, bev_range_x, bev_range_y, bev_resolution=10.0):
    """
    将BEV像素坐标转换为camera坐标系 (仅X, Z)
    
    Returns:
        x_cam, z_cam: camera坐标系下的X(右), Z(前)坐标
    """
    # BEV X -> camera Z (前后)
    z_cam = bev_x / bev_resolution + bev_range_x[0]
    
    # BEV Y -> camera X (左右), 注意左右翻转
    x_cam = -(bev_y / bev_resolution + bev_range_y[0])
    
    return x_cam, z_cam

def load_gt(gt_path,cam_range_x, cam_range_z, distance, bev_resolution=10.0):
    """
    加载GT文件,将camera坐标系转换为BEV像素坐标
    
    Args:
        gt_path: GT文件路径
        cam_range_x: CAMERA的x方向范围 [x_min, x_max] (左右,米)
        cam_range_z: CAMERA的z方向范围 [z_min, z_max] (前后,米)
        distance: 过滤距离 (camera Z轴, 米), 如果为负数,则表示不进行距离过滤,以前视最大的距离进行评估. 超出该距离的bbox不参与评估.
        bev_resolution: BEV分辨率 (pixel/meter)
    """
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    gt_objs = []
    if distance < 0:
        filter_distance = cam_range_z[1]
    else:
        filter_distance = distance
        
    for bbox in gt['cam_bbox_3d']:
        center_cam = bbox['center']  # [x, y, z] in camera coordinate
        x_cam, y_cam, z_cam = center_cam[0], center_cam[1], center_cam[2]
        
        # 转换为BEV像素坐标
        bev_x, bev_y = camera_to_bev_pixel(x_cam, y_cam, z_cam, cam_range_x, cam_range_z, bev_resolution)
        
        category = bbox['category']
        
        # 检查是否在BEV图像范围内
        bev_height = (cam_range_z[1] - cam_range_z[0]) * bev_resolution
        bev_width = (cam_range_x[1] - cam_range_x[0]) * bev_resolution
        
        w,l,h = bbox['wlh']
        bev_box_l = l * bev_resolution
        bev_box_w = w * bev_resolution
        
        yaw = bbox['yaw']  
        if 0 <= bev_x < bev_width and 0 <= bev_y < bev_height and z_cam <= filter_distance:
            gt_objs.append({'center': [bev_x, bev_y], 
                            'obj_height': bev_box_l,
                            'obj_width': bev_box_w,
                            'yaw': yaw,
                            'category': str(category),
                            })
    
    return gt_objs

def load_pd(pd_path, cam_range_x, cam_range_z, distance, bev_resolution=10.0):
    """
    加载PD文件,PD的center已经是BEV像素坐标
    
    Args:
        pd_path: PD文件路径
        cam_range_x: CAMERA的x方向范围 [x_min, x_max] (左右,米)
        cam_range_z: CAMERA的z方向范围 [z_min, z_max] (前后,米)
        distance: 过滤距离 (camera Z轴, 米), 如果为负数,则表示不进行距离过滤,以前视最大的距离进行评估. 超出该距离的bbox不参与评估.
        bev_resolution: BEV分辨率 (pixel/meter)
    """
    with open(pd_path, 'r') as f:
        pd = json.load(f)
    pd_objs = []
    if distance < 0:
        distance = cam_range_z[1]

    filter_distance = (cam_range_z[1] - distance) * bev_resolution       # 过滤距离转换到bev坐标, pixel
    
    
    # 计算BEV图像尺寸
    bev_width = (cam_range_x[1] - cam_range_x[0]) * bev_resolution
    bev_height = (cam_range_z[1] - cam_range_z[0]) * bev_resolution
    
    for box in pd:
        center_bev = box['center']  # [x, y] in BEV pixel coordinates
        w, l, h = box['wlh']
        bev_box_l = l * bev_resolution
        bev_box_w = w * bev_resolution
    
        yaw = box['yaw']
        bev_x, bev_y = center_bev[0], center_bev[1]
        category = box['category']
        if isinstance(category, int):
            category = str(category)
        
        # 检查是否在BEV图像范围内
        if 0 <= bev_x < bev_width and 0 <= bev_y < bev_height and bev_y >= filter_distance:
            pd_objs.append({'center': [bev_x, bev_y], 
                            'obj_height': bev_box_l,
                            'obj_width': bev_box_w,
                            'yaw': yaw,
                            'category': str(category)})
    
    return pd_objs

def create_rotated_bbox(center_x, center_y, width, height, yaw):
    """
    创建旋转的边界框
    
    Args:
        center_x, center_y: 中心点坐标
        width, height: 宽度和高度
        yaw: 旋转角度 (弧度)
    
    Returns:
        Shapely Polygon对象
    """
    # 创建以原点为中心的矩形
    half_w = width / 2
    half_h = height / 2
    corners = [
        (-half_w, -half_h),
        (half_w, -half_h), 
        (half_w, half_h),
        (-half_w, half_h)
    ]
    
    # 旋转角度 (yaw是BEV x轴逆时针旋转到BEV y轴负方向的角度)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    # 旋转并平移到中心点
    rotated_corners = []
    for x, y in corners:
        new_x = cos_yaw * x - sin_yaw * y + center_x
        new_y = sin_yaw * x + cos_yaw * y + center_y
        rotated_corners.append((new_x, new_y))
    
    return Polygon(rotated_corners)

def calculate_iou(bbox1, bbox2):
    """
    计算两个旋转矩形的IoU
    
    Args:
        bbox1, bbox2: Shapely Polygon对象
    
    Returns:
        IoU值 (0-1)
    """
    try:
        intersection = bbox1.intersection(bbox2).area
        union = bbox1.union(bbox2).area
        if union == 0:
            return 0
        return intersection / union
    except:
        return 0

def match_with_iou(gt_objs, pd_objs, iou_thresh=0.5):
    """
    使用IoU进行匹配
    
    Args:
        gt_objs: GT对象列表，每个对象包含center, obj_width, obj_height, yaw
        pd_objs: PD对象列表，格式同GT
        iou_thresh: IoU阈值
    
    Returns:
        tp, fp, fn: 匹配结果
    """
    if not SHAPELY_AVAILABLE:
        print("Error: shapely is required for IoU calculation. Please install with: pip install shapely")
        return 0, len(pd_objs), len(gt_objs)
        
    if len(gt_objs) == 0 or len(pd_objs) == 0:
        return 0, len(pd_objs), len(gt_objs)
    
    # 创建GT和PD的边界框
    gt_bboxes = []
    for obj in gt_objs:
        center_x, center_y = obj['center']
        width = obj['obj_width']
        height = obj['obj_height'] 
        yaw = obj['yaw']
        bbox = create_rotated_bbox(center_x, center_y, width, height, yaw)
        gt_bboxes.append(bbox)
    
    pd_bboxes = []
    for obj in pd_objs:
        center_x, center_y = obj['center']
        width = obj['obj_width']
        height = obj['obj_height']
        yaw = obj['yaw']
        bbox = create_rotated_bbox(center_x, center_y, width, height, yaw)
        pd_bboxes.append(bbox)
    
    # 计算IoU矩阵
    iou_matrix = np.zeros((len(gt_bboxes), len(pd_bboxes)))
    for i, gt_bbox in enumerate(tqdm(gt_bboxes)):
        for j, pd_bbox in enumerate(pd_bboxes):
            iou_matrix[i, j] = calculate_iou(gt_bbox, pd_bbox)
    
    # 使用匈牙利算法进行最优匹配 (最大化IoU)
    # 由于linear_sum_assignment最小化代价，我们使用(1-IoU)作为代价
    cost_matrix = 1 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 统计匹配结果
    matches = 0
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_thresh:
            matches += 1
    
    tp = matches
    fp = len(pd_objs) - matches
    fn = len(gt_objs) - matches

    return tp, fp, fn


def compute_nuscenes_ap(gt_objs, pd_objs, bev_resolution=10.0,
                        dist_thresholds=[0.5, 1.0, 2.0, 4.0]):
    """
    计算 nuScenes 风格的 mAP

    nuScenes 使用 BEV 中心点距离来匹配 GT 和预测框：
    - 如果预测框和 GT 中心点距离 < 阈值，则认为匹配成功
    - 不同距离阈值分别计算 AP，最后取平均

    Args:
        gt_objs: GT对象列表
        pd_objs: PD对象列表
        bev_resolution: BEV分辨率 (pixel/meter)
        dist_thresholds: nuScenes 距离阈值列表 [0.5, 1.0, 2.0, 4.0] 米

    Returns:
        dict: 包含每个阈值下的 AP 和 mAP
    """
    # 如果没有 GT 或没有预测结果，直接返回全0的字典（避免后续除零错误）
    if len(gt_objs) == 0 or len(pd_objs) == 0:
        return {d: 0.0 for d in dist_thresholds} | {'mAP': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}

    # 定义内部函数：将对象从像素坐标转换为米坐标
    # 输入：对象（包含中心点像素坐标 'center'）
    # 输出：中心点的米坐标 (x_meter, y_meter)
    def get_center_meter(obj):
        x, y = obj['center']  # 获取对象的中心点像素坐标 (单位: 像素)
        return (x / bev_resolution, y / bev_resolution)  # 转换为米坐标 (单位: 米)

    # 获取 GT 和预测框的数量
    n_gt = len(gt_objs)  # GT 对象的数量
    n_pd = len(pd_objs)  # 预测对象的数量

    # 计算所有 GT 对象的中心点坐标（转换为米）
    gt_centers = [get_center_meter(obj) for obj in gt_objs]
    # 为每个预测对象添加置信度分数，如果没有置信度则默认设为 1.0（满分）
    pd_objs_with_score = [(obj, obj.get('score', 1.0)) for obj in pd_objs]

    # 按置信度从高到低排序预测结果（这是计算 AP 的关键步骤）
    # 高置信度的预测会被优先考虑，这样才能正确模拟 P-R 曲线
    pd_objs_with_score.sort(key=lambda x: x[1], reverse=True)  # reverse=True 表示降序排列，按照置信度降序排列
    sorted_pd_objs = [x[0] for x in pd_objs_with_score]  # 提取排序后的预测对象
    sorted_pd_centers = [get_center_meter(obj) for obj in sorted_pd_objs]  # 计算排序后的中心点

    # 用于存储每个距离阈值对应的 AP
    ap_per_threshold = {}

    # 遍历每个距离阈值，分别计算对应的 AP
    for dist_thresh in dist_thresholds:
        tp_list = []  # 存储每个预测框是否为 TP（真阳性）
        fp_list = []  # 存储每个预测框是否为 FP（假阳性）

        # 记录每个 GT 是否已被匹配（避免同一个 GT 被多个预测框匹配）
        gt_matched = [False] * n_gt

        # 遍历每个预测框（按置信度从高到低）
        for pd_center in sorted_pd_centers:
            # 找到最近的未匹配 GT
            min_dist = float('inf')  # 初始化最小距离为无穷大
            best_gt_idx = -1  # 初始化最佳 GT 索引为 -1（表示没有找到匹配）

            # 遍历所有 GT，寻找最近的未匹配 GT
            for gt_idx, gt_center in enumerate(gt_centers):
                if gt_matched[gt_idx]:  # 跳过已被匹配的 GT
                    continue
                # 计算预测框与 GT 中心点的欧几里得距离（米）
                dist = np.sqrt((pd_center[0] - gt_center[0])**2 + (pd_center[1] - gt_center[1])**2)
                # 如果当前距离小于最小距离，更新最小距离和最佳 GT 索引
                if dist < min_dist:
                    min_dist = dist
                    best_gt_idx = gt_idx

            # 判断是否匹配成功：找到了最近的 GT 且距离小于阈值
            if best_gt_idx >= 0 and min_dist <= dist_thresh:
                tp_list.append(1)  # 预测正确，记为 TP
                fp_list.append(0)  # 不是 FP
                gt_matched[best_gt_idx] = True  # 标记该 GT 已被匹配
            else:
                tp_list.append(0)  # 不是 TP
                fp_list.append(1)  # 预测错误，记为 FP

        # 计算累积的 TP 和 FP（用于计算 P-R 曲线）
        tp_cumsum = np.cumsum(tp_list)  # 累积真阳性数量
        fp_cumsum = np.cumsum(fp_list)  # 累积假阳性数量

        # 计算召回率（Recall）和精确率（Precision）
        recall = tp_cumsum / n_gt  # 召回率 = 累积 TP / 总 GT 数量
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)  # 精确率 = 累积 TP / (累积 TP + 累积 FP)

        # 计算 AP（使用 11 点插值法，这是 nuScenes 官方推荐的方法）
        ap = 0.0  # 初始化 AP 为 0
        for r in np.linspace(0, 1, 11):  # 在 [0, 1] 区间取 11 个召回率点
            if np.sum(recall >= r) == 0:  # 如果没有召回率 >= 当前阈值 r
                p = 0  # 精确率设为 0
            else:
                # 取所有召回率 >= r 的点中的最大精确率（这是 P-R 曲线的上包络）
                p = np.max(precision[recall >= r])
            ap += p / 11  # 累加所有点的精确率

        # 记录当前距离阈值下的 AP
        ap_per_threshold[dist_thresh] = ap

    # 计算 mAP（所有距离阈值的 AP 的平均值）
    mAP = np.mean(list(ap_per_threshold.values()))

    # 计算 TP/FP/FN（使用默认 2.0 米阈值，这是 nuScenes 常用的阈值）
    default_thresh = 2.0  # 默认距离阈值（米）
    gt_matched = [False] * n_gt  # 重置 GT 匹配状态
    tp_count = 0  # 初始化 TP 计数为 0

    # 再次遍历预测框，计算最终的 TP/FP/FN
    for pd_obj in sorted_pd_objs:
        pd_center = get_center_meter(pd_obj)  # 获取预测框的中心点（米）
        min_dist = float('inf')  # 初始化最小距离
        best_gt_idx = -1  # 初始化最佳 GT 索引

        # 寻找最近的未匹配 GT
        for gt_idx, gt_center in enumerate(gt_centers):
            if gt_matched[gt_idx]:  # 跳过已匹配的 GT
                continue
            # 计算距离
            dist = np.sqrt((pd_center[0] - gt_center[0])**2 + (pd_center[1] - gt_center[1])**2)
            if dist < min_dist:  # 更新最小距离
                min_dist = dist
                best_gt_idx = gt_idx

        # 判断是否匹配成功
        if best_gt_idx >= 0 and min_dist <= default_thresh:
            tp_count += 1  # TP 计数加 1
            gt_matched[best_gt_idx] = True  # 标记 GT 已被匹配

    # 计算最终的 TP、FP、FN 数量
    tp = tp_count  # 真阳性：预测正确且匹配成功
    fp = len(pd_objs) - tp_count  # 假阳性：预测框没有匹配到任何 GT
    fn = n_gt - tp_count  # 假阴性：GT 没有被任何预测框匹配到

    # 返回结果字典，包含每个阈值的 AP、mAP、以及 TP/FP/FN 计数
    return ap_per_threshold | {'mAP': mAP, 'tp': tp, 'fp': fp, 'fn': fn}


def main(gt_dirs, pd_dir, z_min, z_max, x_min, x_max, iou_thresh, distance=-1.0, bev_resolution=10.0, output_dir=None,
         use_nuscenes_map=False, nuscenes_dist_thresholds=[0.5, 1.0, 2.0, 4.0]):
    """
    主函数

    Args:
        gt_dirs: GT文件目录列表（可以是单个目录或多个目录）
        pd_dir: PD文件目录
        z_min, z_max: BEV前后范围 (camera Z轴, 米)
        x_min, x_max: BEV左右范围 (camera X轴, 米)
        iou_thresh: IoU匹配阈值 (0-1),
        distance: 过滤距离 (camera Z轴, 米)
        bev_resolution: BEV分辨率 (pixel/meter)
        output_dir: 输出目录，如果指定则保存结果到文件
        use_nuscenes_map: 是否使用 nuScenes mAP
        nuscenes_dist_thresholds: nuScenes 距离阈值列表
    """
    # 如果gt_dirs是字符串，转换为列表
    if isinstance(gt_dirs, str):
        gt_dirs = [gt_dirs]
    
    # 从所有GT目录收集文件
    gt_files = []
    for gt_dir in gt_dirs:
        gt_files.extend(sorted(glob(os.path.join(gt_dir, '*.json'))))
    gt_files = sorted(gt_files)
    pd_files = sorted(glob(os.path.join(pd_dir, '*.json')))
    assert len(gt_files) == len(pd_files), 'GT和PD文件数量不一致! GT文件数量: {}, PD文件数量: {}'.format(len(gt_files), len(pd_files))

    cam_range_x = [x_min, x_max]
    cam_range_z = [z_min, z_max]

    all_gt = defaultdict(list)
    all_pd = defaultdict(list)

    for gt_file, pd_file in zip(gt_files, pd_files):
        gt_objs = load_gt(gt_file, cam_range_x, cam_range_z, distance, bev_resolution)
        pd_objs = load_pd(pd_file, cam_range_x, cam_range_z, distance, bev_resolution)
        for obj in gt_objs:
            all_gt[obj['category']].append(obj)
        for obj in pd_objs:
            all_pd[obj['category']].append(obj)
    
    results = {}
    total_tp = total_fp = total_fn = 0
    # 对类别进行排序，确保每次运行顺序一致
    all_cats = sorted(set(all_gt.keys()) | set(all_pd.keys()))

    # 打印评估模式信息
    if use_nuscenes_map:
        print(f"\n使用 nuScenes mAP 评估")
        print(f"距离阈值: {nuscenes_dist_thresholds} 米")
    else:
        print(f"\n使用 IoU 匹配: 阈值 {iou_thresh}")

    for cat in all_cats:
        gt_objs = all_gt[cat] if all_gt[cat] else []
        pd_objs = all_pd[cat] if all_pd[cat] else []

        if use_nuscenes_map:
            # 使用 nuScenes mAP
            nuscenes_result = compute_nuscenes_ap(gt_objs, pd_objs, bev_resolution, nuscenes_dist_thresholds)
            tp = nuscenes_result['tp']
            fp = nuscenes_result['fp']
            fn = nuscenes_result['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            results[cat] = {
                'precision': precision, 'recall': recall, 'f1': f1,
                'tp': tp, 'fp': fp, 'fn': fn,
                'mAP': nuscenes_result['mAP'],
                **{f'AP@{d}m': nuscenes_result[d] for d in nuscenes_dist_thresholds}
            }
        else:
            # 使用传统 IoU 匹配
            tp, fp, fn = match_with_iou(gt_objs, pd_objs, iou_thresh)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            results[cat] = {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}

        total_tp += tp
        total_fp += fp
        total_fn += fn

    bev_width = (z_max - z_min) * bev_resolution
    bev_height = (x_max - x_min) * bev_resolution
    
    # 计算总体指标
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
    
    # 准备保存的数据
    metadata = {
        'gt_files_count': len(gt_files),
        'pd_files_count': len(pd_files),
        'bev_range': {
            'z_min': z_min, 'z_max': z_max,
            'x_min': x_min, 'x_max': x_max
        },
        'bev_image_size': {'width': bev_width, 'height': bev_height},
        'bev_resolution': bev_resolution,
        'iou_threshold': iou_thresh,
        'filter_distance': distance
    }
    overall = {
        'precision': micro_p,
        'recall': micro_r,
        'f1': micro_f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }
    
    # 使用结果保存器进行打印和保存
    saver = EvalResultSaver(metadata, overall, results)
    saver.print()
    saver.save_all(output_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='BEV BBox Evaluation by Category')
    parser.add_argument('--config', type=str, default='fastbev_v6', help='Config name (default: fastbev_v6)')
    parser.add_argument('--gt_dir', type=str, nargs='+', default=['select_validation_results/gt_bboxes'], help='GT文件夹路径（可以指定多个）')
    parser.add_argument('--pd_dir', type=str, default = 'select_validation_results/pd_bboxes', help='PD文件夹路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录 (default: outputs/eval/eval_TIMESTAMP)')
    parser.add_argument('--distance', type=float, default=15.0, help='过滤距离 (camera Z轴, 米)')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IoU匹配阈值 (0-1)')
    # 新增：nuScenes mAP 参数
    parser.add_argument('--use_nuscenes_map', action='store_true',
                        help='使用 nuScenes mAP 替代 IoU（更适合小目标评估）')
    parser.add_argument('--nuscenes_dists', type=str, default='0.5,1.0,2.0,4.0',
                        help='nuScenes 距离阈值（米），逗号分隔，默认: 0.5,1.0,2.0,4.0')
    args = parser.parse_args()

    # 解析 nuScenes 距离阈值
    nuscenes_dist_thresholds = [float(x) for x in args.nuscenes_dists.split(',')]
    
    # 加载配置文件
    cfg = get_cfg_defaults(args.config)
    print(f"Using config: {args.config}")
    cfg.freeze()
    
    # 从配置文件读取范围参数
    x_range = cfg.MODEL.FASTBEV.X_RANGE
    z_range = cfg.MODEL.FASTBEV.Z_RANGE
    pixel_per_meter = cfg.MODEL.FASTBEV.PIXEL_PER_METER_H
    
    x_min = x_range[0]
    x_max = x_range[1]
    z_min = z_range[0]
    z_max = z_range[1]
    bev_resolution = pixel_per_meter
    
    print(f"BEV range from config: X[{x_min}, {x_max}]m, Z[{z_min}, {z_max}]m")
    print(f"BEV resolution from config: {bev_resolution} pixel/m")
    print()
    
    # 如果没有指定 output_dir，使用带时间戳的默认目录
    if args.output_dir is None:
        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = f"./outputs/eval/eval_{datetime_str}"
    
    print(f"Results will be saved to: {args.output_dir}\n")

    main(args.gt_dir, args.pd_dir,
         z_min, z_max,
         x_min, x_max,
         args.iou_thresh, args.distance,
         bev_resolution, args.output_dir,
         use_nuscenes_map=args.use_nuscenes_map,
         nuscenes_dist_thresholds=nuscenes_dist_thresholds)
