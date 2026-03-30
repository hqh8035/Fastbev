import json
import logging
import os
import random
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

from utils.heatmap import get_gaussian_radius, set_gaussian_heat, set_standard_heat


class WaymoBEVDataset(Dataset):
    def __init__(self,
                annotations_file: str, 
                category_mapping: Dict = None,
                bev_width_m: float = 30,
                bev_height_m: float = 30,
                pixel_per_meter: float = 10,
                transform: Callable = None,
                generate_det_gt_format: str = 'gaussian',
                complemented_det3d_channel: int = 0,
                complemented_seg_channel: int= 0,
                with_det3d_label: bool = False,
                with_seg_label: bool = False
                ):
        """
        初始化WaymoBEVDataset
        Args:
            data_root: 数据根目录
            annotation_files: 标注JSON文件路径列表
            category_mapping: 类别映射字典，将原始类别名映射到数字ID
            bev_width_m: BEV宽度（米）
            bev_height_m: BEV高度（米）
            pixel_per_meter: 每米像素数
            transform: 图像变换
        """
        self.debug = False
        self.grass_id = 123
        self.annotations_file = annotations_file
        self.category_mapping = category_mapping or self._get_default_category_mapping()
        self.transform = transform
        self.generate_det_gt_format = generate_det_gt_format
        self.complemented_det3d_channel = complemented_det3d_channel
        self.complemented_seg_channel = complemented_seg_channel
        self.with_det3d_label = with_det3d_label
        self.with_seg_label = with_seg_label
        
        # 加载标注数据
        self.annotations = self._load_annotations()
        
        # BEV参数设置
        self.bev_width_m = bev_width_m  # 左右各15米
        self.bev_height_m = bev_height_m  # 前方30米
        self.pixel_per_meter = pixel_per_meter  # 每米10像素
        self.bev_width = int(round(self.bev_width_m * self.pixel_per_meter))  # 300像素
        self.bev_height = int(round(self.bev_height_m * self.pixel_per_meter))  # 300像素
        
        # 相机位置（图像中心）
        self.camera_x = self.bev_width // 2  # 150
        self.camera_y = self.bev_height - 1  # 299 (底部)
        
        # ignore区域参数
        self.ignore_radius = 2  # ignore区域的半径（像素）
        
        logging.info(f"Frame loaded: {len(self.annotations)} samples")
        logging.info(f"BEV size: {self.bev_width}x{self.bev_height} pixels")
        logging.info(f"Categories: {len(self.category_mapping)}")
    
    def _get_default_category_mapping(self) -> List[List[str]]:
        return [['vehicle'],
                ['person'],
                ['cyclist']]
    
    def _load_annotations(self) -> List[Dict]:
        """加载标注文件"""
        with open(self.annotations_file, 'r') as f:
            annotations = f.readlines()
        annotations = [x.strip() for x in annotations]
        return annotations
    
    def world_to_pixel(self, x_world: float, z_world: float) -> Tuple[int, int]:
        """
        将世界坐标转换为像素坐标
        Args:
            x_world: 世界X坐标（米），右为正
            z_world: 世界Z坐标（米），前为正
        Returns:
            (x_pixel, y_pixel): 像素坐标
        """
        x_pixel = self.camera_x + x_world * self.pixel_per_meter
        y_pixel = self.camera_y - z_world * self.pixel_per_meter
        return x_pixel, y_pixel
    
    def filter_boxes_by_fov(self, data: Dict, image_shape: Tuple[int, int]) -> Dict:
        """
        根据FOV过滤边界框
        """
        filtered_data = data.copy()
        
        # 获取相机内参
        cam_intrinsic = np.array(data['cam_intrinsic']) if 'cam_intrinsic' in data else np.array(data['camera_intrinsic'])
        fx = cam_intrinsic[0, 0]
        fy = cam_intrinsic[1, 1]
        cx = cam_intrinsic[0, 2]
        cy = cam_intrinsic[1, 2]
        
        img_height, img_width = image_shape
        
        valid_boxes = []
        total_boxes = len(data['cam_bbox_3d'])
        
        for i, bbox in enumerate(data['cam_bbox_3d']):
            center = np.array(bbox['center'])
            
            # 检查是否在相机前方
            if center[2] <= 0:
                continue
            
            # 投影到2D图像坐标
            x = center[0] / center[2]
            y = center[1] / center[2]
            u = fx * x + cx
            v = fy * y + cy
            
            # 检查图像边界
            if (u < 0 or u >= img_width or v < 0 or v >= img_height):
                continue

            valid_boxes.append(bbox)
        
        filtered_data['cam_bbox_3d'] = valid_boxes
        return filtered_data
    
    
    def encode_center_map(self, gt_map, center, size, overlap=0.7, **kwargs):
        '''Set center on ground-truth map.'''
        if self.generate_det_gt_format == 'gaussian':
            radius = get_gaussian_radius(size, overlap)
            gt_map[0] = set_gaussian_heat(gt_map[0], center, radius)
        else:
            gt_map[0] = set_standard_heat(gt_map[0], center, radius=1)
        return gt_map
        
    def project_bbox3d_to_bev(self, bbox3d):
        '''
            used to check the gt_maps
        '''
        center = np.array(bbox3d['center'])
        w, l, h = np.array(bbox3d['wlh'])
        yaw = bbox3d['yaw']
        
        width = w * self.pixel_per_meter
        height = l * self.pixel_per_meter
        cos_a = np.cos(yaw)
        sin_a = np.sin(yaw)
        
        corners = np.array([
            [-width/2, -height/2],
            [width/2, -height/2],
            [width/2, height/2],
            [-width/2, height/2]
        ])
        
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [-sin_a, -cos_a] # 这里需要修正，因为y轴朝上，行号变化应该是减法
        ])
        
        center_x, center_y = self.world_to_pixel(center[0], center[2])
        rotated_corners = np.dot(corners, rotation_matrix.T)
        world_corners = rotated_corners + np.array([center_x, center_y])
        return world_corners
        
    def densify_polyline(self, points, step=1.0):
        """对点集进行插值，使相邻点距离不超过step像素"""
        dense_points = []
        for i in range(len(points) - 1):
            p1 = np.array(points[i])
            p2 = np.array(points[i+1])
            dist = np.linalg.norm(p2 - p1)
            n_points = max(int(dist // step), 1)
            for j in range(n_points):
                interp = p1 + (p2 - p1) * (j / n_points)
                dense_points.append(tuple(interp))
        dense_points.append(tuple(points[-1]))
        return dense_points
    
    def generate_gt_maps(self, boxes: List[Dict]) -> np.ndarray:
        """
        Args:
            boxes: 过滤后的3D边界框列表
        Returns:
            heatmap: shape (num_classes, bev_height, bev_width)
            regression_map: shape (8, bev_height, bev_width) - [x, y, z, l, w, h, sin_yaw, cos_yaw]
        """
        num_classes = len(self.category_mapping)
        
        center_map = np.zeros([num_classes, self.bev_height, self.bev_width], dtype=np.int32)         # set xy in center heatmap
        center_height_map = np.zeros([num_classes, self.bev_height, self.bev_width], dtype=np.float32)         # set height value in the center heatmap
        offset_map  = np.zeros([num_classes, 2, self.bev_height, self.bev_width], dtype=np.float32)        # x-offset / y-offset
        lwh_map = np.zeros([num_classes, 3, self.bev_height, self.bev_width], dtype=np.float32)            # set l,w,h value in the correspond position in the bev map
        yaw_map = np.zeros([num_classes, 2, self.bev_height, self.bev_width], dtype=np.float32)            # set sin/cos of yaw value for afterwards regression  
        
        for bbox in boxes:
            category = bbox['category']
            category_id = -1
            for i, category_list in enumerate(self.category_mapping):
                if category in category_list:
                    category_id = i
                    break
            if category_id == -1:
                continue
            
            bbox['category_id'] = category_id

        
            center_x_cam, center_y_cam, center_z_cam = bbox['center']
            width, length, height = bbox['wlh']
            yaw = bbox['yaw']
            category_id = bbox['category_id']
            assert category_id >= 0 and category_id < num_classes, "category_id out of range"
            
            # 世界坐标转换为像素坐标
            center_x, center_y = self.world_to_pixel(center_x_cam, center_z_cam)
            center_x_int = int(center_x)
            center_y_int = int(center_y)
            
            # 检查是否在图像范围内
            if (0 <= center_x_int < self.bev_width and 0 <= center_y_int < self.bev_height):
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if center_y_int + i < 0 or center_y_int + i >= self.bev_height or center_x_int + j < 0 or center_x_int + j >= self.bev_width:
                            continue
                        if i == 0 and j == 0:
                            center_map[category_id, center_y_int, center_x_int] = 1
                        elif center_map[category_id, center_y_int + i, center_x_int + j] != 1:
                            center_map[category_id, center_y_int + i, center_x_int + j] = -1
                            
                offset_map[category_id, 0, center_y_int, center_x_int] = float(center_x - center_x_int)  # x偏移
                offset_map[category_id, 1, center_y_int, center_x_int] = float(center_y - center_y_int)  # y偏移
                
                center_height_map[category_id, center_y_int, center_x_int] = float(center_y_cam)  # z坐标
                
                lwh_map[category_id, 0, center_y_int, center_x_int] = length    # bbox 长度
                lwh_map[category_id, 1, center_y_int, center_x_int] = width    # bbox 宽度
                lwh_map[category_id, 2, center_y_int, center_x_int] = height    # bbox 高度
                
                # 角度回归：使用sin/cos
                yaw_map[category_id, 0, center_y_int, center_x_int] = np.sin(yaw)  # sin_yaw
                yaw_map[category_id, 1, center_y_int, center_x_int] = np.cos(yaw)  # cos_yaw
                
        gts_all = np.concatenate([np.expand_dims(center_map, axis=1), offset_map, np.expand_dims(center_height_map, axis=1), lwh_map, yaw_map], axis=1)
        return gts_all

    def draw_polyline_on_map(self, bev_map, bev_points, color=(1), thickness=3):
        for i in range(len(bev_points) - 1):
            cv2.polylines(bev_map, [bev_points.astype(np.int32)], isClosed=False, color=-1, thickness=thickness+2)
            cv2.polylines(bev_map, [bev_points.astype(np.int32)], isClosed=False, color=color, thickness=thickness)
        return bev_map
        
        
    def get_sample(self, idx: int) -> Dict:
        """
        获取单个样本
        Args:
            idx: 样本索引
        Returns:
            sample: 包含图像、heatmap、regression_map等的字典
        """
        if idx >= len(self.annotations):
            raise IndexError(f"Index {idx} out of range")
        
        image_file = self.annotations[idx]
        image_format = os.path.basename(image_file).split('.')[-1]
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Image not found: {image_file}")
        image = cv2.imread(image_file)
        
        if self.with_det3d_label:
            det_3d_label_file = image_file.replace('images/', 'labels/')    
            det_3d_label_file = det_3d_label_file.replace(image_format, 'json')
            
            if not os.path.exists(det_3d_label_file):
                det3d_gt_maps = np.ones([len(self.category_mapping), self.complemented_det3d_channel//len(self.category_mapping), self.bev_height, self.bev_width]) * -1
                boxes = []
                camera_intrinsics = None
            else:
                with open(det_3d_label_file, 'r') as f:
                    data = json.load(f)
                camera_intrinsics = data['cam_intrinsic']
                filtered_data = self.filter_boxes_by_fov(data, image.shape[:2])
                boxes = filtered_data['cam_bbox_3d']
                
                # 生成GT maps
                det3d_gt_maps  = self.generate_gt_maps(boxes)
        else:
            det3d_gt_maps = np.ones([len(self.category_mapping), self.complemented_det3d_channel//len(self.category_mapping), self.bev_height, self.bev_width]) * -1
        
        if self.with_seg_label:
            seg_label_file = image_file.replace('images/', 'labels/') 
            seg_label_file = seg_label_file.replace('.'+image_format, '_seg_mask.png')    # seg_label_file.replace(image_format, 'png')
            
            if not os.path.exists(seg_label_file):
                seg_gt_maps = np.ones([self.complemented_seg_channel, image.shape[0], image.shape[1]]) * -1
            else:
                raw_seg_grass_map = cv2.imread(seg_label_file)[:,:,0]
                # road_seg_map = (raw_seg_grass_map == 1).astype(np.int32)
                # grass_seg_map = (raw_seg_grass_map == 2).astype(np.int32)
                branch_seg_map = (raw_seg_grass_map == 255).astype(np.int32)
                # seg_gt_maps = np.stack([road_seg_map, grass_seg_map, branch_seg_map], axis=0)
                seg_gt_maps = np.stack([branch_seg_map], axis=0)
        
        camera_intrinsics = np.array(camera_intrinsics) if camera_intrinsics is not None else np.ones([3, 3])*-1
        with_camera_intrinsics = not (camera_intrinsics < 0).all()
        sample = {
                'image': image,
                'camera_intrinsics': camera_intrinsics,
                'image_path': image_file,
                'det3d_gt_maps': det3d_gt_maps,
                'seg_gt_maps': seg_gt_maps,
                'boxes_3d': boxes,
                # 'category_mapping': self.category_mapping,
                'with_camera_intrinsics': with_camera_intrinsics,
                'with_seg_label': os.path.exists(seg_label_file)
            }
        
        return sample
    
    def __len__(self):
        return len(self.annotations)
    
    def shuffle(self):
        random.shuffle(self.annotations)
    
    def __getitem__(self, idx):
        sample = self.get_sample(idx)
        if self.transform:
            sample = self.transform(sample)
        return sample