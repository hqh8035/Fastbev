import json
import logging
import os
import random
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

from utils.heatmap import get_gaussian_radius, set_gaussian_heat, set_standard_heat


class NuScenesBEVDataset(Dataset):
    def __init__(self, 
                 annotations_file: str, 
                 category_mapping: Dict = None,
                 bev_width_m: float = 30,
                 bev_height_m: float = 30,
                 pixel_per_meter: float = 10,
                 transform: Callable = None,
                 generate_det_gt_format: str = 'gaussian',
                ):
        """
        初始化NuScenesBEVDataset
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
        self.annotations_file = annotations_file
        self.category_mapping = category_mapping or self._get_default_category_mapping()
        self.transform = transform
        self.generate_det_gt_format = generate_det_gt_format
        
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
        """获取默认的类别映射"""
        # return [['car', 'truck', 'bus', 'trailer', 'construction_vehicle'],
        #         ['pedestrian'],
        #         ['motorcycle', 'bicycle'],
        #         ['barrier']]
        
        return [['car', 'truck', 'bus', 'trailer', 'construction_vehicle'],
                ['pedestrian'],
                ['motorcycle', 'bicycle']]
    
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
        cam_intrinsic = np.array(data['cam_intrinsic'])
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
        
    
    
    def generate_gt_maps(self, boxes: List[Dict]) -> np.ndarray:
        """
        Args:
            boxes: 过滤后的3D边界框列表
        Returns:
            heatmap: shape (num_classes, bev_height, bev_width)
            regression_map: shape (8, bev_height, bev_width) - [x, y, z, l, w, h, sin_yaw, cos_yaw]
        """
        num_classes = len(self.category_mapping)
        
        category_id_box_dict = {}
        
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
            if category_id not in category_id_box_dict:
                category_id_box_dict[category_id] = []
            category_id_box_dict[category_id].append(bbox)
            
        
        gt_map_list = []
        # each class with a group channels 
        for i in range(num_classes):
            # set temp maps to get current category bboxes gt maps
            center_map = np.zeros([1, self.bev_height, self.bev_width])         # set xy in center heatmap
            center_height_map = np.zeros([1, self.bev_height, self.bev_width])         # set height value in the center heatmap
            offset_map  = np.zeros([2, self.bev_height, self.bev_width])        # x-offset / y-offset
            lwh_map = np.zeros([3, self.bev_height, self.bev_width])            # set l,w,h value in the correspond position in the bev map
            yaw_map = np.zeros([2, self.bev_height, self.bev_width])            # set sin/cos of yaw value for afterwards regression    

            if i in category_id_box_dict:
                cur_bboxes = category_id_box_dict[i]
                for bbox in cur_bboxes:                     # bboxes that belong to certain category
                    center = np.array(bbox['center'])
                    wlh = np.array(bbox['wlh'])
                    yaw = bbox['yaw']
                    category_id = bbox['category_id']
                    
                    assert category_id == i, "category error...\n"
                    
                    # 世界坐标转换为像素坐标
                    center_x, center_y = self.world_to_pixel(center[0], center[2])
                    center_x_int = int(center_x)
                    center_y_int = int(center_y)
                    
                    # 检查是否在图像范围内
                    if (0 <= center_x_int < self.bev_width and 0 <= center_y_int < self.bev_height):
                        w, l, h = wlh 
                        size = [w*self.pixel_per_meter, l*self.pixel_per_meter]
                        center_map = self.encode_center_map(center_map, [center_x, center_y], size=size)
                        
                        offset_map[0, center_y_int, center_x_int] = center_x - center_x_int  # x偏移
                        offset_map[1, center_y_int, center_x_int] = center_y - center_y_int  # y偏移
                        
                        center_height_map[0, center_y_int, center_x_int] = center[1]  # y坐标
                        
                        lwh_map[0, center_y_int, center_x_int] = l   # bbox 长度           # 归一化看看是否有改善
                        lwh_map[1, center_y_int, center_x_int] = w   # bbox 宽度
                        lwh_map[2, center_y_int, center_x_int] = h  # bbox 高度
                        
                        # 角度回归：使用sin/cos
                        yaw_map[0, center_y_int, center_x_int] = np.sin(yaw)  # sin_yaw
                        yaw_map[1, center_y_int, center_x_int] = np.cos(yaw)  # cos_yaw
                    
            mask = (center_map > 0.999999)            # float type gt mask
            gt_maps = np.concatenate([mask, center_map, offset_map, center_height_map, lwh_map, yaw_map], 
                                 axis=0) 
            gt_map_list.append(gt_maps)
        
        finnal_gt_maps = np.concatenate(gt_map_list, axis=0)
        
        return finnal_gt_maps 

    
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
        
        bbox_annotation_file = self.annotations[idx]
        with open(bbox_annotation_file, 'r') as f:
            data = json.load(f)
            
        if self.load_seg_data:
            seg_annotation_file = bbox_annotation_file.replace('bbox3d_labels', 'seg_labels').replace('.json', '.npy')
            seg_info = np.load(seg_annotation_file)
            data['seg_mask'] = seg_info            
        
        # 读取图像
        image_path = data['image_path']
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # 过滤边界框
        filtered_data = self.filter_boxes_by_fov(data, image.shape[:2])
        boxes = filtered_data['cam_bbox_3d']
        
        # 生成GT maps
        det_gt_maps  = self.generate_gt_maps(boxes)
        if self.load_seg_data:
            seg_gt_maps = data['seg_mask']
        
        
        # debug here
        if self.debug:
            tmp_bev_img = np.zeros_like(det_gt_maps[0])
            for tmp_box in boxes:
                tmp_bev_corners = self.project_bbox3d_to_bev(tmp_box)
                tmp_bev_corners
                tmp_bev_img = cv2.polylines(tmp_bev_img, [tmp_bev_corners.reshape([-1,1,2]).astype(np.int32)], True, [255], thickness=2)
            
            cv2.imwrite('debug_mask.jpg', det_gt_maps[0]*255)
            cv2.imwrite('tmp_bev_img.jpg', tmp_bev_img*255)
        
        camera_intrinsics = np.array(data['cam_intrinsic'], dtype=np.float32)
        
        if self.load_seg_data:
            sample = {
                'image': image,
                'camera_intrinsics': camera_intrinsics,
                'image_path': image_path,
                'det_gt_maps': det_gt_maps,
                'seg_gt_maps': seg_gt_maps,
                'boxes_3d': boxes,
                'category_mapping': self.category_mapping,
            }
        else:
            sample = {
                'image': image,
                'camera_intrinsics': camera_intrinsics,
                'image_path': image_path,
                'det_gt_maps': det_gt_maps,
                'boxes_3d': boxes,
                'category_mapping': self.category_mapping,
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