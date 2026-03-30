
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.loss import focal_loss, soft_focal_loss

from .backbone import Backbone, BevNeck
from .dinov3 import DINOv3Backbone, Dinov3BEVNeck


class FastBEV(nn.Module):
    def __init__(self, 
                 x_range=(-9.6, 9.6),    # x范围：右侧-9.6~9.6米（相机坐标系X轴）
                 z_range=(0.0, 32.0),      # z范围：前方0~32米（相机坐标系Z轴）
                 y_range=(-3.0, 2.0),      # y范围：高度-3~2米（相机坐标系Y轴）
                 pixel_per_meter_h=10,     # 水平每米像素数
                 pixel_per_meter_v=2,      # 竖直每米像素数
                 detection_channels=128,
                 det3d_output_channels=13,      # 检测3D的通道数量
                 seg_output_channel=3,    # lane/lane_divider/flag_cone
                 backbone='dinov3',
                 neck='dinov3',
                 generate_det_gt_format='gaussian',
                 catetory_num = 3):
        super(FastBEV, self).__init__()
        
        # 计算BEV网格尺寸
        self.x_range = x_range
        self.z_range = z_range
        self.y_range = y_range
        self.pixel_per_meter_h = pixel_per_meter_h
        self.pixel_per_meter_v = pixel_per_meter_v
        self.category_num = catetory_num
        self.generate_det_gt_format = generate_det_gt_format
        self.det3d_output_channels = det3d_output_channels
        self.target_size = [816,960]
        
        
        # 计算网格数量
        self.bev_width = int(round((x_range[1] - x_range[0]) * pixel_per_meter_h))   # x方向网格数（左右）
        self.bev_height = int(round((z_range[1] - z_range[0]) * pixel_per_meter_h))  # z方向网格数（上下）
        self.num_y_planes = int(round((y_range[1] - y_range[0]) * pixel_per_meter_v))  # y方向平面数（高度）
        
        if self.generate_det_gt_format == 'gaussian':
            self.det_focal_loss = soft_focal_loss
        elif self.generate_det_gt_format == 'standard':
            self.det_focal_loss = focal_loss
            
        self.seg_focal_loss = focal_loss
        
        print(f"BEV网格配置 (数据集对齐坐标系):")
        print(f"  X方向(左右): {x_range[0]:.1f}m ~ {x_range[1]:.1f}m, 网格大小: {pixel_per_meter_h:.2f}m, 网格数: {self.bev_width}")
        print(f"  Z方向(上下): {z_range[0]:.1f}m ~ {z_range[1]:.1f}m, 网格大小: {pixel_per_meter_h:.2f}m, 网格数: {self.bev_height}")
        print(f"  Y方向(高度): {y_range[0]:.1f}m ~ {y_range[1]:.1f}m, 网格大小: {pixel_per_meter_v:.2f}m, 平面数: {self.num_y_planes}")
        print(f"  坐标系：原点在底部中心，向右为X正方向，向上为Z正方向")
        
        # Backbone网络
        self.backbone = DINOv3Backbone(in_channels=3) if backbone == 'dinov3' else Backbone(in_channels=3)

        proj_channels = 16
        self.proj_to_bev = nn.ModuleList([
            nn.Conv2d(self.backbone.channels, proj_channels, kernel_size=1, bias=False) for _ in range(self.num_y_planes)
        ])
        
        # BEV特征融合
        self.bev_fusion = Dinov3BEVNeck(proj_channels * self.num_y_planes, detection_channels) if neck == 'dinov3' \
                        else BevNeck(proj_channels * self.num_y_planes, detection_channels, inner_channels=32, expand_ratio=3)
        
        self.det3d_head = nn.Sequential(
            nn.Conv2d(detection_channels, detection_channels, 1),
            nn.ReLU(),
            nn.Conv2d(detection_channels, det3d_output_channels, 1)
        )
        
        self.seg_grass_head = nn.Sequential(
            nn.Conv2d(detection_channels, detection_channels, 1),
            nn.ReLU(),
            nn.Conv2d(detection_channels, seg_output_channel, 1)
        )
        
        
        # Focal Loss参数
        self.alpha = 0.25
        self.gamma = 2.0
        
        # 预计算的采样坐标（部署时固定）
        self.register_buffer('sampling_coords', None)
        
    def _get_bev_grid(self, batch_size, device):
        """生成BEV网格坐标（XZ平面）
        坐标系：原点在底部中心，向右为x，向上为z
        """
        # 计算网格大小
        x_grid_size = (self.x_range[1] - self.x_range[0]) / self.bev_width
        z_grid_size = (self.z_range[1] - self.z_range[0]) / self.bev_height
        
        # 生成网格坐标
        # x方向：从x_range[0]到x_range[1]，向右为正
        x_range = torch.arange(self.bev_width, device=device) * x_grid_size + self.x_range[0] + 0.5 * x_grid_size
        # z方向：从z_range[0]到z_range[1]，向上为正（注意：z_range[0]对应图像底部）
        z_range = torch.arange(self.bev_height, device=device) * z_grid_size + self.z_range[0] + 0.5 * z_grid_size
        
        # 创建网格，注意z方向需要翻转以匹配图像坐标系
        # 图像坐标系：原点在左上角，向下为y正方向
        # BEV坐标系：原点在底部中心，向上为z正方向
        x_grid, z_grid = torch.meshgrid(x_range, z_range, indexing='xy')
        
        # 翻转z方向，使z_range[0]对应图像底部，z_range[1]对应图像顶部
        z_grid = z_grid.flip(dims=[0])
        
        # 扩展为batch维度
        x_grid = x_grid.unsqueeze(0).expand(batch_size, -1, -1)
        z_grid = z_grid.unsqueeze(0).expand(batch_size, -1, -1)
        
        return x_grid, z_grid
    
    def _compute_sampling_coordinates(self, camera_intrinsics, image_height, image_width):
        """预计算采样坐标（归一化到-1到1范围）"""
        batch_size = camera_intrinsics.shape[0]
        device = camera_intrinsics.device
        
        # 生成BEV网格（XZ平面）
        x_grid, z_grid = self._get_bev_grid(batch_size, device)
        y_grid_size = (self.y_range[1] - self.y_range[0]) / self.num_y_planes
        
        # 定义Y平面高度
        y_heights = torch.arange(self.num_y_planes, device=device) * y_grid_size + self.y_range[0] + 0.5 * y_grid_size
        
        # 存储所有y平面的采样坐标
        sampling_coords_list = []
        
        for y_height in y_heights:
            y_cam = y_height * torch.ones_like(x_grid)
            
            # 计算投影坐标
            coords = self._project_to_image(x_grid, y_cam, z_grid, camera_intrinsics, image_height, image_width)
            sampling_coords_list.append(coords)
        
        # 拼接所有y平面的坐标 [B, num_y_planes, H_bev, W_bev, 2]
        sampling_coords = torch.stack(sampling_coords_list, dim=1)
        
        return sampling_coords
    
    def _project_to_image(self, x_cam, y_cam, z_cam, camera_intrinsics, image_height, image_width):
        """将相机坐标系下的3D点投影到图像平面并归一化到-1到1"""
        batch_size = x_cam.shape[0]
        
        # 相机内参矩阵 [B, 3, 3]
        fx = camera_intrinsics[:, 0, 0].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        fy = camera_intrinsics[:, 1, 1].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        cx = camera_intrinsics[:, 0, 2].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        cy = camera_intrinsics[:, 1, 2].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        
        # 相机坐标系：
        # X轴：向右为正
        # Y轴：向下为正  
        # Z轴：向前为正
        
        # 投影公式：u = fx * X/Z + cx, v = fy * Y/Z + cy
        # 注意：需要处理Z=0的情况，避免除零错误
        z_cam_safe = torch.clamp(z_cam, min=0.01)  # 避免Z=0
        
        # 计算像素坐标
        u = fx * x_cam / z_cam_safe + cx
        v = fy * y_cam / z_cam_safe + cy
        
        # 归一化到[-1, 1]范围，用于grid_sample
        # 注意：grid_sample的坐标范围是[-1, 1]，其中：
        # -1 对应图像左边界/上边界
        # 1  对应图像右边界/下边界
        u_norm = 2.0 * (u / image_width) - 1.0
        v_norm = 2.0 * (v / image_height) - 1.0
        
        # 组合坐标 [B, H_bev, W_bev, 2]
        coords = torch.stack([u_norm, v_norm], dim=-1)
        
        return coords
    
    def _sample_features_with_coords(self, features, sampling_coords):
        """使用预计算的坐标采样特征"""
        batch_size, num_y_planes, bev_height, bev_width, _ = sampling_coords.shape
        
        # 重塑坐标以匹配grid_sample的期望格式
        # grid_sample期望: [B, H, W, 2] 的grid，输入: [B, C, H, W]
        # 我们可以直接使用batch维度，无需展开
        
        sampled_features_list = []
        for y in range(num_y_planes):
            # 提取当前y平面的坐标 [B, H_bev, W_bev, 2]
            coords = sampling_coords[:, y]
            
            cur_features = self.proj_to_bev[y](features)
            # 直接采样特征，保持batch维度
            coords = torch.clip(coords, -1.1, 1.1)  # 确保坐标在[-1, 1]范围内, 主要是为了量化考虑进行clip
            coords = torch.from_numpy(coords.cpu().numpy()).to(cur_features.device) # 为了量化编译避开torch.clip算子编译不通过的情况
            sampled = F.grid_sample(cur_features, coords.to(cur_features.dtype), 
                                  mode='bilinear', padding_mode='zeros', align_corners=False)
            # sampled: [B, C, H_bev, W_bev]

            # import cv2
            # import numpy as np
            # cv2.imwrite(f'sampled_{y}.png', (sampled[0].abs().sum(dim=0) > 0.0001).cpu().numpy().astype(np.uint8) * 255)
            
            # 重塑为 [B, C, H_bev, W_bev]
            sampled_features_list.append(sampled)
        
        # 拼接所有y平面的特征 [B, C*num_y_planes, H_bev, W_bev]
        sampled_features = torch.cat(sampled_features_list, dim=1)
        
        return sampled_features
    
    def export(self, x):
        image_width = x.shape[-1] // 2
        images = x[:,:,:,:image_width]
        # target_height, target_width = self.target_size
        # padding_h, padding_width = images.shape[-2]-target_height, images.shape[-1]-target_width
        # images = images[:, :, padding_h:, padding_width:]
        images = F.interpolate(images, size=(self.target_size[0], self.target_size[1]), mode="bilinear", align_corners=False)
        return self._forward(images, None, onnx_export=True)
    
    def forward(self, data_dict):
        images = data_dict['image']
        camera_intrinsics = data_dict['camera_intrinsics']
        
        det3d_heatmap_pred, seg_pred = self._forward(images, camera_intrinsics)
        results = {
            'det3d_pred': det3d_heatmap_pred,
            'seg_pred': seg_pred,
        }
        return results
    
    def _forward(self, images, camera_intrinsics, onnx_export=False):
        """
        Args:
            images: [B, C, H, W] 输入图像
            camera_intrinsics: [B, 3, 3] 相机内参矩阵
        Returns:
            class_pred: [B, 1, H_bev, W_bev] 类别预测
            reg_pred: [B, 4, H_bev, W_bev] 回归预测
        """
        batch_size, _, image_height, image_width = images.shape
        
        # 1. 通过backbone提取特征
        x_4x_fused = self.backbone(images)
        seg_pred = self.seg_grass_head(x_4x_fused)
        
        # 2. 计算或使用预计算的采样坐标
        if self.training:
            assert camera_intrinsics is not None, "camera_intrinsics is required in training mode"
            sampling_coords = self._compute_sampling_coordinates(camera_intrinsics, self.target_size[0], self.target_size[1])
        else:
            sampling_coords = self._compute_sampling_coordinates(camera_intrinsics, self.target_size[0], self.target_size[1]) if self.sampling_coords is None else self.sampling_coords
        
        # 3. 使用相同的采样坐标采样不同尺度的特征
        bev_features_4x = self._sample_features_with_coords(x_4x_fused, sampling_coords)
        
        # 4. 融合特征
        bev_features = self.bev_fusion(bev_features_4x)
    
        # 5. 输出预测
        det3d_heatmap_pred = self.det3d_head(bev_features)
        if onnx_export:
            det3d_heatmap_pred = det3d_heatmap_pred.permute([0,2,3,1])
            seg_pred = seg_pred.permute([0,2,3,1])
        return det3d_heatmap_pred, seg_pred
    
    
    def get_loss(self, results, data_dict):
        """
        Args:
            results: 包含seg_lane_pred, det3d_heatmap_pred, det_reg_pred
            data_dict: 包含heatmap, regression_map
        """
        with_camera_intrinsics = data_dict['with_camera_intrinsics']
        with_seg_label = data_dict['with_seg_label']
        det3d_pred = results['det3d_pred']
        seg_pred = results['seg_pred']
        
        det3d_gt = data_dict['det3d_gt_maps']
        seg_gt = data_dict['seg_gt_maps']
        
        loss_info = {}
        
        # judge the gt is pseudo or not with the channel for det
        det3d_gt = det3d_gt[with_camera_intrinsics]
        det3d_pred = det3d_pred[with_camera_intrinsics]
        if len(det3d_gt) == 0:
            for tmp_key in ['center_loss', 'offset_loss', 'height_loss', 'lwh_loss', 'yaw_loss']:
                loss_info[tmp_key] = torch.tensor(0.0).to(det3d_pred.device)
        else:
            loss_dict = {}
            loss_dict['center_loss'] = []
            loss_dict['offset_loss'] = []
            loss_dict['height_loss'] = []
            loss_dict['lwh_loss'] = []
            loss_dict['yaw_loss'] = []
            
            B, num_det_classes, chanels, H, W = det3d_gt.shape
            det3d_pred = det3d_pred.view(B, self.category_num, chanels, H, W)
            det3d_gt = det3d_gt.view(B, self.category_num, chanels, H, W)
            
            det_foreground_weight = (det3d_gt[:, :, 0:1, :, :] > 0.5).float()
            center_loss = self.det_focal_loss(det3d_pred[:, :, 0:1, :, :], det3d_gt[:, :, 0:1, :, :])
            
            offset_loss = F.l1_loss(det3d_pred[:, :, 1:3, :, :], det3d_gt[:, :, 1:3, :, :], reduction='none')
            offset_loss = (offset_loss * det_foreground_weight).sum() / (det_foreground_weight.sum() * 2.0 + 1e-6)
            
            height_loss = F.l1_loss(det3d_pred[:, :, 3:4, :, :], det3d_gt[:, :, 3:4, :, :], reduction='none')
            height_loss = (height_loss * det_foreground_weight).sum() / (det_foreground_weight.sum() + 1e-6)
            
            lwh_loss = F.l1_loss(det3d_pred[:, :, 4:7, :, :], det3d_gt[:, :, 4:7, :, :], reduction='none')
            lwh_loss = (lwh_loss * det_foreground_weight).sum() / (det_foreground_weight.sum() * 3.0 + 1e-6)
            
            # Yaw loss改进：使用cosine similarity loss替代L1 loss
            # sin(θ), cos(θ)应该满足单位圆约束，L1 loss无法保证这一点
            # 使用1 - cos_similarity可以更好地约束角度预测
            yaw_pred = det3d_pred[:, :, 7:, :, :]  # [B, C, 2, H, W]
            yaw_gt = det3d_gt[:, :, 7:, :, :]
            
            # 归一化预测值到单位圆
            yaw_pred_norm = F.normalize(yaw_pred, p=2, dim=2)  # 在sin/cos维度归一化
            yaw_gt_norm = F.normalize(yaw_gt, p=2, dim=2)
            
            # Cosine similarity loss: 1 - cos(pred, gt)
            # cos_sim = (pred · gt) / (||pred|| ||gt||), 范围[-1, 1]
            # loss = 1 - cos_sim, 范围[0, 2], 完全对齐时为0
            cos_sim = (yaw_pred_norm * yaw_gt_norm).sum(dim=2, keepdim=True)  # [B, C, 1, H, W]
            yaw_loss = (1.0 - cos_sim) * det_foreground_weight
            yaw_loss = yaw_loss.sum() / (det_foreground_weight.sum() + 1e-6)
            
            loss_dict['center_loss'] = center_loss
            loss_dict['offset_loss'] = offset_loss
            loss_dict['height_loss'] = height_loss
            loss_dict['lwh_loss'] = lwh_loss
            loss_dict['yaw_loss'] = yaw_loss
            
            # calculate det loss
            # 权重设置原则（基于实际训练数据调整）：
            # 1. center_loss: ~0.015 → ×15 = 0.225，确保中心点检测得到足够重视
            # 2. offset_loss: ~0.17 → ×2 = 0.34，精确定位关键
            # 3. height_loss: ~0.06 → ×1.5 = 0.09，适度提升
            # 4. lwh_loss: ~0.06 → ×0.3 = 0.018，数据集尺寸变化小，降低权重
            # 5. yaw_loss: ~0.09 → ×2.0 = 0.18，提升权重保证朝向稳定性
            for loss_key, loss_weight in zip(['center_loss', 'offset_loss', 'height_loss', 'lwh_loss', 'yaw_loss'],
                                [15.0, 2.0, 1.5, 0.3, 2.0]):
                loss_info[loss_key] = loss_weight * loss_dict[loss_key]
        
        seg_weights = [1]
        seg_pos_weights = [1]
        # seg_info = ['road_loss', 'grass_loss', 'branch_loss'] 
        seg_info = ['branch_loss'] 
        seg_gt = seg_gt[with_seg_label]
        seg_pred = seg_pred[with_seg_label]
        
        if len(seg_gt) == 0:
            for i in range(len(seg_weights)):
                loss_info[seg_info[i]] = torch.tensor(0.0).to(seg_pred.device)
        else:
            gt_seg_mask = seg_gt
            pd_seg_mask = seg_pred
            
            for i in range(len(seg_weights)):
                cur_seg_loss = self.seg_focal_loss(pd_seg_mask[:,i], gt_seg_mask[:,i], pos_weight=seg_pos_weights[i])
                cur_seg_loss *= seg_weights[i]
                loss_info[seg_info[i]] = cur_seg_loss
                
        total_loss = sum(loss_info.values())
        return total_loss, loss_info

    def decode_det_predictions(self, det3d_pred, score_threshold=0.3):
        """
        解码预测结果
        Args:
            det3d_pred: 1 * pd_channels * feature_height * feature_width
        Returns:
            decoded_boxes: List[List[Dict]] 解码后的3D边界框
        """
        det3d_pred = det3d_pred.squeeze()
        channels_each_group = det3d_pred.shape[0] // self.category_num
        all_decoded_boxes = []
        for i in range(self.category_num):
            det_channels = det3d_pred[i*channels_each_group:(i+1)*channels_each_group,:,:]
            center_map = det_channels[0:1,:,:]
            offset_map = det_channels[1:3,:,:]
            height_map = det_channels[3:4,:,:]
            lwh_map = det_channels[4:7,:,:]
            yaw_map = det_channels[7:9,:,:]
            
            decoded_boxes = []
            self.decode_center_map(center_map, decoded_boxes)
            self.decode_offset_map(offset_map, decoded_boxes)
            self.decode_lwh_map(lwh_map, decoded_boxes)
            self.decode_height_map(height_map, decoded_boxes)
            self.decode_yaw_map(yaw_map, decoded_boxes)
            for obj in decoded_boxes:
                obj['class_id'] = i
            all_decoded_boxes.extend(decoded_boxes)
        return all_decoded_boxes
    
    
    
    def decode_center_map(self, center_map, objects, threshold=0.3):
        center_map = center_map.sigmoid()
        centers = self.decode_center_points(center_map, threshold, distance=5)
        for center in centers:
            obj = {'center_int': (int(center[0]), int(center[1])),
                    'score': float(center[2])}
            objects.append(obj)
        return objects 
    
    
    def decode_offset_map(self, offset_map, objects):
        for obj in objects:
            x, y = obj['center_int']
            offset_x, offset_y = offset_map[:, y, x]
            offset_x, offset_y = float(offset_x), float(offset_y)
            offset_x, offset_y = max(0, min(offset_x, 1)), max(0, min(offset_y, 1))
            obj['center'] = (x + offset_x, y + offset_y)
        return objects
    
    
    def decode_lwh_map(self, lwh_map, objects):
        for obj in objects:
            x, y = obj['center_int']
            l, w, h = [float(elem) for elem in lwh_map[:, y, x]]
            obj['wlh'] = [w,l,h]
        return objects
    
    
    def decode_height_map(self, height_map, objects):
        for obj in objects:
           x, y = obj['center_int']
           height = height_map[:, y, x]
           obj['height'] = float(height)
        return objects

    def decode_yaw_map(self, yaw_map, objects):
        for obj in objects:
            x, y = obj['center_int']
            sin_val, cos_val = yaw_map[:, y, x]
            yaw = torch.atan2(sin_val, cos_val)
            obj['yaw'] = float(yaw)
        return objects
    
    def decode_category_map(self, category_map, objects):    
        for obj in objects:
            x, y = obj['center_int']
            category_vec = category_map[:, y, x]
            cur_category_id = category_vec.argmax()
            obj['class_id'] = int(cur_category_id)
        return objects
    
    
    def decode_center_points(self, center_map, threshold=0.3, distance=2):
        center_map = center_map.squeeze()
        ys, xs = torch.where(center_map >= threshold)
        if len(xs) == 0:
                return torch.zeros((0, 3), dtype=torch.float32)
        centers = torch.zeros((len(xs), 3), dtype=torch.float32)
        centers[:, 0] = xs
        centers[:, 1] = ys
        centers[:, 2] = center_map[ys, xs]
        centers = self.perform_nms_on_points(centers, distance)
        return centers

    
    def perform_nms_on_points(self, points, min_distance):
        number = len(points)
        if number <= 1:
            return points
        indices = torch.argsort(-points[:, 2])
        points = points[indices]
        is_valid = torch.ones(number, dtype=torch.bool)
        min_distance = min_distance ** 2
        for i in range(0, number - 1):
            x1, y1, _ = points[i]
            for j in range(i + 1, number):
                if is_valid[j] == False:
                    continue
                x2, y2, _ = points[j]
                dist = (x1 - x2) ** 2 + (y1 - y2) ** 2
                if dist < min_distance:
                    is_valid[j] = False
        points = points[is_valid]
        return points
    
    
    def _decode_single_box(self, reg_values, x, y, class_id, score):
        """
        解码单个3D边界框
        Args:
            reg_values: [8] 回归值 [x, y, z, l, w, h, sin_yaw, cos_yaw]
            x, y: 像素坐标
            class_id: 类别ID
            score: 置信度分数
        Returns:
            box: Dict 3D边界框信息
        """
        # 将像素坐标转换为世界坐标
        world_x = (x.item() - self.bev_width // 2) / self.pixel_per_meter_h
        world_z = (self.bev_height - 1 - y.item()) / self.pixel_per_meter_h
        
        # 获取回归值
        x_offset = reg_values[0].item()
        y_offset = reg_values[1].item()
        z = reg_values[2].item()
        length = reg_values[3].item()
        width = reg_values[4].item()
        height = reg_values[5].item()
        sin_yaw = reg_values[6]
        cos_yaw = reg_values[7]
        
        # 计算最终的世界坐标
        final_x = world_x + x_offset
        final_z = world_z + y_offset
        print('---------------------------------------------\n\n\n\n\n', final_x, world_x, x_offset)
        
        # 计算yaw角度
        yaw = torch.atan2(sin_yaw, cos_yaw).item()
        
        # 检查边界框尺寸的合理性
        if length <= 0 or width <= 0 or height <= 0:
            return None
        
        # 创建边界框字典
        box = {
            'center': [final_x, z, final_z],
            'wlh': [width, length, height],
            'yaw': yaw,
            'class_id': class_id,
            'score': score.item()
        }
        
        return box
    

def create_fastbev_model(x_range=(-15.0, 15.0), z_range=(0.0, 32.0), 
                         y_range=(-2.0, 3.0), pixel_per_meter_h=10,
                        pixel_per_meter_v=5, 
                        detection_channels = 128,
                        det3d_output_channels = 27,
                        seg_output_channel = 1,
                        backbone='',
                        neck='',
                        generate_det_gt_format='gaussian',
                        catetory_num = 3
                        ):
    """创建FastBEV模型的便捷函数"""
    return FastBEV(
        x_range=x_range,
        z_range=z_range,
        y_range=y_range,
        pixel_per_meter_h=pixel_per_meter_h,
        pixel_per_meter_v=pixel_per_meter_v,
        detection_channels=detection_channels,
        det3d_output_channels=det3d_output_channels,
        seg_output_channel=seg_output_channel,
        backbone=backbone,
        neck=neck,
        generate_det_gt_format=generate_det_gt_format,
        catetory_num=catetory_num
    )
