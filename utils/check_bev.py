import copy
import glob
import json
import os

import cv2
import numpy as np
import torch

from utils.preprocess import preprocess_image


class BEVVisualizer:
    def __init__(self, width_m=30, height_m=30, pixel_per_meter=10, 
                 seg_lane_threshold=0.5, det3d_threshold=0.3, seg_score_list=[0.5,0.5,0.5],
                 distance_threshold=None):
        """
        初始化BEV可视化器
        Args:
            width_m: BEV图像宽度（米），左右各15米
            height_m: BEV图像高度（米），前方30米
            pixel_per_meter: 每米对应的像素数
            distance_threshold: 距离过滤阈值（米），超过该距离的box不显示，None表示不过滤
        """
        self.width_m = width_m
        self.height_m = height_m
        self.pixel_per_meter = pixel_per_meter
        self.seg_lane_threshold = seg_lane_threshold
        self.det3d_threshold = det3d_threshold
        self.seg_score_list = seg_score_list
        self.distance_threshold = distance_threshold
        
        # 计算图像尺寸
        self.img_width = int(width_m * pixel_per_meter)  # 300像素
        self.img_height = int(height_m * pixel_per_meter)  # 300像素
        
        # 相机坐标系：右(X)、下(Y)、前(Z)
        # BEV图像坐标系：X右为正，Y下为正
        # 相机位置在BEV图像底部中心，对应相机坐标系原点(0, 0, 0), 其在 bev 坐标系下坐标如下：
        self.cam_bev_x = self.img_width // 2  # 150 (水平居中，对应相机坐标X=0)
        self.cam_bev_y = self.img_height - 1  # 299 (底部，对应相机坐标Z=0，前方向上)
        self.seg_channels = ['drivable_area', 'ped_crossing', 'walkway',
                             'road_divider', 'lane_divider']                # set the channel
        
        # 类别颜色映射
        self.category_colors = {
            'vehicle': (128, 0, 128),     # 紫色
            'person': (0, 128, 128),  # 深青色
            'cyclelist': (128, 128, 0),     # 橄榄色
        }
        
        # seg category colors
        self.seg_colors = {
            'lane' : [0,255,0], 
            'grass': [0,255,255], 
            'branch': [0,0,255], 
        }
        
        # 默认颜色
        self.default_color = (128, 128, 128)

         # 预计算距离阈值对应的像素阈值，避免在循环中重复计算
        self.pixel_threshold = self._compute_pixel_threshold(distance_threshold)
    
    def _world_to_pixel(self, x_world, z_world):
        """
        将世界坐标（相机坐标系）转换为BEV像素坐标
        Args:
            x_world: 相机坐标系X坐标（米），右为正
            z_world: 相机坐标系Z坐标（米），前为正
        Returns:
            (x_pixel, y_pixel): BEV图像像素坐标
        """
        # 相机坐标系：X右为正，Y下为正，Z前为正
        # BEV像素坐标系：X右为正，Y下为正（相机在底部）
        x_pixel = int(self.cam_bev_x + x_world * self.pixel_per_meter)
        y_pixel = int(self.cam_bev_y - z_world * self.pixel_per_meter)
        return x_pixel, y_pixel
    
    def _compute_pixel_threshold(self, distance_m):
        """
        将物理距离（米）转换为BEV像素坐标的y阈值（内部方法，初始化时调用）
        
        Args:
            distance_m: 距离阈值（米），相机坐标系Z轴方向
        
        Returns:
            pixel_threshold: BEV像素坐标的y阈值，小于该y值的box会被过滤
        
        说明：
            在BEV图像中：
            - 相机位置在底部（cam_bev_y = img_height - 1）
            - 前方距离越远，y坐标越小
            - 因此：pixel_threshold = bbox_bev_y - distance_m * pixel_per_meter
            - 只保留 bbox_bev_y >= pixel_threshold 的box（即距离 <= distance_m）
            - 小于 pixel_threshold 的box会被过滤（距离 > distance_m）
        """
        if distance_m is None:
            return None

        # 在BEV图像中，cam_bev_y 是底部，减去距离对应的像素值得到阈值
        pixel_threshold = self.cam_bev_y - distance_m * self.pixel_per_meter
        return pixel_threshold
    
    def filter_boxes_by_fov(self, data, data_root, distance_threshold=30):
        """
        根据FOV和距离过滤边界框
        """
        filtered_data = data.copy()
        
        # 获取相机内参
        cam_intrinsic = np.array(data['cam_intrinsic'])
        fx = cam_intrinsic[0, 0]
        fy = cam_intrinsic[1, 1]
        cx = cam_intrinsic[0, 2]
        cy = cam_intrinsic[1, 2]
        
        # 读取图像获取尺寸
        image_path = os.path.join(data_root, data['image_path'])
        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]
        
        valid_boxes = []
        total_boxes = len(data['cam_bbox_3d'])
        
        for i, bbox in enumerate(data['cam_bbox_3d']):
            center = np.array(bbox['center'])
            
            # 检查距离
            distance = center[2]
            if distance > distance_threshold:
                print(f"  Box {i+1} ({bbox['category']}) filtered: distance {distance:.1f}m > {distance_threshold}m")
                continue
            
            # 检查是否在相机前方
            if center[2] <= 0:
                print(f"  Box {i+1} ({bbox['category']}) filtered: behind camera (z={center[2]:.1f}m)")
                continue
            
            # 投影到2D图像坐标
            x = center[0] / center[2]
            y = center[1] / center[2]
            u = fx * x + cx
            v = fy * y + cy
            
            # 检查图像边界
            if (u < 0 or u >= img_width or 
                v < 0 or v >= img_height):
                print(f"  Box {i+1} ({bbox['category']}) filtered: outside image bounds (u={u:.1f}, v={v:.1f})")
                continue
            
            valid_boxes.append(bbox)
            print(f"  Box {i+1} ({bbox['category']}) kept: distance={distance:.1f}m")
        
        filtered_data['cam_bbox_3d'] = valid_boxes
        print(f"\nFiltering results: {len(valid_boxes)}/{total_boxes} boxes kept")
        
        return filtered_data
    
    def draw_rotated_rectangle(self, img, center, width, height, angle, color, thickness=2):
        """
        绘制旋转矩形
        Args:
            img: 图像
            center: 中心点 (x, y)
            width: 宽度（像素）
            height: 高度（像素）
            angle: 旋转角度（弧度）
            color: 颜色
            thickness: 线宽
        """
        # 计算旋转矩形的四个顶点
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # 相对于中心的四个顶点
        corners = np.array([
            [-width/2, -height/2],
            [width/2, -height/2],
            [width/2, height/2],
            [-width/2, height/2]
        ])
        
        # 旋转矩阵
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [-sin_a, -cos_a] # 这里需要修正，因为y轴朝上，行号变化应该是减法
        ])
        
        # 旋转顶点
        rotated_corners = np.dot(corners, rotation_matrix.T)
        
        # 平移到世界坐标
        world_corners = rotated_corners + np.array(center)
        
        # 转换为整数坐标
        corners_int = world_corners.astype(np.int32)
        
        # 绘制矩形
        cv2.polylines(img, [corners_int], True, color, thickness)
        
        return corners_int
    
    def draw_arrow(self, img, start_point, end_point, color, thickness=2):
        """
        绘制箭头
        """
        cv2.arrowedLine(img, start_point, end_point, color, thickness, 
                        tipLength=0.3, line_type=cv2.LINE_AA)
    
    def draw_distance_grid(self, bev_img, interval_m=5):
        """
        在BEV图像上绘制距离标尺线
        
        Args:
            bev_img: BEV图像
            interval_m: 距离间隔（米），默认5米
        """
        # 标尺线颜色和样式
        line_color = (100, 100, 100)  # 灰色
        text_color = (200, 200, 200)  # 浅灰色
        line_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        # 从相机位置开始，每隔interval_m米绘制一条横线
        distance = interval_m
        while distance < self.height_m:
            # 计算该距离对应的BEV图像y坐标
            y_pixel = int(self.cam_bev_y - distance * self.pixel_per_meter)
            
            # 如果y坐标在图像范围内
            if 0 <= y_pixel < self.img_height:
                # 绘制横线（从左到右）
                cv2.line(bev_img, (0, y_pixel), (self.img_width, y_pixel), 
                        line_color, line_thickness, cv2.LINE_AA)
                
                # 在左侧添加距离标注
                label = f"{distance}m"
                # 计算文本尺寸以便更好地定位
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thickness)
                
                # 文本位置：左侧边缘，略微偏上
                text_x = 5
                text_y = y_pixel - 5
                
                # 确保文本不会超出图像范围
                if text_y - text_height < 0:
                    text_y = y_pixel + text_height + 5
                
                # 绘制文本背景（半透明黑色矩形）
                overlay = bev_img.copy()
                cv2.rectangle(overlay, 
                            (text_x - 2, text_y - text_height - 2),
                            (text_x + text_width + 2, text_y + baseline + 2),
                            (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, bev_img, 0.5, 0, bev_img)
                
                # 绘制距离标注文本
                cv2.putText(bev_img, label, (text_x, text_y), 
                           font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            
            distance += interval_m
    
    def create_bev_frame(self, data):
        """
        创建单帧BEV图像
        """
        # 创建空白图像
        bev_img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        
        # 处理每个3D边界框
        if 'cam_bbox_3d' in data:
            bboxes = data['cam_bbox_3d']
        elif 'bev_bbox_3d' in data:
            bboxes = data['bev_bbox_3d']
            
        for bbox in bboxes:
            center = np.array(bbox['center'])
            wlh = np.array(bbox['wlh'])
            yaw = bbox['yaw']
            
            # 世界坐标转换为像素坐标
            if 'cam_bbox_3d' in data:
                center_x, center_y = self._world_to_pixel(center[0], center[2])
            else:
                center_x, center_y = center.astype(np.int32)
            
            # 距离过滤：检查是否超过距离阈值
            if self.pixel_threshold is not None:
                # 在BEV图像中，y越小表示距离越远
                # center_y < pixel_threshold 表示距离 > distance_threshold，需要过滤
                if center_y < self.pixel_threshold:
                    continue  # 跳过超出距离的box（小于阈值的会被过滤）
                
            # 检查是否在图像范围内
            if (0 <= center_x < self.img_width and 0 <= center_y < self.img_height):
                # 获取类别和颜色
                category = bbox['category']
                color = self.category_colors.get(category, self.default_color)
                
                # 计算尺寸（像素）
                # CORRECTED: 在BEV中：X方向对应height（长度），Z方向对应width（宽度）
                height_pixels = int(wlh[1] * self.pixel_per_meter)  # 长度（X方向）
                width_pixels = int(wlh[0] * self.pixel_per_meter)   # 宽度（Z方向）
                
                # 绘制旋转矩形 - 修正参数顺序
                self.draw_rotated_rectangle(bev_img, (center_x, center_y), 
                                        height_pixels, width_pixels, yaw, color, 2)
                
                # 绘制朝向箭头
                arrow_length = max(width_pixels, height_pixels) * 0.8
                arrow_end_x = int(center_x + arrow_length * np.cos(yaw))
                arrow_end_y = int(center_y - arrow_length * np.sin(yaw))
                
                # 确保箭头终点在图像范围内
                if (0 <= arrow_end_x < self.img_width and 0 <= arrow_end_y < self.img_height):
                    self.draw_arrow(bev_img, (center_x, center_y), 
                                (arrow_end_x, arrow_end_y), (255, 255, 255), 2)
                
                # 添加标签
                label = f"{category[:3]}"
                cv2.putText(bev_img, label, (center_x + 5, center_y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 绘制距离标尺线（每5米一条）
        self.draw_distance_grid(bev_img)
        
        return bev_img
    
    def visualize_seg_lane_res_in_frame(self, seg_maps):
        h, w = seg_maps.shape[1:]
        background_map = np.ones([h, w, 3]) * 255
        if not (seg_maps == -1).all():
            h, w = seg_maps.shape[1:]
            background_map = np.ones([h, w, 3]) * 255
            if isinstance(seg_maps, torch.Tensor):
                seg_maps = seg_maps.detach().cpu().numpy()
                    
            for idx, (cur_seg_map, cur_class) in enumerate(zip(seg_maps, self.seg_colors)):
                if idx == 0:                            # TODO: just set the config for lane-line, ignore other classes
                    ys, xs = np.where(cur_seg_map != 0)
                    background_map[ys,xs,:] = self.seg_colors[cur_class]
                else:
                    continue
                
        return background_map

    def create_combined_frame(self, data, is_gt=True, down_sample_factor=0.25):
        """
        创建原始图像和BEV图像的组合帧
        """
        # 创建BEV图像
        bev_img = self.create_bev_frame(data)
        if is_gt:
            fov_seg_maps = data['gt_seg_maps']
            if (fov_seg_maps == -1).all():             # no seg grass map
                fov_seg_maps = torch.zeros_like(fov_seg_maps).detach().cpu().numpy()
                down_sample_factor = 1.0
        else:
            if data['pred_seg_maps'] is not None:
                fov_seg_maps = data['pred_seg_maps']
            
        # 读取原始图像
        image_path = data['image_path']
        if os.path.exists(image_path):
            original_img = cv2.imread(image_path)
            original_img = preprocess_image(original_img)
            original_img = cv2.resize(original_img, dsize=None, fx=down_sample_factor, fy=down_sample_factor, interpolation=cv2.INTER_AREA)
            fov_mask = fov_seg_maps.astype(np.bool_)
            tmp_img = copy.deepcopy(original_img)
            for i in range(fov_mask.shape[0]):
                tmp_img[fov_mask[i]] = list(self.seg_colors.values())[i]
            original_img = cv2.addWeighted(original_img, 0.5, tmp_img, 0.5, 0)
            
            if original_img is not None:
                # 调整原始图像大小以匹配BEV图像高度
                target_height = self.img_height
                aspect_ratio = original_img.shape[1] / original_img.shape[0]
                target_width = int(target_height * aspect_ratio)
                
                # 如果调整后的宽度太宽，则按宽度调整
                if target_width > self.img_width * 2:  # 限制最大宽度
                    target_width = self.img_width * 2
                    target_height = int(target_width / aspect_ratio)
                
                original_img_resized = cv2.resize(original_img, (target_width, target_height))
                
                # 创建组合图像
                combined_height = max(self.img_height, target_height)
                combined_width = self.img_width + target_width
                combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
                
                # 放置BEV图像（左侧）
                combined_img[:self.img_height, :self.img_width] = bev_img
                
                # 放置原始图像（右侧）
                combined_img[:target_height, self.img_width:self.img_width + target_width] = original_img_resized
                final_combined_height, final_combined_width = combined_img.shape[:2]
                return combined_img, final_combined_width, final_combined_height
    
    def create_bev_video(self, labels_dir, data_root, output_path="bev_video.mp4", fps=10):
        """
        创建BEV视频
        """
        # 获取所有JSON文件
        json_files = sorted(glob.glob(os.path.join(labels_dir, "*.json")))
        
        if not json_files:
            print(f"No JSON files found in {labels_dir}")
            return
        
        print(f"Found {len(json_files)} JSON files")
        
        # 先处理第一帧来确定视频尺寸
        first_data = None
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    first_data = json.load(f)
                break
            except:
                continue
        
        if first_data is None:
            print("Failed to read any JSON files")
            return
        
        # 创建组合帧来确定尺寸
        _, combined_width, combined_height = self.create_combined_frame(first_data, data_root)
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, 
                                     (combined_width, combined_height))
        
        # 处理每一帧
        for i, json_file in enumerate(json_files):
            print(f"\nProcessing frame {i+1}/{len(json_files)}: {os.path.basename(json_file)}")
            
            try:
                # 读取JSON数据
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # 过滤边界框
                filtered_data = self.filter_boxes_by_fov(data, data_root, 30)
                
                # 创建组合帧
                combined_frame, _, _ = self.create_combined_frame(filtered_data, data_root)
                
                # # 添加帧信息
                # cv2.putText(combined_frame, f"Frame: {i+1}/{len(json_files)}", 
                #           (10, combined_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                #           0.6, (255, 255, 255), 2)
                
                # 写入视频
                video_writer.write(combined_frame)
                
                # 显示进度
                print(f"  Processed frame {i+1}")
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
        # 释放资源
        video_writer.release()
        print(f"\nBEV video saved to: {output_path}")

def main():
    # 配置路径
    labels_dir = "/perception/users/cfx/WorkSpace_2025/nova_fastbev/fastbev/mini_labels"
    data_root = "/perception/users/cfx/WorkSpace_2025/nova_fastbev/fastbev"
    
    
    
    
    # 创建BEV可视化器
    # 30米宽（左右各15米），30米高（前方30米），每米10像素
    bev_viz = BEVVisualizer(width_m=30, height_m=30, pixel_per_meter=10)
    
    # 创建BEV视频
    bev_viz.create_bev_video(labels_dir, data_root, "bev_video.mp4", fps=5)

if __name__ == "__main__":
    main()
