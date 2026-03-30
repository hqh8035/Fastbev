import argparse
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append('.')

from configs import get_cfg_defaults
from datasets.transform import Compose, Normalize, Resize, ToTensor
from datasets.waymo.waymo_dataset import WaymoBEVDataset
from model.fastbev import FastBEV
from utils.check_bev import BEVVisualizer


class FastBEVDemo:
    def __init__(self, model_path, config_name, annotation_file,
        output_dir="demo_results", device="cuda",
        det3d_threshold = 0.3,
        seg_score_list = [0.5],
        save_result=True,
        distance_threshold=None):
        """
        初始化FastBEV演示器
        Args:
            model_path: 训练好的模型路径
            config_name: 配置名称
            output_dir: 输出目录
            device: 设备
            distance_threshold: 距离过滤阈值（米），超过该距离的box不显示，None表示不过滤
        """
        self.annotations_file = annotation_file
        self.model_path = model_path
        self.config_name = config_name
        self.output_dir = output_dir
        self.device = device
        self.det3d_threshold = det3d_threshold
        self.seg_score_list = seg_score_list
        self.save_result = save_result
        self.distance_threshold = distance_threshold
        
        # 类别名称映射
        self.class_names = ['vehicle', 'person', 'cyclist']
        
        # 加载配置
        self.cfg = get_cfg_defaults(config_name)
        self.cfg.freeze()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)
        
        # 初始化数据变换（与训练时完全一致）
        self._initialize_transforms()
        
        # 加载模型
        self.model = self._load_model()
        
        # 创建数据集
        self.dataset = self._create_dataset()
        
        # 创建BEV可视化器（使用配置中的参数）
        self.bev_viz = BEVVisualizer(
            width_m=self.cfg.MODEL.FASTBEV.X_RANGE[1] - self.cfg.MODEL.FASTBEV.X_RANGE[0],
            height_m=self.cfg.MODEL.FASTBEV.Z_RANGE[1] - self.cfg.MODEL.FASTBEV.Z_RANGE[0],
            pixel_per_meter=self.cfg.MODEL.FASTBEV.PIXEL_PER_METER_H,
            det3d_threshold=self.det3d_threshold,
            seg_score_list=self.seg_score_list,
            distance_threshold=self.distance_threshold
        )
        
        print(f"FastBEV Demo initialized")
        print(f"Model: {model_path}")
        print(f"Config: {config_name}")
        print(f"Annotations file: {self.annotations_file}")
        print(f"Dataset: {len(self.dataset)} samples")
        print(f"Output directory: {output_dir}")
        
        # 打印配置信息
        print(f"\nConfiguration:")
        print(f"  X Range: {self.cfg.MODEL.FASTBEV.X_RANGE}")
        print(f"  Z Range: {self.cfg.MODEL.FASTBEV.Z_RANGE}")
        print(f"  Y Range: {self.cfg.MODEL.FASTBEV.Y_RANGE}")
        print(f"  Pixel per meter (H): {self.cfg.MODEL.FASTBEV.PIXEL_PER_METER_H}")
        print(f"  Pixel per meter (V): {self.cfg.MODEL.FASTBEV.PIXEL_PER_METER_V}")
        print(f"  Detection channels: {self.cfg.MODEL.FASTBEV.DETECTION_CHANNELS}")
        print(f"  Det3D channels: {self.cfg.MODEL.FASTBEV.DET3D_OUT_CHANNELS}")
        print(f"  Seg channels: {self.cfg.MODEL.FASTBEV.SEG_OUT_CHANNELS}")
        print(f"  Image size: {self.cfg.DATASET.TRANSFORM.val.shape}")
    
    def _initialize_transforms(self):
        """初始化数据变换（与训练时完全一致）"""
        transform_config = self.cfg.DATASET.TRANSFORM
        
        # 获取验证/测试的配置
        val_config = transform_config.get('val', {})
        
        if 'shape' not in val_config:
            raise ValueError("Transform shape is required for validation")
        
        # 验证/测试模式的变换（与训练时完全一致）
        val_transforms = [
            Resize(size=val_config['shape']),  # 首先resize到指定尺寸
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        self.val_transform = Compose(val_transforms)
        
        print(f"Validation transforms initialized:")
        print(f"  Target shape: {val_config['shape']}")
        print(f"  Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")
    
    def _load_model(self):
        """加载训练好的模型"""
        print(f"Loading model from {self.model_path}")
        
        # 使用配置中的参数创建模型
        model = FastBEV(
            x_range=self.cfg.MODEL.FASTBEV.X_RANGE,
            z_range=self.cfg.MODEL.FASTBEV.Z_RANGE,
            y_range=self.cfg.MODEL.FASTBEV.Y_RANGE,
            pixel_per_meter_h=self.cfg.MODEL.FASTBEV.PIXEL_PER_METER_H,
            pixel_per_meter_v=self.cfg.MODEL.FASTBEV.PIXEL_PER_METER_V,
            detection_channels=self.cfg.MODEL.FASTBEV.DETECTION_CHANNELS,
            det3d_output_channels=self.cfg.MODEL.FASTBEV.DET3D_OUT_CHANNELS,
            seg_output_channel=self.cfg.MODEL.FASTBEV.SEG_OUT_CHANNELS,
            backbone=self.cfg.MODEL.FASTBEV.BACKBONE,
            neck=self.cfg.MODEL.FASTBEV.NECK,
            generate_det_gt_format = self.cfg.MODEL.FASTBEV.generate_det_gt_format,
            catetory_num= self.cfg.MODEL.FASTBEV.CATEGORY_NUM
        )
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 处理分布式训练的state_dict
        if any(k.startswith('module.') for k in state_dict.keys()):
            if not any(k.startswith('module.') for k in model.state_dict().keys()):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device)
        model.eval()
        
        print("Model loaded successfully")
        return model
    
    def _create_dataset(self):
        """创建数据集"""
        # 使用配置中的类别映射
        category_mapping = None
        dataset = WaymoBEVDataset(
            annotations_file=self.annotations_file,
            category_mapping=category_mapping,
            bev_width_m=self.cfg.DATASET.VAL.waymo.bev_width_m,
            bev_height_m=self.cfg.DATASET.VAL.waymo.bev_height_m,
            pixel_per_meter=self.cfg.DATASET.VAL.waymo.pixel_per_meter,
            transform=self.val_transform,  # 使用与训练时完全一致的变换
            generate_det_gt_format = self.cfg.MODEL.FASTBEV.generate_det_gt_format,
            complemented_det3d_channel = self.cfg.MODEL.FASTBEV.DET3D_OUT_CHANNELS,
            complemented_seg_channel = self.cfg.MODEL.FASTBEV.SEG_OUT_CHANNELS,
            with_det3d_label = self.cfg.DATASET.VAL.waymo.with_det3d_label,
            with_seg_label = self.cfg.DATASET.VAL.waymo.with_seg_label,
        )
        
        return dataset
    
    def preprocess_sample(self, sample):
        # 确保数据在正确的设备上
        for k in sample.keys():
            if isinstance(sample[k], torch.Tensor):
                sample[k] = sample[k].to(self.device)
        
        # 将图像和相机内参转换为正确的形状
        sample['image'] = sample['image'].unsqueeze(0)
        sample['camera_intrinsics'] = sample['camera_intrinsics'].unsqueeze(0)
        
        return sample
    
    def postprocess_predictions(self, predictions, sample):
        """后处理模型预测结果"""
        det3d_pred = predictions['det3d_pred']
        seg_pred = predictions['seg_pred']
        
        # 解码预测结果
        if len(sample['boxes_3d']) > 0:
            decoded_boxes = self.model.decode_det_predictions(det3d_pred, score_threshold=self.det3d_threshold)
        else:
            decoded_boxes = []
        
        
        # 转换为可视化格式
        viz_boxes = []
        for box in decoded_boxes:
            if box is not None:
                # 转换为check_bev期望的格式
                viz_box = {
                    'center': box['center'],
                    'wlh': box['wlh'],
                    'yaw': box['yaw'],
                    'height': box['height'],
                    'category': self.class_names[box['class_id']],
                    'score': box['score']
                }
                viz_boxes.append(viz_box)
        
        return viz_boxes, seg_pred
            
    def create_visualization_frame(self, sample, gt_boxes, pred_boxes, frame_idx,
                                gt_seg_maps, pred_seg_maps,
                                down_sample_factor=0.25):
        """创建可视化帧"""
        # 创建GT数据的可视化
        gt_data = {
            'cam_bbox_3d': gt_boxes,
            'image_path': sample['image_path'],
            'cam_intrinsic': sample['camera_intrinsics'].cpu().numpy()[0],
            'gt_seg_maps': gt_seg_maps
        }
        
        # 创建预测数据的可视化
        pred_data = {
            'bev_bbox_3d': pred_boxes,
            'image_path': sample['image_path'],
            'cam_intrinsic': sample['camera_intrinsics'].cpu().numpy()[0],
            'pred_seg_maps': pred_seg_maps
        }
        
        # 创建组合帧：GT + 预测
        gt_frame, gt_width, gt_height = self.bev_viz.create_combined_frame(gt_data,down_sample_factor=down_sample_factor)
        pred_frame, pred_width, pred_height = self.bev_viz.create_combined_frame(pred_data, is_gt=False, down_sample_factor=down_sample_factor)
        
        # 创建最终组合帧：GT在上，预测在下
        final_height = gt_height + pred_height
        final_width = max(gt_width, pred_width)
        final_frame = np.zeros((final_height, final_width, 3), dtype=np.uint8)
        
        # 放置GT帧（上半部分）
        final_frame[:gt_height, :gt_width] = gt_frame
        
        # 放置预测帧（下半部分）
        final_frame[gt_height:gt_height+pred_height, :pred_width] = pred_frame
        
        # 添加分隔线和标签
        cv2.line(final_frame, (0, gt_height), (final_width, gt_height), (255, 255, 255), 3)
        cv2.putText(final_frame, "Ground Truth", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(final_frame, "Predictions", (10, gt_height + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(final_frame, f"Frame {frame_idx}", (final_width - 150, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 添加配置信息
        config_text = f"Config: {self.config_name}"
        cv2.putText(final_frame, config_text, (10, final_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return final_frame, final_width, final_height
    
    def run_demo_on_dataset(self, num_samples=None, save_video=True):
        """在数据集上运行演示"""
        if num_samples is None:
            num_samples = len(self.dataset)
        
        num_samples = min(num_samples, len(self.dataset))
        print(f"Running demo on {num_samples} samples")
        
        # 存储所有帧
        all_frames = []
        frame_width, frame_height = None, None
        
        # 处理每个样本
        for i in tqdm(range(num_samples), desc="Processing samples"):
            # 获取样本
            sample = self.dataset[i]
            image_name = os.path.splitext(os.path.basename(sample['image_path']))[0]
            
            # 预处理（与训练时完全一致）
            processed_sample = self.preprocess_sample(sample)
            
            # 模型推理
            with torch.no_grad():
                predictions = self.model(processed_sample)
            
            # 后处理
            gt_boxes = sample['boxes_3d']
            gt_seg_maps = sample['seg_gt_maps']
                
            pred_boxes, pred_seg_maps = self.postprocess_predictions(predictions, sample)
            
            if self.save_result:
                # 保存结果
                json_dir = os.path.join(self.output_dir, "bboxes")
                os.makedirs(json_dir, exist_ok=True)
                output_result_path = os.path.join(json_dir, f"{image_name}.json")
                with open(output_result_path, 'w') as f:
                    json.dump(pred_boxes, f, indent=4)

                seg_grass_dir = os.path.join(self.output_dir, "seg_grass_masks")
                os.makedirs(seg_grass_dir, exist_ok=True)
                
                tmp_pred_seg_maps = pred_seg_maps.squeeze(axis=0)
                tmp_seg_map_list = []
                for i in range(tmp_pred_seg_maps.shape[0]):
                    tmp_seg_map = (tmp_pred_seg_maps[i] > self.seg_score_list[i]).detach().cpu().numpy().astype(np.uint8)
                    tmp_seg_map_list.append(tmp_seg_map)
                pred_seg_maps = np.stack(tmp_seg_map_list, axis=0)
                np.save(os.path.join(seg_grass_dir, f"{image_name}.npy"), pred_seg_maps)
                
                
                
            # 创建可视化帧
            viz_frame, width, height = self.create_visualization_frame(
                sample, gt_boxes, pred_boxes, i, 
                gt_seg_maps, pred_seg_maps
            )
            
            cv2.putText(viz_frame, image_name, (320, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # 记录尺寸
            if frame_width is None:
                frame_width, frame_height = width, height
            
            # 保存单帧图像
            frame_path = os.path.join(self.output_dir, "images", f"{image_name}.jpg")
            cv2.imwrite(frame_path, viz_frame)
            
            # 存储帧用于视频
            all_frames.append(viz_frame)
            
            # 打印统计信息
            print(f"Frame {i}: GT boxes: {len(gt_boxes)}, Pred boxes: {len(pred_boxes)}")
                
            # except Exception as e:
            #     print(f"Error processing sample {i}: {e}")
            #     continue
        
        # 保存视频
        if save_video and all_frames:
            video_path = os.path.join(self.output_dir, "videos", "demo_comparison.mp4")
            self._save_video(all_frames, video_path, frame_width, frame_height, fps=5)
            print(f"Video saved to: {video_path}")
        
        print(f"Demo completed. Results saved to: {self.output_dir}")
    
    def _save_video(self, frames, output_path, width, height, fps=5):
        """保存视频"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            video_writer.write(frame)
        
        video_writer.release()
    
    def run_single_sample_demo(self, sample_idx=0):
        """运行单个样本的演示"""
        print(f"Running single sample demo on sample {sample_idx}")
        
        try:
            # 获取样本
            sample = self.dataset[sample_idx]
            
            # 预处理（与训练时完全一致）
            processed_sample = self.preprocess_sample(sample)
            
            # 模型推理
            with torch.no_grad():
                predictions = self.model(processed_sample)
            
            # 后处理
            gt_boxes = sample['boxes_3d']
            pred_boxes = self.postprocess_predictions(predictions, sample)
            
            # 创建可视化帧
            viz_frame, width, height = self.create_visualization_frame(
                sample, gt_boxes, pred_boxes, sample_idx
            )
            
            # 保存结果
            output_path = os.path.join(self.output_dir, f"single_sample_{sample_idx}.png")
            cv2.imwrite(output_path, viz_frame)
            
            print(f"Single sample demo completed. Result saved to: {output_path}")
            print(f"GT boxes: {len(gt_boxes)}, Pred boxes: {len(pred_boxes)}")
            
            return viz_frame
            
        except Exception as e:
            print(f"Error in single sample demo: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='FastBEV Demo')
    parser.add_argument('--annotation_file', type=str, default='det3d_overfit.txt', help='annotation file to record validation data')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/fastbev_2025-11-17_14-55-15/checkpoint_step_45000.pth', help='Path to trained model')
    parser.add_argument('--config', type=str, default='fastbev_v6', help='Config name (e.g., fastbev_dinov3)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: outputs/validation/val_TIMESTAMP)')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to process')
    parser.add_argument('--single_sample', type=int, default=None, help='Run demo on single sample')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--det3d_threshold', type=float, default=0.3, help='det threshold to process det maps')
    parser.add_argument("--seg_score_list", type=float, nargs=1, default=[1.5], help="seg score threshold")
    parser.add_argument('--with_det3d_label', action="store_false", help='det threshold to process det maps')
    parser.add_argument('--with_seg_label', action="store_false", help='det iou threshold to process seg grass maps')
    parser.add_argument('--save_result', action="store_false", help='whether to save predicted results or not')
    parser.add_argument('--distance_threshold', type=float, default=None, help='Distance threshold in meters (camera Z axis). Boxes beyond this distance will not be displayed. None means no filtering.')
    args = parser.parse_args()
    
    # 如果没有指定 output_dir，使用带时间戳的默认目录
    if args.output_dir is None:
        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = f"./outputs/validation/val_{datetime_str}"
    
    print(f"Results will be saved to: {args.output_dir}")
    
    # 创建演示器
    demo = FastBEVDemo(
        model_path=args.checkpoint,
        config_name=args.config,
        annotation_file=args.annotation_file,
        output_dir=args.output_dir,
        device=args.device,
        det3d_threshold = args.det3d_threshold,
        seg_score_list = args.seg_score_list,
        save_result=args.save_result,
        distance_threshold=args.distance_threshold
    )
    
    # 运行演示
    if args.single_sample is not None:
        demo.run_single_sample_demo(args.single_sample)
    else:
        # demo.run_demo_on_dataset(num_samples=args.num_samples)
        demo.run_demo_on_dataset(num_samples=None)

if __name__ == "__main__":
    main()
