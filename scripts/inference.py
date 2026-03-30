import argparse
import copy
import glob
import os
import sys
import json
import cv2
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

sys.path.append('.')
from model.fastbev import FastBEV
from utils.postprocess import postprocess_predictions
from utils.preprocess import preprocess
from utils.visualize_utils import generate_video, visualize_det_on_bev


def load_model(model_path,
               x_range=(-16, 16),
               z_range=(0.0, 32.0),
               y_range=(-2.0, 3.0),
               pixel_per_meter_h=10,
               pixel_per_meter_v=2,
               detection_channels=128,
               det3d_output_channels=27,
               seg_output_channel=1,
               backbone = '',
               neck = '',
               device='cuda',
               generate_det_gt_format='standard',
               catetory_num=3):
    """加载训练好的模型"""
    print(f"Loading model from {model_path}")
    
    # 使用配置中的参数创建模型
    model = FastBEV(
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
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 处理分布式训练的state_dict
    if any(k.startswith('module.') for k in state_dict.keys()):
        if not any(k.startswith('module.') for k in model.state_dict().keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    return model


def infer_image_list(model, image_list, camera_intrinsic, device='cuda',
                     det_score=0.3, seg_score_list=[0.5, 0.5, 0.5], down_sample_factor=0.25, 
                     save_dir='infer_results', disable_det=False, disable_seg=False):
    seg_colors = [
            [0,0,255],      # 'lane_line' : 
            [0,255,255],    # 'ped_crossing': 
            [255,0,0],      # 'walkway': 
            [0,0,0],        # 'road_divider': 
            [0,255,0]       # 'drivable_area':
        ]
    os.makedirs(save_dir, exist_ok=True)

    for image_path in tqdm(image_list):
        image_name = os.path.basename(image_path)
        tmp_camera_intrinsic = copy.deepcopy(camera_intrinsic)
        image_tensor, processed_camera_intrinsic = preprocess(image_path=image_path, 
                                                    camera_intrinsic=tmp_camera_intrinsic)
        _, h, w = image_tensor.shape
        image_tensor = image_tensor[:,:,:w//2]
        processed_sample = {}
        processed_sample['image'] = image_tensor.unsqueeze(axis=0).to(device)
        processed_sample['camera_intrinsics'] = processed_camera_intrinsic.unsqueeze(axis=0).to(device)
        
        with torch.no_grad():
            predictions = model(processed_sample)
            
        pred_boxes, pred_seg_maps = postprocess_predictions(predictions,
                                                            det_score_threshold=det_score,
                                                            seg_score_list = seg_score_list,
                                                            class_names=['vehicle', 'person', 'cyclist'])      # 最终结果

        
        # 可视化
        if disable_det:
            pred_boxes = []
        if disable_seg:
            pred_seg_maps = np.zeros_like(pred_seg_maps)
        det_bev_img, fov_img = visualize_det_on_bev(image_path=image_path, 
                                                    pred_boxes=pred_boxes, 
                                                    pred_seg_maps=pred_seg_maps,
                                                    down_sample_factor=down_sample_factor
                                                )
        
        concate_image = np.hstack([det_bev_img, fov_img])
        image_format = image_name.split('.')[-1]
        
        with open(os.path.join(save_dir, image_name.replace(image_format, 'json')), 'w') as f:
            json.dump(pred_boxes, f, indent=4)
        cv2.imwrite(os.path.join(save_dir, image_name), concate_image)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastBEV Inference Script")

    # 模型参数
    parser.add_argument("--model_path", type=str, default='checkpoints/fastbev_2025-11-17_14-55-15/checkpoint_step_45000.pth', help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run model on")

    # 数据参数
    parser.add_argument("--image_list_file", type=str, default="infer_image_list.txt", help="Image list file")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save inference results (default: outputs/infer/infer_TIMESTAMP)")
    parser.add_argument("--det_score", type=float, default=0.3, help="Detection score threshold")
    parser.add_argument("--seg_score_list", type=float, nargs=1, default=[0.5], help="seg score threshold")

    # 相机内参
    parser.add_argument("--camera_intrinsic", type=float, nargs=9, default=[                # set default value here, adapt to your camera intrinsics
        [432.7546294301935, 0, 637.7678519487691],
        [0.0, 588.2841073319762, 536.0641665216108],
        [0, 0, 1]
    ], help="Camera intrinsic matrix (flattened 3x3)")
    
    # 可视化参数
    parser.add_argument("--make_video", action="store_true", help="Whether to generate a video")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--frame_size", type=int, nargs=2, default=[696, 320], help="Video frame size (w h)")
    parser.add_argument("--disable_det", action="store_true", help="Whether to disable detection")
    parser.add_argument("--disable_seg", action="store_true", help="Whether to disable segmentation")

    args = parser.parse_args()
    
    # 生成时间戳，用于目录名和视频文件名
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 如果没有指定 save_dir，使用带时间戳的默认目录
    if args.save_dir is None:
        args.save_dir = f"./outputs/infer/infer_{datetime_str}"
    
    print(f"Results will be saved to: {args.save_dir}")

    # 加载模型
    model = load_model(model_path=args.model_path, device=args.device)

    # 相机参数 reshape
    camera_intrinsic = np.array(args.camera_intrinsic).reshape(3, 3)

    with open(args.image_list_file, 'r') as f:
        info_lines = f.readlines()
    
    image_file_list = [line.strip() for line in info_lines]
    image_file_list = sorted(image_file_list, key=lambda l: float(os.path.splitext(os.path.basename(l))[0]))
    
    # 推理
    infer_image_list(model, image_file_list, camera_intrinsic, device=args.device, 
                     det_score=args.det_score, seg_score_list=args.seg_score_list, 
                     save_dir=args.save_dir,
                     disable_det=args.disable_det,
                     disable_seg=args.disable_seg)
    
    # 生成视频
    if args.make_video:
        result_image_list = glob.glob(os.path.join(args.save_dir, '*.jpg')) + glob.glob(os.path.join(args.save_dir, '*.png'))
        result_image_list = sorted(result_image_list, key=lambda l: float(os.path.splitext(os.path.basename(l))[0]))
        # 创建 videos 子目录
        video_dir = os.path.join(args.save_dir, 'videos')
        os.makedirs(video_dir, exist_ok=True)
        video_file = os.path.join(video_dir, f"infer_video_{datetime_str}.mp4")
        generate_video(result_image_list, video_file, fps=args.fps, frame_size=tuple(args.frame_size))
    