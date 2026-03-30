import argparse
import logging
import os
import sys

sys.path.append('.')
import numpy as np
import torch
from horizon_nn.api import export_onnx

import onnx
from model.fastbev import create_fastbev_model
from utils.preprocess import resize


def setup_logging(log_dir):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'export.log')),
            logging.StreamHandler()
        ]
    )

def load_model(model, checkpoint_path, device):
    """加载模型和检查点"""
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        # checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # 处理模型状态字典中的键名前缀
        if any(k.startswith('module.') for k in state_dict.keys()):
            if not any(k.startswith('module.') for k in model.state_dict().keys()):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        try:
            model.load_state_dict(state_dict)
            logging.info(f"Loaded checkpoint from: {checkpoint_path}")
        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {e}, using random weights")
    else:
        logging.info("No checkpoint provided, using random weights")
    
    model = model.to(device)
    model.eval()
    return model

def export_onnx_self_define(model, x, onnx_path):
    """导出ONNX模型"""
    try:
        export_onnx(
        model, 
        x, 
        onnx_path, 
        verbose=True, 
        opset_version=11,
        input_names=['images'],
        output_names=['det3d_pred', 'seg_pred']
    )
        logging.info("ONNX export completed!")
        
        # 加载并验证ONNX模型
        try:
            onnx_model = onnx.load(onnx_path)
            logging.info("ONNX model loaded successfully!")
            
            # 尝试使用onnxsim（如果安装成功）
            try:
                from onnxsim import simplify
                onnx_model, check = simplify(onnx_model)
                if check:
                    simplified_path = onnx_path.replace(".onnx", "_simplified.onnx")
                    onnx.save(onnx_model, simplified_path)
                    logging.info("ONNX model simplified successfully!")
                    
                    # 尝试生成模型分析报告（修复错误）
                    try:
                        import onnx_tool
                        profile_path = onnx_path.replace(".onnx", "_simplified_profile.txt")
                        onnx_tool.model_profile(simplified_path, save_profile=profile_path)
                        logging.info(f"Model profile saved to: {profile_path}")
                    except Exception as profile_error:
                        logging.warning(f"Failed to generate model profile: {profile_error}")
                        
                else:
                    logging.warning("ONNX simplification failed, using original model")
            except ImportError:
                logging.warning("onnxsim not available, using original ONNX model")
                
        except Exception as e:
            logging.error(f"Error processing ONNX model: {e}")
            
    except Exception as e:
        logging.error(f"Error during ONNX export: {e}")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/fastbev_2025-11-17_14-55-15/checkpoint_step_45000.pth', help='checkpoint path (optional)')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU id to use')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for export')
    parser.add_argument('--height', type=int, default=1088, help='input height')
    parser.add_argument('--width', type=int, default=1280, help='input width')
    parser.add_argument('--x_range', type=float, nargs=2, default=[-16, 16], help='X range (left-right) in meters')
    parser.add_argument('--z_range', type=float, nargs=2, default=[0.0, 32.0], help='Z range (front-back) in meters')
    parser.add_argument('--y_range', type=float, nargs=2, default=[-2.0, 3.0], help='Y range (height) in meters')
    parser.add_argument('--pixel_per_meter_h', type=float, default=10, help='X grid size in meters')
    parser.add_argument('--pixel_per_meter_v', type=float, default=2, help='Z grid size in meters')
    parser.add_argument('--detection_channels', type=int, default=128, help='X grid size in pixels')
    parser.add_argument('--det3d_out_channels', type=int, default=27, help='detection channels')
    parser.add_argument('--seg_output_channel', type=int, default=1, help='seg output channels')
    parser.add_argument('--generate_det_gt_format', type=str, default='standard', help='generate gt-maps format')
    parser.add_argument('--catetory_num', type=int, default=3, help='category nums for detection 3d objects')
    parser.add_argument('--origin_image_height', type=int, default=1088, help='origin image height')
    parser.add_argument('--origin_image_width', type=int, default=1280, help='origin image width')
    parser.add_argument('--target_image_height', type=int, default=816, help='target image height')
    parser.add_argument('--target_image_width', type=int, default=960, help='target image width')
    parser.add_argument("--camera_intrinsic", type=float, nargs=9, default=[                # set default value here, adapt to your camera intrinsics
        [432.7546294301935, 0, 637.7678519487691],
        [0.0, 588.2841073319762, 536.0641665216108],
        [0, 0, 1]
    ], help="Camera intrinsic matrix (flattened 3x3)")
    args = parser.parse_args()
    
    try:
        # 创建输出目录
        onnx_dir = "./onnx"
        os.makedirs(onnx_dir, exist_ok=True)
        
        # 设置日志
        setup_logging(onnx_dir)
        
        # 设置设备
        if torch.cuda.is_available() and args.gpu >= 0:
            device = torch.device(f'cuda:{args.gpu}')
            torch.cuda.set_device(args.gpu)
        else:
            device = torch.device('cpu')
        
        logging.info(f"Using device: {device}")
        logging.info(f"Input shape: ({args.batch_size}, 3, {args.height}, {args.width})")
        logging.info(f"BEV ranges: X={args.x_range}, Z={args.z_range}, Y={args.y_range}")
        logging.info(f"pixel_per_meter: horizon={args.pixel_per_meter_h}, vertical={args.pixel_per_meter_v}")

        # 创建FastBEV模型
        model = create_fastbev_model(
            x_range=tuple(args.x_range),
            z_range=tuple(args.z_range),
            y_range=tuple(args.y_range),
            pixel_per_meter_h=args.pixel_per_meter_h,
            pixel_per_meter_v=args.pixel_per_meter_v,
            detection_channels=args.detection_channels,
            det3d_output_channels=args.det3d_out_channels,
            seg_output_channel=args.seg_output_channel,
            backbone='',
            neck='',
            generate_det_gt_format=args.generate_det_gt_format,
            catetory_num=args.catetory_num
        )
        
        # 加载模型（如果有checkpoint）
        model = load_model(model, args.checkpoint, device)
        
        # 设置输入形状
        input_shape = (args.batch_size, 3, args.height, args.width*2)
        camera_intrinsics_shape = (args.batch_size, 3, 3)
        
        # 创建示例输入,根据需求进行输入尺寸/内参的设置
        # image_array = np.random.rand(args.height, args.width, 3)
        image_array = np.random.rand(args.origin_image_height, args.origin_image_width, 3)      # 原始图像的尺寸
        camera_intrinsics = np.array(args.camera_intrinsic).reshape(3, 3)      # 原始尺寸对应的内参, reshape为3x3矩阵
        
        _, intrinsic = resize(image_array, camera_intrinsics, targe_size=[args.target_image_height, args.target_image_width])
        
        images_tensor = torch.randn(input_shape, device=device)  
        camera_intrinsics = torch.tensor(intrinsic).unsqueeze(0).expand(args.batch_size, -1, -1)
        
        # 预热模型
        model.eval()
        model.sampling_coords = model._compute_sampling_coordinates(camera_intrinsics, args.target_image_height, args.target_image_width)
        model.forward = model.export
        with torch.no_grad():
            _ = model(images_tensor)
        
        # 导出ONNX模型
        onnx_path = os.path.join(onnx_dir, "fastbev_model.onnx")
        logging.info("Exporting ONNX model...")
        export_onnx_self_define(model, images_tensor, onnx_path)
        
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
