import argparse
import glob
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append('.')

import onnxruntime as ort
import torch

from utils.postprocess import postprocess_predictions
from utils.preprocess import preprocess
from utils.visualize_utils import generate_video, visualize_det_on_bev


def run_onnx_inference(onnx_model_path, image_path, camera_intrinsic, 
                       det3d_threshold, seg_thres_list, 
                       class_names=['vehicle', 'person', 'cyclist'],
                       onnx_infer =False):
    """
    使用ONNX模型推理
    """
    # 创建 ONNX Runtime Session
    sess = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    # 预处理
    image_tensor, camera_intrinsic = preprocess(image_path, camera_intrinsic, onnx_infer=onnx_infer)

    # onnxruntime 输入必须是 numpy
    input_image = image_tensor.unsqueeze(0).numpy()        # shape: (1, 3, H, W)
    input_intrinsics = camera_intrinsic.unsqueeze(0).to(torch.float32).numpy()  # shape: (1, 3, 3)

    # 获取输入输出名字
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    # print("ONNX Input names:", input_names)
    # print("ONNX Output names:", output_names)

    # 构造输入字典
    ort_inputs = {
        input_names[0]: input_image,
        # 如果模型需要 intrinsics，可以加上:
        # input_names[1]: input_intrinsics
    }

    # 推理
    ort_outs = sess.run(output_names, ort_inputs)

    predictions = {
        'det3d_pred': torch.from_numpy(ort_outs[0]).permute([0,3,1,2]),
        'seg_pred': torch.from_numpy(ort_outs[1]).permute([0,3,1,2])
    }

    # 后处理
    pred_boxes, pred_seg_maps = postprocess_predictions(
        predictions,
        det3d_threshold,
        seg_thres_list,
        class_names=class_names
    )

    return pred_boxes, pred_seg_maps


def infer_image_list(onnx_model, image_list, camera_intrinsic, 
                     det3d_thres, seg_thres_list,
                     fps, frame_size, save_dir='onnx_infer_results', 
                     make_video=False, onnx_infer=False,
                     video_name='onnx_infer_video.mp4'):
    os.makedirs(save_dir, exist_ok=True)
    for image_path in tqdm(image_list):
        image_name = os.path.basename(image_path)
        pred_boxes, pred_seg_maps = run_onnx_inference(
            onnx_model,
            image_path,
            camera_intrinsic,
            det3d_thres,
            seg_thres_list,
            onnx_infer=onnx_infer
        )

        bev_image, fov_image = visualize_det_on_bev(image_path, pred_boxes, pred_seg_maps)
        concate_image = np.hstack([bev_image, fov_image])
        cv2.imwrite(os.path.join(save_dir, image_name), concate_image)

    if make_video:
        result_image_list = glob.glob(os.path.join(save_dir, '*.jpg'))
        result_image_list = sorted(result_image_list, key=lambda l: float(os.path.basename(l).replace('.jpg', '')))
        generate_video(result_image_list, video_name, fps=fps, frame_size=frame_size)


if __name__ == "__main__":
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="ONNX Inference for FastBEV")
    parser.add_argument("--onnx_model_path", type=str,default='onnx/fastbev_model_simplified.onnx',
                        help="Path to the ONNX model file")
    parser.add_argument("--image_dir", type=str, default='infer_images',)
    parser.add_argument("--image_path", type=str,default='infer_images/0.jpg',
                        help="Path to the input image")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Path to save visualization result (default: outputs/test_onnx/test_TIMESTAMP)")    
    parser.add_argument("--det3d_threshold", type=float, default=0.3, help="Detection 3d objects score threshold")
    parser.add_argument("--seg_thres_list", type=float, nargs=1, default=[0.5], help="segmentation score threshold")
    parser.add_argument("--camera_intrinsic", type=float, nargs=9, default=[                # set default value here, adapt to your camera intrinsics
        [432.7546294301935, 0, 637.7678519487691],
        [0.0, 588.2841073319762, 536.0641665216108],
        [0, 0, 1]
    ], help="Camera intrinsic matrix (flattened 3x3)")
    
    parser.add_argument("--fps", type=int, default=20, help="Video FPS")
    parser.add_argument("--frame_size", type=int, nargs=2, default=[1280, 800], help="Video frame size (w h)")
    parser.add_argument("--make_video", action="store_true", help="Whether to generate a video")
    parser.add_argument("--onnx_infer", action="store_false", help="Whether to use ONNX inference preprocess")

    args = parser.parse_args()
    
    # 如果没有指定 save_dir，使用带时间戳的默认目录
    if args.save_dir is None:
        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.save_dir = f"./outputs/test_onnx/test_{datetime_str}"
    
    print(f"Results will be saved to: {args.save_dir}")

    
    # # use single image for onnx inference demo
    image_list = ['/perception/users/chenfuxuan/WorkSpace_2025/Data/liufen/accepted_data/batch6/images/caijiche/1764751207.256007680.png']
    exist_image_list = [img for img in image_list if os.path.exists(img)]
    if len(exist_image_list) == 0:
        print("No image found in the image list")
        exit(1)
    else:
        print(f"Found {len(exist_image_list)} images")

    # 如果需要生成视频，视频文件名也带时间戳
    video_name = "infer_video.mp4" if not args.make_video else os.path.join(args.save_dir, "videos", f"test_video_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4")
    if args.make_video:
        os.makedirs(os.path.join(args.save_dir, "videos"), exist_ok=True)
    
    infer_image_list(args.onnx_model_path, exist_image_list, np.array(args.camera_intrinsic).reshape(3, 3), 
                     args.det3d_threshold, args.seg_thres_list,
                     args.fps, args.frame_size, save_dir=args.save_dir, make_video=args.make_video,
                     onnx_infer=args.onnx_infer, video_name=video_name)