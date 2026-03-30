
import cv2
import numpy as np
import torch


def resize(image_array, camera_intrinsic=None, 
           targe_size=[816,960], pad_value=0):
    target_h, target_w = targe_size
    img_h, img_w = image_array.shape[:2]
    
    # 计算缩放比例，保持宽高比
    scale_h = target_h / img_h
    scale_w = target_w / img_w
    assert scale_h == scale_w, print("scale_h not equals scale_w...\n")
    scale = scale_h
    
    new_image = cv2.resize(image_array, dsize=None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_LINEAR)
    intrinsics = camera_intrinsic.copy() if camera_intrinsic is not None else None
    
    if camera_intrinsic is not None:    
        # 缩放调整：fx, fy需要乘以缩放比例
        intrinsics[0, 0] *= scale  # fx
        intrinsics[1, 1] *= scale  # fy
        
        # 平移调整：cx, cy需要加上padding偏移
        intrinsics[0, 2] = intrinsics[0, 2] * scale  # cx
        intrinsics[1, 2] = intrinsics[1, 2] * scale # cy
    
    return new_image, intrinsics


def to_tensor(image_array, camera_intrinsic):
    if isinstance(image_array, np.ndarray):
        image_array = np.transpose(image_array, [2,0,1])
        image_tensor = torch.from_numpy(image_array).float() / 255.0
    else:
        image_tensor = image_array.float() / 255.0
        
    if isinstance(camera_intrinsic, np.ndarray):
        camera_intrinsic = torch.from_numpy(camera_intrinsic)
    return image_tensor, camera_intrinsic

def normalize(image_tensor, 
              mean = torch.tensor([0.4850, 0.4560, 0.4060]),
              std = torch.tensor([0.2290, 0.2240, 0.2250])):
    mean = mean.view(3,1,1)
    std = std.view(3,1,1)
    image_tensor = (image_tensor - mean) / std
    return image_tensor


def bgr_to_yuv_bt601_full_range_128(bgr: np.ndarray) -> np.ndarray:
    """
    BGR[H,W,3] -> YUV_BT601_FULL_RANGE[H,W,3]  (Y,U,V顺序，Full Range)
    输出dtype为int8，范围为[-128, 127]
    """
    assert bgr.ndim == 3 and bgr.shape[2] == 3, "输入必须是H×W×3的BGR图像"
    # return (bgr.astype(np.int32) - 128).astype(np.int8)
    b, g, r = cv2.split(bgr.astype(np.float32))
    y  =  0.299    * r + 0.587    * g + 0.114    * b
    cb = -0.168736 * r - 0.331264 * g + 0.500000 * b + 128
    cr =  0.500000 * r - 0.418688 * g - 0.081312 * b + 128
    yuv = cv2.merge((y, cb, cr))
    yuv = np.clip(yuv, 0, 255)            # 先clip到[0,255]范围
    yuv = yuv - 128                        # 减去128，结果范围[-128, 127]
    yuv = np.clip(yuv, -128, 127)         # 确保在int8范围内
    return yuv.astype(np.int8)             # 最后转换为int8


def preprocess(image_path, camera_intrinsic, onnx_infer=False, quant=False):
    image = cv2.imread(image_path)
    raw_image_height, raw_image_width = image.shape[:2]
    resized_image, processed_camera_intrinsic = resize(image, camera_intrinsic)
    if not quant:
        image_tensor, processed_camera_intrinsic = to_tensor(image, processed_camera_intrinsic)
        image_tensor = normalize(image_tensor)
        
        resized_image, processed_camera_intrinsic = to_tensor(resized_image, processed_camera_intrinsic)
        resized_image = normalize(resized_image)
    else:
        image = bgr_to_yuv_bt601_full_range_128(image)
        image_tensor = torch.from_numpy(image)
        processed_camera_intrinsic = torch.from_numpy(processed_camera_intrinsic)
    
    if onnx_infer:
        empty_tensor = torch.zeros_like(image_tensor)
        image_tensor = torch.concatenate([image_tensor, empty_tensor], axis=-1)
        return image_tensor, processed_camera_intrinsic
    else:
        empty_tensor = torch.zeros_like(resized_image)
        resized_image = torch.concatenate([resized_image, empty_tensor], axis=-1)
        return resized_image, processed_camera_intrinsic


def preprocess_image(image_array):
    image_array, _ = resize(image_array)
    return image_array