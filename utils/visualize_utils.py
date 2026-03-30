import copy

import cv2
import numpy as np

from utils.preprocess import preprocess_image


def draw_rotated_rectangle(img, center, width, height, angle, color, thickness=2):
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
    

def visualize_det_on_bev(image_path, pred_boxes,
                         pred_seg_maps,
                         down_sample_factor=0.25):
    det3d_category_colors = {
        'vehicle': (128, 0, 128),
        'person': (0, 128, 128), 
        'pedestrian': (0, 128, 128),
        'cyclist': (128,128,0)
    }
            
    seg_colors = [
            [0,255,0],      # 'branch'
        ]
    
    
    cam_x_range = [-16, 16]
    cam_z_range = [0, 32]
    cam_y_range = [-3, 2]
    pixel_per_meter_h = 10      # bev图像分辨率
    pixel_per_meter_v  = 2      # 高度比例尺
    
    bev_width = int((cam_x_range[1] - cam_x_range[0]) * pixel_per_meter_h)
    bev_height = int((cam_z_range[1] - cam_z_range[0]) * pixel_per_meter_h)
    bev_img = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
    
    # visualize bev result
    bev_cam_point = [bev_width//2, bev_height-1]
    cv2.circle(bev_img, (bev_cam_point[0], bev_cam_point[1]), radius=3, color=(0,0,255), thickness=-1)
    
    for bbox in pred_boxes:
        center = np.array(bbox['center'])
        wlh = np.array(bbox['wlh'])
        yaw = bbox['yaw']
        center_x, center_y = center.astype(np.int32)
        
        if (0 <= center_x < bev_width and 0 <= center_y < bev_height):
            # 获取类别和颜色
            category = bbox['category']
            color = det3d_category_colors[category]
            
            # 计算尺寸（像素）
            # CORRECTED: 在BEV中：X方向对应height（长度），Z方向对应width（宽度）
            height_pixels = int(wlh[1] * pixel_per_meter_h)  # 长度（X方向）
            width_pixels = int(wlh[0] * pixel_per_meter_h)   # 宽度（Z方向）
            
            # 绘制旋转矩形 - 修正参数顺序
            draw_rotated_rectangle(bev_img, (center_x, center_y), 
                                    height_pixels, width_pixels, yaw, color, 2)
            
            # 绘制朝向箭头
            arrow_length = max(width_pixels, height_pixels) * 0.8
            arrow_end_x = int(center_x + arrow_length * np.cos(yaw))
            arrow_end_y = int(center_y - arrow_length * np.sin(yaw))
            
            # 确保箭头终点在图像范围内
            if (0 <= arrow_end_x < bev_width and 0 <= arrow_end_y < bev_height):
                cv2.arrowedLine(bev_img, (center_x, center_y), 
                            (arrow_end_x, arrow_end_y), color, 2)
            
            # 添加标签
            label = f"{category[:3]}"
            cv2.putText(bev_img, label, (center_x + 5, center_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    fov_image = cv2.imread(image_path)
    fov_image = preprocess_image(fov_image)         # resize and padding
    fov_image = cv2.resize(fov_image, dsize=None, fx=down_sample_factor, fy=down_sample_factor)     # down_sampl_factor为0.25
    copy_image = copy.deepcopy(fov_image)
    
    for i in range(len(seg_colors)):
        copy_image[pred_seg_maps[i].astype(np.bool_)] = seg_colors[i]
    
    fov_image = cv2.addWeighted(copy_image, 0.5, fov_image, 0.5, 0)
    ratio = bev_img.shape[0] / fov_image.shape[0]
    fov_image = cv2.resize(fov_image, dsize=None, fx=ratio, fy=ratio)
    return bev_img, fov_image


def generate_video(image_list, video_file, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_file, fourcc, fps, frame_size)
    for idx, image_path in enumerate(image_list):
        print(idx)
        image = cv2.imread(image_path)
        image = cv2.putText(image, str(idx), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 255), lineType=cv2.LINE_AA, thickness=3)
        video.write(image)
    
    video.release()
    print(f"Video saved to {video_file}")