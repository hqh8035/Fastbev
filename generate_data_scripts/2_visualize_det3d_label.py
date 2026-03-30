import json
import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_corners(wlh, center, yaw):
    '''
        camera坐标系为右下前, l与x轴绑定; w和z轴绑定; h和y轴绑定;
        yaw角表示x轴绕z轴逆时针向z轴旋转的角度;
    '''
    w, l, h = wlh
    x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
    y_corners = h / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
    z_corners = w / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    rotation_matrix = [
        [np.cos(yaw), 0, -np.sin(yaw)],
        [0, 1, 0],
        [np.sin(yaw), 0, np.cos(yaw)],
        
    ]
    corners = rotation_matrix @ corners
    # Translate
    x, y, z = center
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners


if __name__ == "__main__":
    show_dir = 'visualize_mini_label_res'
    os.makedirs(show_dir, exist_ok=True)
    
    json_list = glob.glob(os.path.join('mini_labels/nuscenes_bbox3d_labels', '*.json'))
    
    for json_path in json_list:
        data = load_json(json_path)
        
        img_path = data['image_path']
        img = cv2.imread(img_path)
        cam_intrinsic = data['cam_intrinsic']
        for bbox in data['cam_bbox_3d']:
            category = bbox['category']
            center = bbox['center']
            wlh = bbox['wlh']
            yaw = bbox['yaw']
            corners_3d = get_corners(wlh, center, yaw)
            
            view = np.eye(4)
            view[:3, :3] = np.array(cam_intrinsic)
            in_front = corners_3d[2, :] > 0.1
            if all(in_front) is False:
                continue
            points = corners_3d
            points = np.concatenate((points, np.ones((1, points.shape[1]))), axis=0)
            points = np.dot(view, points)[:3, :]
            points /= points[2, :]
                        
            box_img = points.astype(np.int32)
            if category != 'pedestrian':
                color = (0, 0, 255)
            else:
                color = [255,0,0]
                
            tmp_pnts = points.T[:,:2].astype(np.int32)                
            
            for i in range(4):
                j = (i + 1) % 4
                # 下底面
                cv2.line(img, (box_img[0, i], box_img[1, i]), (box_img[0, j], box_img[1, j]), color, thickness=1)
                # 上底面
                cv2.line(img, (box_img[0, i + 4], box_img[1, i + 4]), (box_img[0, j + 4], box_img[1, j + 4]), color, thickness=1)
                # 侧边线
                cv2.line(img, (box_img[0, i], box_img[1, i]), (box_img[0, i + 4], box_img[1, i + 4]), color, thickness=1)
        
        cv2.imwrite(os.path.join(show_dir, os.path.basename(img_path)), img)       
