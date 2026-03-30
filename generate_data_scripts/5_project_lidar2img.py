import numpy as np
import cv2
import os
import json
import glob
import copy


def load_lidar_points(lidar_path):
    # nuscenes lidar .bin: 每个点 5个float32 [x, y, z, intensity, ring_index]
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    return points[:, :3]  # 取 xyz

def filter_points_in_camera_frame(pts_cam, horizon=[-9.6, 9.6], vertical=[0,32], altitude=[-3,2]):
    '''
        pts_cam:(x,y,z), 3*N, camera frame, 右下前
    '''
    mask_x = (pts_cam[0,:] > horizon[0]) & (pts_cam[0,:] < horizon[1])
    mask_y = (pts_cam[1,:] > altitude[0]) & (pts_cam[1,:] < altitude[1])
    mask_z = (pts_cam[2,:] > vertical[0]) & (pts_cam[2,:] < vertical[1])
    
    mask = mask_x & mask_y & mask_z
    filter_pts = pts_cam.T[mask].T
    return filter_pts
    
def project_distance(pnts, x_range=[-9.6, 9.6], z_range=[0,32], y_range=[-3,2] ):
    '''
        pnts: 3*N
        将pnts对应的x,y,z投影到[0,255]
    '''
    x_scale = 255.0 / (x_range[1]-x_range[0])
    x_bias = 255.0/2
    
    y_scale = 255.0 / (y_range[1]-y_range[0])
    y_bias = 3*y_scale
    
    z_scale = 255.0 / (z_range[1]-z_range[0])
    z_bias = 0.0
    
    pseduo_pnts = copy.deepcopy(pnts)
    pseduo_pnts[0,:] = x_scale * pnts[0, :] + x_bias
    pseduo_pnts[1,:] = y_scale * pnts[1, :] + y_bias
    pseduo_pnts[2,:] = z_scale * pnts[2, :] + z_bias
    
    return pseduo_pnts.T
    
    
    
def project_lidar_to_image(points, cam_intrinsic, R_lidar2cam, T_lidar2cam, image_shape):
    """
    points: (N,3) lidar系点云
    cam_intrinsic: (3,3)
    R_lidar2cam: (3,3) rotation (lidar->cam)
    T_lidar2cam: (3,)   translation (lidar->cam)
    image_shape: (H,W)
    """
    # step1: lidar->cam
    R = np.array(R_lidar2cam)
    t = np.array(T_lidar2cam).reshape(3,1)
    pts_lidar = points.T  # (3,N)
    pts_cam = R @ pts_lidar + t  # (3,N)

    # 只保留在相机前方的点
    mask = pts_cam[2,:] > 0
    pts_cam = pts_cam[:, mask]
    pts_cam = filter_points_in_camera_frame(pts_cam)
    
    # step2: cam->image (pinhole)
    uv = cam_intrinsic @ pts_cam
    uv = uv[:2] / uv[2]  # (2,N)

    # step3: 过滤图像边界
    H, W = image_shape
    u, v = uv
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid].astype(np.int32)
    v = v[valid].astype(np.int32)
    pts_cam = pts_cam[:, valid]  # (3, M)

    # step4: 构建一个图像 (H,W,3)，每个点填 xyz
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    
    pseudo_pixels = project_distance(pts_cam)
    xyz_map[v, u] = pseudo_pixels
    return xyz_map

def process_data_dict(frame):
    lidar_path = frame["lidar_path"]
    image_path = frame["image_path"]

    points = load_lidar_points(lidar_path)

    cam_intrinsic = np.array(frame["cam_intrinsic"])
    cam2lidar_R = np.array(frame["cam2lidar_rotation"])
    cam2lidar_t = np.array(frame["cam2lidar_translation"])

    # 注意：json里给的是 cam->lidar，需要取逆变换 lidar->cam
    R_cam2lidar = np.array(cam2lidar_R)
    T_cam2_lidar = np.array(cam2lidar_t)
    R_lidar2cam = np.linalg.inv(R_cam2lidar)
    T_lidar2cam = -R_lidar2cam @ T_cam2_lidar

    # 读取图像，获取大小
    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    xyz_map = project_lidar_to_image(points, cam_intrinsic, R_lidar2cam, T_lidar2cam, (H,W))
    return xyz_map

if __name__ == "__main__":
    
    dst_dir = 'all_labels/synthetic_data'
    os.makedirs(dst_dir, exist_ok=True)
    
    label_list = glob.glob(os.path.join('all_labels/nuscenes_bbox3d_labels', '*.json'))
    for idx in range(len(label_list)):
        print(f'process {idx}/{len(label_list)}...\n')
        label_file = label_list[idx]
        
        with open(label_file, 'r') as f:
            json_dict = json.load(f)
        cur_res = process_data_dict(json_dict)
        assert cur_res.min() >= 0, "lower than 0!!!"
        assert cur_res.max() <= 255, "larger than 255!!!"
        
        cur_name = os.path.basename(label_file).replace('.json', '.npy')
        dst_file = os.path.join(dst_dir, cur_name)
        np.save(dst_file, cur_res)

    
    
    
