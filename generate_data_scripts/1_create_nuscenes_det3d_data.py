import os
import sys
import ujson

sys.path.append('.')
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from tqdm import tqdm
from auxiliary_scripts.utils import NameMapping

def to_serializable(obj):
    """Convert numpy types to Python types for ujson serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # 保留 numpy 默认精度
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)    # 转成 Python float
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not os.path.isfile(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes



def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',                   # camera2ego
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix       # camera2ego rotation
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix       # ego2global rotation
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)            # R @ X_lidar = X_sensor, lidar->ego->global->ego->sensor, pre ego related to lidar, post ego relate to sensor
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (                             # 行向量的表示用右乘完成
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def fill_trainval_infos(nusc, label_dir):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    os.makedirs(label_dir, exist_ok=True)
    
    for sample in tqdm(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])       # sample_data_record
        cs_record = nusc.get('calibrated_sensor',                           # calibrate_sample_record
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])        # ego-car pose
        lidar_path, _, _ = nusc.get_sample_data(lidar_token)

        assert os.path.exists(lidar_path), "f{lidar_path} not exists..."

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],       # sample_token
            'sweeps': [],
            'cams': dict(),
            'lidar2ego_translation': cs_record['translation'],      # lidar2ego translation
            'lidar2ego_rotation': cs_record['rotation'],            # lidar2ego rotation quaternion
            'ego2global_translation': pose_record['translation'],   # ego2global translation
            'ego2global_rotation': pose_record['rotation'],         # ego2global rotation
            'timestamp': sample['timestamp'],
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)            # camera path / camera intrinsic matrix
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
        
        
        # 当前所需的各种变换矩阵
        cam2lidar_rotation = cam_info['sensor2lidar_rotation']
        cam2lidar_translation = cam_info['sensor2lidar_translation']
        cur_cam_intrinsic = cam_intrinsic
        cur_image_path = cam_path
        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        cam2ego_translation = cam_info['sensor2ego_translation']
        cam2ego_rotation = cam_info['sensor2ego_rotation']
        
        # 获取3D bbox,   lidar坐标系下, 然后转换到camera坐标系下
        sd_rec = nusc.get('sample_data', cam_token)
        s_rec = nusc.get('sample', sd_rec['sample_token'])
        ann_recs = [nusc.get('sample_annotation', token) for token in s_rec['anns']]

        label_dict = {}
        label_dict['sensor_type'] = 'cam_front'
        label_dict['cam2lidar_rotation'] = cam2lidar_rotation
        label_dict['cam2lidar_translation'] = cam2lidar_translation
        label_dict['ego2global_translation'] = ego2global_translation
        label_dict['ego2global_rotation'] = ego2global_rotation
        label_dict['cam2ego_translation'] = cam2ego_translation
        label_dict['cam2ego_rotation'] = cam2ego_rotation
        label_dict['cam_intrinsic'] = cur_cam_intrinsic
        label_dict['image_path'] = cur_image_path
        label_dict['lidar_path'] = lidar_path 
        label_dict['cam_timestamp'] = sample['timestamp']
        label_dict['cam_bbox_3d'] = []
        
        for ann_rec in ann_recs:
            # 世界坐标系下的box标注信息
            box = nusc.get_box(ann_rec['token'])

            # 从世界坐标系->车身坐标系
            box.translate(-np.array(ego2global_translation))
            box.rotate(Quaternion(ego2global_rotation).inverse)

            # 从车身坐标系->相机坐标系
            box.translate(-np.array(cam2ego_translation))
            box.rotate(Quaternion(cam2ego_rotation).inverse)
            
            corners_3d = box.corners()
            in_front = corners_3d[2, :] > 0.1
            if all(in_front) is False:
                continue
            
            category_name = ann_rec['category_name']
            if category_name  in NameMapping:
                general_category_name = NameMapping[category_name]
                if general_category_name in ['barrier', 'traffic_cone']:      # 忽略掉barrier类别
                    continue
                bottom_corners = box.bottom_corners()       # 逆时针的底部四个点
                
                arrow_start = bottom_corners[:,-1]
                arrow_end = bottom_corners[:,0]
                direction_vector = (np.array(arrow_end)-np.array(arrow_start))[[0,2]]
                yaw = np.arctan2(direction_vector[1], direction_vector[0])
            
                xyz = box.center
                wlh = box.wlh
                orientation = box.orientation
                
                tmp_box_dict= {
                    'category': NameMapping[category_name],
                    'center': xyz,
                    'wlh': wlh,
                    'orientation': list(orientation),
                    'yaw': yaw,
                }
                label_dict['cam_bbox_3d'].append(tmp_box_dict)
        label_file = os.path.basename(cur_image_path).replace('.jpg', '.json')
        label_file = os.path.join(label_dir, label_file)
        with open(label_file, 'w') as f:
            ujson.dump(label_dict, f, indent=4, default=to_serializable, escape_forward_slashes=False)



if __name__ == "__main__":
    version = 'v1.0-trainval'
    root_path = 'data/nuscenes'
    label_dir = 'all_labels/nuscenes_bbox3d_labels'
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    fill_trainval_infos(nusc, label_dir)