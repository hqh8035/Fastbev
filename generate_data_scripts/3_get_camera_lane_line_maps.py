import os
import json
import glob
import cv2
import numpy as np
from pyquaternion import Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

def generate_video(image_files):
    tmp_image_file = image_files[0]
    tmp_image = cv2.imread(tmp_image_file)
    h, w = tmp_image.shape[:2]
    video_path = 'bev_lane_map.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 20, (w, h))
    for idx, tmp_path in enumerate(image_files):
        print(idx)
        src_path = os.path.join(tmp_path)
        image = cv2.imread(src_path)
        image = cv2.putText(image, str(idx), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=(0, 0, 255), lineType=cv2.LINE_AA, thickness=3)
        video.write(image)
        
        
locations = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']

def get_nuscenes(version = 'v1.0-mini', root_path='mini_data/nuscenes', verbose=True):
       
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')
    
    nusc = NuScenes(version=version, dataroot=root_path, verbose=verbose)
    return nusc, train_scenes, val_scenes

def get_scene2map(nusc):
    scene2map = {}
    for rec in nusc.scene:
        log = nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']
    return scene2map


def get_nusc_maps(map_folder):
    nusc_maps = {
        map_name: NuScenesMap(dataroot=map_folder, map_name=map_name) for map_name in locations
    }
    return nusc_maps



def get_lidar_line_mask(TransMat_lidar2global, patch_size, maps_dict, location, classes=['divider'], canvas_size=None):

    map_pose = TransMat_lidar2global[:2, 3]
    patch_box = (map_pose[0], map_pose[1], patch_size[0], patch_size[1])

    rotation = TransMat_lidar2global[:3, :3]
    v = np.dot(rotation, np.array([1, 0, 0]))
    yaw = np.arctan2(v[1], v[0])
    patch_angle = yaw / np.pi * 180
    
    masks = maps_dict[location].get_map_mask(
        patch_box=patch_box,
        patch_angle=patch_angle,
        layer_names=classes,
        canvas_size=canvas_size,
    )        
    return masks



if __name__ == "__main__":
    
    nusc, train_scenes, val_scenes = get_nuscenes(version='v1.0-trainval',
                                                  root_path='data/nuscenes')
    scene2map = get_scene2map(nusc)
    maps_dict = get_nusc_maps(map_folder='data/nuscenes')
    
    visualize = False
    if visualize:
        dst_dir = 'visualize_lane_line_mask'
        os.makedirs(dst_dir, exist_ok=True)
    
    label_dir = 'all_labels/nuscenes_seg_labels'
    os.makedirs(label_dir, exist_ok=True)
    
    for sample_idx, sample in enumerate(nusc.sample):
        print(f'processing {sample_idx}/{len(nusc.sample)}')
        sample = nusc.sample[sample_idx]
        sample_token = sample['token']

        sample_record = nusc.get('sample', sample_token)
        cam_token = sample_record['data']['CAM_FRONT']
        cam_record = nusc.get('sample_data', cam_token)
        cam_path = nusc.get_sample_data_path(cam_token)
        cam_name = os.path.basename(cam_path)
        
        scene_record = nusc.get('scene', sample_record['scene_token'])    
        scene_name = scene_record['name']
        location = scene2map[scene_name]
        
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])       # sample_data_record
        cs_record = nusc.get('calibrated_sensor',                           # calibrate_sample_record
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])  
        
        lidar2ego_translation = cs_record['translation']      # lidar2ego translation
        lidar2ego_rotation = cs_record['rotation']           # lidar2ego rotation quaternion
        ego2global_translation = pose_record['translation']   # ego2global translation
        ego2global_rotation = pose_record['rotation']     
        
        lidar2global_rotation = Quaternion(ego2global_rotation).rotation_matrix @ Quaternion(lidar2ego_rotation).rotation_matrix
        lidar2gloabl_translation = Quaternion(ego2global_rotation).rotation_matrix @ lidar2ego_translation + ego2global_translation
        
        
        # lidar
        TransMat_lidar2global = np.zeros([4,4])
        TransMat_lidar2global[:3,:3] = lidar2global_rotation
        TransMat_lidar2global[:3,3] = lidar2gloabl_translation
        TransMat_lidar2global[3,3] = 1
        
        
        # get the relative position of lidar to camera
        label_file = os.path.join('all_labels/nuscenes_bbox3d_labels', os.path.basename(cam_path).replace('.jpg', '.json'))
        with open(label_file, 'r') as f:
            json_dict = json.load(f)
            
        T_cam2lidar = json_dict['cam2lidar_translation'] 
        
        
        actual_bev_x_distance = 19.2            # m， 左右对称取19.2m
        actual_bev_y_distance = 32              # m, 以cam作为远点向前取32m
        patch_h = 40*2                # actual_bev_y_distance,便于后续截取, 以中心截取，所以需要*2
        patch_w = 30                # 大于actual_bev_x_distance,便于后续截取
        pixels_per_meter = 10
        # classes = ['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'road_divider', 'lane_divider']
        classes = ['drivable_area', 'ped_crossing', 'walkway', 'road_divider', 'lane_divider']
        
        line_mask = get_lidar_line_mask(TransMat_lidar2global, [patch_h, patch_w], maps_dict, location=location, classes=classes)

        # lidar in the center, use the offset to get the position of camera
        x_offset = T_cam2lidar[0] * pixels_per_meter
        y_offset = T_cam2lidar[1] * pixels_per_meter
        lidar_center = np.array([line_mask.shape[2]//2, line_mask.shape[1] //2])
        cam_center = lidar_center + np.array([x_offset, y_offset])
        
        # cam center tor crop
        left_x, right_x = int(cam_center[0]-actual_bev_x_distance/2*pixels_per_meter), int(cam_center[0]+actual_bev_x_distance/2*pixels_per_meter)
        start_y, forward_y = int(cam_center[1]), int(cam_center[1]+ actual_bev_y_distance*pixels_per_meter)
        
        actual_mask = np.zeros([len(classes), 
                                int(actual_bev_y_distance*pixels_per_meter),
                                int(actual_bev_x_distance*pixels_per_meter)])
        for i in range(len(classes)):
            cur_class = classes[i]
            cur_mask = line_mask[i]
            cur_mask = cur_mask[start_y:forward_y, left_x:right_x]
            cur_mask = cur_mask[::-1,:]
            actual_mask[i][cur_mask.astype(np.bool_)] = i+1
  
        np.save(os.path.join(label_dir, os.path.basename(cam_path).replace('.jpg', '.npy')),
                    actual_mask)
        
        if visualize:
            image = cv2.imread(cam_path)
            background_map = np.ones([patch_h*pixels_per_meter,patch_w*pixels_per_meter,3])*255
            
            
            color_list = [
                [0,255,0],          # green     drivable_area
                [0,0,255],          # red       ped_crossing
                [255,0,0],           # blue     walkway
                # [0,255,255],        # yellow    stop_line           # we dont't need that so discard
                [0,0,0],         # cyan     road_divider
                [0,0,0]               # black   land_divider
            ]
            
            for map_idx in range(line_mask.shape[0]):
                cur_map = line_mask[map_idx]
                ys, xs = np.where(cur_map != 0)
                background_map[ys,xs,:] = color_list[map_idx] 
                
            cv2.circle(background_map, cam_center.astype(np.int32), radius=3, color=[226, 43, 138], thickness=-1)
            actual_background_map = background_map[start_y:forward_y, left_x:right_x, :]
            actual_background_map = actual_background_map[::-1,:,:]
            ratio = actual_background_map.shape[0] / image.shape[0]
            image = cv2.resize(image, dsize=None, fx=ratio, fy=ratio)
            concate_image = np.hstack([actual_background_map, image])
            
            
            dst_image_file = os.path.join(dst_dir, cam_name.replace('.jpg', '.png'))
            cv2.imwrite(dst_image_file, concate_image)
    
    # generate_video()
    if visualize:
        image_files = glob.glob(os.path.join('test_lane_line_mask', '*.png'))
        image_files = sorted(image_files, key=lambda l:float(l.split('__')[-1].replace('.png', '')))
        generate_video(image_files)
    