import json
import os

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits


def extract_cam_front_images(nusc_root, out_root, version='v1.0-trainval'):
    """
    提取 nuScenes 数据集 train/val 的 CAM_FRONT 图像并分开保存

    Args:
        nusc_root (str): nuScenes 数据集根目录，比如 '/data/nuscenes'
        out_root (str): 输出文件夹，比如 './nuscenes_front_images'
        version (str): nuScenes 版本，默认 'v1.0-trainval'
    """
    nusc = NuScenes(version=version, dataroot=nusc_root, verbose=True)

    # 训练集和验证集场景
    train_scenes = splits.train
    val_scenes = splits.val

    # 建立输出目录
    train_out = os.path.join(out_root, "train")
    val_out = os.path.join(out_root, "val")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)

    # 获取所有 scene 名称到 token 的映射
    scene_name2token = {s['name']: s['token'] for s in nusc.scene}

    def save_images(scene_names):
        label_list = []
        for scene_name in scene_names:
            scene = nusc.get('scene', scene_name2token[scene_name])
            first_sample = nusc.get('sample', scene['first_sample_token'])

            sample_token = scene['first_sample_token']
            while sample_token != "":
                sample = nusc.get('sample', sample_token)
                cam_front_token = sample['data']['CAM_FRONT']
                cam_front = nusc.get('sample_data', cam_front_token)

                src_path = os.path.join(nusc_root, cam_front['filename'])
                cur_src_name = os.path.basename(src_path).replace('.jpg', '')
                label_list.append(cur_src_name)
                sample_token = sample['next']
        return label_list

    print("Extracting train images...")
    train_label_list = save_images(train_scenes)
    
    print("Extracting val images...")
    val_label_list = save_images(val_scenes)

    return train_label_list, val_label_list


if __name__ == "__main__":
    nusc_root = "data/nuscenes"        # 修改成你的 nuScenes 路径
    out_root = "./nuscenes_cam_front"      # 输出路径
    train_label_list, val_label_list = extract_cam_front_images(nusc_root, out_root)
    
    with open('train_label_list.json', 'w') as f:
        json.dump(train_label_list, f)
    
    with open('val_label_list.json', 'w') as f:
        json.dump(val_label_list, f)