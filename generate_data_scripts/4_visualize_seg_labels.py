import os
import glob
import cv2
import json
import numpy as np


color_list = [
    [0,0,0],         # black     road_divider
    [0,0,0],         # black   land_divider
    [0,255,0],       # green     drivable_area
    [0,0,255],       # red       ped_crossing
    [255,0,0],       # blue     walkway
]


seg_labels = glob.glob(os.path.join('mini_labels/nuscenes_seg_labels', '*.npy'))
seg_labels = sorted(seg_labels, key=lambda l:float(os.path.basename(l).split('__')[-1].replace('.npy', '')))
det_labels = [x.replace('.npy', '.json').replace('seg_labels','bbox3d_labels') for x in seg_labels]


dst_visualize_image_dir = 'visualize_mini_seg_label_res'
os.makedirs(dst_visualize_image_dir, exist_ok=True)

for idx in range(len(seg_labels)):
    print(f'process {idx}...')
    cur_seg_label = seg_labels[idx]
    cur_det_label = det_labels[idx]
    with open(cur_det_label, 'r') as f:
        json_dict = json.load(f)
    cam_path = json_dict['image_path']
    image_name = os.path.basename(cam_path)
    image = cv2.imread(cam_path)
    
    line_mask = np.load(cur_seg_label)
    
    background_map = np.ones([320,192,3])*255
    for map_idx in range(line_mask.shape[0]):
        cur_map = line_mask[map_idx]
        ys, xs = np.where(cur_map != 0)
        background_map[ys,xs,:] = color_list[map_idx] 
    cv2.circle(background_map, [192//2, 320], radius=3, color=[226, 43, 138], thickness=-1)

    ratio = line_mask.shape[-2] / image.shape[0]
    image = cv2.resize(image, dsize=None, fx=ratio, fy=ratio)
    concate_image = np.hstack([background_map, image])
    concate_image = cv2.putText(concate_image, str(idx), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=(0, 0, 255), lineType=cv2.LINE_AA, thickness=3)
    cv2.imwrite(os.path.join(dst_visualize_image_dir, image_name), concate_image)
    # video.write(concate_image)

