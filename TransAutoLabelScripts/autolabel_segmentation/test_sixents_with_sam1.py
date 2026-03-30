import cv2
import os
import time
import pickle
import copy
import glob
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def get_color_from_id(cls_id: int, num_classes: int = 20):
        """
        根据类别 id 生成颜色（BGR格式）
        :param cls_id: 类别索引（0 ~ num_classes-1）
        :param num_classes: 类别总数，用于分配色相空间
        :return: (B, G, R) 三元组
        """
        # 将类别映射到 0~179 (OpenCV HSV色相范围)
        hue = int(179 * cls_id / num_classes)
        color = np.uint8([[[hue, 255, 255]]])  # 饱和度和亮度固定为最大
        bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(int(c) for c in bgr)

def visualize_segmentation(image_array, masks):
    num = len(masks)
    for i in range(num):
        cur_seg_part = masks[i]['segmentation']
        cur_color = get_color_from_id(i, num)
        image_array[cur_seg_part] = cur_color
    return image_array


def get_segmentation_maps(image_array, masks):
    num = len(masks)
    map_list = []
    
    for i in range(num):
        cur_map= copy.deepcopy(image_array)
        img_h, img_w = cur_map.shape[:2]
        cur_seg_part = masks[i]['segmentation']
        x0, y0, w, h = masks[i]['bbox']
        x1 = x0 + w
        y1 = y0 + h
        
        x0 = max(0, int(x0))
        x1 = min(int(x1), img_w)
        
        y0 = max(0, int(y0))
        y1 = min(int(y1), img_h)
        
         
        
        cur_color = get_color_from_id(i, num)
        cur_map[cur_seg_part] = cur_color
        cur_map = cv2.addWeighted(image_array, 0.5, cur_map, 0.5, 0)
        
        cv2.rectangle(cur_map, pt1=[x0, y0], pt2=[x1,y1], color=[0,0,255], thickness=2)
        map_list.append(cur_map)
    return map_list 



if __name__ == "__main__":

    sam = sam_model_registry["vit_h"](checkpoint="TransAutoLabelScripts/autolabel_models/sam_vit_h_4b8939.pth")
    # mask_generator = SamAutomaticMaskGenerator(sam)         # version3
    
    mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)
    
    
    
    src_dir = 'results'
    dst_dir = 'segmentation_maps'
    
    os.makedirs(dst_dir, exist_ok=True)
    
    infer_time = 0
    src_image_list = glob.glob(os.path.join(src_dir, '*.png'))
    for idx in range(len(src_image_list)):
        print(f'idx is {idx}')
        img_file = src_image_list[idx]
        img_name = os.path.basename(img_file)
        dst_map_dir = os.path.join(dst_dir, img_name + '_maps')
        os.makedirs(dst_map_dir, exist_ok=True)
        
        # img_file = "results/946685984.561260032.png"
        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        star_time = time.time()
        masks = mask_generator.generate(image)              # 生成mask
        end_time = time.time()
        during_time = end_time - star_time
        infer_time += during_time
        
        with open(os.path.join(dst_dir, img_name+'.pkl'), 'wb') as f:
            pickle.dump(masks, f)
        
        
        image_array = cv2.imread(img_file)
        
        map_list = get_segmentation_maps(image_array, masks)
        
        for i in range(len(map_list)):
            cur_map_name = img_name + '_map'+str(i)+'.jpg'
            cur_map_file = os.path.join(dst_map_dir, cur_map_name)
            cv2.imwrite(cur_map_file, map_list[i])
        
        # visualize the generated maps
        background_image = copy.deepcopy(image_array)
        visual_res = visualize_segmentation(background_image, masks)
        concate_image = cv2.addWeighted(visual_res, 0.5, image_array, 0.5, 0)
        dst_img_file = os.path.join(dst_dir, img_name+'_visualize_res.jpg')
        cv2.imwrite(dst_img_file, concate_image)
        
    avg_infer_time = infer_time / len(src_image_list)
    print(f'average infer time is {avg_infer_time}')
    a = 1