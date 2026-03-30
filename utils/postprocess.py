import torch


def decode_det_predictions(det3d_pred, category_num = 3, score_threshold=0.3):
    """
    解码预测结果
    Args:
        det3d_pred: 1 * pd_channels * feature_height * feature_width
    Returns:
        decoded_boxes: List[List[Dict]] 解码后的3D边界框
    """
    det3d_pred = det3d_pred.squeeze()
    channels_each_group = det3d_pred.shape[0] // category_num
    all_decoded_boxes = []
    
    for i in range(category_num):
        det_channels = det3d_pred[i*channels_each_group:(i+1)*channels_each_group,:,:]
        center_map = det_channels[0:1,:,:]
        offset_map = det_channels[1:3,:,:]
        height_map = det_channels[3:4,:,:]
        lwh_map = det_channels[4:7,:,:]
        yaw_map = det_channels[7:9,:,:]
        
        decoded_boxes = []
        decode_center_map(center_map, decoded_boxes, threshold=score_threshold)
        decode_offset_map(offset_map, decoded_boxes)
        decode_lwh_map(lwh_map, decoded_boxes)
        decode_height_map(height_map, decoded_boxes)
        decode_yaw_map(yaw_map, decoded_boxes)
        for obj in decoded_boxes:
            obj['class_id'] = i
        all_decoded_boxes.extend(decoded_boxes)
    return all_decoded_boxes

def decode_seg_predictions(seg_pred, score_threshold_list=[0.5,0.5,0.5]):
    '''
        seg_pred: [channel_num, height, width]
    '''
    seg_pred = seg_pred.sigmoid().squeeze(axis=0)
    for i in range(seg_pred.shape[0]):
        seg_pred[i] = seg_pred[i] > score_threshold_list[i]
    seg_pred = seg_pred.to(torch.float).cpu().numpy()
    return seg_pred


def decode_center_map(center_map, objects, threshold=0.3):
    center_map = center_map.sigmoid()
    centers = decode_center_points(center_map, threshold, distance=5)
    for center in centers:
        obj = {'center_int': (int(center[0]), int(center[1])),
                'score': float(center[2])}
        objects.append(obj)
    return objects 


def decode_offset_map(offset_map, objects):
    for obj in objects:
        x, y = obj['center_int']
        offset_x, offset_y = offset_map[:, y, x]
        offset_x, offset_y = float(offset_x), float(offset_y)
        offset_x, offset_y = max(0, min(offset_x, 1)), max(0, min(offset_y, 1))
        obj['center'] = (x + offset_x, y + offset_y)
    return objects


def decode_lwh_map(lwh_map, objects):
    for obj in objects:
        x, y = obj['center_int']
        l, w, h = [float(elem) for elem in lwh_map[:, y, x]]
        obj['wlh'] = [w,l,h]
    return objects


def decode_height_map(height_map, objects):
    for obj in objects:
        x, y = obj['center_int']
        height = height_map[:, y, x]
        obj['height'] = float(height)
    return objects

def decode_yaw_map(yaw_map, objects):
    for obj in objects:
        x, y = obj['center_int']
        sin_val, cos_val = yaw_map[:, y, x]
        yaw = torch.atan2(sin_val, cos_val)
        obj['yaw'] = float(yaw)
    return objects

def decode_category_map(category_map, objects):    
    for obj in objects:
        x, y = obj['center_int']
        category_vec = category_map[:, y, x]
        cur_category_id = category_vec.argmax()
        obj['class_id'] = int(cur_category_id)
    return objects


def decode_center_points(center_map, threshold=0.3, distance=2):
    center_map = center_map.squeeze()
    ys, xs = torch.where(center_map >= threshold)
    if len(xs) == 0:
            return torch.zeros((0, 3), dtype=torch.float32)
    centers = torch.zeros((len(xs), 3), dtype=torch.float32)
    centers[:, 0] = xs
    centers[:, 1] = ys
    centers[:, 2] = center_map[ys, xs]
    centers = perform_nms_on_points(centers, distance)
    return centers


def perform_nms_on_points(points, min_distance):
    number = len(points)
    if number <= 1:
        return points
    indices = torch.argsort(-points[:, 2])
    points = points[indices]
    is_valid = torch.ones(number, dtype=torch.bool)
    min_distance = min_distance ** 2
    for i in range(0, number - 1):
        x1, y1, _ = points[i]
        for j in range(i + 1, number):
            if is_valid[j] == False:
                continue
            x2, y2, _ = points[j]
            dist = (x1 - x2) ** 2 + (y1 - y2) ** 2
            if dist < min_distance:
                is_valid[j] = False
    points = points[is_valid]
    return points


def postprocess_predictions(predictions, 
                            det_score_threshold=0.3, 
                            seg_score_list = [0.5, 0.5, 0.5],
                            class_names=['vehicle', 'person', 'cyclist']):
    """后处理模型预测结果"""
    det3d_pred = predictions['det3d_pred']
    seg_pred = predictions['seg_pred']
       
    # 解码预测结果
    decoded_boxes = decode_det_predictions(det3d_pred, category_num=len(class_names), score_threshold=det_score_threshold)
    seg_lane_maps = decode_seg_predictions(seg_pred, score_threshold_list=seg_score_list)
    
    
    # 转换为可视化格式
    viz_boxes = []
    for box in decoded_boxes:
        if box is not None:
            # 转换为check_bev期望的格式
            viz_box = {
                'center': box['center'],
                # 'height': box['height'],
                'wlh': box['wlh'],
                'yaw': box['yaw'],
                'category': class_names[box['class_id']],
                'score': box['score']
            }
            # print(viz_box)
            viz_boxes.append(viz_box)
    
    return viz_boxes, seg_lane_maps