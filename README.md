# nova_fastbev

## 数据

### bev 检测数据

当前数据包含了两部分：
- 一部分为开源的 Waymo 数据
- 一部分为六分的数据

#### 开源 Waymo 数据

- 路径：`/perception/users/chenfuxuan/WorkSpace_2025/Data/Public`

细节内容可以参考该路径下的 `README.md`，其中包含了数据的处理 / 可视化等说明。

#### 六分数据

- 用于 FastBEV 的数据需要逐渐积累迭代模型，累计过程参考：  
<https://lekqjyg0qj.feishu.cn/wiki/GA5vw7bIcihIwakGByIcFFPRnXg>

- 六分的数据抽帧之后会上传数据平台，从数据平台处理数据，参考路径：

`/perception/users/chenfuxuan/WorkSpace_2025/Data/liufen/scripts_to_process_data/trans_lidar2cam_label_from_platform.py` 用于完成数据平台上的数据转换为训练所需的数据。

- golf/采集车数据管理

   `/perception/users/chenfuxuan/WorkSpace_2025/Data/liufen/scripts_to_process_data/trans_lidar2cam_label_from_platform.py`中的collector_type参数，会根据采集平台类型(golf/采集车)进行区分

### 分割数据

树枝分割数据一般采用大模型进行处理，参考路径：

`TransAutoLabelScripts/autolabel_segmentation`

- `qwen_api_sim.py`：  
  利用 Qwen 大模型在前景中添加树枝，可以根据场景调整其中的 prompt

- `test_sixents_with_sam1.py`：  
  利用 SAM1 对生成的数据进行自动分割，请 QA 同学帮忙进行筛选，获取分割 label



## 环境配置

- 创建conda环境
```
conda env create -f environment.yml -n nova_fastbev
```
- pip一些包
```
pip install -r requirements.txt
```
- 安装数据平台的包 dp-data-common参考：https://lekqjyg0qj.feishu.cn/wiki/XTfpwSSUYiWKEhkCidwcd7fqnfc
```
pip install --upgrade dp-data-common --extra-index-url http://__token__:4mdAaaCbmWZRLnVpcVj9@192.168.3.224:8081/api/v4/projects/1450/packages/pypi/simple --trusted-host 192.168.3.224
```
- 配置环境变量
```
nano ~/.bashrc
```
在文件末尾加入：
```
# 数据库地址，固定填写
export DATABASE_SERVER="192.168.3.67:27017"

# 数据库用户名，前往数据平台“用户中心”/“个人信息”/“开发设置”查看
export DATABASE_USERNAME="数据库用户名"

# 数据库密码，前往数据平台“用户中心”/“个人信息”/“开发设置”查看
export DATABASE_PASSWORD="数据库密码"
```

验证：

保存后执行：
```
source ~/.bashrc

python -c "from dp_data_common.client.data_client import DataClient; print(f'found {len(DataClient().list_collections(\"db_shared\"))} collections.')"
```
应能看到如 `Found xxx collections.`的日志输出 

## 启动训练

### 配置数据
- `configs/config_fastbev_v6.py` 中 `_C.DATASET.TRAIN.waymo.annotations_file`记录对应的训练图像的路径，每次启动训练可以配置新的训练数据的路径

- 一些参数说明： 
    - `_C.TRAIN.SAVE_FREQ`: 设置为存储model的频率，例如2000次iter存储模型
    - `_C.TRAIN.LR`: 学习率
    -  `_C.TRAIN.BATCH_SIZE`: batch_size
    - `_C.PATH.CHECKPOINT_DIR`: 保存模型的路径

    其他参数一般无需改动

### 启动训练

- 单卡训练: `python scripts/train.py --config fastbev_v6` 

- 多卡训练：  
  ```bash
  export CUDA_VISIBLE_DEVICES="1,2"     # 根据gpu数量配置
  python scripts/train.py --config fastbev_v6 --distributed

## 模型导出
- torch环境导出：scripts/export_onnx.py
- 参数说明：
    ```bash
  --checkpoint: 训练的模型路径
  --camera_intrinsic: 相机内参
  --height: 原始数据的高度
  --width: 原始数据的宽度
  --origin_image_height: 原始数据的高度
  --origin_image_width: 原始数据的宽度
  --target_image_height：前处理之后的输入的高度
  --target_image_width: 前处理之后的输入的宽度

  当前输入尺寸为1088*1280，前处理resize之后的尺寸为816*960，参数配置一般只需要改动checkpoint和camera_intrinsic即可； 后续有改动同步改动其他参数即可

- onnx模型测试：scripts/test_onnx.py
- 参数说明：
    ```bash
    --onnx_model_path: onnx模型的路径
    --save_dir: 推理生成的保存结果
    --det3d_threshold：检测结果的阈值
    --seg_thres_list：分割结果的阈值
    --camera_intrinsic: 相机内参

    --image_dir:要验证的多张图像的路径
    --image_path: 单张验证图像的路径, 根据推理的图像设置推理使用image_path/image_dir参数


    --make_video: 推理结果是否要生成视频, 后续参数为生成video的参数配置
    --video_file
    --fps
    --frame_size
    
    --onnx_infer：默认为True, onnx模型推理
    
    当前参数只需改动对应的onnx路径/save_dir路径和相机内参，其他默认

- 地平线工具链参数导出：scripts/export_onnx_with_horizon.py, 参数和scripts/export_onnx.py相同

- 地平线工具链导出的onnx模型测试：scripts/test_onnx_horizon.py, 参数和 scripts/test_onnx.py相同

## 模型推理

- scripts/inference.py

- 参数说明：
    ```bash
    --model_path: 模型的路径
    --image_dir: 需要推理的图像的路径
    --save_dir：推理结果保存的路径

    --det_score：检测的阈值
    --seg_score_list：分割的阈值
    --camera_intrinsic: 相机内参

    --make_video: 推理结果是否要生成视频, 后续参数为生成video的参数配置
    --video_file
    --fps
    --frame_size
    
    当前参数只需改动对应的模型路径/save_dir路径和相机内参，其他默认

## 模型验证

- scripts/validation.py

- 参数说明：
    ```bash
    --annotation_file: 需要验证的带有标签的数据的记录
    --checkpoint: 模型路径
    --config：默认保持和训练时的fastbev_v6一致
    --output_dir：验证结果输出的路径

    --det3d_threshold： 检测的阈值
    --seg_score_list: 分割的阈值
    --with_det3d_label： 默认为True
    --with_seg_label: 默认为True
    --save_result: 是否保存推理的json结果
    
    当前参数只需改动对应的模型路径/s验证结果输出的路径和相机内参，其他默认,这里没有设置内参，因为验证过程的label带有内参

## 模型评估
- scripts/eval.py

- 参数说明：
    ```bash
    --gt_dir: 保存gt label的目录
    --pd_dir：模型预测的label的目录
    --distance：参与评估的bbox的范围，例distance设置为10, 只有10米内的bbox参与evaluation

    --z_min
    --z_max
    --x_min
    --x_max: 表征了bev的范围，一般无需进行改动

    --iou_thres：gt和pd匹配的阈值
    --bev_resolution: bev分辨率，当前为10 pixel/meter

## 制作推理效果视频

用于展示模型的效果，需要连续帧展示。

1. 拉取数据。目前选取的数据，拉取的命令为：

  ```shell
  conda activate nova_fastbev
  python auxilary_scripts/get_data_from_platform.py --database db_dev --collection demo_sixents_20260115 --query '{ "info.meta.car": "caijiche", "info.meta.collect_date": { "$in": [ "20251226","20251229"] } }' --dst_dir demo_dataset/
  ```

2. 生成图像列表。

  ```shell
  conda activate nova_fastbev
  python auxilary_scripts/gen_image_list_txt.py -d demo_dataset/left -o demo_image_list.txt
  ```

3. 推理并生成 mp4 视频。

  ```shell
  conda activate nova_fastbev
  python scripts/inference.py \
    --model_path checkpoints/fastbev_2026-01-14_16-31-11/checkpoint_step_48000.pth \
    --save_dir infer_result \
    --camera_intrinsic 524.31359863281250 0 657.63726806640625 0 524.31359863281250 519.53070068359375 0 0 1 \
    --frame_size 696 320 \
    --image_list_file demo_image_list.txt \
    --disable_seg
  ```