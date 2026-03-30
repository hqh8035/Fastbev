import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append('.')
from configs import get_cfg_defaults
from datasets import UnifiedBEVDataset
from model.fastbev import FastBEV

# 抑制libibverbs相关警告
os.environ['IBV_FORK_SAFE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

def setup_validation_logging(log_dir, is_main_process=True):
    """设置验证日志"""
    if is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'validation.log')),
                logging.StreamHandler()
            ]
        )
        return SummaryWriter(log_dir)
    return None

def setup_distributed(rank, world_size):
    """设置分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def validate(model, val_loader, cfg, writer=None, epoch=None, is_main_process=True, distributed=False, rank=0):
    """验证函数"""
    model.eval()
    device = next(model.parameters()).device
    
    # 初始化指标记录
    epoch_metrics = {
        'loss': {'values': []},
        'heatmap_loss': {'values': []},
        'reg_loss_6d': {'values': []},
        'yaw_loss': {'values': []}
    }
    
    if is_main_process:
        pbar = tqdm(total=len(val_loader), desc='Validating')
    
    with torch.no_grad():
        for batch in val_loader:
            # 将数据移到正确的设备上
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device, non_blocking=True)
            
            # 模型推理
            outputs = model(batch)
            
            # 获取损失
            if hasattr(model, 'module'):
                loss, loss_info = model.module.get_loss(outputs, batch)
            else:
                loss, loss_info = model.get_loss(outputs, batch)
            
            # 记录各种损失
            epoch_metrics['loss']['values'].append(loss.item())
            epoch_metrics['heatmap_loss']['values'].append(loss_info.get('heatmap_loss', 0.0))
            epoch_metrics['reg_loss_6d']['values'].append(loss_info.get('reg_loss_6d', 0.0))
            epoch_metrics['yaw_loss']['values'].append(loss_info.get('yaw_loss', 0.0))
            
            if is_main_process:
                pbar.update(1)
    
    if is_main_process:
        pbar.close()
    
    # 在分布式模式下同步指标
    if distributed:
        for metric_name in epoch_metrics:
            gathered_values = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_values, epoch_metrics[metric_name]['values'])
            
            if is_main_process:
                all_values = []
                for values in gathered_values:
                    all_values.extend(values)
                epoch_metrics[metric_name]['values'] = all_values
    
    # 计算平均值
    results = {}
    if is_main_process:
        for metric_name in epoch_metrics:
            values = epoch_metrics[metric_name]['values']
            if values:
                results[metric_name] = sum(values) / len(values)
                
                if writer is not None and epoch is not None:
                    writer.add_scalar(f'Val/{metric_name}', results[metric_name], epoch)
        
        # 记录到日志
        log_str = f"Validation metrics - "
        for metric_name, metric_value in results.items():
            log_str += f"{metric_name}: {metric_value:.4f}, "
        logging.info(log_str[:-2])
    
    return results if is_main_process else None

def validate_checkpoint(rank, world_size, cfg, checkpoint_path, distributed=False):
    """验证指定检查点的性能"""
    try:
        is_main_process = (not distributed) or (rank == 0)
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        if distributed:
            setup_distributed(rank, world_size)
        
        # 设置日志
        writer = setup_validation_logging(cfg.PATH.LOG_DIR, is_main_process)
        
        if is_main_process:
            logging.info("Starting validation with config:")
            logging.info(f"Checkpoint: {checkpoint_path}")
            logging.info(f"Distributed: {distributed}")
            if distributed:
                logging.info(f"World size: {world_size}")
        
        # 创建验证数据加载器
        dataset_config = {
            'train': cfg.DATASET.TRAIN,
            'val': cfg.DATASET.VAL,
            'transform': {
                'train': {
                    'shape': cfg.DATASET.TRANSFORM.train.shape,
                    'augmentation': {
                        'enabled': cfg.DATASET.TRANSFORM.train.augmentation.enabled,
                        'severity': cfg.DATASET.TRANSFORM.train.augmentation.severity,
                        'color_prob': cfg.DATASET.TRANSFORM.train.augmentation.color_prob,
                        'noise_prob': cfg.DATASET.TRANSFORM.train.augmentation.noise_prob,
                        'blur_prob': cfg.DATASET.TRANSFORM.train.augmentation.blur_prob,
                    }
                },
                'val': {
                    'shape': cfg.DATASET.TRANSFORM.val.shape
                }
            }
        }
        
        # 设置 DataLoader 的共享内存参数（与train.py保持一致）
        torch.multiprocessing.set_sharing_strategy('file_system')
        
        # 创建数据集
        dataset = UnifiedBEVDataset(dataset_config, distributed=distributed)
        val_loader = dataset.get_dataloader('val', 
                                          batch_size=8, 
                                          num_workers=cfg.TRAIN.NUM_WORKERS)
        
        if is_main_process:
            logging.info(f"Validation dataset size: {len(val_loader.dataset)}")
        
        # 创建模型并加载检查点
        if cfg.MODEL.NAME == 'FastBEV':
            # 使用与train.py相同的模型创建方式
            model = FastBEV(
                x_range=cfg.MODEL.FASTBEV.X_RANGE,
                z_range=cfg.MODEL.FASTBEV.Z_RANGE,
                y_range=cfg.MODEL.FASTBEV.Y_RANGE,
                pixel_per_meter_h=cfg.MODEL.FASTBEV.PIXEL_PER_METER_H,
                pixel_per_meter_v=cfg.MODEL.FASTBEV.PIXEL_PER_METER_V,
                detection_channels=cfg.MODEL.FASTBEV.DETECTION_CHANNELS,
                seg_classes=cfg.MODEL.FASTBEV.SEG_CLASSES,
                det_classes=cfg.MODEL.FASTBEV.DET_CLASSES,
                backbone=cfg.MODEL.FASTBEV.BACKBONE,
                neck=cfg.MODEL.FASTBEV.NECK
            )
        else:
            raise ValueError(f"Invalid model name: {cfg.MODEL.NAME}")
        
        model = model.to(device)
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 处理分布式训练的state_dict
        if any(k.startswith('module.') for k in state_dict.keys()):
            if not any(k.startswith('module.') for k in model.state_dict().keys()):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=True)
        
        if is_main_process:
            logging.info(f"Loaded checkpoint from: {checkpoint_path}")
            if 'epoch' in checkpoint:
                logging.info(f"Checkpoint epoch: {checkpoint['epoch']}")
            if 'best_loss' in checkpoint:
                logging.info(f"Checkpoint best loss: {checkpoint['best_loss']:.4f}")
        
        # 如果使用分布式，包装模型
        if distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[rank], output_device=rank)
        
        # 运行验证
        metrics = validate(model, val_loader, cfg, writer, 
                         is_main_process=is_main_process,
                         distributed=distributed,
                         rank=rank)
        
        # 输出结果
        if is_main_process and metrics is not None:
            logging.info("\nValidation Results:")
            for metric_name, metric_value in metrics.items():
                logging.info(f"{metric_name}: {metric_value:.4f}")
        
        if is_main_process and writer is not None:
            writer.close()
        
        return metrics if is_main_process else None
        
    except Exception as e:
        logging.error(f"Error in validate_checkpoint: {str(e)}")
        if distributed:
            cleanup_distributed()
        raise e
    finally:
        if distributed:
            cleanup_distributed()

def main():
    """独立运行时的主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='fastbev_v1', help='config name')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint path')
    parser.add_argument('--distributed', action='store_true', help='whether to use distributed validation')
    args = parser.parse_args()
    
    try:
        # 加载配置
        cfg = get_cfg_defaults(args.config)
        cfg.freeze()
        
        if args.distributed:
            # 获取可见的GPU数量
            n_gpus = torch.cuda.device_count()
            if n_gpus < 2:
                logging.warning(f'Distributed validation requires multiple GPUs, but only found {n_gpus} GPU(s).')
                logging.info('Falling back to single GPU validation.')
                validate_checkpoint(0, 1, cfg, args.checkpoint, distributed=False)
            else:
                logging.info(f'Starting distributed validation with {n_gpus} GPUs')
                mp.spawn(
                    validate_checkpoint,
                    args=(n_gpus, cfg, args.checkpoint, True),
                    nprocs=n_gpus
                )
        else:
            validate_checkpoint(0, 1, cfg, args.checkpoint, distributed=False)
            
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
