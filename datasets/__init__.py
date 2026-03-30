import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as data
from torch.utils.data import ConcatDataset, DistributedSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from .nuscenes.nuscenes_dataset import NuScenesBEVDataset
from .transform import (
    BEVAugmentation,
    ColorEnhancement,
    Compose,
    ContrastEnhancement,
    LightNoiseEnhancement,
    Normalize,
    Resize,
    ToTensor,
)
from .waymo.waymo_dataset import WaymoBEVDataset
#from .pedcorner.pedcorner_dataset import PedcornerBEVDataset


class UnifiedBEVDataset:
    """统一的BEV数据集加载器，支持多GPU分布式训练"""
    
    DATASET_MAPPING = {
        'nuscenes': NuScenesBEVDataset,
        'waymo': WaymoBEVDataset,
        'pedestrian': WaymoBEVDataset,
        'corner': WaymoBEVDataset,
    }
    
    def __init__(self, config: Dict, distributed: bool = False):
        """
        Args:
            config: 数据集配置字典
            distributed: 是否使用分布式训练
        """
        self.config = config
        self.distributed = distributed
        self.complemented_det3d_channel = config['complemented_det3d_channel']
        self.complemented_seg_channel = config['complemented_seg_channel']
        self.down_scale_factor = config['down_scale_factor']
        self._initialize_transforms()
        
    def _initialize_transforms(self):
        """初始化数据变换"""
        if 'transform' not in self.config:
            raise ValueError("Transform configuration is required")
        
        transform_config = self.config['transform']
        
        # 分别获取训练和验证的配置
        train_config = transform_config.get('train', {})
        val_config = transform_config.get('val', {})
        
        if 'shape' not in train_config or 'shape' not in val_config:
            raise ValueError("Transform shape is required for both train and val")
        
        # 获取增强配置
        aug_config = train_config.get('augmentation', {})
        if aug_config.get('enabled', False):
            # 训练模式的变换（包含增强）
            train_transforms = [
                Resize(size=train_config['shape']),  # 首先resize到指定尺寸
                BEVAugmentation(
                    severity=aug_config.get('severity', 0.5),
                    color_prob=aug_config.get('color_prob', 0.8),
                    noise_prob=aug_config.get('noise_prob', 0.6),
                    blur_prob=aug_config.get('blur_prob', 0.4)
                ),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        else:
            # 训练模式的变换（无增强）
            train_transforms = [
                Resize(size=train_config['shape'], scale_factor=self.down_scale_factor),  # 首先resize到指定尺寸
                # 增加图像变换
                ColorEnhancement(
                        brightness_limit=0.15,
                        contrast_limit=0.15,
                        saturation_limit=0.2,
                        hue_shift_limit=15,
                        prob=0.7,
                    ),
                ContrastEnhancement(
                        gamma_limit=(85, 115),
                        clahe_clip_limit=1.5,
                        clahe_tile_grid_size=(6, 6),
                        prob=0.6,
                    ),
                LightNoiseEnhancement(
                        gauss_noise_var_limit=(5.0, 25.0),
                        multiplicative_noise_multiplier=(0.95, 1.05),
                        iso_noise_color_shift=(0.005, 0.02),
                        prob=0.4,
                    ), 
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        
        # 验证/测试模式的变换
        val_transforms = [
            Resize(size=val_config['shape']),  # 首先resize到指定尺寸
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        self.transforms = {
            'train': Compose(train_transforms),
            'val': Compose(val_transforms)
        }
    
    def _load_single_dataset(self, dataset_name: str, dataset_config: Dict, mode: str) -> data.Dataset:
        """加载单个数据集
        
        Args:
            dataset_name: 数据集名称
            dataset_config: 数据集配置
            mode: 'train' 或 'val'
        """
        if dataset_name not in self.DATASET_MAPPING:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        dataset_class = self.DATASET_MAPPING[dataset_name]
        
        # 修复：直接创建数据集实例，不使用 BEVDataset 包装
        dataset_instance = dataset_class(
            annotations_file=dataset_config['annotations_file'],
            category_mapping=dataset_config.get('category_mapping', None),
            bev_width_m=dataset_config.get('bev_width_m', 30),
            bev_height_m=dataset_config.get('bev_height_m', 30),
            pixel_per_meter=dataset_config.get('pixel_per_meter', 10),
            transform=self.transforms[mode],
            generate_det_gt_format = dataset_config['generation_gt'],
            complemented_det3d_channel = self.complemented_det3d_channel,
            complemented_seg_channel = self.complemented_seg_channel,
            with_det3d_label = dataset_config['with_det3d_label'],
            with_seg_label = dataset_config['with_seg_label']
        )
        
        return dataset_instance
    
    def _create_distributed_sampler(self, dataset, mode: str) -> Optional[DistributedSampler]:
        """创建分布式采样器"""
        if not self.distributed:
            return None
            
        return DistributedSampler(
            dataset,
            shuffle=(mode == 'train'),
            drop_last=(mode == 'train')
        )
        
    def _create_weighted_sampler(self, datasets, weights, total_size: int) -> WeightedRandomSampler:
        """创建加权采样器"""
        sample_weights = []
        for dataset, weight in zip(datasets, weights):
            sample_weights.extend([weight] * len(dataset))
        sample_weights = torch.DoubleTensor(sample_weights)

        # 输出权重统计信息
        logging.info("Sample Weights Statistics:")
        for i, (dataset, weight) in enumerate(zip(datasets, weights)):
            dataset_name = list(self.config['train'].keys())[i] if i < len(self.config['train']) else f"dataset_{i}"
            logging.info(f"  [{dataset_name}] samples: {len(dataset)}, weight: {weight}")
        logging.info(f"  Total sample_weights: {len(sample_weights)}")
        logging.info(f"  Weight distribution: {dict(zip([list(self.config['train'].keys())[i] if i < len(self.config['train']) else f'dataset_{i}' for i in range(len(datasets))], [w.item() for w in torch.tensor(weights)]))}")
        
        if self.distributed:
            # 在分布式训练中，每个进程只需要处理部分数据
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            samples_per_gpu = total_size // world_size
            offset = samples_per_gpu * rank
            
            # 确保每个GPU获得不同的数据
            g = torch.Generator()
            g.manual_seed(torch.initial_seed() + rank)
            
            return WeightedRandomSampler(
                sample_weights,
                num_samples=samples_per_gpu,
                replacement=True,
                generator=g
            )
        else:
            return WeightedRandomSampler(
                sample_weights,
                num_samples=total_size,
                replacement=True
            )
    
    def get_dataloader(self, 
                      mode: str,
                      batch_size: int = 1,
                      num_workers: int = 4,
                      shuffle: bool = None) -> torch.utils.data.DataLoader:
        """
        获取指定模式的数据加载器
        
        Args:
            mode: 'train' 或 'val'
            batch_size: batch大小
            num_workers: 数据加载线程数
            shuffle: 是否打乱数据。若为None，则train模式下为True，val模式下为False
        """
        if mode not in self.config:
            raise ValueError(f"Mode {mode} not found in config")
            
        if shuffle is None:
            shuffle = (mode == 'train')
            
        # 加载所有数据集
        datasets = []
        weights = []
        total_size = 0
        
        for dataset_name, dataset_config in self.config[mode].items():
            dataset = self._load_single_dataset(dataset_name, dataset_config, mode)
            datasets.append(dataset)
            weights.append(dataset_config.get('weight', 1.0))
            total_size += len(dataset)
            
            # 添加调试信息
            logging.info(f"Loaded {dataset_name} dataset: {len(dataset)} samples")
        
        if len(datasets) == 1:
            combined_dataset = datasets[0]
            if self.distributed:
                sampler = self._create_distributed_sampler(combined_dataset, mode)
            else:
                sampler = None
        else:
            combined_dataset = ConcatDataset(datasets)
            
            if self.distributed:
                # 分布式训练时，根据是否需要加权采样选择采样器
                if mode == 'train' and any(w != 1.0 for w in weights):
                    sampler = self._create_weighted_sampler(datasets, weights, total_size)
                else:
                    sampler = self._create_distributed_sampler(combined_dataset, mode)
            else:
                # 非分布式训练时，只在需要加权采样时使用WeightedRandomSampler
                if mode == 'train' and any(w != 1.0 for w in weights):
                    sampler = self._create_weighted_sampler(datasets, weights, total_size)
                else:
                    sampler = None
        
        # 添加调试信息
        logging.info(f"Combined dataset size: {len(combined_dataset)}")
        logging.info(f"Total samples: {total_size}")
        
        return torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(mode == 'train'),
            collate_fn=collate_fn
        )

def collate_fn(batch):
    """自定义的collate函数，处理BEV数据"""
    keys = batch[0].keys()
    outputs = {}
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            outputs[k] = torch.stack([b[k] for b in batch])
        elif isinstance(batch[0][k], np.ndarray):
            outputs[k] = torch.stack([torch.from_numpy(b[k]) for b in batch])
        elif isinstance(batch[0][k], bool):
            outputs[k] = torch.tensor([b[k] for b in batch])
        elif isinstance(batch[0][k], list):
            outputs[k] = [b[k] for b in batch]
        else:
            outputs[k] = [b[k] for b in batch]
    return outputs

# 使用示例:
if __name__ == "__main__":
    # 测试配置
    config = {
        'train': {
            'nuscenes': {
                'data_root': './sample_data',
                'annotations_dir': 'sample_annotations',
                'category_mapping': [['car', 'truck', 'bus', 'trailer', 'construction_vehicle'],
                                    ['pedestrian'],
                                    ['motorcycle', 'bicycle'],
                                    ['barrier']],
                'bev_width_m': 30,
                'bev_height_m': 30,
                'pixel_per_meter': 10,
                'weight': 1.0
            }
        },
        'val': {
            'nuscenes': {
                'data_root': './sample_data',
                'annotations_dir': 'sample_annotations',
                'category_mapping': [['car', 'truck', 'bus', 'trailer', 'construction_vehicle'],
                                    ['pedestrian'],
                                    ['motorcycle', 'bicycle'],
                                    ['barrier']],
                'bev_width_m': 30,
                'bev_height_m': 30,
                'pixel_per_meter': 10
            }
        },
        'transform': {
            'train': {
                'shape': (352, 640),
                'augmentation': {
                    'enabled': False,
                    'severity': 0.5,
                    'color_prob': 0.8,
                    'noise_prob': 0.6,
                    'blur_prob': 0.4
                }
            },
            'val': {
                'shape': (352, 640)
            }
        }
    }
    
    # 创建数据集
    dataset = UnifiedBEVDataset(config, distributed=False)
    print("BEV dataset loader created successfully!")
    
    # 获取数据加载器
    train_loader = dataset.get_dataloader('train', batch_size=2, num_workers=2)
    val_loader = dataset.get_dataloader('val', batch_size=1, num_workers=2)
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")