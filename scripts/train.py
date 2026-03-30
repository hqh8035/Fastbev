import os
import socket
import sys

os.environ["IBV_FORK_SAFE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# 设置多进程策略 - 必须在主进程开始时设置
import torch

torch.multiprocessing.set_sharing_strategy("file_system")
torch.multiprocessing.set_start_method("spawn", force=True)

sys.path.append(".")
import argparse
import logging
from datetime import datetime

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from configs import get_cfg_defaults
from datasets import UnifiedBEVDataset
from model.fastbev import FastBEV


def find_free_port():
    """查找空闲端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class Trainer:
    """训练器类，支持单卡和多卡训练"""

    def __init__(self, cfg, rank=0, world_size=1, distributed=False):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.distributed = distributed
        self.is_main_process = (not distributed) or (rank == 0)

        # 设置设备 - 修正CUDA设备映射
        if torch.cuda.is_available():
            # rank是相对于可见GPU的索引，不是绝对GPU ID
            self.device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(rank)
        else:
            self.device = torch.device("cpu")

        # 初始化分布式训练
        if distributed:
            self._setup_distributed()

        # 设置日志和checkpoint目录
        self.writer, self.checkpoint_save_dir = self._setup_logging()

        # 初始化训练状态
        self.current_step = 0
        self.best_metrics = {"loss": float("inf")}
        
        # EMA模型（用于平滑参数，提升稳定性）
        self.ema_model = None
        self.ema_decay = getattr(self.cfg.TRAIN, 'EMA_DECAY', 0.0)

    def _setup_distributed(self):
        """设置分布式训练环境"""
        # 分布式训练要求：所有进程必须使用同一个MASTER_PORT
        # 
        # 端口分配策略：
        # 1. 主进程在启动时自动找到空闲端口（通过find_free_port()）
        # 2. 端口通过环境变量MASTER_PORT传递给所有子进程
        # 3. 这样可以同时运行多个训练任务而不会端口冲突
        master_port = int(os.environ.get("MASTER_PORT", "18888"))

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)

        try:
            # 明确指定设备ID以消除警告
            torch.cuda.set_device(self.rank)
            
            dist.init_process_group(
                backend="nccl",
                rank=self.rank,
                world_size=self.world_size,
            )

            if self.is_main_process:
                logging.info(f"Initialized distributed training on port {master_port}")
            
            # 每个进程都打印自己的信息，便于调试
            logging.info(
                f"[Rank {self.rank}/{self.world_size}] Device: {self.device}, GPU: {torch.cuda.current_device()}"
            )

        except Exception as e:
            logging.error(f"Failed to initialize distributed training: {e}")
            raise

    def _cleanup_distributed(self):
        """清理分布式训练资源"""
        if dist.is_initialized():
            dist.destroy_process_group()

    def _setup_logging(self):
        """设置日志系统"""
        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_dir = self.cfg.PATH.CHECKPOINT_DIR + f"_{datetime_str}"
        
        # 为每次训练创建独立的TensorBoard日志目录
        tensorboard_log_dir = os.path.join("./outputs/train/tensorboard", f"run_{datetime_str}")

        if self.is_main_process:
            os.makedirs(self.cfg.PATH.LOG_DIR, exist_ok=True)
            os.makedirs(tensorboard_log_dir, exist_ok=True)
            os.makedirs(checkpoint_dir, exist_ok=True)

            # 创建logger
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            logger.handlers = []  # 清空旧handler

            # 文件handler
            class FlushFileHandler(logging.FileHandler):
                def emit(self, record):
                    super().emit(record)
                    self.flush()

            log_file = os.path.join(self.cfg.PATH.LOG_DIR, f"train_{datetime_str}.log")
            file_handler = FlushFileHandler(log_file, mode="a")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )

            # 控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

            return SummaryWriter(tensorboard_log_dir), checkpoint_dir
        else:
            logging.getLogger().disabled = True
            return None, checkpoint_dir

    def _create_model(self):
        """创建模型"""
        if self.cfg.MODEL.NAME == "FastBEV":
            model = FastBEV(
                x_range=self.cfg.MODEL.FASTBEV.X_RANGE,
                z_range=self.cfg.MODEL.FASTBEV.Z_RANGE,
                y_range=self.cfg.MODEL.FASTBEV.Y_RANGE,
                pixel_per_meter_h=self.cfg.MODEL.FASTBEV.PIXEL_PER_METER_H,
                pixel_per_meter_v=self.cfg.MODEL.FASTBEV.PIXEL_PER_METER_V,
                detection_channels=self.cfg.MODEL.FASTBEV.DETECTION_CHANNELS,
                det3d_output_channels=self.cfg.MODEL.FASTBEV.DET3D_OUT_CHANNELS,
                seg_output_channel=self.cfg.MODEL.FASTBEV.SEG_OUT_CHANNELS,
                backbone=self.cfg.MODEL.FASTBEV.BACKBONE,
                neck=self.cfg.MODEL.FASTBEV.NECK,
                generate_det_gt_format=self.cfg.MODEL.FASTBEV.generate_det_gt_format,
                catetory_num=self.cfg.MODEL.FASTBEV.CATEGORY_NUM,
            )
        else:
            raise ValueError(f"Invalid model name: {self.cfg.MODEL.NAME}")

        # 加载预训练模型
        if (
            hasattr(self.cfg.MODEL, "PRETRAINED_PATH")
            and self.cfg.MODEL.PRETRAINED_PATH
        ):
            self._load_pretrained(model, self.cfg.MODEL.PRETRAINED_PATH, strict=False)

        model = model.to(self.device)
        
        # 初始化EMA模型（在DDP包装之前！），理由见: docs/DDP和EMA详解.md
        if self.ema_decay > 0:
            import copy
            # 复制原始模型，不要复制DDP包装的模型
            self.ema_model = copy.deepcopy(model)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False
            if self.is_main_process:
                logging.info(f"EMA model initialized with decay={self.ema_decay}")

        # 分布式训练包装（在EMA初始化之后）
        if self.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(
                model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=True,
            )

        return model

    def _create_datasets(self):
        """创建数据集"""
        # 数据集配置
        dataset_config = {
            "train": self.cfg.DATASET.TRAIN,
            "val": self.cfg.DATASET.VAL,
            "down_scale_factor": self.cfg.MODEL.FASTBEV.DOWN_SCALE_FACTOR,
            "complemented_det3d_channel": self.cfg.MODEL.FASTBEV.DET3D_OUT_CHANNELS,
            "complemented_seg_channel": self.cfg.MODEL.FASTBEV.SEG_OUT_CHANNELS,
            "transform": {
                "train": {
                    "shape": self.cfg.DATASET.TRANSFORM.train.shape,
                    "augmentation": {
                        "enabled": self.cfg.DATASET.TRANSFORM.train.augmentation.enabled,
                        "severity": self.cfg.DATASET.TRANSFORM.train.augmentation.severity,
                        "color_prob": self.cfg.DATASET.TRANSFORM.train.augmentation.color_prob,
                        "noise_prob": self.cfg.DATASET.TRANSFORM.train.augmentation.noise_prob,
                        "blur_prob": self.cfg.DATASET.TRANSFORM.train.augmentation.blur_prob,
                    },
                },
                "val": {"shape": self.cfg.DATASET.TRANSFORM.val.shape},
            },
        }

        dataset = UnifiedBEVDataset(dataset_config, distributed=self.distributed)

        # 分布式训练时减少worker数量，避免进程过多
        if self.distributed:
            num_workers = min(self.cfg.TRAIN.NUM_WORKERS // self.world_size, 2)
            num_workers = max(num_workers, 1)  # 至少1个worker
        else:
            num_workers = self.cfg.TRAIN.NUM_WORKERS

        train_loader = dataset.get_dataloader(
            "train", batch_size=self.cfg.TRAIN.BATCH_SIZE, num_workers=num_workers
        )

        val_loader = dataset.get_dataloader(
            "val", batch_size=1, num_workers=num_workers
        )

        if self.is_main_process:
            logging.info(f"Train dataset size: {len(train_loader.dataset)}")
            logging.info(f"Val dataset size: {len(val_loader.dataset)}")
            logging.info(f"Using {num_workers} workers per dataloader")

        return train_loader, val_loader

    def _create_optimizers(self):
        """创建优化器和调度器"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )

        # 使用CosineAnnealingLR替代OneCycleLR，更平滑稳定
        # 添加warmup支持
        warmup_iters = getattr(self.cfg.TRAIN, 'WARMUP_ITERS', 0)
        
        if warmup_iters > 0:
            # 使用lambda scheduler实现warmup + cosine annealing
            def lr_lambda(current_step):
                if current_step < warmup_iters:
                    # Warmup阶段：线性增长
                    return float(current_step) / float(max(1, warmup_iters))
                else:
                    # Cosine annealing阶段
                    progress = float(current_step - warmup_iters) / float(max(1, self.cfg.TRAIN.MAX_ITERS - warmup_iters))
                    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            # 不使用warmup，直接使用cosine annealing
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.TRAIN.MAX_ITERS,
                eta_min=self.cfg.TRAIN.LR * 0.01  # 最小学习率为初始的1%
            )

        return optimizer, scheduler

    def _load_pretrained(self, model, pretrained_path, strict=True):
        """加载预训练模型"""
        if not os.path.exists(pretrained_path):
            if self.is_main_process:
                logging.warning(
                    f"Pretrained model path {pretrained_path} does not exist!"
                )
            return

        if self.is_main_process:
            logging.info(f"Loading pretrained model from {pretrained_path}")

        checkpoint = torch.load(pretrained_path, map_location="cpu")

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # 处理DDP模型名称
        if any(k.startswith("module.") for k in state_dict.keys()):
            if not any(k.startswith("module.") for k in model.state_dict().keys()):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

        try:
            model.load_state_dict(state_dict, strict=strict)
            if self.is_main_process:
                logging.info("Successfully loaded pretrained model")
        except Exception as e:
            if self.is_main_process:
                logging.warning(f"Error loading pretrained model: {str(e)}")
            if not strict:
                model_dict = model.state_dict()
                pretrained_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if k in model_dict and model_dict[k].shape == v.shape
                }
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                if self.is_main_process:
                    logging.info(
                        f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained model"
                    )

    def save_checkpoint(self, checkpoint_name=None, best_loss=None):
        """保存checkpoint"""
        if not self.is_main_process:
            return

        if best_loss is None:
            best_loss = float("inf")

        checkpoint = {
            "step": self.current_step,
            "model_state_dict": (
                self.model.module.state_dict()
                if hasattr(self.model, "module")
                else self.model.state_dict()
            ),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": best_loss,
        }
        
        # 保存EMA模型（EMA模型不应该被DDP包装，直接保存）
        if self.ema_model is not None:
            checkpoint["ema_model_state_dict"] = self.ema_model.state_dict()

        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_step_{self.current_step}.pth"

        path = os.path.join(self.checkpoint_save_dir, checkpoint_name)
        try:
            torch.save(checkpoint, path)
            logging.info(f"Saved checkpoint to {path}")
            
            # 额外保存一个仅包含EMA模型的轻量级checkpoint（用于推理）
            if self.ema_model is not None:
                ema_only_path = os.path.join(self.checkpoint_save_dir, f"ema_only_step_{self.current_step}.pth")
                torch.save({
                    "step": self.current_step,
                    "model_state_dict": checkpoint["ema_model_state_dict"],
                }, ema_only_path)
                logging.info(f"Saved EMA-only checkpoint to {ema_only_path}")
        except Exception as e:
            logging.error(f"Error saving checkpoint: {str(e)}")

    def train_step(self, batch):
        """单步训练"""
        # 数据移到GPU
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device, non_blocking=True)

        # 前向传播
        self.optimizer.zero_grad()
        outputs = self.model(batch)

        # 计算loss
        if hasattr(self.model, "module"):
            loss, loss_info = self.model.module.get_loss(outputs, batch)
        else:
            loss, loss_info = self.model.get_loss(outputs, batch)

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            max_norm=self.cfg.TRAIN.CLIP_GRAD_NORM,
        )

        self.optimizer.step()
        self.scheduler.step()
        
        # 更新EMA模型
        if self.ema_model is not None and self.ema_decay > 0:
            self._update_ema()

        return loss, loss_info
    
    def _update_ema(self):
        """更新EMA模型参数和buffers（包括BatchNorm的running_mean/running_var）"""
        # 训练模型：如果是DDP包装的，需要访问.module
        model_to_update = self.model.module if hasattr(self.model, "module") else self.model
        # EMA模型：不应该被DDP包装，直接使用
        ema_model_to_update = self.ema_model
        
        with torch.no_grad():
            # 更新parameters（权重和偏置）
            for ema_param, model_param in zip(ema_model_to_update.parameters(), model_to_update.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)
            
            # 更新buffers（BatchNorm的running_mean和running_var等）
            for ema_buffer, model_buffer in zip(ema_model_to_update.buffers(), model_to_update.buffers()):
                ema_buffer.data.copy_(model_buffer.data)  # buffers直接复制，不做EMA

    def train(self, resume_from=None):
        """主训练循环"""
        try:
            if self.is_main_process:
                logging.info("Starting training with config:")
                logging.info(f"Batch size: {self.cfg.TRAIN.BATCH_SIZE}")
                logging.info(f"Learning rate: {self.cfg.TRAIN.LR}")
                logging.info(f"Total iterations: {self.cfg.TRAIN.MAX_ITERS}")
                logging.info(f"Distributed: {self.distributed}")
                if self.distributed:
                    logging.info(f"World size: {self.world_size}")

            # 初始化模型、数据集、优化器
            self.model = self._create_model()
            self.train_loader, self.val_loader = self._create_datasets()
            self.optimizer, self.scheduler = self._create_optimizers()

            if self.is_main_process:
                logging.info("All processes ready, starting training...")

            # 添加调试信息
            logging.info(f"[Rank {self.rank}] Creating data iterator...")
            
            # 初始化数据迭代器
            train_iter = iter(self.train_loader)
            
            logging.info(f"[Rank {self.rank}] Data iterator created successfully")
            
            # 在数据加载之后再同步，确保所有进程都完成了第一次数据加载
            if self.distributed:
                logging.info(f"[Rank {self.rank}] Waiting at barrier...")
                dist.barrier()
                logging.info(f"[Rank {self.rank}] Barrier passed, starting training loop")

            # 训练循环
            while self.current_step < self.cfg.TRAIN.MAX_ITERS:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    # 重新创建迭代器，避免调用shuffle（可能在分布式下有问题）
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                # 训练一步
                loss, loss_info = self.train_step(batch)
                self.current_step += 1

                # 记录日志
                if (
                    self.is_main_process
                    and self.current_step % self.cfg.TRAIN.PRINT_FREQ == 0
                ):
                    self._log_training_info(loss, loss_info)

                # 保存checkpoint
                if (
                    self.is_main_process
                    and self.current_step % self.cfg.TRAIN.SAVE_FREQ == 0
                    and self.current_step > 0
                ):
                    self.save_checkpoint()

            if self.is_main_process:
                logging.info("Training completed!")
                if self.writer:
                    self.writer.close()

        except Exception as e:
            if self.is_main_process:
                logging.error(f"Error in training: {str(e)}")
            raise e
        finally:
            if self.distributed:
                self._cleanup_distributed()

    def _log_training_info(self, loss, loss_info):
        """记录训练信息"""
        current_lr = self.optimizer.param_groups[0]["lr"]
        loss_str = " ".join([f"{k}: {v.item():.4f}" for k, v in loss_info.items()])

        logging.info(
            f"Step [{self.current_step}/{self.cfg.TRAIN.MAX_ITERS}] "
            f"Loss: {loss.item():.4f} LR: {current_lr:.6f} "
            f"Loss_info: {loss_str}"
        )

        if self.writer:
            self.writer.add_scalar("Loss/train_step", loss.item(), self.current_step)
            self.writer.add_scalar("LR", current_lr, self.current_step)
            for k, v in loss_info.items():
                self.writer.add_scalar(f"Loss/{k}", v.item(), self.current_step)


def train_worker(rank, world_size, cfg, distributed=True, resume_from=None):
    """训练工作进程"""
    # 在worker中不要重复设置多进程策略
    trainer = Trainer(cfg, rank, world_size, distributed)
    trainer.train(resume_from=resume_from)


def main():
    parser = argparse.ArgumentParser(description="FastBEV Training Script")
    parser.add_argument("--config", type=str, default="fastbev_v6", help="Config name")
    parser.add_argument(
        "--distributed", action="store_true", help="Use distributed training"
    ) 
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    args = parser.parse_args()

    try:
        # 加载配置
        cfg = get_cfg_defaults(args.config)
        print(f"Using config: {args.config}")
        print(f"Distributed training: {args.distributed}")
        cfg.freeze()

        if args.distributed:
            n_gpus = torch.cuda.device_count()
            if n_gpus < 2:
                print(
                    f"Distributed training requires multiple GPUs, but only found {n_gpus} GPU(s)."
                )
                print("Falling back to single GPU training.")
                train_worker(0, 1, cfg, distributed=False, resume_from=args.resume)
            else:
                # 在主进程中找到一个空闲端口，所有子进程将共享这个端口
                # 这样可以同时运行多个训练任务而不会端口冲突
                if "MASTER_PORT" not in os.environ:
                    free_port = find_free_port()
                    os.environ["MASTER_PORT"] = str(free_port)
                    print(f"Auto-allocated MASTER_PORT: {free_port}")
                else:
                    print(f"Using existing MASTER_PORT: {os.environ['MASTER_PORT']}")
                
                print(f"Starting distributed training with {n_gpus} GPUs")
                mp.spawn(
                    train_worker,
                    args=(n_gpus, cfg, True, args.resume),
                    nprocs=n_gpus,
                    join=True,
                )
                print(f"All spawned processes completed")
        else:
            print("Starting single GPU training...")
            train_worker(0, 1, cfg, distributed=False, resume_from=args.resume)

    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback

        traceback.print_exc()
        raise e


if __name__ == "__main__":
    main()
