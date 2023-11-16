import random
import torch
import torch.utils.data
import torch.distributed

from . import data_transforms
from .model import FrameFieldModel
from .trainer import Trainer
from . import losses
from . import local_utils

from lydorn_utils import print_utils

try:
    import apex
    from apex import amp

    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


def count_trainable_params(model):
    count = 0
    for param in model.parameters():
        if param.requires_grad:
            count += param.numel()
    return count

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class LearningRateBase(object):
    def __init__(self, base_lr):
        self._base_lr = base_lr

    @property
    def base_lr(self):
        return self._base_lr

    def step(self, global_step, optimizer):
        raise NotImplementedError

class PolyLearningRate(LearningRateBase):
    def __init__(self,
                 base_lr,
                 power,
                 max_iters,
                 ):
        super(PolyLearningRate, self).__init__(base_lr)
        self.power = power
        self.max_iters = max_iters

    def step(self, global_step, optimizer):
        factor = (1 - global_step / self.max_iters) ** self.power
        cur_lr = self.base_lr * factor
        set_lr(optimizer, cur_lr)

def train(gpu, config, shared_dict, barrier, train_ds, val_ds, backbone):
    # --- Set seeds --- #
    torch.manual_seed(2)  # For DistributedDataParallel: make sure all models are initialized identically
    torch.autograd.set_detect_anomaly(True)

    # --- Setup DistributedDataParallel --- #
    rank = config["nr"] * config["gpus"] + gpu
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=config["world_size"],
        rank=rank
    )

    if gpu == 0:
        print("# --- Start training --- #")

    # --- Setup run --- #
    # Setup run on process 0:
    if gpu == 0:
        shared_dict["run_dirpath"], shared_dict["init_checkpoints_dirpath"] = local_utils.setup_run(config)
    barrier.wait()  # Wait on all processes so that shared_dict is synchronized.

    # Choose device
    torch.cuda.set_device(gpu)

    # --- Online transform performed on the device (GPU):
    train_online_cuda_transform = data_transforms.get_online_cuda_transform(config,
                                                                            augmentations=config["data_aug_params"][
                                                                                "enable"])
    if val_ds is not None:
        eval_online_cuda_transform = data_transforms.get_online_cuda_transform(config, augmentations=False)
    else:
        eval_online_cuda_transform = None

    if "samples" in config:
        rng_samples = random.Random(0)
        train_ds = torch.utils.data.Subset(train_ds, rng_samples.sample(range(len(train_ds)), config["samples"]))
        if val_ds is not None:
            val_ds = torch.utils.data.Subset(val_ds, rng_samples.sample(range(len(val_ds)), config["samples"]))
        # test_ds = torch.utils.data.Subset(test_ds, list(range(config["samples"])))

    if gpu == 0:
        print(f"Train dataset has {len(train_ds)} samples.")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds,
                                                                    num_replicas=config["world_size"], rank=rank)
    val_sampler = None
    if val_ds is not None:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds,
                                                                      num_replicas=config["world_size"], rank=rank)
    if "samples" in config:
        eval_batch_size = min(2 * config["optim_params"]["batch_size"], config["samples"])
    else:
        # eval_batch_size = 2 * config["optim_params"]["batch_size"]
        eval_batch_size = config["optim_params"]["batch_size"]

    init_dl = torch.utils.data.DataLoader(train_ds, batch_size=eval_batch_size, pin_memory=True,
                                          sampler=train_sampler, num_workers=config["num_workers"], drop_last=True)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config["optim_params"]["batch_size"], shuffle=False,
                                           pin_memory=True, sampler=train_sampler, num_workers=config["num_workers"],
                                           drop_last=True)
    if val_ds is not None:
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=eval_batch_size, pin_memory=True,
                                             sampler=val_sampler, num_workers=config["num_workers"], drop_last=True)
    else:
        val_dl = None

    model = FrameFieldModel(config, backbone=backbone, train_transform=train_online_cuda_transform,
                            eval_transform=eval_online_cuda_transform)
    model.cuda(gpu)
    if gpu == 0:
        print("Model has {} trainable params".format(count_trainable_params(model)))

    loss_func = losses.build_combined_loss(config).cuda(gpu)
    # Compute learning rate
    lr = min(config["optim_params"]["base_lr"] * config["optim_params"]["batch_size"] * config["world_size"], config["optim_params"]["max_lr"])

    if config["optim_params"]["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    if config["use_amp"] and APEX_AVAILABLE:
        amp.register_float_function(torch, 'sigmoid')
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    elif config["use_amp"] and not APEX_AVAILABLE and gpu == 0:
        print_utils.print_warning("WARNING: Cannot use amp because the apex library is not available!")

    # Wrap the model for distributed training
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    iter_num = int(1. * config["optim_params"]["max_epoch"] * len(train_ds) / config["optim_params"]["batch_size"])
    lr_scheduler = PolyLearningRate(base_lr=0.003, power=0.9, max_iters=iter_num)

    trainer = Trainer(rank, gpu, config, model, optimizer, loss_func,
                      run_dirpath=shared_dict["run_dirpath"],
                      init_checkpoints_dirpath=shared_dict["init_checkpoints_dirpath"],
                      lr_scheduler=lr_scheduler)
    trainer.fit(train_dl, val_dl=val_dl, init_dl=init_dl)
