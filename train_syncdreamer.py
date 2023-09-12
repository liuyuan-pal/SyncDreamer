import argparse, os, sys
import numpy as np
import time
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl

from omegaconf import OmegaConf
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

from ldm.util import instantiate_from_config


@rank_zero_only
def rank_zero_print(*args):
    print(*args)

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-r", "--resume", dest='resume', action='store_true', default=False)
    parser.add_argument("-b", "--base", type=str, default='configs/syncdreamer-training.yaml',)
    parser.add_argument("-l", "--logdir", type=str, default="ckpt/logs", help="directory for logging data", )
    parser.add_argument("-c", "--ckptdir", type=str, default="ckpt/models", help="directory for checkpoint data", )
    parser.add_argument("-s", "--seed", type=int, default=6033, help="seed for seed_everything", )
    parser.add_argument("--finetune_from", type=str, default="/cfs-cq-dcc/rondyliu/models/sd-image-conditioned-v2.ckpt", help="path to checkpoint to load model state from" )
    parser.add_argument("--gpus", type=str, default='0,')
    return parser

def trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if hasattr(opt, k))

class SetupCallback(Callback):
    def __init__(self, resume, logdir, ckptdir, cfgdir, config):
        super().__init__()
        self.resume = resume
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            rank_zero_print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config, os.path.join(self.cfgdir, "configs.yaml"))

            if not self.resume and os.path.exists(os.path.join(self.logdir,'checkpoints','last.ckpt')):
                raise RuntimeError(f"checkpoint {os.path.join(self.logdir,'checkpoints','last.ckpt')} existing")

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, log_images_kwargs=None):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}

    @rank_zero_only
    def log_to_logger(self, pl_module, images, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)

    @rank_zero_only
    def log_to_file(self, save_dir, split, images, global_step, current_epoch):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{:06}-{:06}-{}.jpg".format(global_step, current_epoch, k)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    @rank_zero_only
    def log_img(self, pl_module, batch, split="train"):
        if split == "val": should_log = True
        else: should_log = self.check_frequency(pl_module.global_step)

        if should_log:
            is_train = pl_module.training
            if is_train: pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    images[k] = torch.clamp(images[k], -1., 1.)

            self.log_to_file(pl_module.logger.save_dir, split, images, pl_module.global_step, pl_module.current_epoch)
            # self.log_to_logger(pl_module, images, split)

            if is_train: pl_module.train()

    def check_frequency(self, check_idx):
        if (check_idx % self.batch_freq) == 0 and check_idx > 0:
            return True
        else:
            return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, split="train")

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # print('validation ....')
        # print(dataloader_idx)
        # print(batch_idx)
        if batch_idx==0: self.log_img(pl_module, batch, split="val")

class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.strategy.reduce(max_memory)
            epoch_time = trainer.strategy.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass

def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

class ResumeCallBacks(Callback):
    def on_train_start(self, trainer, pl_module):
        pl_module.optimizers().param_groups = pl_module.optimizers()._optimizer.param_groups

def load_pretrain_stable_diffusion(new_model, finetune_from):
    rank_zero_print(f"Attempting to load state from {finetune_from}")
    old_state = torch.load(finetune_from, map_location="cpu")
    if "state_dict" in old_state: old_state = old_state["state_dict"]

    in_filters_load = old_state["model.diffusion_model.input_blocks.0.0.weight"]
    new_state = new_model.state_dict()
    if "model.diffusion_model.input_blocks.0.0.weight" in new_state:
        in_filters_current = new_state["model.diffusion_model.input_blocks.0.0.weight"]
        in_shape = in_filters_current.shape
        ## because the model adopts additional inputs as conditions.
        if in_shape != in_filters_load.shape:
            input_keys = ["model.diffusion_model.input_blocks.0.0.weight", "model_ema.diffusion_modelinput_blocks00weight",]
            for input_key in input_keys:
                if input_key not in old_state or input_key not in new_state:
                    continue
                input_weight = new_state[input_key]
                if input_weight.size() != old_state[input_key].size():
                    print(f"Manual init: {input_key}")
                    input_weight.zero_()
                    input_weight[:, :4, :, :].copy_(old_state[input_key])
                old_state[input_key] = torch.nn.parameter.Parameter(input_weight)

    new_model.load_state_dict(old_state, strict=False)

def get_optional_dict(name, config):
    if name in config:
        cfg = config[name]
    else:
        cfg =  OmegaConf.create()
    return cfg

if __name__ == "__main__":
    # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    opt = get_parser().parse_args()

    assert opt.base != ''
    name = os.path.split(opt.base)[-1]
    name = os.path.splitext(name)[0]
    logdir = os.path.join(opt.logdir, name)

    # logdir: checkpoints+configs
    ckptdir = os.path.join(opt.ckptdir, name)
    cfgdir = os.path.join(logdir, "configs")

    if opt.resume:
        ckpt = os.path.join(ckptdir, "last.ckpt")
        opt.resume_from_checkpoint = ckpt
        opt.finetune_from = "" # disable finetune checkpoint

    seed_everything(opt.seed)

    ###################config#####################
    config = OmegaConf.load(opt.base)  # loade default configs
    lightning_config = config.lightning
    trainer_config = config.lightning.trainer
    for k in trainer_args(opt): # overwrite trainer configs
        trainer_config[k] = getattr(opt, k)

    ###################trainer#####################
    # training framework
    gpuinfo = trainer_config["gpus"]
    rank_zero_print(f"Running on GPUs {gpuinfo}")
    ngpu = len(trainer_config.gpus.strip(",").split(','))
    trainer_config['devices'] = ngpu

    ###################model#####################
    model = instantiate_from_config(config.model)
    model.cpu()
    # load stable diffusion parameters
    if opt.finetune_from != "":
        load_pretrain_stable_diffusion(model, opt.finetune_from)

    ###################logger#####################
    # default logger configs
    default_logger_cfg = {"target": "pytorch_lightning.loggers.TensorBoardLogger",
                          "params": {"save_dir": logdir, "name": "tensorboard_logs", }}
    logger_cfg = OmegaConf.create(default_logger_cfg)
    logger = instantiate_from_config(logger_cfg)

    ###################callbacks#####################
    # default ckpt callbacks
    default_modelckpt_cfg = {"target": "pytorch_lightning.callbacks.ModelCheckpoint",
                             "params": {"dirpath": ckptdir, "filename": "{epoch:06}", "verbose": True, "save_last": True, "every_n_train_steps": 5000}}
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, get_optional_dict("modelcheckpoint", lightning_config))  # overwrite checkpoint configs
    default_modelckpt_cfg_repeat = {"target": "pytorch_lightning.callbacks.ModelCheckpoint",
                                    "params": {"dirpath": ckptdir, "filename": "{step:08}", "verbose": True, "save_last": False, "every_n_train_steps": 5000, "save_top_k": -1}}
    modelckpt_cfg_repeat = OmegaConf.merge(default_modelckpt_cfg_repeat)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "train_syncdreamer.SetupCallback",
            "params": {"resume": opt.resume, "logdir": logdir, "ckptdir": ckptdir, "cfgdir": cfgdir, "config": config}
        },
        "learning_rate_logger": {
            "target": "train_syncdreamer.LearningRateMonitor",
            "params": {"logging_interval": "step"}
        },
        "cuda_callback": {"target": "train_syncdreamer.CUDACallback"},
    }
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, get_optional_dict("callbacks", lightning_config))
    callbacks_cfg['model_ckpt'] = modelckpt_cfg  # add checkpoint
    callbacks_cfg['model_ckpt_repeat'] = modelckpt_cfg_repeat  # add checkpoint
    callbacks = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]  # construct all callbacks
    if opt.resume:
        callbacks.append(ResumeCallBacks())

    trainer = Trainer.from_argparse_args(args=argparse.Namespace(), **trainer_config,
                                         accelerator='cuda', strategy=DDPStrategy(find_unused_parameters=False), logger=logger, callbacks=callbacks)
    trainer.logdir = logdir

    ###################data#####################
    config.data.params.seed = opt.seed
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup('fit')

    ####################lr#####################
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    accumulate_grad_batches = trainer_config.accumulate_grad_batches if hasattr(trainer_config, "trainer_config") else 1
    rank_zero_print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    model.learning_rate = base_lr
    rank_zero_print("++++ NOT USING LR SCALING ++++")
    rank_zero_print(f"Setting learning rate to {model.learning_rate:.2e}")
    model.image_dir = logdir # used in output images during training

    # run
    trainer.fit(model, data)
