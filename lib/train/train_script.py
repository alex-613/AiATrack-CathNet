# For import modules
import importlib
import os

from torch import nn
from torch.nn.functional import l1_loss
# Distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP

# Network related
from lib.models.aiatrack import build_aiatrack
# Forward propagation related
from lib.train.actors import AIATRACKActor
# Train pipeline related
from lib.train.trainers import LTRTrainer
# Loss function related
from lib.utils.box_ops import giou_loss
# Some more advanced functions
from .base_functions import *


def run(settings):
    torch.cuda.empty_cache()
    settings.description = 'training script'

    # Update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("ERROR: %s doesn't exist" % settings.cfg_file)
    # import the config_module library that is specific to the script name
    config_module = importlib.import_module('lib.config.%s.config' % settings.script_name)
    # Creates a configuration edict
    cfg = config_module.cfg
    # Updates the configuration using the configuration file that is defined

    # Commenting this line out because we wont use a configuration file
    config_module.update_config_from_file(settings.cfg_file)

    # Update settings based on cfg
    # Update those setting again according to that specific in the settings
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    settings.log_file = os.path.join(log_dir, '%s-%s.log' % (settings.script_name, settings.config_name))

    # Create network.
    net = build_aiatrack(cfg)

    # Wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank])  # find_unused_parameters=True
        settings.device = torch.device('cuda:%d' % settings.local_rank)
    else:
        settings.device = torch.device('cuda:0')

    # Loss functions and actors
    objective = {'giou': giou_loss, 'l1': l1_loss, 'iou': nn.MSELoss()}
    loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'iou': cfg.TRAIN.IOU_WEIGHT}

    # The actor carries out the actions of the AiA network, such as forward pass, calculate losses, then backprop and update with optimizer
    actor = AIATRACKActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    # Grab the dataloader
    loader_train = build_dataloaders(cfg, settings)

    # Grab the trainer
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler)

    # Train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
