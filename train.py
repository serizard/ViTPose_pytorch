# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
import click
import yaml

from glob import glob

import torch
import torch.distributed as dist
import torch.nn.functional as F

from utils.util import init_random_seed, set_random_seed
from utils.dist_util import get_dist_info, init_dist
from utils.logging import get_root_logger

import configs.ViTPose_base_coco_256x192 as b_cfg
import configs.ViTPose_large_coco_256x192 as l_cfg
import configs.ViTPose_huge_coco_256x192 as h_cfg
import configs.ViTPose_base_simple_coco_256x192 as b_cfg_simple
import configs.ViTPose_large_simple_coco_256x192 as l_cfg_simple
import configs.ViTPose_huge_simple_coco_256x192 as h_cfg_simple

from models.model import ViTPose
from datasets.COCO import COCODataset
from utils.train_valid_fn import train_model

CUR_PATH = osp.dirname(__file__)

@click.command()
@click.option('--config-path', type=click.Path(exists=True), default='config.yaml', required=True, help='train config file path')
@click.option('--model-name', type=str, default='b', required=True, help='[b: ViT-B, l: ViT-L, h: ViT-H, b-simple: ViT-B-simple, l-simple: ViT-L-simple, h-simple: ViT-H-simple]')
@click.option('--experiment', type=int, default=1, help='1~12')
@click.option('--batch-size', type=int, default=32, help='batch size')
@click.option('--epochs', type=int, default=210, help='epochs')
@click.option('--random-seed', type=int, default=None, help='random seed')
@click.option('--max-images', type=int, default=None, help='max images')

def main(config_path, model_name, experiment, batch_size, epochs, random_seed, max_images):
        
    cfg = {'b':b_cfg,
           'l':l_cfg,
           'h':h_cfg,
           'b-simple':b_cfg_simple,
           'l-simple':l_cfg_simple,
           'h-simple':h_cfg_simple}.get(model_name.lower())
    # Load config.yaml
    with open(config_path, 'r') as f:
        cfg_yaml = yaml.load(f, Loader=yaml.SafeLoader)
        
    for k, v in cfg_yaml.items():
        if hasattr(cfg, k):
            raise ValueError(f"Already exsist {k} in config")
        else:
            cfg.__setattr__(k, v)

    cfg.data['samples_per_gpu'] = batch_size

    cfg.total_epochs = epochs

    # set cudnn_benchmark
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    
    # Set work directory (session-level)
    if not hasattr(cfg, 'work_dir'):
        cfg.__setattr__('work_dir', f"{CUR_PATH}/runs/train")
        
    if not osp.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir)
    session_dir = osp.join(cfg.work_dir, str(experiment).zfill(3))
    os.makedirs(session_dir, exist_ok=True)
    cfg.__setattr__('work_dir', session_dir)
        

    if cfg.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if cfg.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f"We treat {cfg['gpu_ids']} as gpu-ids, and reset to "
                f"{cfg['gpu_ids'][0:1]} as gpu-ids to avoid potential error in "
                "non-distribute training time.")
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(cfg.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(session_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()

    # log some basic info
    logger.info(f'Distributed training: {distributed}')

    # set random seeds
    seed = init_random_seed(random_seed)
    logger.info(f"Set random seed to {seed}, "
                f"deterministic: {cfg.deterministic}")
    set_random_seed(seed, deterministic=cfg.deterministic)
    meta['seed'] = seed


    model = ViTPose(cfg.model)

    pretrained_path = cfg.model['pretrained']
    if pretrained_path:
        pretrained_backbone = torch.load(pretrained_path)
        print('Load pretrained model from', pretrained_path)
        width, height = cfg.model['backbone']['img_size']
        patch_size = cfg.model['backbone']['patch_size']
        embed_dim = cfg.model['backbone']['embed_dim']
        new_pos_embed = resize_pos_embed(pretrained_backbone['model']['pos_embed'], height, width, patch_size, embed_dim)
        pretrained_backbone['model']['pos_embed'] = new_pos_embed

        pretrained_backbone['model']['last_norm.weight'] = pretrained_backbone['model']['norm.weight']
        pretrained_backbone['model']['last_norm.bias'] = pretrained_backbone['model']['norm.bias']
        del pretrained_backbone['model']['norm.weight']
        del pretrained_backbone['model']['norm.bias']

        modified_dict = {f'backbone.{key}': value for key, value in pretrained_backbone['model'].items()}
        
        for name, param in model.named_parameters():
            if name in modified_dict:
                param.data.copy_(modified_dict[name])
                param.requires_grad = False

    if cfg.resume_from:
        model.load_state_dict(torch.load(cfg.resume_from)['state_dict'])
    
    
    # Set dataset
    datasets_train = COCODataset(
        root_path=cfg.data_root, 
        data_version="train2017",
        is_train=True, 
        use_gt_bboxes=True,
        image_width=192, 
        image_height=256,
        scale=True, 
        scale_factor=0.35, 
        flip_prob=0.5, 
        rotate_prob=0.5, 
        rotation_factor=45., 
        half_body_prob=0.3,
        use_different_joints_weight=True, 
        heatmap_sigma=3, 
        soft_nms=False,
        max_images = max_images
        )
    
    datasets_valid = COCODataset(
        root_path=cfg.data_root, 
        data_version="val2017",
        is_train=False, 
        use_gt_bboxes=True,
        image_width=192, 
        image_height=256,
        scale=False, 
        scale_factor=0.35, 
        flip_prob=0.5, 
        rotate_prob=0.5, 
        rotation_factor=45., 
        half_body_prob=0.3,
        use_different_joints_weight=True, 
        heatmap_sigma=3, 
        soft_nms=False,
        max_images=max_images
        )

    train_model(
        model=model,
        datasets_train=datasets_train,
        datasets_valid=datasets_valid,
        cfg=cfg,
        distributed=distributed,
        validate=cfg.validate,
        timestamp=timestamp,
        meta=meta,
        experiment=experiment,
        seed=random_seed
        )

def resize_pos_embed(pos_embed, new_height, new_width, patch_size, embed_dim):
    cls_token = pos_embed[:, 0]
    pos_tokens = pos_embed[:, 1:]

    orig_num_patches = pos_tokens.shape[1]
    orig_size = int(orig_num_patches ** 0.5)  # Assuming square grid

    pos_tokens = pos_tokens.reshape(1, orig_size, orig_size, embed_dim).permute(0, 3, 1, 2)
    new_pos_tokens = F.interpolate(pos_tokens, size=(new_height // patch_size, new_width // patch_size), mode='bicubic', align_corners=False)
    new_pos_tokens = new_pos_tokens.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)

    new_pos_embed = torch.cat((cls_token.unsqueeze(0), new_pos_tokens), dim=1)

    return new_pos_embed


if __name__ == '__main__':
    main()
