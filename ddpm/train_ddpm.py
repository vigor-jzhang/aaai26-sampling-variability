import os
import argparse
import wandb

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from utils import get_config, folder_create, setup_logger
from dataloader import AmosCTMRIDataloader
from model import create_diffusion
from test_ddpm import test_ddpm


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default=None, help='config file path')


def init_wandb(config):
    wandb.init(
        dir=config.experiment.work_dir,
        project=config.experiment.task_name + '-' + config.experiment.net_name,
        name=config.experiment.name,
        config={
            'unet-dim': config.unet.dim,
        },
    )
    return None


def create_scratch_model(ckpt_path, config):
    # define diffusion model
    diffusion = create_diffusion(config)
    # create optimizer
    optimizer = torch.optim.AdamW(
        diffusion.parameters(), 
        lr=config.optimizer.lr,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        eps=1e-08,
        weight_decay=config.optimizer.weight_decay
    )
    # create scheduler
    sche_diffusion = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.lr_scheduler.factor,
        patience=config.lr_scheduler.patience,
        cooldown=config.lr_scheduler.cooldown,
        min_lr=config.lr_scheduler.min_lr,
    )
    # save needed modules
    torch.save({
        'epoch': 0,
        'diffusion': diffusion.state_dict(),
        'optimizer': optimizer.state_dict(),
        'sche_diffusion': sche_diffusion.state_dict(),
    }, ckpt_path)
    return None


def train(previous_ckpt, config, logger):
    # define the device
    device = torch.device(config.training.device)
    # define diffusion model
    diffusion = create_diffusion(config)
    diffusion = diffusion.to(device)
    # create optimizer
    optimizer = torch.optim.AdamW(
        diffusion.parameters(), 
        lr=config.optimizer.lr,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        eps=1e-08,
        weight_decay=config.optimizer.weight_decay
    )
    # create scheduler
    sche_diffusion = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.lr_scheduler.factor,
        patience=config.lr_scheduler.patience,
        cooldown=config.lr_scheduler.cooldown,
        min_lr=config.lr_scheduler.min_lr,
    )
    # load checkpoint
    checkpoint = torch.load(previous_ckpt, weights_only=True)
    diffusion.load_state_dict(checkpoint['diffusion'])
    diffusion.train()
    optimizer.load_state_dict(checkpoint['optimizer'])
    sche_diffusion.load_state_dict(checkpoint['sche_diffusion'])
    start_epoch = checkpoint['epoch']
    # initial dataloader
    dataloader = AmosCTMRIDataloader(config)
    # define gradscaler
    scaler = torch.amp.GradScaler('cuda')
    # start training
    for epoch in range(start_epoch, start_epoch + config.experiment.ckpt_intervals):
        list_diff_loss = []
        for (img, edge) in dataloader:
            img, edge = img.to(device), edge.to(device)
            # zero grad optimizer
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = diffusion(img, edge)
                list_diff_loss.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        # finish one epoch
        avg_diff_loss = sum(list_diff_loss) / len(list_diff_loss)
        print(f'Epoch {epoch} -- diffusion loss: {avg_diff_loss:.4f}', flush=True)
        logger.info(f'Epoch {epoch} -- diffusion loss: {avg_diff_loss:.4f}')
        wandb.log({'diffusion loss': avg_diff_loss}, step=epoch)
        # update the scheduler
        if epoch > config.lr_scheduler.warmup_epochs:
            sche_diffusion.step(avg_diff_loss)
    # save checkpoint
    checkpoint_path = config.experiment.ckpt_dir + '/ckpt_epoch{:0>6d}.pt'.format(epoch+1)
    logger.info('Save model in ['+checkpoint_path+']')
    print('Saving model in ['+checkpoint_path+']', end='', flush=True)
    torch.save({
        'epoch': epoch+1,
        'diffusion': diffusion.state_dict(),
        'optimizer': optimizer.state_dict(),
        'sche_diffusion': sche_diffusion.state_dict(),
    }, checkpoint_path)
    print(' ... END', flush=True)
    return checkpoint_path


if __name__ == '__main__':
    # get config
    args = parser.parse_args()
    config = get_config(args.config_path)
    # init random seed for torch
    torch.manual_seed(config.training.seed)
    # folders create
    config = folder_create(config)
    print(config)
    # setup logger
    logger = setup_logger(config)
    # init wandb
    init_wandb(config)
    # restore ckpt or create new
    if not config.experiment.restoring:
        create_scratch_model(config.experiment.ckpt_dir + '/ckpt_epoch{:0>6d}.pt'.format(0), config)
        previous_ckpt = config.experiment.ckpt_dir + '/ckpt_epoch{:0>6d}.pt'.format(0)
    else:
        previous_ckpt = config.experiment.restore_path
    # load epochs
    checkpoint = torch.load(previous_ckpt, weights_only=True)
    start_epoch = checkpoint['epoch']
    del checkpoint
    # training procedure
    for epoch in range(start_epoch, config.experiment.max_epochs, config.experiment.ckpt_intervals):
        now_ckpt = train(previous_ckpt, config, logger)
        # testing
        if epoch % int(config.experiment.test_intervals) == 0:
            test_examples = test_ddpm(now_ckpt, config, target=3)
            test_images = wandb.Image(test_examples, caption="T2 test Epoch {:0>6d}".format(epoch))
            wandb.log({'T2 test': test_images}, step=epoch+config.experiment.ckpt_intervals)
            test_examples = test_ddpm(now_ckpt, config, target=2)
            test_images = wandb.Image(test_examples, caption="T1 test Epoch {:0>6d}".format(epoch))
            wandb.log({'T1 test': test_images}, step=epoch+config.experiment.ckpt_intervals)
            test_examples = test_ddpm(now_ckpt, config, target=1)
            test_images = wandb.Image(test_examples, caption="AMOS T1 test Epoch {:0>6d}".format(epoch))
            wandb.log({'AMOS T1 test': test_images}, step=epoch+config.experiment.ckpt_intervals)
            test_examples = test_ddpm(now_ckpt, config, target=0)
            test_images = wandb.Image(test_examples, caption="CT test Epoch {:0>6d}".format(epoch))
            wandb.log({'CT test': test_images}, step=epoch+config.experiment.ckpt_intervals)
        previous_ckpt = now_ckpt
        # delete 5 intervals before checkpoint
        temp_path = config.experiment.ckpt_dir+'/ckpt_epoch{:0>6d}.pt'.format(int(epoch-5*config.experiment.ckpt_intervals))
        if os.path.exists(temp_path):
            os.remove(temp_path)
    # finally, finish wandb
    wandb.finish()
