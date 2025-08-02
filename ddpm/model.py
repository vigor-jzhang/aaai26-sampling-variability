import torch
from unet import Unet
from diffusion import GaussianDiffusion


def create_unet_model(config):
    model = Unet(
        dim = config.unet.dim,
        dim_mults = config.unet.dim_mults,
        flash_attn = config.unet.flash_attn,
    )
    return model


def create_diffusion(config):
    unet = create_unet_model(config)
    #print(f"unet size: {sum(p.numel() for p in unet.parameters())}")
    diffusion = GaussianDiffusion(
        unet,
        image_size = config.diffusion.image_size,
        timesteps = config.diffusion.timesteps,
        sampling_timesteps = config.diffusion.sampling_timesteps,
    )
    return diffusion
    


if __name__ == '__main__':
    from utils import get_config
    config = get_config('./configs/init-ddpm-config.yaml')

    diffusion = create_diffusion(config).to('cuda:0')
    
    x = torch.rand(5, 1, 256, 256).to('cuda:0')
    edge = torch.rand(5, 1, 256, 256).to('cuda:0')
    print('for training')
    y = diffusion(x, edge)
    print(y)
    print('for prediction')
    y = diffusion.sample(edge)
    print(y.shape)