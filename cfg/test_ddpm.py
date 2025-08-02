import numpy as np
import random
import pickle

from dataloader import load_data

import torch
from model import create_diffusion


class AmosCTMRITestDataset(torch.utils.data.Dataset):
    def __init__(self, target):
        super(AmosCTMRITestDataset).__init__()
        # data parameters
        self.target = target
        self.dataloader_len = 20
        self.nii_prefix = "path/to/real/data/folder"
        self.image_size = 256
        with open('ct-mri-split.pkl', 'rb') as fh:
            datadict = pickle.load(fh)
        self.fdict = {}
        # load AMOS CT
        self.fdict[0] = {}
        for nowf in datadict['ct']['amos']['test'].keys():
            self.fdict[0][nowf] = datadict['ct']['amos']['test'][nowf]
            self.fdict[0][nowf+'-datatype'] = 'ct'
        # load AMOS MRI
        self.fdict[1] = {}
        for nowf in datadict['mri']['amos']['test'].keys():
            self.fdict[1][nowf] = datadict['mri']['amos']['test'][nowf]
            self.fdict[1][nowf+'-datatype'] = 'mri'
        # load Panc T1
        self.fdict[2] = {}
        for nowf in datadict['mri']['panc-t1']['test'].keys():
            self.fdict[2][nowf] = datadict['mri']['panc-t1']['test'][nowf]
            self.fdict[2][nowf+'-datatype'] = 'mri'
        # load Panc T2
        self.fdict[3] = {}
        for nowf in datadict['mri']['panc-t2']['test'].keys():
            self.fdict[3][nowf] = datadict['mri']['panc-t2']['test'][nowf]
            self.fdict[3][nowf+'-datatype'] = 'mri'
    
    def __len__(self):
        return int(self.dataloader_len)
    
    def __getitem__(self, idx):
        # only use T2
        group = self.target
        img_path = random.choice(list(self.fdict[group].keys()))
        img_path = img_path.replace('-datatype', '')
        datatype = self.fdict[group][img_path+'-datatype']
        img_path = self.nii_prefix + img_path
        img, sobel_img = load_data(img_path, datatype, image_size=self.image_size, aug=False)
        return img.to(torch.float32), sobel_img.to(torch.float32)


def AmosCTMRITestDataloader(target=3):
    dataloader = AmosCTMRITestDataset(target)
    return torch.utils.data.DataLoader(
        dataloader,
        batch_size=4,
        collate_fn=None,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=False)


def simple_norm(x):
    if x.min() == x.max():
        return np.zeros(x.shape)
    return (x - x.min()) / (x.max() - x.min())


def norm_stack(a, b, c, d):
    a = simple_norm(a)
    b = simple_norm(b)
    c = simple_norm(c)
    d = simple_norm(d)
    comb = np.hstack((a, b, c, d))
    return comb


def test_ddpm(ckpt_path, config, target):
    # define the device
    device = torch.device('cuda:0')
    # define vqgan model
    diffusion = create_diffusion(config)
    diffusion = diffusion.to(device)
    checkpoint = torch.load(ckpt_path, weights_only=True)
    diffusion.load_state_dict(checkpoint['diffusion'])
    diffusion.eval()
    # define dataloader
    dataloader = AmosCTMRITestDataloader(target)
    for (img, edge) in dataloader:
        img, edge = img.to(device), edge.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.no_grad():
                diff_img = diffusion.sample(edge)
        break

    img = img.detach().cpu().numpy()
    edge = edge.detach().cpu().numpy()
    diff_img = diff_img.detach().cpu().numpy()

    row1 = norm_stack(img[0,0], img[1,0], img[2,0], img[3,0])
    row2 = norm_stack(diff_img[0,0], diff_img[1,0], diff_img[2,0], diff_img[3,0])
    row3 = norm_stack(edge[0,0], edge[1,0], edge[2,0], edge[3,0])
    combined = np.vstack((row1, row2, row3))
    return combined
