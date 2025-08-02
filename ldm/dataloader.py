import glob
import numpy as np
import random
import pickle
import nibabel as nib
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2


def resize_long_side(img, target_length = 256):
    long_side_length = target_length
    oldh, oldw = img.shape[1], img.shape[2]
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    newh, neww = int(newh + 0.5), int(neww + 0.5)
    target_size = (newh, neww)
    # resize transform instance
    resize_trans = transforms.Resize(size=target_size, antialias=True)
    img_resized = resize_trans(img)
    return img_resized


def pad_image(img, target_size = 256):
    h, w = img.shape[1], img.shape[2]
    pad_l = int((target_size - w) / 2)
    pad_r = int(target_size - w - pad_l)
    pad_t = int((target_size - h) / 2)
    pad_b = int(target_size - h - pad_t)
    pad_size = (pad_l, pad_t, pad_r, pad_b)
    pad_trans = transforms.Pad(pad_size)
    img_padded = pad_trans(img)
    return img_padded


def img_aug(img):
    # random augmentation
    # apply random horizontal flip
    if torch.rand(1) > 0.5:
        img = TF.hflip(img)
    # apply random vertical flip
    if torch.rand(1) > 0.5:
        img = TF.vflip(img)
    # apply random rotation
    if torch.rand(1) > 0.5:
        angle = transforms.RandomRotation.get_params(degrees=(-180, 180))
        img = TF.rotate(img, angle)
    return img


def ct_mask(image):
    image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(image_uint8, threshold1=3, threshold2=250)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    mask = np.zeros_like(image_uint8, dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, (255), thickness=-1)  # Fill the convex hull
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image


def value_norm(img, datatype):
    # abdominal soft tissue, WW: 400 HU, WL: 50 HU
    WINDOW_LEVEL = 50
    WINDOW_WIDTH = 400
    # normalize the image
    if datatype == 'ct':
        lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
        upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
        img_pre = np.clip(img, lower_bound, upper_bound)
        img_pre = ((img_pre - np.min(img_pre)) / (np.max(img_pre) - np.min(img_pre)) * 1.0)
        img_pre = ct_mask(img_pre)
        img_pre = ((img_pre - np.min(img_pre)) / (np.max(img_pre) - np.min(img_pre)) * 1.0)
    elif datatype == 'mri':
        try:
            lower_bound, upper_bound = np.percentile(img[img > 0], 1.0), np.percentile(img[img > 0], 99.0)
            img_pre = np.clip(img, lower_bound, upper_bound)
            img_pre = ((img_pre - np.min(img_pre)) / (np.max(img_pre) - np.min(img_pre)) * 1.0)
        except:
            return None
    else:
        raise TypeError('incorrect data type (must be "ct" or "mri")')
    # img to tensor
    img_pre = np.squeeze(img_pre)
    img_3c = np.expand_dims(img_pre, axis=0)
    img_3c = torch.from_numpy(img_3c)
    return img_3c


def preprocess(img, image_size):
    # reshape the image
    img = resize_long_side(img, image_size)
    # pad the image
    img = pad_image(img, image_size)
    # return the image
    return img


def create_gaussian_kernel(kernel_size, sigma):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    # make a grid of (x, y) coordinates
    ax = torch.linspace(-(kernel_size // 2), kernel_size // 2, steps=kernel_size)
    xx, yy = torch.meshgrid([ax, ax], indexing='ij')
    xx = xx.float()
    yy = yy.float()
    # compute the 2D Gaussian function
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    # normalize the kernel so that sum is 1
    kernel = kernel / torch.sum(kernel)
    # check the shape before reshaping
    if kernel.numel() != kernel_size * kernel_size:
        raise RuntimeError(f"Kernel size mismatch: expected {kernel_size * kernel_size}, got {kernel.numel()}")
    # reshape to be compatible with conv2d (out_channels, in_channels, H, W)
    return kernel.view(1, 1, kernel_size, kernel_size)


def create_laplacian_kernel():
    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
    #laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
    return laplacian_kernel.view(1, 1, 3, 3)


def laplacian_of_gaussian(image, kernel_size=3, sigma=0.6):
    # create the Gaussian and Laplacian kernels
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma).to(image.device)
    laplacian_kernel = create_laplacian_kernel().to(image.device)
    # apply the Gaussian smoothing
    smoothed = F.conv2d(image, gaussian_kernel, padding=kernel_size // 2)
    # apply the Laplacian filter
    log_one = F.conv2d(smoothed, laplacian_kernel, padding=1)
    # abs and norm
    log_one = torch.abs(log_one)
    log_one = (log_one - log_one.min()) / (log_one.max() - log_one.min())
    # smooth again
    smoothed = F.conv2d(log_one, gaussian_kernel, padding=kernel_size // 2)
    log_two = F.conv2d(smoothed, laplacian_kernel, padding=1)
    # abs and norm
    log_two = torch.abs(log_two)
    edges = (log_two - log_two.min()) / (log_two.max() - log_two.min())
    return edges


def load_data(img_path, datatype, image_size=256, aug=True):
    # load nii data
    img_nib = nib.load(img_path)
    img_all = img_nib.get_fdata().astype(np.float32)
    # choose slice index (avoid all 0 slice)
    good_slice = False
    while not good_slice:
        slice_idx = np.random.choice(img_all.shape[-1], size=1, replace=False)
        img = np.squeeze(img_all[:, :, slice_idx])
        img = value_norm(img, datatype)
        if (img.min() == img.max()):
            continue
        else:
            good_slice = True
    # load slice
    img = preprocess(img, image_size)
    # augment flips and rotatio
    if aug:
        img = img_aug(img)
    # then create a edge image
    edge_img = laplacian_of_gaussian(img)
    return img, edge_img


class AmosCTMRIDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(AmosCTMRIDataset).__init__()
        batchs_num = 1000
        batch_size = 21
        image_size = 256
        # data parameters
        self.dataloader_len = batchs_num * batch_size
        self.nii_prefix = "path/to/real/data/folder"
        self.image_size = image_size
        with open('ct-mri-split.pkl', 'rb') as fh:
            datadict = pickle.load(fh)
        self.fdict = {}
        # load AMOS CT
        self.fdict[0] = {}
        for nowf in datadict['ct']['amos']['train'].keys():
            self.fdict[0][nowf] = datadict['ct']['amos']['train'][nowf]
            self.fdict[0][nowf+'-datatype'] = 'ct'
        # load AMOS MRI
        self.fdict[1] = {}
        for nowf in datadict['mri']['amos']['train'].keys():
            self.fdict[1][nowf] = datadict['mri']['amos']['train'][nowf]
            self.fdict[1][nowf+'-datatype'] = 'mri'
        # load Panc T1
        self.fdict[2] = {}
        for nowf in datadict['mri']['panc-t1']['train'].keys():
            self.fdict[2][nowf] = datadict['mri']['panc-t1']['train'][nowf]
            self.fdict[2][nowf+'-datatype'] = 'mri'
        # load Panc T2
        self.fdict[3] = {}
        for nowf in datadict['mri']['panc-t2']['train'].keys():
            self.fdict[3][nowf] = datadict['mri']['panc-t2']['train'][nowf]
            self.fdict[3][nowf+'-datatype'] = 'mri'
    
    def __len__(self):
        return int(self.dataloader_len)
    
    def __getitem__(self, idx):
        #group = random.choice([0, 1, 2, 3])
        # only use T2
        group = 3
        img_path = random.choice(list(self.fdict[group].keys()))
        img_path = img_path.replace('-datatype', '')
        datatype = self.fdict[group][img_path+'-datatype']
        img_path = self.nii_prefix + img_path
        img, sobel_img = load_data(img_path, datatype, image_size=self.image_size)
        return sobel_img.to(torch.float32).repeat(3, 1, 1), img.to(torch.float32).repeat(3, 1, 1)


def AmosCTMRIDataloader():
    batch_size = 8
    num_workers = 1
    dataloader = AmosCTMRIDataset()
    return torch.utils.data.DataLoader(
        dataloader,
        batch_size=batch_size,
        collate_fn=None,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False)


if __name__ == '__main__':
    dataloader = AmosCTMRIDataloader()
    count = 0
    for (img, edge) in dataloader:
        print(img.shape)
        print(edge.shape)
        count += 1
        if count > 10:
            break
