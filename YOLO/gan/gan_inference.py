import numpy as np
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import os
import random
import glob
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from gan import Generator
torch.manual_seed(0)


sintec_dir='Gan_Sintec_0_30'
T_NUM = 'T11'
best_model = 201
show =True
save = True
n_epochs = 1
dim_A = 3
dim_B = 3
batch_size = 1
load_shape = 224
target_shape = 224
device = 'cuda'
plt.rcParams["figure.figsize"] = (10, 10)

def show_tensor_images(image_tensor, curr_step, real_fake, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)[0]
    if show:
        fig_1,ax_1 = plt.subplots(figsize=(2.91, 2.91))
        ax_1.imshow(image_unflat.permute(1, 2, 0).squeeze())
        ax_1.axis('off')
        if save:
            # fig_1.savefig(os.path.join(os.getcwd(), 'GanResults','inference',sintec_dir+'_fake2real',
            #         '{:06d}_{}.png'.format(curr_step[0][-10:])), bbox_inches='tight', pad_inches=0)
            fig_1.savefig(os.path.join(os.getcwd(), 'GanResults','inference',sintec_dir+'_new',
                    '{}'.format(curr_step[0][-10:])), bbox_inches='tight', pad_inches=0)
    # plt.show()
    # plt.pause(3)
    plt.close()

class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.transform = transform
        self.files_A = sorted(glob.glob(os.path.join(
            root, sintec_dir, 'images',mode) + '\\*.*'))
        print("Found {}".format(len(self.files_A)))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(
            self.files_A[index % len(self.files_A)]))
        if item_A.shape[0] != 3:
            item_A = item_A.repeat(3, 1, 1)
        # Old versions of PyTorch didn't support normalization for different-channeled images
        return (item_A - 0.5) * 2, self.files_A[index % len(self.files_A)]


    def __len__(self):
        return len(self.files_A)

def inference():
    transform = transforms.Compose([
        transforms.Resize(target_shape),
        transforms.ToTensor(),
    ])

    dataset_0 = ImageDataset(os.path.join(
        os.getcwd(), "datasets"), transform=transform)
    dataset_1 = ImageDataset(os.path.join(
        os.getcwd(), "datasets"), transform=transform, mode='test')
    gen_AB = Generator(dim_A, dim_B).to(device)
    gen_BA = Generator(dim_B, dim_A).to(device)


    # Feel free to change pretrained to False if you're training the model from scratch
    pretrained = True
    if pretrained:
        pre_dict = torch.load('GanResults/models/'+T_NUM+'/cycleGAN_G1_a_'+str(best_model)+'.pth')
        gen_AB.load_state_dict(pre_dict['gen_AB'])
        gen_BA.load_state_dict(pre_dict['gen_BA'])
        
    dataloader_0 = DataLoader(dataset_0, batch_size=batch_size, shuffle=False)
    dataloader_1 = DataLoader(dataset_1, batch_size=batch_size, shuffle=False)
    cur_step = 0

    for real_A, file in tqdm(dataloader_0):
        # image_width = image.shape[3]
        real_A = nn.functional.interpolate(real_A, size=target_shape)
        real_A = real_A.to(device)
        # with torch.no_grad():
        #     fake_A = gen_BA(real_B)
        ### Update discriminator B ###
        gen_BA.eval()
        with torch.no_grad():
            fake_B = gen_BA(real_A)
        ### Visualization code ###
        if show:
            show_tensor_images(fake_B, file, 'fake', size=(
                dim_B, target_shape, target_shape))
        # You can change save_model to True if you'd like to save the model
        cur_step += 1
        
    for real_A, file in tqdm(dataloader_1):
        # image_width = image.shape[3]
        real_A = nn.functional.interpolate(real_A, size=target_shape)
        real_A = real_A.to(device)
        # with torch.no_grad():
        #     fake_A = gen_BA(real_B)
        ### Update discriminator B ###
        gen_BA.eval()
        with torch.no_grad():
            fake_B = gen_BA(real_A)
        ### Visualization code ###
        if show:
            show_tensor_images(fake_B, file, 'fake', size=(
                dim_B, target_shape, target_shape))
        # You can change save_model to True if you'd like to save the model
        cur_step += 1

if __name__=="__main__":
    inference()
