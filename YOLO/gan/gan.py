from PIL import Image
from torch.utils.data import Dataset
import os
import glob
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(42)
import csv
import pandas as pd
import numpy as np

im_num = 0
T_NUM = "T12"

def show_tensor_images(image_tensor, epoch, real_fake, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    real_0 = image_unflat[0,:,:,:]
    sintec_0 = image_unflat[1,:,:,:]
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    fig_0,ax_0 = plt.subplots()
    ax_0.imshow(image_grid.permute(1, 2, 0).squeeze())
    fig_0.savefig(os.path.join(os.getcwd(), 'GanResults', 'images',T_NUM,
                '{:05d}_{}_batch.png'.format(epoch, real_fake)))
    if real_fake == 'real':
        fig_1,ax_1 = plt.subplots(figsize=(2.91, 2.91))
        ax_1.imshow(sintec_0.permute(1, 2, 0).squeeze())
        ax_1.axis('off')
        fig_1.savefig(os.path.join(os.getcwd(), 'GanResults', 'images',T_NUM,
                    '{:05d}_sintec.png'.format(epoch)), bbox_inches='tight', pad_inches=0)
        fig_2,ax_2 = plt.subplots(figsize=(2.91, 2.91))
        ax_2.imshow(real_0.permute(1, 2, 0).squeeze())
        ax_2.axis('off')
        fig_2.savefig(os.path.join(os.getcwd(), 'GanResults', 'images',T_NUM,
                    '{:05d}_real.png'.format(epoch)), bbox_inches='tight', pad_inches=0)
    else:
        fig_1,ax_1 = plt.subplots(figsize=(2.91, 2.91))
        ax_1.imshow(sintec_0.permute(1, 2, 0).squeeze())
        ax_1.axis('off')
        fig_1.savefig(os.path.join(os.getcwd(), 'GanResults', 'images',T_NUM,
                    '{:05d}_sintec2real.png'.format(epoch)), bbox_inches='tight', pad_inches=0)
        fig_2,ax_2 = plt.subplots(figsize=(2.91, 2.91))
        ax_2.axis('off')
        ax_2.imshow(real_0.permute(1, 2, 0).squeeze())
        fig_2.savefig(os.path.join(os.getcwd(), 'GanResults', 'images',T_NUM,
                    '{:05d}_real2sintec.png'.format(epoch)), bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.pause(3)
    plt.close('all')
    

class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.transform = transform
        self.files_A = sorted(glob.glob(os.path.join(
            root, 'Gan_Real', 'images', mode, '*.*')))
        self.files_B = sorted(
            glob.glob(os.path.join(root, 'Gan_Sintec_0_30', 'images', mode, '*.*')))
        # if len(self.files_A) > len(self.files_B):
        #     self.files_A, self.files_B = self.files_B, self.files_A
        self.new_perm()

    def new_perm(self):
        self.randperm = torch.randperm(len(self.files_B))[:len(self.files_A)]

    def __getitem__(self, index):
        item_A = self.transform(Image.open(
            self.files_A[index % len(self.files_A)]))
        item_B = self.transform(Image.open(self.files_B[self.randperm[index]]))
        if item_A.shape[0] != 3:
            item_A = item_A.repeat(3, 1, 1)
        if item_B.shape[0] != 3:
            item_B = item_B.repeat(3, 1, 1)
        if index == len(self) - 1:
            self.new_perm()
        # Old versions of PyTorch didn't support normalization for different-channeled images
        return (item_A - 0.5) * 2, (item_B - 0.5) * 2

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))

class DemodConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=1) -> None:
        super().__init__()
        
        self.epsilon = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = input_channels
        self.out_channel = output_channels
        
        self.weights = nn.Parameter(torch.randn(1,output_channels,input_channels,kernel_size,kernel_size))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.randn(output_channels))
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
    def forward(self, input):
        batch, in_channel, height, width = input.shape
        
        demod = torch.rsqrt(self.weights.pow(2).sum([2,3,4])+ self.epsilon)
        weight = self.weights * demod.view(batch,self.out_channel,1,1,1)
        
        weight = weight.view(batch*self.out_channel,
                             in_channel,
                             self.kernel_size,
                             self.kernel_size)
        input = input.view(1,batch*in_channel,height,width)
        if self.bias is None:
            out = torch.conv2d(input,weight=weight,padding=self.padding,groups=batch,dilation=self.dilation,stride=self.stride)
        else:
            out = torch.conv2d(input,weight=weight,bias=self.bias, padding=self.padding,groups=batch,dilation=self.dilation,stride=self.stride)
        _, _, height, width = out.shape  
        out = out.view(batch, self.out_channel,height,width)
        return out
    
class ResidualBlock(nn.Module):
    '''
    ResidualBlock Class:
    Performs two convolutions and an instance normalization, the input is added
    to this output to form the residual block output.
    Values:
        input_channels: the number of channels to expect from a given input
    '''

    def __init__(self, input_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = DemodConv2d(input_channels, input_channels,
                               kernel_size=3, padding=1)
        self.conv2 = DemodConv2d(input_channels, input_channels,
                               kernel_size=3, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        '''
        Function for completing a forward pass of ResidualBlock: 
        Given an image tensor, completes a residual block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        original_x = x.clone()
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return original_x + x

class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs a convolution followed by a max pool operation and an optional instance norm.
    Values:
        input_channels: the number of channels to expect from a given input
    '''

    def __init__(self, input_channels, use_bn=True, kernel_size=3, activation='relu'):
        super(ContractingBlock, self).__init__()
        self.conv1 = DemodConv2d(input_channels, input_channels * 2,
                               kernel_size=kernel_size, padding=1, stride=2)
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)


    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock: 
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x)
        x = self.activation(x)
        return x

class ExpandingBlock(nn.Module):
    '''
    ExpandingBlock Class:
    Performs a convolutional transpose operation in order to upsample, 
        with an optional instance norm
    Values:
        input_channels: the number of channels to expect from a given input
    '''

    def __init__(self, input_channels, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            input_channels, input_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.up1 = nn.Upsample(scale_factor = 2, mode='nearest'),
        # self.ref1 = nn.ReflectionPad2d(1)
        # self.conv2 = nn.Conv2d(input_channels, int(input_channels / 2),
        #                                      kernel_size=3, stride=1, padding=0)
        
        self.activation = nn.ReLU()

    def forward(self, x):
        '''
        Function for completing a forward pass of ExpandingBlock: 
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
            skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        x = self.conv1(x)
        
        # x = self.up1(x)
        # x = self.ref1(x)
        # x = self.conv2(x)
        
        x = self.activation(x)
        return x

class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a Generator - 
    maps each the output to the desired number of output channels
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''

    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = DemodConv2d(input_channels, output_channels,
                              kernel_size=7, padding=3)

    def forward(self, x):
        '''
        Function for completing a forward pass of FeatureMapBlock: 
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x

class Generator(nn.Module):
    '''
    Generator Class
    A series of 2 contracting blocks, 9 residual blocks, and 2 expanding blocks to 
    transform an input image into an image from the other class, with an upfeature
    layer at the start and a downfeature layer at the end.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''

    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super(Generator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        res_mult = 4
        self.res0 = ResidualBlock(hidden_channels * res_mult)
        self.res1 = ResidualBlock(hidden_channels * res_mult)
        self.res2 = ResidualBlock(hidden_channels * res_mult)
        self.res3 = ResidualBlock(hidden_channels * res_mult)
        self.res4 = ResidualBlock(hidden_channels * res_mult)
        self.res5 = ResidualBlock(hidden_channels * res_mult)
        self.res6 = ResidualBlock(hidden_channels * res_mult)
        self.res7 = ResidualBlock(hidden_channels * res_mult)
        self.res8 = ResidualBlock(hidden_channels * res_mult)
        self.expand2 = ExpandingBlock(hidden_channels * 4)
        self.expand3 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        '''
        Function for completing a forward pass of Generator: 
        Given an image tensor, passes it through the U-Net with residual blocks
        and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x0 = self.upfeature(x)
        x0 = self.contract1(x0)
        x0 = self.contract2(x0)
        x0 = self.res0(x0)
        x0 = self.res1(x0)
        x0 = self.res2(x0)
        x0 = self.res3(x0)
        x0 = self.res4(x0)
        x0 = self.res5(x0)
        x0 = self.res6(x0)
        x0 = self.res7(x0)
        x0 = self.res8(x0)
        x0 = self.expand2(x0)
        x0 = self.expand3(x0)
        xn = self.downfeature(x0)
        return self.tanh(xn)

class Discriminator(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake. 
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''

    def __init__(self, input_channels, hidden_channels=64):
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(
            hidden_channels, use_bn=False, kernel_size=4, activation='lrelu')
        self.contract2 = ContractingBlock(
            hidden_channels * 2, kernel_size=4, activation='lrelu')
        self.contract3 = ContractingBlock(
            hidden_channels * 4, kernel_size=4, activation='lrelu')
        self.final = nn.Conv2d(hidden_channels * 8, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        xn = self.final(x3)
        return xn

adv_criterion = nn.MSELoss()
recon_criterion = nn.L1Loss()

n_epochs = 500
dim_A = 3
dim_B = 3
display_step = 200
batch_size = 1
lr = 5e-4
load_shape = 224
target_shape = 224
device = 'cuda'

transform = transforms.Compose([
    # transforms.Resize(load_shape),
    transforms.Resize(target_shape),
    # transforms.RandomCrop(target_shape),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

dataset_train = ImageDataset(os.path.join(
    os.getcwd(), "datasets"), transform=transform)
dataset_test = ImageDataset(os.path.join(
    os.getcwd(), "datasets"), transform=transform, mode="test")

gen_AB = Generator(dim_A, dim_B).to(device)
gen_BA = Generator(dim_B, dim_A).to(device)
gen_opt = torch.optim.Adam(list(gen_AB.parameters()) +
                           list(gen_BA.parameters()), lr=lr, betas=(0.5, 0.999))
disc_A = Discriminator(dim_A).to(device)
disc_A_opt = torch.optim.Adam(disc_A.parameters(), lr=lr, betas=(0.5, 0.999))
disc_B = Discriminator(dim_B).to(device)
disc_B_opt = torch.optim.Adam(disc_B.parameters(), lr=lr, betas=(0.5, 0.999))


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


# Feel free to change pretrained to False if you're training the model from scratch
pretrained = False
if pretrained:
    pre_dict = torch.load('./YOLO/cycleGAN_100000.pth')
    gen_AB.load_state_dict(pre_dict['gen_AB'])
    gen_BA.load_state_dict(pre_dict['gen_BA'])
    gen_opt.load_state_dict(pre_dict['gen_opt'])
    disc_A.load_state_dict(pre_dict['disc_A'])
    disc_A_opt.load_state_dict(pre_dict['disc_A_opt'])
    disc_B.load_state_dict(pre_dict['disc_B'])
    disc_B_opt.load_state_dict(pre_dict['disc_B_opt'])
else:
    gen_AB = gen_AB.apply(weights_init)
    gen_BA = gen_BA.apply(weights_init)
    disc_A = disc_A.apply(weights_init)
    disc_B = disc_B.apply(weights_init)

def get_disc_loss(real_X, fake_X, disc_X, adv_criterion):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        real_X: the real images from pile X
        fake_X: the generated images of class X
        disc_X: the discriminator for class X; takes images and returns real/fake class X
            prediction matrices
        adv_criterion: the adversarial loss function; takes the discriminator 
            predictions and the target labels and returns a adversarial 
            loss (which you aim to minimize)
    '''
    #### START CODE HERE ####
    disc_fake_X_hat = disc_X(fake_X.detach())  # Detach generator
    disc_fake_X_loss = adv_criterion(
        disc_fake_X_hat, torch.zeros_like(disc_fake_X_hat))
    disc_real_X_hat = disc_X(real_X)
    disc_real_X_loss = adv_criterion(
        disc_real_X_hat, torch.ones_like(disc_real_X_hat))
    disc_loss = (disc_fake_X_loss + disc_real_X_loss) / 2
    #### END CODE HERE ####
    return disc_loss

def get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion):
    '''
    Return the adversarial loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        disc_Y: the discriminator for class Y; takes images and returns real/fake class Y
            prediction matrices
        gen_XY: the generator for class X to Y; takes images and returns the images 
            transformed to class Y
        adv_criterion: the adversarial loss function; takes the discriminator 
                  predictions and the target labels and returns a adversarial 
                  loss (which you aim to minimize)
    '''
    #### START CODE HERE ####
    fake_Y = gen_XY(real_X)
    disc_fake_Y_hat = disc_Y(fake_Y)
    adversarial_loss = adv_criterion(
        disc_fake_Y_hat, torch.ones_like(disc_fake_Y_hat))
    #### END CODE HERE ####
    return adversarial_loss, fake_Y

def get_identity_loss(real_X, gen_YX, identity_criterion):
    '''
    Return the identity loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        gen_YX: the generator for class Y to X; takes images and returns the images 
            transformed to class X
        identity_criterion: the identity loss function; takes the real images from X and
                        those images put through a Y->X generator and returns the identity 
                        loss (which you aim to minimize)
    '''
    #### START CODE HERE ####
    identity_X = gen_YX(real_X)
    identity_loss = identity_criterion(identity_X, real_X)
    #### END CODE HERE ####
    return identity_loss, identity_X

def get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion):
    '''
    Return the cycle consistency loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        fake_Y: the generated images of class Y
        gen_YX: the generator for class Y to X; takes images and returns the images 
            transformed to class X
        cycle_criterion: the cycle consistency loss function; takes the real images from X and
                        those images put through a X->Y generator and then Y->X generator
                        and returns the cycle consistency loss (which you aim to minimize)
    '''
    #### START CODE HERE ####
    cycle_X = gen_YX(fake_Y)
    cycle_loss = cycle_criterion(cycle_X, real_X)
    #### END CODE HERE ####
    return cycle_loss, cycle_X

def get_gen_loss(real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, identity_criterion, cycle_criterion, lambda_identity=0.05, lambda_cycle=10):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        real_A: the real images from pile A
        real_B: the real images from pile B
        gen_AB: the generator for class A to B; takes images and returns the images 
            transformed to class B
        gen_BA: the generator for class B to A; takes images and returns the images 
            transformed to class A
        disc_A: the discriminator for class A; takes images and returns real/fake class A
            prediction matrices
        disc_B: the discriminator for class B; takes images and returns real/fake class B
            prediction matrices
        adv_criterion: the adversarial loss function; takes the discriminator 
            predictions and the true labels and returns a adversarial 
            loss (which you aim to minimize)
        identity_criterion: the reconstruction loss function used for identity loss
            and cycle consistency loss; takes two sets of images and returns
            their pixel differences (which you aim to minimize)
        cycle_criterion: the cycle consistency loss function; takes the real images from X and
            those images put through a X->Y generator and then Y->X generator
            and returns the cycle consistency loss (which you aim to minimize).
            Note that in practice, cycle_criterion == identity_criterion == L1 loss
        lambda_identity: the weight of the identity loss
        lambda_cycle: the weight of the cycle-consistency loss
    '''
    # Hint 1: Make sure you include both directions - you can think of the generators as collaborating
    # Hint 2: Don't forget to use the lambdas for the identity loss and cycle loss!
    #### START CODE HERE ####
    # Adversarial Loss -- get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion)
    adv_loss_BA, fake_A = get_gen_adversarial_loss(
        real_B, disc_A, gen_BA, adv_criterion)
    adv_loss_AB, fake_B = get_gen_adversarial_loss(
        real_A, disc_B, gen_AB, adv_criterion)
    gen_adversarial_loss = adv_loss_BA + adv_loss_AB

    # Identity Loss -- get_identity_loss(real_X, gen_YX, identity_criterion)
    # identity_loss_A, identity_A = get_identity_loss(
    #     real_A, gen_BA, identity_criterion)
    # identity_loss_B, identity_B = get_identity_loss(
    #     real_B, gen_AB, identity_criterion)
    # gen_identity_loss = identity_loss_A + identity_loss_B

    # Cycle-consistency Loss -- get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion)
    cycle_loss_BA, cycle_A = get_cycle_consistency_loss(
        real_A, fake_B, gen_BA, cycle_criterion)
    cycle_loss_AB, cycle_B = get_cycle_consistency_loss(
        real_B, fake_A, gen_AB, cycle_criterion)
    gen_cycle_loss = cycle_loss_BA + cycle_loss_AB

    # Total loss
    # gen_loss = lambda_identity * gen_identity_loss +  lambda_cycle * gen_cycle_loss + gen_adversarial_loss
    gen_loss = lambda_cycle * gen_cycle_loss + gen_adversarial_loss
    #### END CODE HERE ####
    return gen_loss, fake_A, fake_B

plt.rcParams["figure.figsize"] = (10, 10)

def train(save_model=False):
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    train_size = len(dataloader_train)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    test_size = len(dataloader_test)
    gen_loss_list_train = []
    gen_loss_list_test = []
    disc_loss_list_train = []
    disc_loss_list_test = []
    show_a = None
    show_b = None
    for epoch in range(n_epochs):
        # Dataloader returns the batches
        # for image, _ in tqdm(dataloader):
        for real_A, real_B in tqdm(dataloader_train):
            # image_width = image.shape[3]
            real_A = nn.functional.interpolate(real_A, size=target_shape)
            real_B = nn.functional.interpolate(real_B, size=target_shape)
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            ### Update discriminator A ###
            disc_A_opt.zero_grad()  # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake_A = gen_BA(real_B)
            disc_A_loss = get_disc_loss(real_A, fake_A, disc_A, adv_criterion)
            disc_A_loss.backward(retain_graph=True)  # Update gradients
            disc_A_opt.step()  # Update optimizer

            ### Update discriminator B ###
            disc_B_opt.zero_grad()  # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake_B = gen_AB(real_A)
            disc_B_loss = get_disc_loss(real_B, fake_B, disc_B, adv_criterion)
            disc_B_loss.backward(retain_graph=True)  # Update gradients
            disc_B_opt.step()  # Update optimizer

            ### Update generator ###
            gen_opt.zero_grad()
            gen_loss, fake_A, fake_B = get_gen_loss(
                real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, recon_criterion, recon_criterion
            )
            gen_loss.backward()  # Update gradients
            gen_opt.step()  # Update optimizer

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_A_loss.item()/train_size
            
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item()/train_size
            show_a = real_A
            show_b = real_B
    
        gen_loss_list_train.append(mean_generator_loss)
        disc_loss_list_train.append(mean_discriminator_loss)
        print(
            f"Epoch {epoch}:Train Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
        show_tensor_images(torch.cat([show_a, show_b]), epoch, 'real', size=(
            dim_A, target_shape, target_shape))
        show_tensor_images(torch.cat([fake_B, fake_A]), epoch, 'fake', size=(
            dim_B, target_shape, target_shape))
        mean_generator_loss = 0
        mean_discriminator_loss = 0
        
        # You can change save_model to True if you'd like to save the model
        if save_model:
            torch.save({
                'gen_AB': gen_AB.state_dict(),
                'gen_BA': gen_BA.state_dict(),
                'gen_opt': gen_opt.state_dict(),
                'disc_A': disc_A.state_dict(),
                'disc_A_opt': disc_A_opt.state_dict(),
                'disc_B': disc_B.state_dict(),
                'disc_B_opt': disc_B_opt.state_dict()
            }, f"./GanResults/models/"+T_NUM+f"/cycleGAN_G1_a_{epoch}.pth")
            
        for real_A, real_B in tqdm(dataloader_test):
            # image_width = image.shape[3]
            real_A = nn.functional.interpolate(real_A, size=target_shape)
            real_B = nn.functional.interpolate(real_B, size=target_shape)
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            ### Update discriminator A ###
            with torch.no_grad():
                fake_A = gen_BA(real_B)
            disc_A_loss = get_disc_loss(real_A, fake_A, disc_A, adv_criterion)

            ### Update discriminator B ###
            with torch.no_grad():
                fake_B = gen_AB(real_A)
            disc_B_loss = get_disc_loss(real_B, fake_B, disc_B, adv_criterion)

            ### Update generator ###
            gen_loss, fake_A, fake_B = get_gen_loss(
                real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, recon_criterion, recon_criterion
            )
            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_A_loss.item() / test_size
            
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / test_size

        ### Visualization code ###
        gen_loss_list_test.append(mean_generator_loss)
        disc_loss_list_test.append(mean_discriminator_loss)
        
        print(
            f"Epoch {epoch}: Validation Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
        mean_generator_loss = 0
        mean_discriminator_loss = 0
        
        fig, ax = plt.subplots()
        ax.plot(list(range(epoch+1)),gen_loss_list_train, label='Generator Train Loss')
        ax.plot(list(range(epoch+1)),disc_loss_list_train, label='Discriminator Train Loss')
        ax.plot(list(range(epoch+1)),gen_loss_list_test, label='Generator Test Loss')
        ax.plot(list(range(epoch+1)),disc_loss_list_test, label='Discriminator Test Loss')
        ax.set_title('Training/Test Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        fig.savefig('./GanResults/Loss/'+T_NUM+'/TrainLoss.png')
        plt.close(fig)
        
        log_dict = {
                "Gen_Train_loss": gen_loss_list_train,
                "Gen_Val_loss": gen_loss_list_test,
                "Disc_Train_loss": disc_loss_list_train,
                "Disc_Val_loss": disc_loss_list_test,
            }
        df = pd.DataFrame(log_dict)
        df.to_csv('./GanResults/Loss/'+T_NUM+'/TrainLoss.csv', index=False)

if __name__ == "__main__":
    if not os.path.exists(os.path.join(os.getcwd(), 'GanResults', 'images', T_NUM)):
        os.makedirs(os.path.join(os.path.join(os.getcwd(), 'GanResults', 'images', T_NUM)))
        os.makedirs(os.path.join(os.path.join(os.getcwd(), 'GanResults', 'Loss', T_NUM)))
        os.makedirs(os.path.join(os.path.join(os.getcwd(), 'GanResults', 'models', T_NUM)))
    train(save_model=True)
