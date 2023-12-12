import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import pickle
from scipy import ndimage
import argparse
import matplotlib.pyplot as plt

### Define transforms ###

class Random_Crop(object):
    def __call__(self, sample):
        image, label = sample['image'],sample['label']

        Max_C,max_H,max_W,max_D = image.shape
        H = random.randint(0, max_H - 128)
        W = random.randint(0, max_W - 128)
        D = random.randint(0, max_D - 128)

        image = image[...,H: H + 128, W: W + 128, D: D + 128] 
        
        #image = image.transpose(1, 2, 3, 0) # so it fits dimensions of code 
        # actually since they re-transpose it later (in brats ToTensorboth) will just not transpose ever
        
        label = label[...,H: H + 128, W: W + 128, D: D + 128]

        #label = label.reshape(1,128,128,128) # not sure if this will work
        
        return {'image': image, 'label': label}


class Random_Crop_128_oversample(object):
    def __call__(self, sample):
        image, label = sample['image'],sample['label']
        Max_C,max_H,max_W,max_D = image.shape

        if np.max(label)>0:
            # if there is some positive regions in the image, should happen 1/2 of time, make sure random cro[ includes pos point
            indices = np.argwhere(label == 1)
            random_index = np.random.choice(len(indices))
            random_point = indices[random_index] # get 4d coordinates - but how do i know they they are at the edges
            C,H,W,D = random_point[0],random_point[1],random_point[2],random_point[3]
            if H + 64 > max_H: # check if point is not at edge of cube
                H = H - 64
            elif H - 64 < 0:
                H = H + 64
            if W + 64 > max_W:
                W = W - 64
            elif W - 64 < 0:
                W = W+64
            if D + 64 > max_D:
                D = D-64
            elif D - 64 < 0:
                D = D+64
            
        else:
            H = random.randint(64, max_H - 64)
            W = random.randint(64, max_W - 64)
            D = random.randint(64, max_D - 64)
    
        image = image[...,H-64: H + 64, W-64: W + 64, D-64: D + 64] 
        label = label[...,H-64: H + 64, W-64: W + 64, D-64: D + 64] 
        
        return {'image': image, 'label': label}

class Random_Crop_192_oversample(object):
    def __call__(self, sample):
        image, label = sample['image'],sample['label']
        Max_C,max_H,max_W,max_D = image.shape

        if np.max(label)>0:
            # if there is some positive regions in the image, should happen 1/2 of time, make sure random cro[ includes pos point
            indices = np.argwhere(label == 1)
            random_index = np.random.choice(len(indices))
            random_point = indices[random_index] # get 4d coordinates - but how do i know they they are at the edges
            C,H,W,D = random_point[0],random_point[1],random_point[2],random_point[3]
            if H + 96 > max_H: # check if point is not at edge of cube
                H = H - 96
            elif H - 96 < 0:
                H = H + 96
            if W + 96 > max_W:
                W = W - 96
            elif W - 96 < 0:
                W = W+96
            if D + 96 > max_D:
                D = D-96
            elif D - 96 < 0:
                D = D+96
            
        else:
            H = random.randint(96, max_H - 96)
            W = random.randint(96, max_W - 96)
            D = random.randint(96, max_D - 96)
    
        image = image[...,H-96: H + 96, W-96: W + 96, D-96: D + 96] 
        label = label[...,H-96: H + 96, W-96: W + 96, D-96: D + 96] 
        
        return {'image': image, 'label': label}


class Random_Crop_192(object):
    def __call__(self, sample):
        image, label = sample['image'],sample['label']

        Max_C,max_H,max_W,max_D = image.shape
        H = random.randint(0, max_H - 192)
        W = random.randint(0, max_W - 192)
        D = random.randint(0, max_D - 192)

        image = image[...,H: H + 192, W: W + 192, D: D + 192] 
        
        #image = image.transpose(1, 2, 3, 0) # so it fits dimensions of code 
        # actually since they re-transpose it later (in brats ToTensorboth) will just not transpose ever
        
        label = label[...,H: H + 192, W: W + 192, D: D + 192]

        #label = label.reshape(1,128,128,128) # not sure if this will work
        
        return {'image': image, 'label': label}

class Fix_Size_400(object):
    def __call__(self, sample):
        image, label = sample['image'],sample['label']

        Max_C,max_H,max_W,max_D = image.shape

        if max_H < 400:
            # pad
            diff_H = 400 - max_H
            image = np.pad(image, ((0, 0), (0, diff_H), (0, 0), (0, 0)), mode='constant') # only padding at top of image
        elif max_H > 400:
            start_H = (max_H - 400)/2
            
            image = image[...,start_H: start_H + 400, :, :] 
            label = label[...,start_H: start_H + 400, :, :] 
        # if max_H == 400 do nothing
        
        return {'image': image, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, label = sample['image'],sample['label']
        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)
        
        image = torch.from_numpy(image).float() #.unsqueeze(0)
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}

def transform(image,label):
    sample = {'image':image,'label':label}
    
    trans = transforms.Compose([
        Random_Crop(),
        ToTensor()
    ]) # can add more data augmentation later

    new_sample = trans(sample)
    new_image, new_label = new_sample['image'],new_sample['label']

    return new_image, new_label

def transform_128_oversample(image,label):
    sample = {'image':image,'label':label}
    
    trans = transforms.Compose([
        Random_Crop_128_oversample(),
        ToTensor()
    ]) # can add more data augmentation later

    new_sample = trans(sample)
    new_image, new_label = new_sample['image'],new_sample['label']

    return new_image, new_label
    
def transform_192(image,label):
    sample = {'image':image,'label':label}
    
    trans = transforms.Compose([
        Random_Crop_192(),
        ToTensor()
    ]) # can add more data augmentation later

    new_sample = trans(sample)
    new_image, new_label = new_sample['image'],new_sample['label']

    return new_image, new_label


def transform_192_oversample(image,label):
    sample = {'image':image,'label':label}
    
    trans = transforms.Compose([
        Random_Crop_192_oversample(),
        ToTensor()
    ]) # can add more data augmentation later

    new_sample = trans(sample)
    new_image, new_label = new_sample['image'],new_sample['label']

    return new_image, new_label

def transform_val(image,label):
    sample = {'image':image,'label':label}
    
    trans = transforms.Compose([
        Random_Crop_192(),
        ToTensor()
    ])

    new_sample = trans(sample)
    new_image, new_label = new_sample['image'],new_sample['label']

    return new_image, new_label

def transform_test(image,label):
    sample = {'image':image,'label':label}
    
    trans = transforms.Compose([
        ToTensor()
    ])
    #         Fix_Size_400(),

    new_sample = trans(sample)
    new_image, new_label = new_sample['image'],new_sample['label']

    return new_image, new_label


### Define dataset class ###

class Autopet(Dataset):
    def __init__(self, list_file, root='train_data', mode='train', modal='t1',OOD_Condition = 'normal',folder='folder0',level = 0):
        #paths, names,only_paths = [], [], []
        names =[]
        with open(list_file) as f:
            for line in f:
                # list of files will have names of all images without file ending
                line = line.strip()
                name = line
                names.append(name)
                #path = os.path.join(root, name)
                #paths.append(path)
        self.root = root
        self.mode = mode
        self.modal = modal
        self.names = names
        self.image_list = names
        #self.paths = paths
        self.OOD_Condition = OOD_Condition
        self.folder = folder
        self.level = level

    def __getitem__(self, item):
        name = self.names[item]
        if self.mode in ['train']:
            label_path = os.path.join(self.root,'pet_mask',name+'_mask.npy')
            image_path = os.path.join(self.root,'petct',name+'_petct.npy')
               
            label = np.load(label_path)
            image = np.load(image_path)
        
            image,label = transform_128_oversample(image,label)

            return image, label

        elif self.mode in ['val']:
            label_path = os.path.join(self.root,'pet_mask',name+'_mask.npy')
            image_path = os.path.join(self.root,'petct',name+'_petct.npy')
               
            label = np.load(label_path)
            image = np.load(image_path)
            
            image,label = transform_128_oversample(image,label)

            return image, label

        elif self.mode in ['test']:
            label_path = os.path.join('test_data','pet_mask',name+'_mask.npy')
            image_path = os.path.join('test_data','petct',name+'_petct.npy')
               
            label = np.load(label_path)
            image = np.load(image_path)

            image,label = transform_test(image,label) # anissa: maybe no random cropping?
            #image,label = transform_128_oversample(image,label) # anissa: maybe no random cropping?


            return image, label

            # have not included anything about ood stuff
            
    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]

class Autopet2(Dataset):
    def __init__(self, list_file, root='train_data', mode='train', modal='t1',OOD_Condition = 'normal',folder='folder0',level = 0):
        #paths, names,only_paths = [], [], []
        names =[]
        with open(list_file) as f:
            for line in f:
                # list of files will have names of all images without file ending
                line = line.strip()
                name = line
                names.append(name)
                #path = os.path.join(root, name)
                #paths.append(path)
        self.root = root
        self.mode = mode
        self.modal = modal
        self.names = names
        self.image_list = names
        #self.paths = paths
        self.OOD_Condition = OOD_Condition
        self.folder = folder
        self.level = level

    def __getitem__(self, item):
        name = self.names[item]
        if self.mode in ['train']:
            label_path = os.path.join(self.root,'pet_mask',name+'_mask.npy')
            image_path = os.path.join(self.root,'petct',name+'_petct.npy')
               
            label = np.load(label_path)
            image = np.load(image_path)
        
            image,label = transform_192_oversample(image,label)

            return image, label

        elif self.mode in ['val']:
            label_path = os.path.join(self.root,'pet_mask',name+'_mask.npy')
            image_path = os.path.join(self.root,'petct',name+'_petct.npy')
               
            label = np.load(label_path)
            image = np.load(image_path)
            
            image,label = transform_192_oversample(image,label)

            return image, label

        elif self.mode in ['test']:
            label_path = os.path.join('test_data','pet_mask',name+'_mask.npy')
            image_path = os.path.join('test_data','petct',name+'_petct.npy')
               
            label = np.load(label_path)
            image = np.load(image_path)

            image,label = transform_test(image,label) # anissa: maybe no random cropping?

            return image, label

            # have not included anything about ood stuff
            
    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/gpfs3/well/papiez/users/hri611/python/UMIS/', type=str)
    parser.add_argument('--train_dir', default='train_data', type=str)
    parser.add_argument('--train_file', default='/gpfs3/well/papiez/users/hri611/python/UMIS/train_imgs.txt', type=str)
    parser.add_argument('--dataset', default='autopet', type=str)
    parser.add_argument('--num_gpu', default= 4, type=int)
    parser.add_argument('--valid_dir', default='train_data', type=str)
    parser.add_argument('--valid_file', default='val_imgs.txt', type=str)
    parser.add_argument('--test_dir', default='test_data', type=str)
    parser.add_argument('--test_file', default='test_imgs.txt', type=str)


    # anissa: ones i haven't changed:
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--num_gpu', default= 4, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--batch_size', default=2, type=int) # anissa: maybe should change this
    parser.add_argument('--modal', default='both', type=str)
    parser.add_argument('--Variance', default=0.1, type=int)
    
    args = parser.parse_args()
    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    val_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    val_root = os.path.join(args.root, args.valid_dir)
    test_list = os.path.join(args.root, args.test_dir, args.test_file) #anissa: commented out anything to do with testing for now
    test_root = os.path.join(args.root, args.test_dir)
    
    train_set = Autopet(train_list, train_root, args.mode,args.modal)
    val_set = Autopet(val_list, val_root, args.mode, args.modal)
    test_set = Autopet(test_list, test_root, args.mode, args.modal)


    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size)
    val_loader = DataLoader(dataset=val_set, batch_size=1)
    test_loader = DataLoader(dataset=test_set, batch_size=1)
    for i, data in enumerate(train_loader):
        x, target = data
        if args.mode == 'test':
            noise = torch.clamp(torch.randn_like(x) * args.Variance, -args.Variance * 2, args.Variance * 2)
            x += noise
        # x_no = np.unique(x.numpy())
        # target_no = np.unique(target.numpy())
