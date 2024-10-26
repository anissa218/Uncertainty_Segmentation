import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.optim as optim
from torch.autograd import Variable
import logging
import time
from torch.utils.data import DataLoader
from PIL import Image
from models.TrustworthySeg import TMSU
from data.transform import ISIC2018_transform,LiTS2017_transform,DRIVE2004_transform,TOAR2019_transform,HC_2018_transform
from data.BraTS2019 import BraTS
from data.Autopet2023 import Autopet, Autopet2 # anissa: added this

# from data.ISIC2018 import ISIC|
# from data.COVID19 import Covid
# from data.CHAOS20 import CHAOS
# from data.LiTS17 import LiTS
# from data.DRIVE04 import DRIVE
# from data.TOAR19 import TOAR
# from data.HC2018 import HC

import cv2
from thop import profile
from models.criterions import get_soft_label
from predict import tailor_and_concat,RandomMaskingGenerator,softmax_output_litsdice,softmax_output_litshd,softmax_assd_litsscore,softmax_mIOU_litsscore,Uentropy_our,cal_ueo,cal_ece_our,softmax_mIOU_score,softmax_output_dice,softmax_output_hd,dice_isic,iou_isic,HD_isic,Dice_isic,IOU_isic,softmax_assd_score
from binary import assd,hd95
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import argparse
import nibabel as nib
import imageio
from plot import loss_plot,metrics_plot
from tensorboardX import SummaryWriter

import pandas as pd

torch.cuda.empty_cache()
def create_args(user='anissa', experiment='U192_valfix2', date=None, description='Trustworthy medical image segmentation by coco, training on trai_imgsn.txt!',
                mode='train', dataset='autopet2', crop_H=192, crop_W=192, crop_D=192, num_classes=2,
                input_modality='petct', folder='folder0', input_C=2, input_H=192, input_W=192, input_D=192,
                output_D=192, lr=0.0002, weight_decay=1e-5, submission='./results', seed=1000, no_cuda=False,
                batch_size=2, start_epoch=0, end_epochs=10, lambda_epochs = 50,save_freq=5, resume='', load=True, model_name='U',
                en_time=10, OOD_Condition='normal', OOD_Level=1, use_TTA=False, snapshot=True, save_format='nii',
                test_date='2023-01-01', test_epoch=199, n_skip=3, vit_name='R50-ViT-B_16', vit_patches_size=16,train_file = 'pos_train_imgs.txt',val_file = 'pos_val_imgs.txt',test_file = 'pos_test_imgs.txt',Uncertainty_Loss=False,save_dir='./results'):
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    date = local_time.split(' ')[0] if date is None else date

    args = argparse.Namespace(user=user, experiment=experiment, date=date, description=description,
                              mode=mode, dataset=dataset, crop_H=crop_H, crop_W=crop_W, crop_D=crop_D,
                              num_classes=num_classes, input_modality=input_modality, folder=folder,
                              input_C=input_C, input_H=input_H, input_W=input_W, input_D=input_D,
                              output_D=output_D, lr=lr, weight_decay=weight_decay, submission=submission,
                              seed=seed, no_cuda=no_cuda, batch_size=batch_size, start_epoch=start_epoch,
                              end_epochs=end_epochs, save_freq=save_freq, resume=resume, load=load,
                              model_name=model_name, en_time=en_time, OOD_Condition=OOD_Condition,
                              OOD_Level=OOD_Level, use_TTA=use_TTA, snapshot=snapshot, save_format=save_format,
                              test_date=test_date, test_epoch=test_epoch, n_skip=n_skip, vit_name=vit_name,
                              vit_patches_size=vit_patches_size,train_file=train_file,val_file=val_file,test_file=test_file,lambda_epochs=lambda_epochs,Uncertainty_Loss=Uncertainty_Loss,save_dir=save_dir)

    return args
def getDataset(args):
    if args.dataset =='autopet':
        base_folder = args.folder

        # anissa
        root_path = '/gpfs3/well/papiez/users/hri611/python/UMIS/'
        train_file = args.train_file
        #train_file = 'train_imgs.txt'
        train_dir='train_data'
        train_list = os.path.join(root_path, train_file)
        train_root = os.path.join(root_path, train_dir)
        train_set = Autopet(train_list, train_root, args.mode,args.input_modality,OOD_Condition=args.OOD_Condition, level=args.OOD_Level) # removed folder = base_folder
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size)
        print('Samples for train = {}'.format(len(train_loader.dataset)))

        valid_file=args.val_file
        valid_dir='train_data'
        valid_list = os.path.join(root_path, valid_file)
        valid_root = os.path.join(root_path, valid_dir)
        valid_set = Autopet(valid_list, valid_root,'val',args.input_modality,OOD_Condition=args.OOD_Condition, level=args.OOD_Level)
        valid_loader = DataLoader(valid_set, batch_size=1)
        print('Samples for valid = {}'.format(len(valid_loader.dataset)))

        # left just in case
        test_file=args.test_file
        test_dir='train_data'
        test_list = os.path.join(root_path, test_file)
        test_root = os.path.join(root_path, test_dir)
        test_set = Autopet(test_list, test_root,'test',args.input_modality,OOD_Condition=args.OOD_Condition, level=args.OOD_Level)
        test_loader = DataLoader(test_set, batch_size=1)
        print('Samples for test = {}'.format(len(test_loader.dataset)))
    
    elif args.dataset =='autopet2':
        base_folder = args.folder

        # anissa
        root_path = '/gpfs3/well/papiez/users/hri611/python/UMIS/'
        train_file = args.train_file
        #train_file = 'train_imgs.txt'
        train_dir='train_data'
        train_list = os.path.join(root_path, train_file)
        train_root = os.path.join(root_path, train_dir)
        train_set = Autopet2(train_list, train_root, args.mode,args.input_modality,OOD_Condition=args.OOD_Condition, level=args.OOD_Level) # removed folder = base_folder
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size)
        print('Samples for train = {}'.format(len(train_loader.dataset)))

        valid_file=args.val_file
        valid_dir='train_data'
        valid_list = os.path.join(root_path, valid_file)
        valid_root = os.path.join(root_path, valid_dir)
        valid_set = Autopet2(valid_list, valid_root,'val',args.input_modality,OOD_Condition=args.OOD_Condition, level=args.OOD_Level)
        valid_loader = DataLoader(valid_set, batch_size=1)
        print('Samples for valid = {}'.format(len(valid_loader.dataset)))

        # left just in case
        test_file=args.test_file
        test_dir='train_data'
        test_list = os.path.join(root_path, test_file)
        test_root = os.path.join(root_path, test_dir)
        test_set = Autopet2(test_list, test_root,'test',args.input_modality,OOD_Condition=args.OOD_Condition, level=args.OOD_Level)
        test_loader = DataLoader(test_set, batch_size=1)
        print('Samples for test = {}'.format(len(test_loader.dataset)))    
    else:
        train_loader=None
        valid_loader=None
        test_loader=None
        print('There is no this dataset')
        raise NameError
    return train_loader,valid_loader,test_loader

def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)

def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - (epoch) / max_epoch, power), 8)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
args = create_args()

#args = getArgs()
print(args.dataset)
if args.dataset == 'autopet':
    args.batch_size = 2
    args.num_classes = 2
    args.out_size = (128, 128,128)
elif args.dataset == 'autopet2':
    args.batch_size = 2
    args.num_classes = 2
    args.out_size = (192, 192,192)
elif args.dataset == 'BraTS':
    args.batch_size = 2
    args.num_classes = 4
    args.out_size = (240, 240,160)
    input_tensor = torch.randn(1, 4, args.out_size[0], args.out_size[1], args.out_size[2]).cuda()
else:
    print('There is no this dataset')
    raise NameError

train_loader, valid_loader, test_loader = getDataset(args)
model = TMSU(args)
# calculate model's Params & Flops
total = sum([param.nelement() for param in model.parameters()])
print("Number of model's Params: %.2fM" % (total / 1e6))

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
model.cuda()

log_dir = os.path.join(os.path.abspath((os.getcwd())), 'log', args.experiment + args.date)
log_file = log_dir + '.txt'
log_args(log_file)
epoch_loss = 0
best_dice = 0
loss_list = []
dice_list = []
OOD_Condition = ['noise','blur','mask']

resume = ''
if os.path.isfile(resume):
    logging.info('loading checkpoint {}'.format(resume))
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    args.start_epoch = checkpoint['epoch']
    logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                .format(resume, args.start_epoch))
else:
    logging.info('re-training!!!')
#writer = SummaryWriter()

#model_name ='U_pos_192_valfix2_U_autopet2_folder0_epoch_199.pth'
#model_name = 'U_autopet_folder0_epoch_199.pth'
model_name = 'U_pos_192_valfix2_U_autopet2_folder0_epoch_133.pth'
Net_name = args.model_name
snapshot = args.snapshot  # False
logging.info('===========>Test begining!===========')
logging.info('--------------------------------------This is all argsurations----------------------------------')
for arg in vars(args):
    logging.info('{}={}'.format(arg, getattr(args, arg)))
logging.info('----------------------------------------This is a halving line----------------------------------')
logging.info('{}'.format(args.description))    
    
if args.Uncertainty_Loss:
    savepath = args.submission + '/'+ str(Net_name) + 'eviloss'  + '/' + str(args.dataset) +'/' + str(args.OOD_Condition) +'/'+ str(args. OOD_Level)
else:
    savepath = args.submission + '/' + str(Net_name)+ 'evi' + '/' + str(args.dataset) +'/' + str(args.OOD_Condition) + '/' + str(args.OOD_Level)
dice_total = 0
dice_total_WT = 0
dice_total_TC = 0
dice_total_ET = 0
hd_total = 0
hd95_total = 0
assd_total = 0
hd_total_WT = 0
hd_total_TC = 0
hd_total_ET = 0
assd_total_WT = 0
assd_total_TC = 0
assd_total_ET = 0
noise_dice_total = 0
noise_dice_total_WT = 0
noise_dice_total_TC = 0
noise_dice_total_ET = 0
iou_total = 0
iou_total_WT = 0
iou_total_TC = 0
iou_total_ET = 0
noise_iou_total = 0
noise_iou_total_WT = 0
noise_iou_total_TC = 0
noise_iou_total_ET = 0
noise_hd_total = 0
noise_assd_total = 0
noise_hd_total_WT = 0
noise_hd_total_TC = 0
noise_hd_total_ET = 0
noise_assd_total_WT = 0
noise_assd_total_TC = 0
noise_assd_total_ET = 0
runtimes = []
certainty_total = 0
noise_certainty_total = 0
mne_total = 0
noise_mne_total = 0
ece_total = 0
noise_ece_total = 0
ueo_total = 0
noise_ueo_total = 0
step = 0

dt_size = len(test_loader.dataset)
if args.Uncertainty_Loss:
    #load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                #args.save_dir,
                                #args.experiment + '_' + args.model_name + '_' + args.dataset +'_'+ args.folder + '_DUloss' + '_epoch_{}.pth'.format(args.test_epoch))
    load_file = os.path.join('/gpfs3/well/papiez/users/hri611/python/UMIS/results/',model_name) # changed anissa

else:
    #load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                #args.save_dir,
                                #args.experiment + '_' + args.model_name + '_' + args.dataset +'_'+ args.folder + '_epoch_{}.pth'.format(args.test_epoch))
    load_file = os.path.join('/gpfs3/well/papiez/users/hri611/python/UMIS/results/',model_name) # changed anissa


if os.path.exists(load_file):
    checkpoint = torch.load(load_file)
    model.load_state_dict(checkpoint['state_dict'])
    args.start_epoch = checkpoint['epoch']
    print('Successfully load checkpoint {}'.format(load_file))
else:
    print('There is no resume file to load!')

names = test_loader.dataset.image_list

model.eval()

dice_list = []
iou_list = []
uncertainty_list =[]
num_classes_list=[]
hd_list = []
ueo_list = []
name_list =[]

for i, data in enumerate(test_loader):
    msg = 'Subject {}/{}, '.format(i + 1, len(test_loader))

    step += 1
    #input, noised_input, target = data  # input ground truth
    input, target = data

    if args.dataset == 'BraTS':
        num_classes = 4
    elif args.dataset == 'autopet' or args.dataset == 'autopet2':
        num_classes = 2
    else:
        num_classes = 2
    
    x = input.cuda()
    #noised_x = noised_input.cuda()
    target = target.cuda()
    torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
    start_time = time.time()
    with torch.no_grad():
        args.mode = 'test'
        if args.dataset =='BraTS':
            evidences = model(x, target[:, :, :, :155], 0, args.mode, args.dataset)
            noised_evidences = model(noised_x, target[:, :, :, :155], 0, args.mode, args.dataset)
        else:
            evidences = model(x, target, 0, args.mode, args.dataset)
            #noised_evidences = model(noised_x, target, 0, args.mode, args.dataset)
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        logging.info('Single sample test time consumption {:.2f} seconds!'.format(elapsed_time))
        runtimes.append(elapsed_time)
        alpha = evidences + 1
        uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)

        S = torch.sum(alpha, dim=1, keepdim=True)
        prob = alpha / S
        mne = Uentropy_our(prob, num_classes)
        # min_mne = torch.min(mne)
        # max_mne = torch.max(mne)

        _, predicted = torch.max(alpha / S, 1)
        output = torch.squeeze(predicted).cpu().detach().numpy()
        # n of unique classes
        num_class = target.max()

        # for noise_x
        # noised_alpha = noised_evidences + 1
        # noised_uncertainty = num_classes / torch.sum(noised_alpha, dim=1, keepdim=True)

        # _, noised_predicted = torch.max(noised_evidences.data, 1)
        # noised_prob = noised_alpha / torch.sum(noised_alpha, dim=1, keepdim=True)
        # noised_mne = Uentropy_our(noised_prob, num_classes)
        # noised_output = torch.squeeze(noised_predicted).cpu().detach().numpy()

        if args.dataset=='autopet' or args.dataset=='autopet2':
            ece = cal_ece_our(torch.squeeze(predicted), torch.squeeze(target.cpu()))
            H, W, T = args.crop_H, args.crop_W, args.crop_D # not sure about this anissa
            
            Otarget = torch.squeeze(target).cpu().numpy()
            target = torch.squeeze(target).cpu().numpy()  # .cpu().numpy(dtype='float32')
            hd_res = softmax_output_hd(output, target)
            assd_res = softmax_assd_score(output, target)
            iou_res = softmax_mIOU_score(output, target)
            dice_res = softmax_output_dice(output, target)
            dice_total_WT += dice_res[0]
            iou_total_WT += iou_res[0]
            hd_total_WT += hd_res[0]
            assd_total_WT += assd_res[0]

            mean_uncertainty = torch.mean(uncertainty)
            
            # mne & ece
            mne_total += torch.mean(mne)  # mix _uncertainty mean_uncertainty mean_uncertainty_succ
            
            # ece
            ece_total += ece
            # U ece ueo
            certainty_total += mean_uncertainty  # mix _uncertainty mean_uncertainty mean_uncertainty_succ
            pc = output

            # ueo
            thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
            to_evaluate = dict()
            to_evaluate['target'] = target
            u = torch.squeeze(uncertainty)
            U = u.cpu().detach().numpy()
            to_evaluate['prediction'] = pc
            to_evaluate['uncertainty'] = U
            UEO = cal_ueo(to_evaluate, thresholds)
            ueo_total += UEO
            print('current_UEO:{}; current_num:{}'.format(UEO, i))

            # confidence map
            conf = 1-uncertainty
            confe = torch.exp(-uncertainty)
            mean_conf = torch.mean(conf)
            mean_confe = torch.mean(confe)
            conf_output = torch.squeeze(conf).cpu().detach().numpy()
            confe_output = torch.squeeze(confe).cpu().detach().numpy()
            
            print('current_U:{};current_num:{}'.format(mean_uncertainty, i))
            print('current_conf:{};current_num:{}'.format(mean_conf, i))
            print('current_confe:{};current_num:{}'.format(mean_confe, i))
            # logging.info('current_U:{};current_num:{}'.format(mean_uncertainty,i))
            # logging.info('current_conf:{};current_num:{}'.format(mean_conf,i))
            # logging.info('current_confe:{};current_num:{}'.format(mean_confe, i))
            
            # uncertainty np
            Otarget = target
            Oinput = torch.squeeze(x,0).cpu().detach().numpy()
            U_output = torch.squeeze(uncertainty).cpu().detach().numpy()
            # U_output = torch.squeeze(mne).cpu().detach().numpy()
            
            name = str(i)
            if names:
                name = names[i]
                msg += '{:>20}, '.format(name)

                # save each image

            output_dir = 'test_output_' + str(model_name)
            sub_dir = name  # Assuming 'name' is the directory you want to create

                # Check if the directories exist before creating them
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            if not os.path.exists(os.path.join(output_dir,sub_dir)):
                os.mkdir(os.path.join(output_dir,sub_dir))
                
                # save inputs
            ct = x[0,0,:,:,:].cpu().numpy()
            pet = x[0,1,:,:,:].cpu().numpy()
            ct_nii = nib.Nifti1Image(ct, affine=np.eye(4))  # You might need to adjust the 'affine' matrix
            pet_nii = nib.Nifti1Image(pet, affine=np.eye(4))  # You might need to adjust the 'affine' matrix
                # Save the NIfTI image to a .nii.gz file
            nib.save(ct_nii, os.path.join(output_dir,sub_dir, 'ct_input.nii.gz'))
            nib.save(pet_nii, os.path.join(output_dir,sub_dir, 'pet_input.nii.gz'))
                
            target = target.astype(np.float32)
            target_nii = nib.Nifti1Image(target, affine=np.eye(4))  # You may need to adjust the affine matrix
                # Save the NIfTI image to a file
            nib.save(target_nii, os.path.join(output_dir,sub_dir, 'target.nii.gz'))
            
            output = output.astype(np.float32)
            output_nii = nib.Nifti1Image(output, affine=np.eye(4))  # You may need to adjust the affine matrix
                # Save the NIfTI image to a file
            nib.save(output_nii, os.path.join(output_dir,sub_dir, 'output.nii.gz'))

            uncertainty = uncertainty[0,0,:,:,:].cpu().numpy()
            uncertainty_nii = nib.Nifti1Image(uncertainty, affine=np.eye(4))  # You might need to adjust the 'affine' matrix
            # Save the NIfTI image to a .nii.gz file
            nib.save(uncertainty_nii, os.path.join(output_dir,sub_dir, 'uncertainty.nii.gz'))
                
            dice_list.append(dice_res[0])
            iou_list.append(iou_res[0])
            uncertainty_list.append(mean_uncertainty.cpu())
            num_classes_list.append(num_class.cpu()) # 0 if only background, 1 if both
            hd_list.append(hd_res[0])
            ueo_list.append(UEO)
            name_list.append(name)

# save results as df
results_df = pd.DataFrame()
results_df['image']=name_list
results_df['dice_list']=dice_list
results_df['iou_list']=iou_list
results_df['uncertainty_list']=[float(tensor) for tensor in uncertainty_list]
results_df['num_classes_list']=[float(tensor) for tensor in num_classes_list]
results_df['hd_list']=hd_list
results_df['ueo_list']=ueo_list

df_name = str(model_name)+'_test_results.csv'
results_df.to_csv(df_name)
