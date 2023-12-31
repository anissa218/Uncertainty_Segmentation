# python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train_spup3.py

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # changed to 0 
import random
import logging
import onnx
import numpy as np
import pandas as pd
import time
import setproctitle
import torch
import torch.optim
# from models import criterions
from models.lib.VNet3D import VNet
from plot import loss_plot,metrics_plot
from models.lib.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from models.lib.vit_seg_modeling import ViT
from models.lib.probabilistic_unet import ProbabilisticUnet
from models.lib.probabilistic_unet2D import ProbabilisticUnet2D
from models.lib.VNet3D import VNet
from models.lib.VNet2D import VNet2D
from models.lib.TransU_zoo import Transformer_U
from models.lib.UNet3DZoo import Unet,AttUnet,Unetdrop
from models.lib.UNet2DZoo import Unet2D,AttUnet2D,resnet34_unet,Unet2Ddrop
# removed import below (anissa)
#from models.lib.GetPromptModel import build_promptmodel
#from models.criterions import softmax_dice,FocalLoss,DiceLoss,DC_and_BCE_loss,SDiceLoss,get_soft_label
from models.criterions import softmax_dice,FocalLoss,DiceLoss,SDiceLoss,get_soft_label

from data.transform import ISIC2018_transform,LiTS2017_transform
from data.BraTS2019 import BraTS
from data.ISIC2018 import ISIC
#from data.COVID19 import Covid
#from data.CHAOS20 import CHAOS
from data.LiTS17 import LiTS
from data.Autopet2023 import Autopet # anissa: added this
from torch.utils.data import DataLoader
from torch.nn import functional as F
from models.lib.utils import l2_regularisation
from tensorboardX import SummaryWriter
from predict import validate_softmax,test_softmax,one_hot,one_hot_co,one_hot_co2D,testensemblemax
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def getArgs():
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    parser = argparse.ArgumentParser()

    # anissa changes:
    parser.add_argument('--user', default='anissa', type=str)
    parser.add_argument('--experiment', default='UMIS', type=str)
    parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
    parser.add_argument('--description',
                        default='Trustworthy medical image segmentation by coco,'
                                'training on trai_imgsn.txt!',
                        type=str)
    parser.add_argument("--mode", default="train", type=str, help="train/test/train&test")
    parser.add_argument('--dataset', default='autopet', type=str, help="autopet/BraTS/ISIC/COVID/CHAOS/LiTS")
    parser.add_argument('--crop_H', default=192, type=int)
    parser.add_argument('--crop_W', default=192, type=int)
    parser.add_argument('--crop_D', default=192, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--input_modality', default='petct', type=str, help="petct/t1/t2/both/four")

    
    parser.add_argument("--folder", default="folder0", type=str, help="folder0/folder1/folder2/folder3/folder4")
    parser.add_argument('--input_C', default=4, type=int)
    parser.add_argument('--input_H', default=240, type=int)
    parser.add_argument('--input_W', default=240, type=int)
    parser.add_argument('--input_D', default=160, type=int)  # 155
    parser.add_argument('--output_D', default=155, type=int)
    # Training Information
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    # parser.add_argument('--amsgrad', default=True, type=bool)
    parser.add_argument('--submission', default='./results', type=str)
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument('--no_cuda', default=False, type=bool)
    parser.add_argument('--batch_size', default=2, type=int, help="2/4/8/16")
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--end_epochs', default=200, type=int)
    parser.add_argument('--save_freq', default=5, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--load', default=True, type=bool)
    parser.add_argument('--model_name', default='U', type=str, help="AU/V/U/PU/ResU/Udrop/UE0/")
    parser.add_argument('--en_time', default=10, type=int)
    parser.add_argument('--OOD_Condition', default='normal', type=str, help="normal/noise/mask/blur/spike/ghost/")
    parser.add_argument('--OOD_Level', default=1, type=int, help="0: 'No',1:'Low', 2:'Upper Low', 3:'Mid', 4:'Upper Mid', 5:'High'")
    parser.add_argument('--use_TTA', default=False, type=bool, help="True/False")
    parser.add_argument('--snapshot', default=True, type=bool, help="True/False") # visualization results
    parser.add_argument('--save_format', default='nii', type=str)
    parser.add_argument('--test_date', default='2023-01-01', type=str)
    parser.add_argument('--test_epoch', default=199, type=int)
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
         default=16, help='vit_patches_size, default is 16')

    # anissa added thiese: 
    parser.add_argument('--train_file', default='/gpfs3/well/papiez/users/hri611/python/UMIS/train_imgs.txt', type=str)
    parser.add_argument('--val_file', default='val_imgs.txt', type=str)
    parser.add_argument('--test_file', default='val_imgs.txt', type=str)

    args = parser.parse_args()

    return args

def getModel(args):
    if args.dataset == 'autopet':
        if args.model_name == 'attU':
            model = AttUnet(in_channels=2, base_channels=16, num_classes=2)
        else:
            model = Unet(in_channels=2, base_channels=16, num_classes=2) # changed num of channels and classes


    # if args.dataset=='BraTS':
    #     if args.model_name == 'TransU' and args.input_modality == 'four':
    #         _, model = Transformer_U(dataset='BraTS', _conv_repr=True, _pe_type="learned")
    #     elif args.model_name == 'AU' and args.input_modality == 'four':
    #         model = AttUnet(in_channels=4, base_channels=16, num_classes=args.num_classes)
    #     elif args.model_name == 'AU':
    #         model = AttUnet(in_channels=1, base_channels=16, num_classes=args.num_classes)
    #     elif args.model_name == 'V' and args.input_modality == 'four':
    #         model = VNet(n_channels=4, n_classes=args.num_classes, n_filters=16, normalization='gn', has_dropout=False)
    #     elif args.model_name == 'UE0' or 'UE00' or 'UE01' or 'UE02' or 'UE03' or 'UE04' or 'UE05' or 'UE06' or 'UE07' or 'UE08' or 'UE09':
    #         # args.lr = 0.0002 # O:0.0002  0.0001 0.0002 0.0003 0.001
    #         # args.batch_size = 4  # O:8    4 8 16
    #         model = Unet(in_channels=4, base_channels=16, num_classes=4)
    #     elif args.model_name == 'U' and args.input_modality == 'four':
    #         model = Unet(in_channels=4, base_channels=16, num_classes=4)
    #     elif args.model_name == 'U':
    #         model = Unet(in_channels=1, base_channels=16, num_classes=4)
    
    else:
        print('There is no this dataset')
        raise NameError
    return model

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
        
    else:
        train_loader=None
        valid_loader=None
        test_loader=None
        print('There is no this dataset')
        raise NameError
    return train_loader,valid_loader,test_loader

def val(args,model,checkpoint_dir,epoch,best_dice,valid_loader):

    print('Samples for valid = {}'.format(len(valid_loader.dataset)))

    start_time = time.time()
    model.eval()
    with torch.no_grad():
        best_dice,aver_dice,aver_iou = validate_softmax(args,save_dir = checkpoint_dir,
                                                        best_dice = best_dice,
                                                current_epoch = epoch,
                                                valid_loader = valid_loader,
                                                model = model,
                                                names = valid_loader.dataset.image_list,
                                                )
        # dice_list.append(aver_dice)
        # iou_list.append(aver_iou)
    end_time = time.time()
    full_test_time = (end_time-start_time)/60
    average_time = full_test_time/len(valid_loader.dataset)
    print('{:.2f} minutes!'.format(average_time))
    return best_dice,aver_dice,aver_iou

def train(args,model,train_loader,valid_loader,criterion_dl,model_name):
    print('Samples for train = {}'.format(len(train_loader.dataset)))

    logging.info('--------------------------------------This is all argsurations----------------------------------')
    for arg in vars(args):
        logging.info('{}={}'.format(arg, getattr(args, arg)))
    logging.info('----------------------------------------This is a halving line----------------------------------')
    logging.info('{}'.format(args.description))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint', args.experiment+args.date)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    resume = ''

    writer = SummaryWriter()

    if os.path.isfile(resume) and args.load:
        logging.info('loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])

        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.resume, args.start_epoch))
    else:
        logging.info('re-training!!!')

    start_time = time.time()

    torch.set_grad_enabled(True)
    loss_list = []
    dice_list = []
    iou_list = []
    best_dice =0
    # args.start_epoch
    for epoch in range(args.start_epoch, args.end_epochs):
        epoch_loss = 0
        loss = 0
        runtimes=[]
        # loss1 = 0
        # loss2 = 0
        # loss3 = 0
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch+1, args.end_epochs))
        start_epoch = time.time()
        for i, data in enumerate(train_loader):

            adjust_learning_rate(optimizer, epoch, args.end_epochs, args.lr)

            x, target = data
            x = x.cuda()
            target = target.cuda()

            if model_name =='PU': # anissa: not relevant
                if args.dataset == 'BraTS':
                    onehot_target = one_hot_co(target, args.num_classes)
                else:
                    onehot_target = get_soft_label(target, args.num_classes).permute(0, 3, 1, 2)
                # output = model(x)
                model.forward(x, onehot_target, training=True)
                elbo = model.elbo(onehot_target)
                reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior) + l2_regularisation(
                    model.fcomb.layers)
                loss = -elbo + 1e-5 * reg_loss
            else:
                torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
                start_time = time.time()
                output = model(x)
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                # logging.info('Single sample train time consumption {:.2f} minutes!'.format(elapsed_time / 60))
                runtimes.append(elapsed_time)
                # if args.dataset == 'LiTS':
                #     target = target.unsqueeze(1)

                output = F.softmax(output,1)
                #target = target.unsqueeze(1) # for SDiceloss # anissa: uncommented this as seemed the label was already in correct dimensions 
                soft_target = get_soft_label(target, args.num_classes) # for SDiceloss: mean loss
                loss = criterion_dl(output, soft_target)  # for SDiceloss


            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            reduce_loss = loss.data.cpu().numpy()

            logging.info('Epoch: {}_Iter:{}  loss: {:.5f}'
                        .format(epoch, i, reduce_loss))
            epoch_loss += reduce_loss

        end_epoch = time.time()
        loss_list.append(epoch_loss)

        writer.add_scalar('lr', optimizer.defaults['lr'], epoch)
        writer.add_scalar('loss', loss, epoch)

        epoch_time_minute = (end_epoch-start_epoch)/60
        remaining_time_hour = (args.end_epochs-epoch-1)*epoch_time_minute/60
        logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
        logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))
        best_dice,aver_dice,aver_iou = val(args,model,checkpoint_dir,epoch,best_dice,valid_loader)
        dice_list.append(aver_dice)
        iou_list.append(aver_iou)
    writer.close()
    # validation

    end_time = time.time()
    total_time = (end_time-start_time)/3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))
    logging.info('----------------------------------The training process finished!-----------------------------------')

    loss_plot(args, loss_list)
    metrics_plot(args, 'dice',dice_list.cpu())
    loss_list.to_csv('loss_list')

def test(args,model,test_loader):
    for arg in vars(args):
        logging.info('{}={}'.format(arg, getattr(args, arg)))
    logging.info('----------------------------------------This is a halving line----------------------------------')
    logging.info('{}'.format(args.description))

    print('Samples for test = {}'.format(len(test_loader.dataset)))

    logging.info('final test........')
    load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                             'checkpoint', args.experiment + args.test_date, args.model_name + '_' +args.dataset +'_'+ args.folder + '_epoch_{}.pth'.format(args.test_epoch))
    # load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
    #                          'checkpoint', args.experiment + args.test_date, args.model_name  + '_epoch_{}.pth'.format(args.test_epoch))

    if os.path.exists(load_file):
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(os.path.join(args.experiment + args.test_date, args.model_name + '_' + args.dataset+'_'+ args.folder  + '_epoch_{}.pth')))
    else:
        print('There is no resume file to load!')


    start_time = time.time()
    model.eval()
    with torch.no_grad():
        aver_dice,aver_noise_dice,aver_hd,aver_noise_hd,aver_assd,aver_noise_assd  = test_softmax(args, test_loader = test_loader,
                                            model = model,
                                            load_file=load_file,
                                            names = test_loader.dataset.image_list,
                                            )
    end_time = time.time()
    full_test_time = (end_time-start_time)/60
    average_time = full_test_time/len(test_loader.dataset)
    print('{:.2f} minutes!'.format(average_time))
    if args.dataset=='BraTS':
        logging.info('aver_dice_WT=%f,aver_dice_TC = %f,aver_dice_ET = %f' % (aver_dice[0],aver_dice[1],aver_dice[2]))
        logging.info('aver_noise_dice_WT=%f,aver_noise_dice_TC = %f,aver_noise_dice_ET = %f' % (aver_noise_dice[0], aver_noise_dice[1], aver_noise_dice[2]))
        logging.info('aver_hd_WT=%f,aver_hd_TC = %f,aver_hd_ET = %f' % (aver_hd[0],aver_hd[1],aver_hd[2]))
        logging.info('aver_noise_hd_WT=%f,aver_noise_hd_TC = %f,aver_noise_hd_ET = %f' % (aver_noise_hd[0], aver_noise_hd[1], aver_noise_hd[2]))
        logging.info('aver_assd_WT=%f,aver_assd_TC = %f,aver_assd_ET = %f' % (aver_assd[0],aver_assd[1],aver_assd[2]))
        logging.info('aver_noise_assd_WT=%f,aver_noise_assd_TC = %f,aver_noise_assd_ET = %f' % (aver_noise_assd[0], aver_noise_assd[1], aver_noise_assd[2]))
    elif args.dataset == 'LiTS':
        logging.info('aver_dice_Liver=%f,aver_dice_Tumor = %f' % (aver_dice[0], aver_dice[1]))
        logging.info('aver_noise_dice_Liver=%f,aver_noise_dice_Tumor = %f' % (
        aver_noise_dice[0], aver_noise_dice[1]))
        logging.info('aver_hd_Liver=%f,aver_hd_Tumor = %f' % (aver_hd[0], aver_hd[1]))
        logging.info('aver_noise_hd_Liver=%f,aver_noise_hd_Tumor = %f' % (
        aver_noise_hd[0], aver_noise_hd[1]))
        logging.info('aver_assd_Liver=%f,aver_assd_Tumor = %f' % (aver_assd[0], aver_assd[1]))
        logging.info('aver_noise_assd_Liver=%f,aver_noise_assd_Tumor = %f' % (
        aver_noise_assd[0], aver_noise_assd[1]))
    else:
        logging.info('aver_dice=%f' % (aver_dice))
        logging.info('aver_noise_dice=%f' % (aver_noise_dice))
        logging.info('aver_hd=%f' % (aver_hd))
        logging.info('aver_noise_hd=%f' % (aver_noise_hd))
        logging.info('aver_hd=%f' % (aver_assd))
        logging.info('aver_noise_hd=%f' % (aver_noise_assd))

def test_ensemble(args,models,test_loader):

    print('Samples for test = {}'.format(len(test_loader.dataset)))

    logging.info('final test........')


    # load ensemble models
    load_model=[]
    # load_model[0]=.23
    for i in range(args.en_time):
        load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                             'checkpoint', args.experiment + args.test_date, args.model_name + str(i) + '_' + args.dataset+ '_' + args.folder +'_epoch_' +'199' +'.pth')
        load_model.append(torch.load(load_file))
        # KK =model[i]
        models[i].load_state_dict(load_model[i]['state_dict'])
    print('Successfully load all ensemble models')
    start_time = time.time()
    for model in models:
        model.eval()
    with torch.no_grad():
        aver_dice,aver_noise_dice,aver_hd,aver_noise_hd,aver_assd,aver_noise_assd = testensemblemax( test_loader = test_loader,
                                            model = models,
                                            args=args,
                                            names=test_loader.dataset.image_list,
                                            )
    end_time = time.time()
    full_test_time = (end_time-start_time)/60
    average_time = full_test_time/len(test_loader.dataset)
    print('average pics {:.2f} minutes!'.format(average_time))
    if args.dataset=='BraTS':
        logging.info('aver_dice_WT=%f,aver_dice_TC = %f,aver_dice_ET = %f' % (aver_dice[0],aver_dice[1],aver_dice[2]))
        logging.info('aver_noise_dice_WT=%f,aver_noise_dice_TC = %f,aver_noise_dice_ET = %f' % (aver_noise_dice[0], aver_noise_dice[1], aver_noise_dice[2]))
        logging.info('aver_hd_WT=%f,aver_hd_TC = %f,aver_hd_ET = %f' % (aver_hd[0],aver_hd[1],aver_hd[2]))
        logging.info('aver_noise_hd_WT=%f,aver_noise_hd_TC = %f,aver_noise_hd_ET = %f' % (aver_noise_hd[0], aver_noise_hd[1], aver_noise_hd[2]))
        logging.info('aver_assd_WT=%f,aver_assd_TC = %f,aver_assd_ET = %f' % (aver_assd[0],aver_assd[1],aver_assd[2]))
        logging.info('aver_noise_assd_WT=%f,aver_noise_assd_TC = %f,aver_noise_assd_ET = %f' % (aver_noise_assd[0], aver_noise_assd[1], aver_noise_assd[2]))
    elif args.dataset=='LiTS':
        logging.info('aver_dice_WT=%f,aver_dice_TC = %f' % (aver_dice[0],aver_dice[1]))
        logging.info('aver_noise_dice_WT=%f,aver_noise_dice_TC = %f' % (aver_noise_dice[0], aver_noise_dice[1]))
        logging.info('aver_hd_WT=%f,aver_hd_TC = %f' % (aver_hd[0],aver_hd[1]))
        logging.info('aver_noise_hd_WT=%f,aver_noise_hd_TC = %f' % (aver_noise_hd[0], aver_noise_hd[1]))
        logging.info('aver_assd_WT=%f,aver_assd_TC = %f' % (aver_assd[0],aver_assd[1]))
        logging.info('aver_noise_assd_WT=%f,aver_noise_assd_TC = %f' % (aver_noise_assd[0], aver_noise_assd[1]))

    else:
        logging.info('aver_dice=%f' % (aver_dice))
        logging.info('aver_noise_dice=%f' % (aver_noise_dice))
        logging.info('aver_hd=%f' % (aver_hd))
        logging.info('aver_noise_hd=%f' % (aver_noise_hd))
        logging.info('aver_hd=%f' % (aver_assd))
        logging.info('aver_noise_hd=%f' % (aver_noise_assd))
    # logging.info('aver_dice_WT=%f,aver_dice_TC = %f,aver_dice_ET = %f' % (aver_dice[0],aver_dice[1],aver_dice[2]))
    # logging.info('aver_noise_dice_WT=%f,aver_noise_dice_TC = %f,aver_noise_dice_ET = %f' % (aver_noise_dice[0], aver_noise_dice[1], aver_noise_dice[2]))
    # logging.info('aver_hd_WT=%f,aver_hd_TC = %f,aver_hd_ET = %f' % (aver_hd[0],aver_hd[1],aver_hd[2]))
    # logging.info('aver_noise_hd_WT=%f,aver_noise_hd_TC = %f,aver_noise_hd_ET = %f' % (aver_noise_hd[0], aver_noise_hd[1], aver_noise_hd[2]))

def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)


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

if __name__ == '__main__':
    args = getArgs()
    print(args.dataset)
    if args.dataset == 'autopet':
        args.batch_size = 2
        args.num_classes = 2
        args.out_size = (192, 192,192) # anissa: not sure if this is what i want? #(240, 240,160), changed again from (128,128,128)
    elif args.dataset == 'BraTS':
        args.batch_size = 2
        args.num_classes = 4
        args.out_size = (240, 240,160)
    else:
        print('There is no this dataset')
        raise NameError
    train_loader, valid_loader, test_loader = getDataset(args)

    # criterion = softBCE_dice(aggregate="sum")
    # criterion = softmax_dice
    # criterion_fl = FocalLoss(4)
    criterion_dl = SDiceLoss()
    # criterion_dl = DiceLoss()
    # criterion_dl = DC_and_BCE_loss(bce_kwargs={}, soft_dice_kwargs={})

    num = 2

    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment + args.date)
    log_file = log_dir + '.txt'
    log_args(log_file)
    # Net model choose
    model = getModel(args)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of model's parameter: %.2fM" % (total / 1e6))
    # OOD_Condition = ['noise','blur','mask']
    OOD_Condition = ['mask']
    # OOD_Condition = ['ghost','mask']

    if 'train' in args.mode:
        train(args,model,train_loader,valid_loader,criterion_dl,args.model_name)
    
    # anissa : commented this out for now
    # args.mode = 'test'
    # if args.model_name == 'UE0' or args.model_name =='UED':
    #     models = []
    #     for j in range(0,5):
    #         args.OOD_Condition = OOD_Condition[j]
    #         print("arg.OOD_Condition: %s" % (OOD_Condition[j]))
    #         if args.OOD_Condition =='noise':
    #             start = 5
    #             end = 6
    #         elif args.OOD_Condition =='blur':
    #             start = 5
    #             end = 6
    #         else:
    #             start = 4
    #             end = 6
    #         # start = 1
    #         # end = 5
    #         for i in range(start,end):
    #             print("arg.OOD_Level: %d" % (i))
    #             args.OOD_Level = i
    #             train_loader, valid_loader, test_loader = getDataset(args)
    #             for m in range(args.en_time):
    #                 models.append(model)
    #             test_ensemble(args,models,test_loader)
    # else:
    #     for j in range(0,5):
    #         args.OOD_Condition = OOD_Condition[j]
    #         print("arg.OOD_Condition: %s" % (OOD_Condition[j]))
    #         if args.OOD_Condition =='spike':
    #             start = 1
    #             end = 2
    #         elif args.OOD_Condition == 'ghost':
    #             start = 6
    #             end = 7
    #         else:
    #             start = 3
    #             end = 5
    #         for i in range(start,end):
    #             print("arg.OOD_Level: %d" % (i))
    #             args.OOD_Level = i
    #             train_loader, valid_loader, test_loader = getDataset(args)
    #             test(args, model, test_loader)
