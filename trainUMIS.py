import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from data.Autopet2023 import Autopet, Autopet2
# from data.ISIC2018 import ISIC
# from data.COVID19 import Covid
# from data.CHAOS20 import CHAOS
# from data.LiTS17 import LiTS
# from data.DRIVE04 import DRIVE
# from data.TOAR19 import TOAR
# from data.HC2018 import HC

import cv2
from thop import profile
from models.criterions import get_soft_label
from predict import RandomMaskingGenerator,softmax_output_litsdice,softmax_output_litshd,softmax_assd_litsscore,softmax_mIOU_litsscore,Uentropy_our,cal_ueo,cal_ece_our,softmax_mIOU_score,softmax_output_dice,softmax_output_hd,dice_isic,iou_isic,HD_isic,Dice_isic,IOU_isic,softmax_assd_score
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

torch.cuda.empty_cache()

def getArgs():
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    parser = argparse.ArgumentParser()
    # Basic Information
    parser.add_argument('--user', default='anissa', type=str)
    parser.add_argument('--experiment', default='UMIS', type=str) # BraTS ISIC COVID
    parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
    parser.add_argument('--description',
                        default='Trustworthy medical image segmentation by coco,'
                                'training on train.txt!',
                        type=str)
    # training detalis
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--end_epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--submission', default='./results', type=str)

    parser.add_argument('--lambda-epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--save_freq', default=5, type=int)
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate')# BraTS: 0.0002 # ISIC: 0.0002
    # DataSet Information
    parser.add_argument('--savepath', default='./results/plot/output', type=str)
    parser.add_argument('--save_dir', default='./results', type=str)
    parser.add_argument("--mode", default="train", type=str, help="train/test/train&test")
    parser.add_argument('--dataset', default='autopet', type=str, help="BraTS/ISIC/LiTS/DRIVE/HC") #
    parser.add_argument("--folder", default="folder0", type=str, help="folder0/folder1/folder2/folder3/folder4")
    parser.add_argument('--input_H', default=300, type=int)
    parser.add_argument('--input_W', default=400, type=int)
    parser.add_argument('--input_D', default=400, type=int)  # 155
    parser.add_argument('--crop_H', default=128, type=int)
    parser.add_argument('--crop_W', default=128, type=int)
    parser.add_argument('--crop_D', default=128, type=int)
    parser.add_argument('--output_D', default=155, type=int)
    parser.add_argument('--batch_size', default=2, type=int, help="2/4/8/16")
    parser.add_argument('--OOD_Condition', default='normal', type=str, help="normal/noise/mask/blur/")
    parser.add_argument('--OOD_Level', default=0, type=int, help="0: 'No',1:'Low', 2:'Upper Low', 3:'Mid', 4:'Upper Mid', 5:'High'")
    # parser.add_argument('--OOD_Variance', default=2, type=int)
    parser.add_argument('--snapshot', default=True, type=bool, help="True/False")  # visualization results
    parser.add_argument('--Uncertainty_Loss', default=False, type=bool, help="True/False")  # adding uncertainty_loss
    parser.add_argument('--input_modality', default='petct', type=str, help="t1/t2/both/four")  # Single/multi-modal
    parser.add_argument('--model_name', default='U', type=str, help="U/V/AU/TransU/ViT/")  # multi-modal
    parser.add_argument('--test_epoch', type=int, default=199, metavar='N',
                        help='best epoch')
    # for ViT
    # parser.add_argument('--n_skip', type=int,
    #                     default=3, help='using number of skip-connect, default is num')
    # parser.add_argument('--vit_name', type=str,
    #                     default='R50-ViT-B_16', help='select one vit model')
    # parser.add_argument('--vit_patches_size', type=int,
    #                     default=8, help='vit_patches_size, default is 16')
    # anissa added thiese: 
    parser.add_argument('--train_file', default='/gpfs3/well/papiez/users/hri611/python/UMIS/train_imgs.txt', type=str)
    parser.add_argument('--val_file', default='val_imgs.txt', type=str)
    parser.add_argument('--test_file', default='val_imgs.txt', type=str)
    
    args = parser.parse_args()
    # args.dims = [[240,240,160], [240,240,160]]
    # args.modes = len(args.dims)

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
        logging.info('Samples for train = {}'.format(len(train_loader.dataset)))

        valid_file=args.val_file
        valid_dir='train_data'
        valid_list = os.path.join(root_path, valid_file)
        valid_root = os.path.join(root_path, valid_dir)
        valid_set = Autopet(valid_list, valid_root,'val',args.input_modality,OOD_Condition=args.OOD_Condition, level=args.OOD_Level)
        valid_loader = DataLoader(valid_set, batch_size=1)
        logging.info('Samples for valid = {}'.format(len(valid_loader.dataset)))

        test_file=args.test_file
        test_dir='train_data'
        test_list = os.path.join(root_path, test_file)
        test_root = os.path.join(root_path, test_dir)
        test_set = Autopet(test_list, test_root,'test',args.input_modality,OOD_Condition=args.OOD_Condition, level=args.OOD_Level)
        test_loader = DataLoader(test_set, batch_size=1)
        logging.info('Samples for test = {}'.format(len(test_loader.dataset)))
    
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
        logging.info('Samples for train = {}'.format(len(train_loader.dataset)))

        valid_file=args.val_file
        valid_dir='train_data'
        valid_list = os.path.join(root_path, valid_file)
        valid_root = os.path.join(root_path, valid_dir)
        valid_set = Autopet2(valid_list, valid_root,'val',args.input_modality,OOD_Condition=args.OOD_Condition, level=args.OOD_Level)
        valid_loader = DataLoader(valid_set, batch_size=1)
        logging.info('Samples for valid = {}'.format(len(valid_loader.dataset)))

        test_file=args.test_file
        test_dir='train_data'
        test_list = os.path.join(root_path, test_file)
        test_root = os.path.join(root_path, test_dir)
        test_set = Autopet2(test_list, test_root,'test',args.input_modality,OOD_Condition=args.OOD_Condition, level=args.OOD_Level)
        test_loader = DataLoader(test_set, batch_size=1)
        logging.info('Samples for test = {}'.format(len(test_loader.dataset)))    
    else:
        train_loader=None
        valid_loader=None
        test_loader=None
        logging.info('There is no this dataset')
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

def train(args,model,optimizer,epoch,train_loader):
    
    model.train()
    loss_meter = AverageMeter()
    step = 0
    dt_size = len(train_loader.dataset)
    for i, data in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, args.end_epochs, args.lr)
        step += 1
        input, target = data
        x = input.cuda()  # for multi-modal combine train
        target = target.cuda()
        # refresh the optimizer

        args.mode = 'train'
        # n = torch.unique(target)
        # print('n target classes:')
        # print(n)
        evidences, loss = model(x, target, epoch, args.mode,args.dataset)

        print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_loader.batch_size + 1, loss.item()))
        # compute gradients and take step
        optimizer.zero_grad()
        loss.requires_grad_(True).backward()
        optimizer.step()
        loss_meter.update(loss.item())
    return loss_meter.avg

def val(args, model, current_epoch, best_dice, valid_loader):
    print('===========>Validation begining!===========')
    model.eval()
    loss_meter = AverageMeter()
    num_classes = args.num_classes
    dice_total, iou_total = 0, 0
    step = 0
    # model.eval()
    pos_patch_count = 0
    for i, data in enumerate(valid_loader):
        step += 1
        input, target = data

        x = input.cuda()  # for multi-modal combine train
        target = target.cuda()
        gt = target.cuda()

        n = torch.unique(target)
        # logging.info('unique elements in target image')
        # logging.info(n)

        if n.size(0) > 1: # anissa: check whether there is a tumour class in your image
            pos_patch_count +=1
            pos_patch = True
        else:
            pos_patch = False
            
        with torch.no_grad():
            args.mode = 'val'
            if args.dataset =='BraTS': # anissa - have added evidence loss
                evidence,loss = model(x, target[:, :, :, :155], current_epoch, args.mode, args.dataset)  # two input_modality 4
            else:
                evidence,loss = model(x, target, current_epoch, args.mode,  args.dataset)
            
            loss_meter.update(loss.item()) #anissa

            alpha = evidence + 1

            S = torch.sum(alpha, dim=1, keepdim=True)
            prob = alpha / S
            _, predicted = torch.max(prob, 1)

            output = torch.squeeze(predicted).cpu().detach().numpy()
            target = torch.squeeze(target).cpu().numpy()
                        
            if args.dataset == 'autopet' or args.dataset=='autopet2':
                if pos_patch:
                    iou_res = softmax_mIOU_score(output, target) #anissa not sure if this will work
                    dice_res = softmax_output_dice(output, target)
                    dice_total += dice_res[0]
                    iou_total += iou_res[0] #changed to 0 but dont think it makes a diff as either way 1 and 1 is included
                    logging.info('current_iou:{} ; current_dice:{}'.format(iou_res[0], dice_res[0]))
                else:
                    logging.info('negative image, no dice/iou')
            elif args.dataset == 'BraTS':
                iou_res = softmax_mIOU_score(output, target[:, :, :155])
                dice_res = softmax_output_dice(output, target[:, :, :155])
                dice_total += dice_res[1]
                iou_total += iou_res[1]
            else:
                soft_gt = get_soft_label(gt, num_classes)
                soft_predicted = get_soft_label(predicted.unsqueeze(0), num_classes)
                iou_res = IOU_isic(soft_predicted, soft_gt,num_classes)
                dice_res = Dice_isic(soft_predicted, soft_gt,num_classes)
                dice_total += dice_res
                iou_total += iou_res

    aver_dice = dice_total / pos_patch_count
    aver_iou = iou_total / pos_patch_count

    # aver_dice = dice_total / len(valid_loader)
    # aver_iou = iou_total / len(valid_loader)
    if aver_dice > best_dice \
            or (current_epoch + 1) % int(args.end_epochs - 1) == 0 \
            or (current_epoch + 1) % int(args.end_epochs - 2) == 0 \
            or (current_epoch + 1) % int(args.end_epochs - 3) == 0:
        logging.info('aver_dice:{} > best_dice:{}'.format(aver_dice, best_dice))
        best_dice = aver_dice
        logging.info('===========>save best model!')
        # added args.experiment (anissa)
        if args.Uncertainty_Loss:
            file_name = os.path.join(args.save_dir, args.experiment + '_'+ args.model_name +'_'+args.dataset +'_'+ args.folder + '_DUloss'+'_epoch_{}.pth'.format(current_epoch))
        else:
            file_name = os.path.join(args.save_dir,args.experiment + '_'+ args.model_name + '_' + args.dataset +'_'+ args.folder + '_epoch_{}.pth'.format(current_epoch))
        torch.save({
            'epoch': current_epoch,
            'state_dict': model.state_dict(),
        },
            file_name)
    return loss_meter.avg, best_dice

def test(args,model,test_loader):
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
        load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 args.save_dir,
                                 args.experiment + '_' + args.model_name + '_' + args.dataset +'_'+ args.folder + '_DUloss' + '_epoch_{}.pth'.format(args.test_epoch))
        load_file = '/gpfs3/well/papiez/users/hri611/python/UMIS/results/U_autopet_folder0_epoch_199.pth'

    else:
        load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 args.save_dir,
                                 args.experiment + '_' + args.model_name + '_' + args.dataset +'_'+ args.folder + '_epoch_{}.pth'.format(args.test_epoch))
        load_file = '/gpfs3/well/papiez/users/hri611/python/UMIS/results/U_autopet_folder0_epoch_199.pth'


    if os.path.exists(load_file):
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(
            os.path.join(args.save_dir + '/' + args.model_name +'_'+args.dataset+ '_epoch_' + str(args.test_epoch))))
    else:
        print('There is no resume file to load!')
    names = test_loader.dataset.image_list

    model.eval()
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

                if snapshot:
                    """ --- colorful figure--- don't fully understand what this does """
                    Snapshot_img = np.zeros(shape=(H, W, 2, T), dtype=np.float32)
                    # K = [np.where(output[0,:,:,:] == 1)]
                    #Snapshot_img[:, :, 0, :][np.where(pc == 0)] = 255
                    Snapshot_img[:, :, 1, :][np.where(pc == 1)] = 255

                    Snapshot_img = Snapshot_img.astype(np.uint8)
                    print("Shape of Snapshot_img:", Snapshot_img.shape)
                    print("Minimum value:", np.min(Snapshot_img))
                    print("Maximum value:", np.max(Snapshot_img))

                    target_img = np.zeros(shape=(H, W, 2, T), dtype=np.float32)
                    # K = [np.where(output[0,:,:,:] == 1)]
                    #target_img[:, :, 0, :][np.where(Otarget == 0)] = 255
                    target_img[:, :, 1, :][np.where(Otarget == 1)] = 255
                    target_img = target_img.astype(np.uint8)

                    print("Shape of target img:", target_img.shape)
                    print("Minimum value:", np.min(target_img))
                    print("Maximum value:", np.max(target_img))


                    for frame in range(T):
                        if frame%25 == 0: # don't want to do too much in case wrong
                            if not os.path.exists(os.path.join(savepath, name)):
                                os.makedirs(os.path.join(savepath,  name))
    
                            # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
                            imageio.imwrite(os.path.join(savepath, name, str(frame) + '.png'),
                                            Snapshot_img[:, :, :, frame])
                            imageio.imwrite(os.path.join(savepath, name, str(frame) + '_gt.png'),
                                            target_img[:, :, :, frame])
                            imageio.imwrite(os.path.join(savepath, name, str(frame) + '_input_ct.png'),
                                            Oinput[0,:, :, frame].astype(np.uint8))
                            imageio.imwrite(os.path.join(savepath, name, str(frame) + '_input_pet.png'),
                                            Oinput[1,:, :, frame].astype(np.uint8))
                            imageio.imwrite(
                                os.path.join(savepath, name, str(frame) + '_uncertainty.png'),
                                U_output[:, :, frame].astype(np.uint8))
                            imageio.imwrite(
                                os.path.join(savepath, name, str(frame) + '_conf.png'),
                                conf_output[:, :, frame].astype(np.uint8))
                            
                            imageio.imwrite(
                                os.path.join(savepath, name, str(frame) + '_confe.png'),
                                confe_output[:, :, frame].astype(np.uint8))
    
                            U_img = cv2.imread(
                                os.path.join(savepath, name, str(frame) + '_uncertainty.png'))
                            U_heatmap = cv2.applyColorMap(U_img, cv2.COLORMAP_JET)
                            
                            cv2.imwrite(
                                os.path.join(savepath, name,
                                             str(frame) + '_colormap_uncertainty.png'),
                                U_heatmap)
    
                            conf_img = cv2.imread(
                                os.path.join(savepath, name, str(frame) + '_conf.png'))
                            conf_heatmap = cv2.applyColorMap(conf_img, cv2.COLORMAP_JET)
                            cv2.imwrite(
                                os.path.join(savepath, name,
                                             str(frame) + '_colormap_conf.png'),
                                conf_heatmap)
                        
                            confe_img = cv2.imread(
                                os.path.join(savepath, name, str(frame) + '_confe.png'))
                            confe_heatmap = cv2.applyColorMap(confe_img, cv2.COLORMAP_JET)
                            cv2.imwrite(
                                os.path.join(savepath, name,
                                             str(frame) + '_colormap_confe.png'),
                                confe_heatmap)
                        
            print('current_dice:{}'.format(dice_res[0]))
            print('current_iou:{}'.format(iou_res[0]))
            print('current_hd:{}'.format(hd_res[0]))

    num = len(test_loader)
    if args.dataset == 'autopet' or args.dataset=='autopet2':
        aver_certainty = certainty_total / num
        aver_mne = mne_total / num
        aver_ece = ece_total / num
        aver_ueo = ueo_total / num
        aver_dice_WT = dice_total_WT / num
        aver_iou_WT = iou_total_WT / num
        aver_hd_WT = hd_total_WT / num
        aver_assd_WT = assd_total_WT / num
        print('aver_dice_WT=%f' % (
        aver_dice_WT * 100))
        print('aver_iou_WT=%f' % (
        aver_iou_WT * 100))
        
        #print('aver_hd_WT=%f,aver_hd_TC = %f,aver_hd_ET = %f' % (aver_hd_WT, aver_hd_TC, aver_hd_ET))
        #print('aver_assd_WT=%f,aver_assd_TC = %f,aver_assd_ET = %f' % (aver_assd_WT, aver_assd_TC, aver_assd_ET))
        print('aver_certainty=%f' % (aver_certainty))
        print('aver_mne=%f' % (aver_mne,))
        print('aver_ece=%f' % (aver_ece))
        print('aver_ueo=%f' % (aver_ueo))

        # logging.info('aver_dice_WT=%f' % (
        # aver_dice_WT * 100))
        # logging.info('aver_iou_WT=%f' % (
        # aver_iou_WT * 100))

        # logging.info('aver_certainty=%f' % (aver_certainty))
        # logging.info('aver_mne=%f' % (aver_mne,))
        # logging.info('aver_ece=%f' % (aver_ece))
        # logging.info('aver_ueo=%f' % (aver_ueo))
        
        return aver_dice_WT,aver_hd_WT,aver_assd_WT,aver_certainty
   
if __name__ == "__main__":
    args = getArgs()
    
    logging.info('--------------------------------------This is all argsurations----------------------------------')
    for arg in vars(args):
        logging.info('{}={}'.format(arg, getattr(args, arg)))
    logging.info('----------------------------------------This is a halving line----------------------------------')
    logging.info('{}'.format(args.description))
    
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

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4) # anissa - changed this
    model.cuda()

    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment + args.date)
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
    writer = SummaryWriter()


    if args.mode =='train&test':
        for epoch in range(args.start_epoch, args.end_epochs + 1):
            logging.info('===========Train begining!===========')
            logging.info('Epoch {}/{}'.format(epoch, args.end_epochs - 1))
            epoch_loss = train(args,model,optimizer,epoch,train_loader)
            writer.add_scalar('loss',epoch_loss,epoch)
            logging.info("epoch %d avg_loss:%0.3f" % (epoch, epoch_loss))
            val_loss, best_dice = val(args, model, epoch, best_dice, valid_loader)
            writer.add_scalar('val_loss',val_loss,epoch)
            loss_list.append(epoch_loss)
            dice_list.append(best_dice)
        loss_list.cpu() # anissa: added these
        dice_list.cpu()
        loss_plot(args, loss_list)
        metrics_plot(args, 'dice', dice_list)
        test_dice,test_noise_dice,test_hd,test_noise_hd,test_assd,test_noise_assd = test(args,model,test_loader)

    elif args.mode =='train':
        for epoch in range(1, args.end_epochs + 1):
            logging.info('===========Train begining!===========')
            logging.info('Epoch {}/{}'.format(epoch, args.end_epochs - 1))
            epoch_loss = train(args,model,optimizer,epoch,train_loader)
            logging.info("epoch %d avg_loss:%0.3f" % (epoch, epoch_loss))
            writer.add_scalar('train_loss',epoch_loss,epoch)
            val_loss, best_dice = val(args, model, epoch, best_dice, valid_loader)
            writer.add_scalar('val_loss',val_loss,epoch)
            loss_list.append(epoch_loss)
            dice_list.append(best_dice)
        loss_list.cpu() # anissa: added these
        dice_list.cpu()
        dice_list.to_csv(args.experiment+'_dice_list.csv')
        loss_plot(args, loss_list)
        metrics_plot(args, 'dice', dice_list)
    elif args.mode =='test':
        train_loader, valid_loader, test_loader = getDataset(args)
        aver_dice_WT,aver_hd_WT,aver_assd_WT,aver_certainty = test(args, model, test_loader)
        
        # for j in range(0,2):
        #     args.OOD_Condition = OOD_Condition[j]
        #     print("arg.OOD_Condition: %s" % (OOD_Condition[j]))
        #     start = 1
        #     end = 4
        #     for i in range(start,end):
        #         print("arg.OOD_Level: %d" % (i))
        #         args.OOD_Level = i
        #         train_loader, valid_loader, test_loader = getDataset(args)
        #         test_dice, test_noise_dice, test_hd, test_noise_hd, test_assd, test_noise_assd = test(args, model, test_loader)
    else:
        print('There is no this mode')
        raise NameError
