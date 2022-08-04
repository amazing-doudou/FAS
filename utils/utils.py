import toml
import yaml
import torch
from torch import optim
from transformers import AdamW, get_linear_schedule_with_warmup
from models.densenet_161 import DeepPixBis
from models.mobilenet_v2 import DeepPixBis_mbv2
from models.mobilenet_v2_small import Mbv2_small
from models.LGSC import LGSC_small
import argparse, os, json
import torchvision as tv
from os import path
#import torchvision.transforms as transforms
import collections

# python 3.8+ compatibility
try:
    collectionsAbc = collections.abc
except:
    collectionsAbc = collections
    
    
def get_opts(cfg):
    opt = argparse.Namespace()
    
    opt.task_name = ''
    #opt.exp_name = 'test0610_mobilenetv2_small_dropout_0d1_1024_Adam_128_bn_smooth_30e-4_0604_train_new_data_add_3dmodel/'
   # opt.exp_name = 'test0902_fintune_real_0901_mobilenetv2_small_dropout_0d2_128_Adam_128_bn_smooth_cce_5e-5_20200902_train_list_add_lyb_live'
    opt.exp_name = 'test_1209_01'
    opt.fold = 1
    opt.data_root = '' 
    #opt.data_list = '2get_list/train_result/1230_train_test_for_learning.txt' 
    #opt.val_list = '2get_list/train_result/20200724_0406_end_to_end_test_data.txt' 
    #opt.data_list = '2get_list/train_result/20200902_train_list_add_lyb_live.txt' 
    opt.data_list = '2get_list/train_result/0818_gloabal_train_new_data_add_low_phone.txt'
    
    #train1204  0107_train.txt 0102_test_data_djy_less.txt 0323_train_data 0408_train_new_data_add_iphone 1230_train_test_for_learning  0415_train_new_data
    opt.val_list = '2get_list/test_result/20200604_0406_end_to_end_test_data.txt' #test1209_shorter  0101_test_data_djy.txt  0120_test_data_djy.txt  0120_test_data.txt 0406_end_to_end_test_data 1230_test_data_for_learning.txt
 
    opt.out_root = '/mnt/sdb4/2d_project/sigle_result/'
    opt.out_path = os.path.join(opt.out_root,opt.exp_name)
    opt.out_path_test = '/mnt/sda1/nb_project/sigle_result/test12_31_test/'
    
    ### Dataloader options ###
    opt.nthreads = 16 #32 16
    opt.batch_size = 1024 # 256 1024
    opt.ngpu = 1 
    opt.img_size = 128  #change to 224 128 160 144

    ### Learning ###
    opt.freeze_epoch = 0
    opt.optimizer_name = 'Adam' #Adam  SGD
    opt.weight_decay = 0 #1e-4
    opt.lr = 4e-4 #2e-4  1e-4 2e-3 2e-3 1e-4 6e-4 
    opt.lr_decay_lvl = 0.5  #0.5 0.1
    opt.lr_decay_period = 4 # 4 2
    opt.lr_type = 'step_lr'#'cosine_repeat_lr'# step_lr ReduceLROnPlateau
    opt.lr_reduce_mode = 'None' #max for acc min for the min loss
    opt.num_epochs=300  #50
    opt.resume = None # 'model_40.pth'
    opt.debug = 0 
    opt.start_record_epoch = 1
    opt.num_itr= 250  #500 250 1000
      
    ### Other ###   
    opt.manual_seed = 704
    opt.log_batch_interval= 125  ##display batch's process 300 125
    opt.log_checkpoint = 1
    opt.net_type = 'mobilenetv2_small' # ResNet34CaffeSingle_fc ResNet34CaffeMulti_fc  ResNet18Siam  Resnet18_single_online se_resnet_18 densenet_121 Resnet18_single_online_bn Resnet18_single_online_bn_sig mobilenetv2 ghost_net
    opt.pretrained = True
    opt.pretrained_path = "/mnt/sdb4/2d_project/fintune_model/0805/model_8_67000"
    opt.isSaveTrainHard = False
    opt.isSaveTrainHardThresh = 0.3
    opt.SaveTrainHardPath = "/mnt/sdb5/20200723_train_hard_img"
    opt.classifier_type = 'linear'
    opt.smooth_rate = 0.02
    opt.loss_type= 'smooth_cce'  #smooth_cce cce  smooth_focal_cce circle focal_loss am_softmax
    opt.alpha_scheduler_type = None
    opt.nclasses = 2
    opt.fake_class_weight = 1
    opt.visdom_port = 8097
    #opt.isFintune = 'true'
    
    
    opt.train_transform = tv.transforms.Compose([
                tv.transforms.Resize(cfg['model']['image_size']),
                #transforms.RandomCrop(opt.img_size),   
#                 tv.transforms.RandomApply([
                   #transforms.ColorJitter(0.25,0.25,0.25,0.125)],p=0.2),
                	#tv.transforms.RandomRotation(45)],p=0.3),
#                  tv.transforms.RandomRotation(15)],p=0.2),
                #tv.transforms.RandomRotation(45,resample=2,p = 0.3),
                tv.transforms.RandomApply([
                   #transforms.ColorJitter(0.25,0.25,0.25,0.125)],p=0.2),
                tv.transforms.RandomAffine(degrees=15, translate=(0, 0.2), scale=(0.9, 1), shear=(6, 9), fillcolor=0)],p=0.1),                
                tv.transforms.RandomHorizontalFlip(p=0.2),
#                 tv.transforms.RandomApply([
#                     transforms.CustomCutout(1, 25, 75)],p=0.1),
#                 tv.transforms.CustomGaussianBlur(max_kernel_radius=3, p=0.2),
                #transforms.CustomRandomResizedCrop(128, scale=(0.5, 1.0)),
                #tv.transforms.RandomApply([
                #   tv.transforms.Resize((int(opt.img_size*1.2),int(opt.img_size*1.2))),
                 #  tv.transforms.CenterCrop(opt.img_size)],p=0.2),#center_crop must be right 128
                #tv.transforms.RandomApply([
                #    tv.transforms.Resize((int(opt.img_size*1.2),int(opt.img_size*1.2))),
                #   tv.transforms.RandomCrop(opt.img_size,4,True,0,'constant')],p=0.2),#center_crop must be right 128
                tv.transforms.RandomApply([
                    tv.transforms.ColorJitter(0.25,0.25,0.25,0.5)],p=0.3),
                #tv.transforms.ColorJitter(0.25,0.25,0.25,0.5)],p=0.4),
                tv.transforms.RandomGrayscale(p=0.02),
                
#                 tv.transforms.RandomApply([
#                     tv.transforms.CustomGaussianNoise(var = 0.0025, p = 0.3),
#                     tv.transforms.CustomPoissonNoise(p = 0.3)],p = 0.05),
                    
#                 tv.transforms.Motion_blur(degree = 15, angle = 15, p = 0.05),
                
                #tv.transforms.RandomApply([
                #tv.transforms.ColorJitter(0.5,0.0,0.0,0.3),
                #transforms.CustomGaussianNoise(var = 0.0025, p = 0.5),
                #transforms.CustomPoissonNoise(p = 0.5)],p=0.10),
                
                #center_crop must be right 128
                #transforms.LabelSmoothing(eps=0.1, p=0.3),
                #transforms.CenterCrop(112),
                
                #transforms.GaussianBlur(max_kernel_radius=3, p=0.2),
                #transforms.CustomRandomResizedCrop(128, scale=(0.5, 1.0)),
                #transforms.RandomApply([
                #transforms.CenterCrop(200)],p=0.2),
                #transforms.RandomHorizontalFlip(), 
                #transforms.CenterCrop(112),
                #transforms.RandomGrayscale(p=0.3),
              #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                tv.transforms.ToTensor(),
                #transforms.CustomNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                tv.transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
        ])
    opt.test_transform = tv.transforms.Compose([
          #  transforms.CustomResize((125,125)),
            # transforms.CustomRotate(0),
            # transforms.CustomRandomHorizontalFlip(p=0),
            # transforms.CustomCrop((112,112), crop_index=0),
            tv.transforms.ToTensor(),
            #transforms.CustomNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    
    return opt


def update(d, u):
    for k, v in u.items():
        dv = d.get(k, {})
        if not isinstance(dv, collectionsAbc.Mapping):
            d[k] = v
        elif isinstance(v, collectionsAbc.Mapping):
            d[k] = update(dv, v)
        else:
            d[k] = v
    return d

def load_config(config, new_config):
    print("reading config from <{}>\n".format(path.abspath(config)))
    try:
        with open(config) as fp:
            cfg = yaml.load(fp, Loader=yaml.CLoader)  
            
        if new_config is not None:
            print('toml: ', new_config)
            new_cfg = toml.loads('\n'.join(new_config))
            update(cfg, new_cfg)
        return cfg

#         with open(config_path, "r") as f:
#             config = toml.load(f)
#             return config
    except FileNotFoundError as e:
        print("can not find config file")
        raise e


def read_cfg(cfg_file):
    """
    Read configurations from yaml file
    Args:
        cfg_file (.yaml): path to cfg yaml
    Returns:
        (dict): configuration in dict
    """
    with open(cfg_file,'r') as rf:
        cfg = yaml.safe_load(rf)
    return cfg
    # temp=open(cfg_file,'r')
    # cfg=yaml.safe_load(temp)
    # temp.close()
    return cfg

def get_optimizer(cfg, network, len_DataLoader):
    """ Get optimizer based on the configuration
    Args:
        cfg (dict): a dict of configuration
        network: network to optimize
    Returns:
        optimizer 
    """
    optimizer = None
    if cfg['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=cfg['train']['lr'])
    if cfg['train']['optimizer'] == 'AdamW':
        optimizer = AdamW(network.parameters(), lr=cfg['train']['lr'], eps=1e-6)
        warm_up_ratio = cfg['train']['warm_up_ratio'] # 定义要预热的step
        epoch = cfg['train']['num_epochs']
        len_DataLoader = len_DataLoader # 可以根据pytorch中的len(DataLoader)计算
        total_steps = len_DataLoader * epoch # 每一个epoch中有多少个step可以根据len(DataLoader)计算：total_steps = len(DataLoader) * epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)
    else:
        raise NotImplementedError
    return optimizer, scheduler
def get_device(cfg):
    """ Get device based on configuration
    Args: 
        cfg (dict): a dict of configuration
    Returns:
        torch.device
    """
    device = None
    if cfg['device'] == 'cpu':
        device = torch.device("cpu")
    elif cfg['device'] == 'gpu':
        device = torch.device("cuda")
    else:
        raise NotImplementedError
    return device
def build_network(cfg):
    """ Build the network based on the cfg
    Args:
        cfg (dict): a dict of configuration
    Returns:
        network (nn.Module) 
    """
    network = None
    if cfg['model']['base'] == 'LGSC':
        network = LGSC_small(cfg['model'])
#     if cfg['model']['backbone']=='MobileNetV2_small':
#         network=Mbv2_small()
    else:
        raise NotImplementedError
    return network