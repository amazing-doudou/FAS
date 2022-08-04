import os
import cv2
import torch
from torchvision import transforms, datasets
from models.loss import PixWiseBCELoss
from models.densenet_161 import DeepPixBis
from models.liveness_net import LivenessNet
from datasets.PixWiseDataset import PixWiseDataset
from utils.utils import read_cfg, get_optimizer, build_network, get_device
import numpy as np
from PIL import Image

cfg = read_cfg(cfg_file='config/densenet_161_adam_lr1e-3.yaml')

network = build_network(cfg)

checkpoint = torch.load(
    os.path.join(cfg['output_dir'], '{}_{}_{}.pth'.format(cfg['model']['base'], cfg['dataset']['name'], 1)),
    map_location=torch.device('cpu'))

network.load_state_dict(checkpoint['state_dict'])

network.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

files=open('/data/e6a31fbc6b594c0795e6add0230d4bab/labelnew.txt','r')
imgs=files.readlines()
total=len(imgs)
true=0
false=0
right=0
wrong=0
for i in imgs:
#     img=os.path.join('/data/e6a31fbc6b594c0795e6add0230d4bab/',i.split()[0])
    img=i.split()[0]
    label=str(i.split()[1])
    print(label)
    img_det = cv2.imread(img)

    anti_img = transform(img_det)

            # print(anti_img.shape)
    anti_img = anti_img.unsqueeze(0)

            # print(anti_img.shape)

    dec, binary = network.forward(anti_img)
    res = torch.mean(dec).item()
    if res < 0.5 and label=='1':
        true+=1
        right+=1
        print('yes')
    elif res>=0.5 and label=='0':
        false+=1
        right+=1
        print('no')
    else:
        wrong+=1
        print('error')
print('全部：{}，正确：{},错误:{},真人：{},假人：{}'.format(total,right,wrong,true,false))