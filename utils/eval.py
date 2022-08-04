import torch
from torchvision import transforms
import numpy as np
from PIL import ImageDraw
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def predict(output, mask, label, threshold=0.5, score_type='combined'):
#     mask torch.Size([512, 1, 16, 16])
#     label torch.Size([512, 1])
    with torch.no_grad():
        if score_type == 'pixel':
            preds = torch.mean(output, axis=(1,2,3)).squeeze() # [N]
            targets = label.squeeze() # [N]
        elif score_type == 'binary':
            score = torch.mean(label, axis=1)
        elif score_type == 'combined':
            score = torch.mean(mask, axis=(1,2)) + torch.mean(label, axis=1)
        elif score_type == 'patch_pixel':  # [N, 1, 16, 16]
            preds = output
            targets = mask
        elif score_type == 'binary_classification':  # [N]
            if len(output.shape) == 4:
                preds = torch.mean(output, axis=(1,2,3)).squeeze()
            elif len(output.shape) == 2:
                preds = torch.argmax(output, dim=-1) # [N]
            targets = label.squeeze() # [N]
        else:
            raise NotImplementedError

        preds = (preds > threshold).type(torch.FloatTensor)  #二值化
        targets = (targets > threshold).type(torch.FloatTensor)  #二值化
        return preds, targets
    

def calc_acc(pred, target, score_type):
    """
    把二值化以后的预测值和目标值进行equal比较，相同数值的位置==1，否则==0
    得到equal矩阵
    equal矩阵算平均值，得到accuracy
    """
    if score_type == 'patch_pixel':
        equal = torch.mean(pred.eq(target).type(torch.FloatTensor))  # [N, 1, 16, 16] [N, 1, 16, 16]      
    elif score_type == 'binary_classification':
        equal = torch.mean(pred.eq(target).type(torch.FloatTensor))  # [N] [N]

    return equal.item()

def calc_tprfpr(pred, target):
    """
    pred: (N, 1, map_size, map_size)
    target: (N, 1, map_size, map_size)
    pred: (N)
    target: (N)    
    """   
    with torch.no_grad():
        if len(pred.shape) == 4:
            N, C, map_size, map_size = pred.shape
            pred = pred.reshape(N*C*map_size*map_size)
            target = target.reshape(N*C*map_size*map_size)
            
        tn = 0
        fp = 0
        fn = 0
        tp = 0
        
        for i in range(len(pred)):
            if target[i] == 0:
                if pred[i] ==1:
                    fp += 1
                elif pred[i] ==0:
                    tn += 1
            elif target[i] == 1:
                if pred[i] ==1:
                    tp += 1
                elif pred[i] ==0:
                    fn += 1                    
                
        batch_tpr = tp / (tp + fn)
        batch_fpr = fp / (fp + tn) 
        return batch_tpr, batch_fpr, tn, fp, fn, tp
    
    
def cal_metric(predicted, target, show = False):
    fpr, tpr, thresholds = roc_curve(target, predicted)
    _tpr = (tpr)
    _fpr = (fpr)
    tpr = tpr.reshape((tpr.shape[0],1))
    fpr = fpr.reshape((fpr.shape[0],1))
    scale = np.arange(0, 1, 0.00000001)
    function = interpolate.interp1d(_fpr, _tpr)
    y = function(scale)
    znew = abs(scale + y -1)
    eer = scale[np.argmin(znew)]
    FPRs = {"TPR@FPR=10E-2": 0.01, "TPR@FPR=10E-3": 0.001, "TPR@FPR=10E-4": 0.0001}
    TPRs = {"TPR@FPR=10E-2": 0.01, "TPR@FPR=10E-3": 0.001, "TPR@FPR=10E-4": 0.0001}
    for i, (key, value) in enumerate(FPRs.items()):
        index = np.argwhere(scale == value)
        score = y[index] 
        TPRs[key] = float(np.squeeze(score))
    auc = roc_auc_score(target, predicted)
    if show:
        plt.plot(scale, y)
        plt.show()
    return eer, TPRs, auc,{'x':scale, 'y':y}
    
    
def add_images_tb(cfg, epoch, img_batch, preds, targets, score, writer):
    """ Do the inverse transformation
    x = z*sigma + mean
      = (z + mean/sigma) * sigma
      = (z - (-mean/sigma)) / (1/sigma),
    Ref: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/6
    """
    mean = [-cfg['dataset']['mean'][i] / cfg['dataset']['sigma'][i] for i in range(len(cfg['dataset']['mean']))]
    sigma = [1 / cfg['dataset']['sigma'][i] for i in range(len(cfg['dataset']['sigma']))]
    img_transform = transforms.Compose([
        transforms.Normalize(mean, sigma),
        transforms.ToPILImage()
    ])
    ts_transform = transforms.ToTensor()
    for idx in range(img_batch.shape[0]):
        vis_img = img_transform(img_batch[idx].cpu())
        ImageDraw.Draw(vis_img).text((0,0), 'pred: {} vs gt: {}'.format(int(preds[idx]), int(targets[idx])), (255,0,255))
        ImageDraw.Draw(vis_img).text((20,20), 'score {}'.format(score[idx]), (255,0,255))
        tb_img = ts_transform(vis_img)
        writer.add_image('Prediction visualization/{}'.format(idx), tb_img, epoch)