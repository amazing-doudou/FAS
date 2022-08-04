import os
from random import randint
import torch
import torchvision
from trainer.base import BaseTrainer
from utils.meters import AverageMeter
from utils.eval import predict, calc_acc, calc_tprfpr, cal_metric, add_images_tb
from models.LGSC import patchify, unpatchify, get_random_index, get_fusion_imgs_labels
from torchvision import transforms
from collections import OrderedDict
import torch.onnx as onnx
from torch.autograd import Variable

class Trainer(BaseTrainer):
    def __init__(self, cfg, network, optimizer, loss, lr_scheduler, device, trainloader, testloader, writer):
        super(Trainer, self).__init__(cfg, network, optimizer, loss, lr_scheduler, device, trainloader, testloader, writer)
        # self.network = self.network.to(device)
        self.train_loss_metric = AverageMeter(writer=writer, name='Loss/train', length=len(self.trainloader))
        self.train_acc_metric = AverageMeter(writer=writer, name='Accuracy/train', length=len(self.trainloader))
        self.train_fpr_metric = AverageMeter(writer=writer, name='Fpr/train', length=len(self.trainloader))
        self.train_tpr_metric = AverageMeter(writer=writer, name='Tpr/train', length=len(self.trainloader))

        self.val_loss_metric = AverageMeter(writer=writer, name='Loss/val', length=len(self.testloader))
        self.val_acc_metric = AverageMeter(writer=writer, name='Accuracy/val', length=len(self.testloader))
        self.val_fpr_metric = AverageMeter(writer=writer, name='Fpr/val', length=len(self.testloader))
        self.val_tpr_metric = AverageMeter(writer=writer, name='Tpr/val', length=len(self.testloader))

        self.best_val_acc = 0.0
    def load_model(self,epoch_n):
        if False:
            saved_name = '/data/model/lcnet.224.16.14.v0404_model/LGSC_lcnet_small_rose_1.pth'
            print('Loading resume network %s...' % (saved_name))
            pretrain_dict = torch.load(saved_name)['state_dict']
            model_dict = self.network.state_dict()
            pretrain_dict = {k: v for k, v in pretrain_dict.items()
                             if k in model_dict
                             and model_dict[k].shape == pretrain_dict[k].shape}
            assert len(pretrain_dict) > 0
            print('Loading load pretrain params length: '
                             '%d, model_dict length: %d ...'
                             % (len(pretrain_dict), len(model_dict)))
                
            model_dict.update(pretrain_dict)
            self.network.load_state_dict(model_dict)
            print('load_model: ', saved_name)
            return
        
        saved_name = '/home/output/LGSC_lcnet_small_rose_0.pth'
        state = torch.load(saved_name)
#         state_dict_opt = state['optimizer']
#         new_state_dict_opt = []
#         for k,v in state_dict_opt:
#             k = k[7:]
#             new_state_dict_opt[k] = v
        self.optimizer.load_state_dict(state['optimizer'])
        self.network.load_state_dict(state['state_dict'])
        print('load_model: ', saved_name)
        
        
    def save_model(self, epoch):
        if not os.path.exists(self.cfg['output_dir']):
            os.makedirs(self.cfg['output_dir'])
        saved_name = os.path.join(self.cfg['output_dir'], '{}_{}_{}_{}'.format(self.cfg['model']['base'], self.cfg['model']['backbone'], self.cfg['dataset']['name'],epoch))
        state = {
            'epoch': epoch,
            'state_dict': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, os.path.join(saved_name + ".pth"))
            
        if False:
            dummy_input = torch.Tensor(1, 3, int(self.cfg['model']['image_size'][0]), int(self.cfg['model']['image_size'][1])).type(torch.FloatTensor).cuda()
            onnx.export(self.network.module, dummy_input, os.path.join(saved_name + ".onnx"), verbose=True)
        
        
    def train_one_epoch(self, epoch):
        self.network.train()
        self.train_loss_metric.reset(epoch)
        self.train_acc_metric.reset(epoch)
        self.train_tpr_metric.reset(epoch)
        self.train_fpr_metric.reset(epoch)
        # torch.backends.cudnn.enabled = False
        TN = 0 
        FP = 0 
        FN = 0 
        TP = 0        
        for i, (img, label, img_name) in enumerate(self.trainloader): # [N, 3, 224, 224] [N, 1] [N, 1, map_size, map_size]
            label=label.cuda()
            img=img.cuda()
            ori_img = img
#             img, mask, label = img.to(self.device), mask.to(self.device), label.to(self.device)
            if self.cfg['model']['header_type'] == 'patch_pixel':
                img, mask_14, mask = get_fusion_imgs_labels(self.cfg, imgs=img, ori_labels=label, patch_ratio=0.5, index_ratio=0.5)
#             toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
#             resize = transforms.Resize((224, 224))
#             fusionImg_path = '/home/semtp/notebooks/04016_old14_fusion_patch16_map224/'
#             if not os.path.exists(fusionImg_path):
#                 os.makedirs(fusionImg_path)
#             for j in range(img.shape[0]):
#                 pic = toPIL(img[j].cpu())
#                 ori_pic = toPIL(ori_img[j].cpu())
#                 mask_map = toPIL(mask[j].cpu())
#                 pic.save(fusionImg_path + 'img_{}_{}.jpg'.format(i, j))  
#                 ori_pic.save(fusionImg_path + 'oriImg_{}_{}.jpg'.format(i, j)) 
#                 mask_map.save(fusionImg_path + 'mask_{}_{}.png'.format(i, j)) 
#                 with open(fusionImg_path + 'mask_{}_{}.txt'.format(i, j),"w") as f:
#                     torch.set_printoptions(profile="full")
#                     f.write(str(mask[j]))
#                     torch.set_printoptions(profile="default") # reset
#                 with open(fusionImg_path + 'label_{}_{}.txt'.format(i, j),"w") as f:
#                     torch.set_printoptions(profile="full")
#                     f.write(str(label[j]))
#                     torch.set_printoptions(profile="default") # reset
            output, output_branch = self.network(img)
            if self.cfg['model']['encoder_arch'] == 'one_stream_arch':
                losses = self.loss(output, mask, label)
                losses1 = self.loss(output_branch, mask_14, label) # N 1 14 14 未适配
                if i % 100 == 0:                
                    print('losses[0]:', losses[0])
                    print('losses1[0]:', losses1[0])
                loss = losses[0] + 0.5*losses1[0]
            elif self.cfg['model']['encoder_arch'] == 'two_stream_arch':
                losses = self.loss(output, mask, label)
                losses1 = self.loss(output_branch[0], mask_14, label)
                losses2 = self.loss(output_branch[1], mask_14, label)
                loss = losses[0] + losses1[0] + losses2[0]
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            # Calculate predictions
            output, target = predict(output, mask, label, score_type=self.cfg['model']['header_type'])
            acc = calc_acc(output, target, score_type=self.cfg['model']['header_type'])
            batch_tpr, batch_fpr, tn, fp, fn, tp = calc_tprfpr(output, target)
            TN += tn
            FP += fp
            FN += fn
            TP += tp            
            # Update metrics
            self.train_tpr_metric.update(val=tp, n=(tp+fn))
            self.train_fpr_metric.update(val=fp, n=(fp+tn))            
            self.train_loss_metric.update(loss.item())
            self.train_acc_metric.update(acc)
            if i % 100 == 0:
                print('\rEpoch: {}, iter: {}, loss: {:.3f}, acc: {:.3f}, batch_tpr:{:.3f} tpr:{:.3f}, batch_fpr:{:.3f} fpr:{:.3f}, lr:{:.6f}'.format(
                    epoch, epoch * len(self.trainloader) + i, self.train_loss_metric.avg, self.train_acc_metric.avg, batch_tpr, self.train_tpr_metric.avg, batch_fpr, self.train_fpr_metric.avg, self.lr_scheduler.get_last_lr()[0]),end='')
            if i % 100 == 0:
                self.save_model(epoch)
                
        apcer = FP/(TN + FP)
        npcer = FN/(FN + TP)            
        acer = (apcer + npcer)/2 
        print('\rvalidate Epoch:{}, acer:{:.5f}'.format(epoch, acer),end='')                  
                
    def train(self):
#         self.load_model(0)
        for epoch in range(self.cfg['train']['num_epochs']):
            self.train_one_epoch(epoch)
            epoch_acc = self.validate(epoch)
            if epoch_acc > self.best_val_acc:
                self.best_val_acc = epoch_acc
            self.save_model(epoch)
            
    def validate(self, epoch):
        self.network.eval()
        self.val_loss_metric.reset(epoch)
        self.val_acc_metric.reset(epoch)
        self.val_fpr_metric.reset(epoch)
        self.val_tpr_metric.reset(epoch)
#         seed = randint(0, len(self.testloader)-1)
        result_list = []
        label_list = []
        TN = 0 
        FP = 0 
        FN = 0 
        TP = 0
        for i, (img, mask, label, img_name) in enumerate(self.testloader):
            img, mask, label = img.to(self.device), mask.to(self.device), label.to(self.device)
            # if self.cfg['model']['header_type'] == 'patch_pixel':
            #     img, mask = get_fusion_imgs_labels(self.cfg, imgs=img, ori_labels=label, patch_ratio=0.5, index_ratio=1.0) 
            output, _ = self.network(img)
            losses = self.loss(output, mask, label)
            loss = losses[0]
            # Calculate predictions
            output, target = predict(output, mask, label, score_type='binary_classification')
            acc = calc_acc(output, target, score_type='binary_classification')
            batch_tpr, batch_fpr, tn, fp, fn, tp = calc_tprfpr(output, target)
            TN += tn
            FP += fp
            FN += fn
            TP += tp
#             with open('wrongLog.txt','a+') as wrongLog:
#                 for i in range(len(img_name)):
#                     if preds[i]!=targets[i]:
#                         wrongLog.write(img_name[i])
#                         wrongLog.write('\n')
            # Update metrics
            self.val_loss_metric.update(loss.item())
            self.val_acc_metric.update(acc)
            self.val_tpr_metric.update(val=tp, n=(tp+fn))
            self.val_fpr_metric.update(val=fp, n=(fp+tn))

            preds = output.to('cpu').detach().numpy()
            label = target.to('cpu').detach().numpy()
            for i_batch in range(preds.shape[0]):
                result_list.append(preds[i_batch])
                label_list.append(label[i_batch])
                            
            if i%10==0:
                print('\rvalidate Epoch:{}, loss:{:.5f}, acc:{:.5f}, batch_tpr:{:.5f}, tpr:{:.5f}, batch_fpr:{:.5f} fpr:{:.5f}'.format(epoch, self.val_loss_metric.avg, self.val_acc_metric.avg, batch_tpr, self.val_tpr_metric.avg, batch_fpr, self.val_fpr_metric.avg),end='')
#             if i == seed:
#                 add_images_tb(self.cfg, epoch, img, preds, targets, score, self.writer)

        eer, tprs, auc, xy_dic = cal_metric(result_list, label_list)
        tprfpr2 = tprs['TPR@FPR=10E-2']
        tprfpr3 = tprs['TPR@FPR=10E-3']
        tprfpr4 = tprs['TPR@FPR=10E-4']
        apcer = FP/(TN + FP)
        npcer = FN/(FN + TP)            
        acer = (apcer + npcer)/2 
        print('\rvalidate Epoch:{}, acer:{:.5f}, tpr@fpr10e-2:{:.5f}, tpr@fpr10e-3:{:.5f}, tpr@fpr10e-4:{:.5f}'.format(epoch, acer, tprfpr2, tprfpr3, tprfpr4),end='')       
        return self.val_acc_metric.avg