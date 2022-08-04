import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.onnx as onnx
import torch.nn.functional as F
from models.mobilenet_v2_small import MobileNetV2_Small
from models.LCNet import LCNet_small
from models.Dual_Branch_Network import CDCN_Encoder, LCNet_Encoder, Decoder
from models.Liveness import Liveness_small
import os
import math
from torchvision import transforms

net_set = {"MobileNetV2_small": MobileNetV2_Small, "lcnet_small": LCNet_small, "liveness_small": Liveness_small}

def patchify(cfg, imgs):
    """
    imgs: (N, 3, H, W)
    mask: (N, 1, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = cfg['model']['patch_size']
    c = imgs.shape[1]
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
    return x

def unpatchify(cfg, x, p=8):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    c = x.shape[2] // (p*p)
    x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    return imgs

def get_random_index(N, keep_ratio):
    """
    N: 16
    norandom_num: 4
    ori_index:    tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    index_shuffle:tensor([8, 11, 9, 4, 10, 2, 3, 0, 5, 6, 1, 7])
    index_restore:tensor([12, 13, 14, 15, 16])
    random_index: tensor([8, 11, 9, 4, 10, 2, 3, 0, 5, 6, 1, 7, 12, 13, 14, 15, 16])
    """
    # create random_index from ori_index of a batch indexs
    restore_num = int(N * keep_ratio)
    ori_index = torch.range(0, N-1, dtype=int)
    if N-1 < N-restore_num: # restore_num == 0
        index_restore = torch.tensor([])
    else:
        index_restore = torch.range(N - restore_num, N-1, dtype=int) # 长度为 restore_num
    index_shuffle = torch.randperm(N - restore_num) # 长度为 N - restore_num
    random_index = torch.cat((index_shuffle, index_restore)).long()

    assert len(ori_index) == len(random_index)
    return ori_index, random_index

def get_fusion_imgs_labels(cfg, imgs, ori_labels, patch_ratio, index_ratio):
    """
    L == 16*16个patch，P==patch_size==14
    imgs: [N, 3, H, W] ori_labels:[N]  pixel_label:[N,C,H,W]
    patchify_img:[N,L,P*P*3] patchify_label:[N,L,P*P]
    fusion_masked_img:[N,L,P*P*3] fusion_masked_label:[N,L,P*P*1]

    ori_index:    tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])  num:N
    random_index: tensor([8, 11, 9, 4, 10, 2, 3, 0, 5, 6, 1, 7, 12, 13, 14, 15, 16])  num:N
    mask_index :       tensor([1, 1, 0, 0, 1, 0, 1, 1, 0, 1])   num:L
    pairs_mask_index : tensor([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])   num:L
    """
    N, C, H, W = imgs.shape
    L = (H // cfg['model']['patch_size']) * (W // cfg['model']['patch_size'])
    patch_size = cfg['model']['patch_size']
    ori_index, random_index = get_random_index(N, index_ratio)
    pairs_imgs = imgs[torch.LongTensor(random_index), :, :]  # random imgs's place in a batch
    pair_ori_labels = ori_labels[random_index]  # random labels's place in a batch
    patchify_imgs = patchify(cfg, imgs)
    pairs_patchify_imgs = patchify(cfg, pairs_imgs)
    mask = torch.zeros(patchify_imgs.shape)
    mask2 = torch.zeros(pairs_patchify_imgs.shape)
    scale = H // cfg['model']['map_size']
    patchify_label = torch.zeros(N, L, (patch_size//scale) ** 2 * 1)
    patchify_label_14 = torch.zeros(N, L, (patch_size//16) ** 2 * 1)

    mask_label = torch.zeros(patchify_label.shape)
    mask_label_14 = torch.zeros(patchify_label_14.shape)

    # 对batch中的每张img做fusion
    for i in range(N):
        random_index = torch.randperm(L) # 得到随机的0 - L-1 值的L长度数组 ： 如tensor([1, 2, 4, 0])
        one = torch.LongTensor(random_index[:L // 2])  # 按照patches长度的50% 混合图1和图2的patches
        two = torch.LongTensor(random_index[L // 2:]) # one: tensor([1, 2]) 图1要保留的patches位置  two: tensor([4, 0]) 图2要保留的patches位置
        mask_label[i].index_fill_(dim=0, index=one, value=ori_labels[i].squeeze())  # 把图1的ori_label的值赋给保留图一patches的位置
        mask_label[i].index_fill_(dim=0, index=two, value=pair_ori_labels[i].squeeze())  # 把图2的ori_label的值赋给保留图二patches的位置
        mask_label_14[i].index_fill_(dim=0, index=one, value=ori_labels[i].squeeze())  
        mask_label_14[i].index_fill_(dim=0, index=two, value=pair_ori_labels[i].squeeze())  
        mask[i].index_fill_(dim=0, index=one, value=1)
        mask2[i].index_fill_(dim=0, index=two, value=1)
        

    masked_patchify_img = torch.mul(mask.cuda(), patchify_imgs.cuda())
    pair_masked_patchify_img = torch.mul(mask2.cuda(), pairs_patchify_imgs.cuda())
    fusion_masked_img = torch.add(masked_patchify_img.cuda(), pair_masked_patchify_img.cuda())
    fusion_masked_label = mask_label.cuda()
    fusion_masked_label_14 = mask_label_14.cuda()

    # 反patch化 + 去patch化
    fusion_masked_img = unpatchify(cfg, fusion_masked_img, cfg['model']['patch_size']) # torch.Size([N, L, P*P])
    fusion_masked_label = unpatchify(cfg, fusion_masked_label, cfg['model']['patch_size']//scale) # torch.Size([N, L, P*P])  回到[N, 1, 224, 224]
    fusion_masked_label_14 = unpatchify(cfg, fusion_masked_label_14, cfg['model']['patch_size']//16) # 回到[N, 1, 14, 14]
#     fusion_masked_label = fusion_masked_label.mean(dim=-1).unsqueeze(dim=1).reshape(N, 1, cfg['model']['map_size'], cfg['model']['map_size']) # torch.Size([N, 1, 256]) 
    return fusion_masked_img, fusion_masked_label_14, fusion_masked_label


class LGSC(nn.Module):
    def __init__(self, cfg=None):
        super(LGSC, self).__init__()
        self.encoder_arch = cfg['encoder_arch']
        self.header_type = cfg['header_type']
        self.map_size = cfg['map_size']
#         self.enc = net_set[cfg['backbone']](cfg['image_size'][0])
        if self.encoder_arch == 'one_stream_arch':
            self.enc1 = LCNet_Encoder()
            self.decoder= Decoder(self.enc1.channels, self.enc1.last_channel)
            self.ToOneChannel = nn.Conv2d((self.enc1.last_channel), 1, kernel_size=1, stride=1, padding=0)
            self.sigmoid_map = nn.Sigmoid()
        elif self.encoder_arch == 'two_stream_arch':
            self.enc1 = LCNet_Encoder()
            self.enc2 = CDCN_Encoder()
            self.decoder= Decoder(self.enc1)
#         if self.header_type == 'patch_pixel':
# #             self.dec = nn.Conv2d(self.enc.output_channel, 1, kernel_size=1, stride=1, padding=0)
# #             self.dec = nn.Conv2d((self.enc1.last_channel+self.enc2.last_channel), 1, kernel_size=1, stride=1, padding=0)
#             self.sigmoid_map = nn.Sigmoid()
#     #         self.linear = nn.Linear(cfg['map_size'] * cfg['map_size'], 1)
#     #         self.dropout = nn.Dropout(p=cfg['dropout_prob'])
#     #         self.sigmoid_cls = nn.Sigmoid()
#         if self.header_type == 'binary_classification':
#             self.gap = nn.AdaptiveMaxPool2d(output_size=1)
#             self.linear1 = nn.Linear(self.enc.output_channel, self.enc.output_channel)
#             self.relu = nn.ReLU()
#             self.linear2 = nn.Linear(self.enc.output_channel, 2)
#             self.softmax = nn.Softmax()
# #         assert cfg['image_size'][0] // cfg['patch_size'] == cfg['map_size']
        self._initialize_weights()

    def forward(self, x):
        if self.encoder_arch == 'one_stream_arch':
            out, last = self.enc1(x)
            reg_map_branch = self.ToOneChannel(last)
            reg_map_branch = self.sigmoid_map(reg_map_branch)  # [N, 1, 14, 14]
        elif self.encoder_arch == 'two_stream_arch':
            out, last1 = self.enc1(x)
            out, last2 = self.enc2(x)
            last = torch.cat((last1, last2), dim=1)
            reg_map1 = self.sigmoid_map(out1)  # [N, 1, 14, 14]
            reg_map2 = self.sigmoid_map(out2)  # [N, 1, 14, 14]
            reg_map_branch = [reg_map1, reg_map2]
            out = self.ToOneChannel(out)
            
        out = self.decoder(out, last)
        reg_map = self.sigmoid_map(out)
        return reg_map, reg_map_branch # [N, 1, 112, 112] [N, 1, 14, 14]
#         if self.header_type == 'patch_pixel':
# #             dec = self.dec(out_tir[-1])
#             dec = self.decoder(out)
#             reg_map = self.sigmoid_map(dec)  # [N, 1, 224, 224]
#             print('reg_map:', reg_map.shape)
#             return reg_map1, reg_map2, reg_map 
#         elif self.header_type == 'binary_classification':
#             x = self.gap(out_tir[-1])
#             x = x.view(x.size(0), -1)
#             x = self.linear1(x)
#             x = self.relu(x)
#             x = self.linear2(x)
#             x = self.softmax(x)
#     #         reg = reg_map.view(-1, self.map_size * self.map_size)
#     #         reg = self.dropout(reg)
#     #         cls = self.linear(reg)
#             # dec = self.linear(out_map.view(-1, 8*8))
#     #         cls = torch.Tensor(x.shape[0], 1).type(torch.FloatTensor).cuda()
#             return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def LGSC_small(cfg):
    model = LGSC(cfg)
    return model

def export(dir, cfg):
    dummy_input = Variable(torch.randn(1, 3, 128, 128))
    model = LGSC(cfg)
    model.eval()
    torch.save(model.state_dict(), os.path.join(dir, "MobileNetV2.pth"))
    onnx.export(model, dummy_input, os.path.join(dir, "MobileNetV2.onnx"), verbose=True)


def test():
    model = LGSC()
    rgb = torch.rand(2, 3, 128, 128)
    print(model)
    print('size:', model(rgb).size())
    fin_path = '/mnt/sdb4/2d_project/sigle_result/test/checkpoints/MobileNetV2_small.pth'
    torch.save(model.state_dict(), fin_path)
    # print(model)

if __name__ == '__main__':
    test()