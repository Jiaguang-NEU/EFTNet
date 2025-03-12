import torch
from torch import nn
from torch._C import device
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm

import numpy as np
import random
import time
import cv2

import model.resnet as models
import model.vgg as vgg_models
from model.ASPP import ASPP
from model.PPM import PPM
from model.PSPNet import OneModel as PSPNet
from utils.util import get_train_val_set
from model.feature import extract_feat_res, extract_feat_vgg
from functools import reduce
from operator import add
from model.correlation import Correlation


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def get_gram_matrix(fea):
    b, c, h, w = fea.shape
    fea = fea.reshape(b, c, h * w)  # C*N
    fea_T = fea.permute(0, 2, 1)  # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T) / (torch.bmm(fea_norm, fea_T_norm) + 1e-7)  # C*C
    return gram


def output_filter(output, a=0.5):
    """
    :param output:Probability output
    :param a： threshold
    :return: Filtered binary mask
    """
    out_tensor = output.clone()
    zero_tensor = torch.zeros_like(output)
    filt_tensor = torch.where(out_tensor > 0.5, out_tensor, zero_tensor)
    down = filt_tensor.view(1, -1)
    down_tensor, indices0 = torch.sort(down)
    no_zero_down = torch.nonzero(down_tensor)
    idx = int(no_zero_down.shape[0] * a)
    stand0 = down_tensor[0, -idx] + 1e-10
    # print(stand)
    result_matrix0 = torch.zeros_like(output)
    result_matrix0[out_tensor >= stand0] = 1
    return result_matrix0


def calculate_iou(x, y):
    """
    :param x: [bs, 2, h, w]
    :param y: [bs, h, w]
    :return: iou
    """
    out = x.clone().max(1)[1]
    label = (y == 1)
    x_and_y = out * label
    x_or_y = torch.where((out + label) > 0, torch.ones_like(y), torch.zeros_like(y))
    iou = x_and_y.float().norm(1) / (x_or_y.float().norm(1) + 1e-10)
    return iou


class Attention(nn.Module):
    """
    Guided Attention Module (GAM).

    Args:
        in_channels: interval channel depth for both input and output
            feature map.
        drop_rate: dropout rate.
    """

    def __init__(self, in_channels, drop_rate=0.5):
        super().__init__()
        self.DEPTH = in_channels
        self.DROP_RATE = drop_rate
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=self.DEPTH, out_channels=self.DEPTH,
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.DEPTH, out_channels=self.DEPTH,
                      kernel_size=1),
            nn.Dropout(p=drop_rate),
            nn.Sigmoid())

    @staticmethod
    def mask(embedding, mask):
        h, w = embedding.size()[-2:]
        mask = F.interpolate(mask, size=(h, w), mode='nearest')
        mask = mask
        return mask * embedding

    def forward(self, *x):
        Fs, Ys = x
        att = F.adaptive_avg_pool2d(self.mask(Fs, Ys), output_size=(1, 1))
        g = self.gate(att)
        Fs = g * Fs
        return Fs


class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()
        self.change_weighting = args.change_weighting
        self.q2s_filt_para = args.q2s_filt_para
        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.vgg = args.vgg
        self.dataset = args.data_set
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

        self.print_freq = args.print_freq / 2

        self.pretrained = True
        self.classes = 2
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60

        assert self.layers in [50, 101, 152]

        backbone_str = 'vgg' if args.vgg else 'resnet' + str(args.layers)

        if backbone_str == 'vgg':
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
            self.nsimlairy = [1, 3, 3]
        elif backbone_str == 'resnet50':
            self.feat_ids = list(range(3, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
            self.nsimlairy = [3, 6, 4]
        elif backbone_str == 'resnet101':
            self.feat_ids = list(range(3, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
            self.nsimlairy = [3, 23, 4]
        else:
            raise Exception('Unavailable backbone: %s' % backbone_str)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]

        PSPNet_ = PSPNet(args)
        weight_path = 'initmodel/PSPNet/{}/split{}/{}/best.pth'.format(args.data_set, args.split, backbone_str)
        new_param = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']

        try:
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:  # 1GPU loads mGPU model
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4

        # Base Learner
        self.learner_base = nn.Sequential(PSPNet_.ppm, PSPNet_.cls)

        # Meta Learner
        reduce_dim = 256
        self.low_fea_id = args.low_fea[-1]
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        mask_add_num = 256 + 64 + 1
        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.ASPP_meta = ASPP(reduce_dim)
        self.res1_meta = nn.Sequential(
            nn.Conv2d(reduce_dim * 5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.res2_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.cls_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1))

        # Gram and Meta
        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.gram_merge.weight))

        # Learner Ensemble
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.cls_merge.weight))

        # K-Shot Reweighting
        if args.shot > 1:
            self.kshot_trans_dim = args.kshot_trans_dim
            if self.kshot_trans_dim == 0:
                self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)
                self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
            else:
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))

        self.sigmoid = nn.Sigmoid()
        self.gam = Attention(in_channels=256)
        self.hyper_final = nn.Sequential(
            nn.Conv2d(sum(nbottlenecks[-3:]), 64, kernel_size=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def get_optim(self, model, args, LR):
        if args.shot > 1:
            optimizer = torch.optim.SGD(
                [
                    {'params': model.down_query.parameters()},
                    {'params': model.down_supp.parameters()},
                    {'params': model.init_merge.parameters()},
                    {'params': model.ASPP_meta.parameters()},
                    {'params': model.res1_meta.parameters()},
                    {'params': model.res2_meta.parameters()},
                    {'params': model.cls_meta.parameters()},
                    {'params': model.gram_merge.parameters()},
                    {'params': model.cls_merge.parameters()},
                    {'params': model.kshot_rw.parameters()},
                ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(
                [
                    {'params': model.down_query.parameters()},
                    {'params': model.down_supp.parameters()},
                    {'params': model.init_merge.parameters()},
                    {'params': model.ASPP_meta.parameters()},
                    {'params': model.res1_meta.parameters()},
                    {'params': model.res2_meta.parameters()},
                    {'params': model.cls_meta.parameters()},
                    {'params': model.gram_merge.parameters()},
                    {'params': model.cls_merge.parameters()},
                ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        return optimizer

    def freeze_modules(self, model):
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.learner_base.parameters():
            param.requires_grad = False

    def forward(self, x, s_x, s_y, y_m, y_b, train_round, black_record, cat_idx=None, if_ft=False, fr_output=None, ft_mod='q2s'):
        """
        :param ft_mod:
        :param x: query img[bs, channel, h, w]
        :param s_x: support img[bs, shot, channel, h, w]
        :param s_y: support label[bs, shot, h, w]
        :param y_m: query label[bs, h, w]
        :param y_b: query label b[bs, h, w]
        :param train_round: 0, 1
        :param black_record: black
        :param cat_idx:
        :param if_ft:
        :param fr_output: output of zero round [bs, class, h. w]
        :return:
        """
        if train_round == 0 and ft_mod == 'q2s':
            x_size = x.size()
            bs = x_size[0]
            h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
            w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

            with torch.no_grad():
                query_feats, query_backbone_layers = self.extract_feats(x,
                                                                        [self.layer0, self.layer1, self.layer2,
                                                                         self.layer3,
                                                                         self.layer4], self.feat_ids,
                                                                        self.bottleneck_ids, self.lids)

            if self.vgg:
                query_feat = F.interpolate(query_backbone_layers[2],
                                           size=(query_backbone_layers[3].size(2), query_backbone_layers[3].size(3)), \
                                           mode='bilinear', align_corners=True)
                query_feat = torch.cat([query_backbone_layers[3], query_feat], 1)
            else:
                query_feat = torch.cat([query_backbone_layers[3], query_backbone_layers[2]], 1)

            query_feat = self.down_query(query_feat)

            supp_pro_list = []
            final_supp_list = []
            mask_list = []
            supp_feat_list = []
            supp_pro_list2 = []
            corrs = []
            gams = 0
            for i in range(self.shot):
                mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
                mask_list.append(mask)
                with torch.no_grad():
                    support_feats, support_backbone_layers = self.extract_feats(s_x[:, i, :, :, :],
                                                                                [self.layer0, self.layer1, self.layer2,
                                                                                 self.layer3, self.layer4],
                                                                                self.feat_ids, self.bottleneck_ids,
                                                                                self.lids)
                    final_supp_list.append(support_backbone_layers[4])

                if self.vgg:
                    supp_feat = F.interpolate(support_backbone_layers[2],
                                              size=(
                                                  support_backbone_layers[3].size(2),
                                                  support_backbone_layers[3].size(3)),
                                              mode='bilinear', align_corners=True)
                    supp_feat = torch.cat([support_backbone_layers[3], supp_feat], 1)
                else:
                    supp_feat = torch.cat([support_backbone_layers[3], support_backbone_layers[2]], 1)

                mask_down = F.interpolate(mask,
                                          size=(support_backbone_layers[3].size(2), support_backbone_layers[3].size(3)),
                                          mode='bilinear', align_corners=True)
                supp_feat = self.down_supp(supp_feat)
                supp_pro = Weighted_GAP(supp_feat, mask_down)
                supp_pro_list.append(supp_pro)
                supp_feat_list.append(support_backbone_layers[2])
                support_feats_1 = self.mask_feature(support_feats, mask)
                corr = Correlation.multilayer_correlation(query_feats, support_feats_1, self.stack_ids)
                corrs.append(corr)

                gams += self.gam(supp_feat, mask)

            gam = gams / self.shot
            corrs_shot = [corrs[0][i] for i in range(len(self.nsimlairy))]
            for ly in range(len(self.nsimlairy)):
                for s in range(1, self.shot):
                    corrs_shot[ly] += (corrs[s][ly])

            hyper_4 = corrs_shot[0] / self.shot
            hyper_3 = corrs_shot[1] / self.shot
            if self.vgg:
                hyper_2 = F.interpolate(corr[2], size=(corr[1].size(2), corr[1].size(3)), mode='bilinear',align_corners=True)
            else:
                hyper_2 = corrs_shot[2] / self.shot

            hyper_final = torch.cat([hyper_2, hyper_3, hyper_4], 1)

            hyper_final = self.hyper_final(hyper_final)

            # K-Shot Reweighting
            que_gram = get_gram_matrix(query_backbone_layers[2])  # [bs, C, C] in (0,1)
            norm_max = torch.ones_like(que_gram).norm(dim=(1, 2))
            est_val_list = []
            for supp_item in supp_feat_list:
                supp_gram = get_gram_matrix(supp_item)
                gram_diff = que_gram - supp_gram
                est_val_list.append((gram_diff.norm(dim=(1, 2)) / norm_max).reshape(bs, 1, 1, 1))  # norm2
            est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
            if self.shot > 1:
                val1, idx1 = est_val_total.sort(1)
                val2, idx2 = idx1.sort(1)
                weight = self.kshot_rw(val1)
                idx3 = idx1.gather(1, idx2)
                weight = weight.gather(1, idx3)
                weight_soft = torch.softmax(weight, 1)
            else:
                weight_soft = torch.ones_like(est_val_total)
            est_val = (weight_soft * est_val_total).sum(1, True)  # [bs, 1, 1, 1]

            corr_query_mask_list = []
            cosine_eps = 1e-7
            for i, tmp_supp_feat in enumerate(final_supp_list):
                resize_size = tmp_supp_feat.size(2)
                tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear',
                                         align_corners=True)

                tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
                q = query_backbone_layers[4]
                s = tmp_supp_feat_4
                bsize, ch_sz, sp_sz, _ = q.size()[:]

                tmp_query = q
                tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
                tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

                tmp_supp = s
                tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1)
                tmp_supp = tmp_supp.permute(0, 2, 1)
                tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

                similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
                similarity = similarity.max(1)[0].reshape(bsize, sp_sz * sp_sz)
                similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                        similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
                corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
                corr_query = F.interpolate(corr_query,
                                           size=(query_backbone_layers[3].size()[2], query_backbone_layers[3].size()[3]),
                                           mode='bilinear', align_corners=True)
                corr_query_mask_list.append(corr_query)
            corr_query_mask = torch.cat(corr_query_mask_list, 1)
            corr_query_mask = (weight_soft * corr_query_mask).sum(1, True)

            # Support Prototype
            supp_pro = torch.cat(supp_pro_list, 2)  # [bs, 256, shot, 1]
            supp_pro = (weight_soft.permute(0, 2, 1, 3) * supp_pro).sum(2, True)
            # print(supp_pro.shape)
            supp_pro_list2.append(supp_pro.unsqueeze(dim=2))

            # Tile & Cat
            concat_feat = supp_pro.expand_as(query_feat)
            merge_feat = torch.cat([query_feat, concat_feat, hyper_final, corr_query_mask, gam], 1)  # 256+64+256+64+1
            merge_feat = self.init_merge(merge_feat)

            # Base and Meta
            base_out = self.learner_base(query_backbone_layers[4])

            query_meta = self.ASPP_meta(merge_feat)
            query_meta = self.res1_meta(query_meta)  # 1080->256
            query_meta = self.res2_meta(query_meta) + query_meta
            meta_out = self.cls_meta(query_meta)

            meta_out_soft = meta_out.softmax(1)
            base_out_soft = base_out.softmax(1)

            # Classifier Ensemble
            meta_map_bg = meta_out_soft[:, 0:1, :, :]  # [bs, 1, 60, 60]
            meta_map_fg = meta_out_soft[:, 1:, :, :]  # [bs, 1, 60, 60]
            if self.training and self.cls_type == 'Base':
                c_id_array = torch.arange(self.base_classes + 1, device='cuda')
                base_map_list = []
                for b_id in range(bs):
                    c_id = cat_idx[0][b_id] + 1
                    c_mask = (c_id_array != 0) & (c_id_array != c_id)
                    base_map_list.append(base_out_soft[b_id, c_mask, :, :].unsqueeze(0).sum(1, True))
                base_map = torch.cat(base_map_list, 0)

            else:
                base_map = base_out_soft[:, 1:, :, :].sum(1, True)

            est_map = est_val.expand_as(meta_map_fg)

            meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))
            meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))

            merge_map = torch.cat([meta_map_bg, base_map], 1)
            merge_bg = self.cls_merge(merge_map)  # [bs, 1, 60, 60]

            final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

            # Output Part
            if self.zoom_factor != 1:
                meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
                base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)
                final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)

            # Loss
            if self.training:
                main_loss = self.criterion(final_out, y_m.long())
                aux_loss1 = self.criterion(meta_out, y_m.long())
                aux_loss2 = self.criterion(base_out, y_b.long())
                return final_out.max(1)[1], final_out, main_loss, aux_loss1, aux_loss2
            else:
                if if_ft:
                    pseudo_label = final_out.max(1)[1]
                    pseudo_label[black_record == 255] = 255
                    main_loss = self.criterion(final_out, pseudo_label)
                    aux_loss1 = self.criterion(meta_out, pseudo_label)
                    return final_out, meta_out, base_out, main_loss, aux_loss1
                else:
                    return final_out, meta_out, base_out
        # -------------------------------------------ft round--------------------------------------------
        if train_round > 0 and ft_mod == 'q2s':
            # query feature
            x_size = x.size()
            bs = x_size[0]
            h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
            w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

            with torch.no_grad():
                query_feats, query_backbone_layers = self.extract_feats(x,[self.layer0, self.layer1, self.layer2,
                                                                           self.layer3,self.layer4],
                                                                            self.feat_ids,
                                                                            self.bottleneck_ids, self.lids)

            if self.vgg:
                query_feat = F.interpolate(query_backbone_layers[2],
                                           size=(query_backbone_layers[3].size(2), query_backbone_layers[3].size(3)),
                                           mode='bilinear', align_corners=True)
                query_feat = torch.cat([query_backbone_layers[3], query_feat], 1)
            else:
                query_feat = torch.cat([query_backbone_layers[3], query_backbone_layers[2]], 1)

            query_feat = self.down_query(query_feat)

            # same img
            q2s_corrs = []
            q2s_final_supp_list = []
            q2s_mask_list = []
            q2s_x = x.clone().unsqueeze(dim=1)
            fr_output_softmax = fr_output.softmax(dim=1)
            q2s_y = output_filter(fr_output_softmax[:, 1, :, :], a=self.q2s_filt_para).float().unsqueeze(dim=1)  # [1, 1, h, w] 二值图
            q2s_mask = q2s_y.clone()
            mask = (q2s_y[:, 0, :, :] == 1).float().unsqueeze(1)
            q2s_mask_list.append(mask)
            with torch.no_grad():
                support_feats, support_backbone_layers = self.extract_feats(q2s_x[:, 0, :, :, :],
                                                                [self.layer0, self.layer1, self.layer2,
                                                                 self.layer3, self.layer4],
                                                                self.feat_ids, self.bottleneck_ids,
                                                                self.lids)
                q2s_final_supp_list.append(support_backbone_layers[4])
            if self.vgg:
                supp_feat = F.interpolate(support_backbone_layers[2],
                                          size=(
                                              support_backbone_layers[3].size(2),
                                              support_backbone_layers[3].size(3)),
                                          mode='bilinear', align_corners=True)
                supp_feat = torch.cat([support_backbone_layers[3], supp_feat], 1)
            else:
                supp_feat = torch.cat([support_backbone_layers[3], support_backbone_layers[2]], 1)

            mask_down = F.interpolate(mask,
                                      size=(support_backbone_layers[3].size(2), support_backbone_layers[3].size(3)),
                                      mode='bilinear', align_corners=True)
            supp_feat = self.down_supp(supp_feat)
            supp_pro = Weighted_GAP(supp_feat, mask_down)

            q2s_gam = self.gam(supp_feat, mask)

            q2s_concat_feat = supp_pro.expand_as(query_feat)
            support_feats_1 = self.mask_feature(support_feats, mask)
            corr = Correlation.multilayer_correlation(query_feats, support_feats_1, self.stack_ids)
            q2s_corrs.append(corr)
            corrs_shot = [q2s_corrs[0][i] for i in range(len(self.nsimlairy))]
            for ly in range(len(self.nsimlairy)):
                for s in range(1, 1):
                    corrs_shot[ly] += (q2s_corrs[s][ly])

            hyper_4 = corrs_shot[0] / 1
            hyper_3 = corrs_shot[1] / 1
            if self.vgg:
                hyper_2 = F.interpolate(corr[2], size=(corr[1].size(2), corr[1].size(3)), mode='bilinear',
                                        align_corners=True)
            else:
                hyper_2 = corrs_shot[2] / 1

            hyper_final = torch.cat([hyper_2, hyper_3, hyper_4], 1)

            q2s_hyper_final = self.hyper_final(hyper_final)
            q2s_corr_query_mask_list = []
            cosine_eps = 1e-7
            for i, tmp_supp_feat in enumerate(q2s_final_supp_list):
                resize_size = tmp_supp_feat.size(2)
                tmp_mask = F.interpolate(q2s_mask_list[i], size=(resize_size, resize_size), mode='bilinear',
                                         align_corners=True)

                tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
                q = query_backbone_layers[4]
                s = tmp_supp_feat_4
                bsize, ch_sz, sp_sz, _ = q.size()[:]

                tmp_query = q
                tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
                tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

                tmp_supp = s
                tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1)
                tmp_supp = tmp_supp.permute(0, 2, 1)
                tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

                similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
                similarity = similarity.max(1)[0].reshape(bsize, sp_sz * sp_sz)
                similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                        similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
                corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
                corr_query = F.interpolate(corr_query,
                                           size=(
                                               query_backbone_layers[3].size()[2], query_backbone_layers[3].size()[3]),
                                           mode='bilinear', align_corners=True)
                q2s_corr_query_mask_list.append(corr_query)
            q2s_corr_query_mask = torch.cat(q2s_corr_query_mask_list, 1)
            weight_soft = torch.ones(bs, 1, 1, 1).cuda()
            q2s_corr_query_mask = (weight_soft * q2s_corr_query_mask).sum(1, True)

            # diff imgs
            supp_pro_list = []
            final_supp_list = []
            mask_list = []
            supp_feat_list = []
            corrs = []
            gams = 0
            for i in range(self.shot):
                mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
                mask_list.append(mask)
                with torch.no_grad():
                    support_feats, support_backbone_layers = self.extract_feats(s_x[:, i, :, :, :],
                                                                                [self.layer0, self.layer1, self.layer2,
                                                                                 self.layer3, self.layer4],
                                                                                self.feat_ids, self.bottleneck_ids,
                                                                                self.lids)
                    final_supp_list.append(support_backbone_layers[4])

                if self.vgg:
                    supp_feat = F.interpolate(support_backbone_layers[2],
                                              size=(
                                                  support_backbone_layers[3].size(2),
                                                  support_backbone_layers[3].size(3)),
                                              mode='bilinear', align_corners=True)
                    supp_feat = torch.cat([support_backbone_layers[3], supp_feat], 1)
                else:
                    supp_feat = torch.cat([support_backbone_layers[3], support_backbone_layers[2]], 1)

                mask_down = F.interpolate(mask,
                                          size=(support_backbone_layers[3].size(2), support_backbone_layers[3].size(3)),
                                          mode='bilinear', align_corners=True)
                supp_feat = self.down_supp(supp_feat)
                supp_pro = Weighted_GAP(supp_feat, mask_down)
                supp_pro_list.append(supp_pro)
                supp_feat_list.append(support_backbone_layers[2])

                support_feats_1 = self.mask_feature(support_feats, mask)
                corr = Correlation.multilayer_correlation(query_feats, support_feats_1, self.stack_ids)
                corrs.append(corr)

                gams += self.gam(supp_feat, mask)

            gam = gams / self.shot
            corrs_shot = [corrs[0][i] for i in range(len(self.nsimlairy))]
            for ly in range(len(self.nsimlairy)):
                for s in range(1, self.shot):
                    corrs_shot[ly] += (corrs[s][ly])

            hyper_4 = corrs_shot[0] / self.shot
            hyper_3 = corrs_shot[1] / self.shot
            if self.vgg:
                hyper_2 = F.interpolate(corr[2], size=(corr[1].size(2), corr[1].size(3)), mode='bilinear',
                                        align_corners=True)
            else:
                hyper_2 = corrs_shot[2] / self.shot

            hyper_final = torch.cat([hyper_2, hyper_3, hyper_4], 1)

            hyper_final = self.hyper_final(hyper_final)

            # K-Shot Reweighting
            que_gram = get_gram_matrix(query_backbone_layers[2])  # [bs, C, C] in (0,1)
            norm_max = torch.ones_like(que_gram).norm(dim=(1, 2))
            est_val_list = []
            for supp_item in supp_feat_list:
                supp_gram = get_gram_matrix(supp_item)
                gram_diff = que_gram - supp_gram
                est_val_list.append((gram_diff.norm(dim=(1, 2)) / norm_max).reshape(bs, 1, 1, 1))  # norm2
            est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
            if self.shot > 1:
                val1, idx1 = est_val_total.sort(1)
                val2, idx2 = idx1.sort(1)
                weight = self.kshot_rw(val1)
                idx3 = idx1.gather(1, idx2)
                weight = weight.gather(1, idx3)
                weight_soft = torch.softmax(weight, 1)
            else:
                weight_soft = torch.ones_like(est_val_total)
            est_val = (weight_soft * est_val_total).sum(1, True)  # [bs, 1, 1, 1]

            corr_query_mask_list = []
            cosine_eps = 1e-7
            for i, tmp_supp_feat in enumerate(final_supp_list):
                resize_size = tmp_supp_feat.size(2)
                tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear',
                                         align_corners=True)

                tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
                q = query_backbone_layers[4]
                s = tmp_supp_feat_4
                bsize, ch_sz, sp_sz, _ = q.size()[:]

                tmp_query = q
                tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
                tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

                tmp_supp = s
                tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1)
                tmp_supp = tmp_supp.permute(0, 2, 1)
                tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

                similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
                similarity = similarity.max(1)[0].reshape(bsize, sp_sz * sp_sz)
                similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                        similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
                corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
                corr_query = F.interpolate(corr_query,
                                           size=(
                                               query_backbone_layers[3].size()[2], query_backbone_layers[3].size()[3]),
                                           mode='bilinear', align_corners=True)
                corr_query_mask_list.append(corr_query)
            corr_query_mask = torch.cat(corr_query_mask_list, 1)
            corr_query_mask = (weight_soft * corr_query_mask).sum(1, True)

            # Support Prototype
            supp_pro = torch.cat(supp_pro_list, 2)  # [bs, 256, shot, 1]
            supp_pro = (weight_soft.permute(0, 2, 1, 3) * supp_pro).sum(2, True)

            # Tile & Cat
            concat_feat = supp_pro.expand_as(query_feat)
            concat_feat = concat_feat * self.change_weighting[0] + q2s_concat_feat * self.change_weighting[1]
            gam = gam * self.change_weighting[0] + q2s_gam * self.change_weighting[1]
            hyper_final = hyper_final * self.change_weighting[0] + q2s_hyper_final * self.change_weighting[1]
            corr_query_mask = corr_query_mask * self.change_weighting[0] + q2s_corr_query_mask * self.change_weighting[1]
            merge_feat = torch.cat([query_feat, concat_feat, hyper_final, corr_query_mask, gam], 1)  # 256+256+64+1+256
            merge_feat = self.init_merge(merge_feat)

            # Base and Meta
            sr_base_out = self.learner_base(query_backbone_layers[4])

            query_meta = self.ASPP_meta(merge_feat)
            query_meta = self.res1_meta(query_meta)  # 1080->256
            query_meta = self.res2_meta(query_meta) + query_meta
            sr_meta_out = self.cls_meta(query_meta)

            meta_out_soft = sr_meta_out.softmax(1)
            base_out_soft = sr_base_out.softmax(1)

            # Classifier Ensemble
            meta_map_bg = meta_out_soft[:, 0:1, :, :]  # [bs, 1, 60, 60]
            meta_map_fg = meta_out_soft[:, 1:, :, :]  # [bs, 1, 60, 60]
            if self.training and self.cls_type == 'Base':
                c_id_array = torch.arange(self.base_classes + 1, device='cuda')
                base_map_list = []
                for b_id in range(bs):
                    c_id = cat_idx[0][b_id] + 1
                    c_mask = (c_id_array != 0) & (c_id_array != c_id)
                    base_map_list.append(base_out_soft[b_id, c_mask, :, :].unsqueeze(0).sum(1, True))
                base_map = torch.cat(base_map_list, 0)

            else:
                base_map = base_out_soft[:, 1:, :, :].sum(1, True)

            est_map = est_val.expand_as(meta_map_fg)

            meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))
            meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))

            merge_map = torch.cat([meta_map_bg, base_map], 1)
            merge_bg = self.cls_merge(merge_map)  # [bs, 1, 60, 60]

            sr_final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

            # Output Part
            if self.zoom_factor != 1:
                sr_meta_out = F.interpolate(sr_meta_out, size=(h, w), mode='bilinear', align_corners=True)
                sr_base_out = F.interpolate(sr_base_out, size=(h, w), mode='bilinear', align_corners=True)
                sr_final_out = F.interpolate(sr_final_out, size=(h, w), mode='bilinear', align_corners=True)
            # Loss
            if self.training:
                sr_main_loss = self.criterion(sr_final_out, y_m.long())
                sr_aux_loss1 = self.criterion(sr_meta_out, y_m.long())
                return sr_final_out.max(1)[1], sr_main_loss, sr_aux_loss1
            else:
                if if_ft:
                    sr_ori_pseudo_label = fr_output.clone().max(1)[1].long()
                    sr_ori_pseudo_label[black_record == 255] = 255
                    sr_pseudo_label = sr_ori_pseudo_label
                    sr_main_loss = self.criterion(sr_final_out, sr_pseudo_label)
                    sr_aux_loss1 = self.criterion(sr_meta_out, sr_pseudo_label)
                    return sr_final_out, sr_meta_out, sr_base_out, sr_pseudo_label, sr_main_loss, sr_aux_loss1, q2s_mask
                else:
                    return sr_final_out, sr_meta_out, sr_base_out

        if train_round == 0 and ft_mod == 's2q':
            ious_list = []
            for shot in range(s_x.shape[1]):
                s2q_x = s_x[:, shot, :, :, :]
                s2q_y = s_y[:, shot, :, :]
                s2q_x_size = s2q_x.size()
                bs = s2q_x_size[0]
                h = int((s2q_x_size[2] - 1) / 8 * self.zoom_factor + 1)
                w = int((s2q_x_size[3] - 1) / 8 * self.zoom_factor + 1)
                s_x_1 = s_x.clone()
                s_y_1 = s_y.clone()
                s_x_1[:, shot, :, :, :] = x
                s_y_1[:, shot, :, :] = y_m

                with torch.no_grad():
                    query_feats, query_backbone_layers = self.extract_feats(s2q_x,
                                                                            [self.layer0, self.layer1, self.layer2,
                                                                             self.layer3,
                                                                             self.layer4], self.feat_ids,
                                                                            self.bottleneck_ids, self.lids)

                if self.vgg:
                    query_feat = F.interpolate(query_backbone_layers[2],
                                               size=(query_backbone_layers[3].size(2), query_backbone_layers[3].size(3)),
                                               mode='bilinear', align_corners=True)
                    query_feat = torch.cat([query_backbone_layers[3], query_feat], 1)
                else:
                    query_feat = torch.cat([query_backbone_layers[3], query_backbone_layers[2]], 1)

                query_feat = self.down_query(query_feat)

                supp_pro_list = []
                final_supp_list = []
                mask_list = []
                supp_feat_list = []
                supp_pro_list2 = []
                corrs = []
                gams = 0
                for i in range(self.shot):
                    mask = (s_y_1[:, i, :, :] == 1).float().unsqueeze(1)
                    mask_list.append(mask)
                    with torch.no_grad():
                        support_feats, support_backbone_layers = self.extract_feats(s_x_1[:, i, :, :, :],
                                                                                    [self.layer0, self.layer1,
                                                                                     self.layer2,
                                                                                     self.layer3, self.layer4],
                                                                                    self.feat_ids, self.bottleneck_ids,
                                                                                    self.lids)
                        final_supp_list.append(support_backbone_layers[4])

                    if self.vgg:
                        supp_feat = F.interpolate(support_backbone_layers[2],
                                                  size=(
                                                      support_backbone_layers[3].size(2),
                                                      support_backbone_layers[3].size(3)),
                                                  mode='bilinear', align_corners=True)
                        supp_feat = torch.cat([support_backbone_layers[3], supp_feat], 1)
                    else:
                        supp_feat = torch.cat([support_backbone_layers[3], support_backbone_layers[2]], 1)

                    mask_down = F.interpolate(mask,
                                              size=(
                                                  support_backbone_layers[3].size(2),
                                                  support_backbone_layers[3].size(3)),
                                              mode='bilinear', align_corners=True)
                    supp_feat = self.down_supp(supp_feat)
                    supp_pro = Weighted_GAP(supp_feat, mask_down)
                    supp_pro_list.append(supp_pro)
                    supp_feat_list.append(support_backbone_layers[2])
                    support_feats_1 = self.mask_feature(support_feats, mask)
                    corr = Correlation.multilayer_correlation(query_feats, support_feats_1, self.stack_ids)
                    corrs.append(corr)

                    gams += self.gam(supp_feat, mask)

                gam = gams / self.shot
                corrs_shot = [corrs[0][i] for i in range(len(self.nsimlairy))]
                for ly in range(len(self.nsimlairy)):
                    for s in range(1, self.shot):
                        corrs_shot[ly] += (corrs[s][ly])

                hyper_4 = corrs_shot[0] / self.shot
                hyper_3 = corrs_shot[1] / self.shot
                if self.vgg:
                    hyper_2 = F.interpolate(corr[2], size=(corr[1].size(2), corr[1].size(3)), mode='bilinear',
                                            align_corners=True)
                else:
                    hyper_2 = corrs_shot[2] / self.shot

                hyper_final = torch.cat([hyper_2, hyper_3, hyper_4], 1)

                hyper_final = self.hyper_final(hyper_final)

                # K-Shot Reweighting
                que_gram = get_gram_matrix(query_backbone_layers[2])  # [bs, C, C] in (0,1)
                norm_max = torch.ones_like(que_gram).norm(dim=(1, 2))
                est_val_list = []
                for supp_item in supp_feat_list:
                    supp_gram = get_gram_matrix(supp_item)
                    gram_diff = que_gram - supp_gram
                    est_val_list.append((gram_diff.norm(dim=(1, 2)) / norm_max).reshape(bs, 1, 1, 1))  # norm2
                est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
                if self.shot > 1:
                    val1, idx1 = est_val_total.sort(1)
                    val2, idx2 = idx1.sort(1)
                    weight = self.kshot_rw(val1)
                    idx3 = idx1.gather(1, idx2)
                    weight = weight.gather(1, idx3)
                    weight_soft = torch.softmax(weight, 1)
                else:
                    weight_soft = torch.ones_like(est_val_total)
                est_val = (weight_soft * est_val_total).sum(1, True)  # [bs, 1, 1, 1]

                corr_query_mask_list = []
                cosine_eps = 1e-7
                for i, tmp_supp_feat in enumerate(final_supp_list):
                    resize_size = tmp_supp_feat.size(2)
                    tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear',
                                             align_corners=True)

                    tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
                    q = query_backbone_layers[4]
                    s = tmp_supp_feat_4
                    bsize, ch_sz, sp_sz, _ = q.size()[:]

                    tmp_query = q
                    tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
                    tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

                    tmp_supp = s
                    tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1)
                    tmp_supp = tmp_supp.permute(0, 2, 1)
                    tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

                    similarity = torch.bmm(tmp_supp, tmp_query) / (
                            torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
                    similarity = similarity.max(1)[0].reshape(bsize, sp_sz * sp_sz)
                    similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                            similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
                    corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
                    corr_query = F.interpolate(corr_query,
                                               size=(
                                                   query_backbone_layers[3].size()[2],
                                                   query_backbone_layers[3].size()[3]),
                                               mode='bilinear', align_corners=True)
                    corr_query_mask_list.append(corr_query)
                corr_query_mask = torch.cat(corr_query_mask_list, 1)
                corr_query_mask = (weight_soft * corr_query_mask).sum(1, True)

                # Support Prototype
                supp_pro = torch.cat(supp_pro_list, 2)  # [bs, 256, shot, 1]
                supp_pro = (weight_soft.permute(0, 2, 1, 3) * supp_pro).sum(2, True)
                supp_pro_list2.append(supp_pro.unsqueeze(dim=2))

                # Tile & Cat
                concat_feat = supp_pro.expand_as(query_feat)
                merge_feat = torch.cat([query_feat, concat_feat, hyper_final, corr_query_mask, gam],1)  # 256+64+256+64+1
                merge_feat = self.init_merge(merge_feat)

                # Base and Meta
                base_out = self.learner_base(query_backbone_layers[4])

                query_meta = self.ASPP_meta(merge_feat)
                query_meta = self.res1_meta(query_meta)  # 1080->256
                query_meta = self.res2_meta(query_meta) + query_meta
                meta_out = self.cls_meta(query_meta)

                meta_out_soft = meta_out.softmax(1)
                base_out_soft = base_out.softmax(1)

                # Classifier Ensemble
                meta_map_bg = meta_out_soft[:, 0:1, :, :]  # [bs, 1, 60, 60]
                meta_map_fg = meta_out_soft[:, 1:, :, :]  # [bs, 1, 60, 60]
                if self.training and self.cls_type == 'Base':
                    c_id_array = torch.arange(self.base_classes + 1, device='cuda')
                    base_map_list = []
                    for b_id in range(bs):
                        c_id = cat_idx[0][b_id] + 1
                        c_mask = (c_id_array != 0) & (c_id_array != c_id)
                        base_map_list.append(base_out_soft[b_id, c_mask, :, :].unsqueeze(0).sum(1, True))
                    base_map = torch.cat(base_map_list, 0)

                else:
                    base_map = base_out_soft[:, 1:, :, :].sum(1, True)

                est_map = est_val.expand_as(meta_map_fg)

                meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))
                meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))

                merge_map = torch.cat([meta_map_bg, base_map], 1)
                merge_bg = self.cls_merge(merge_map)  # [bs, 1, 60, 60]

                final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

                # Output Part
                if self.zoom_factor != 1:
                    meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
                    base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)
                    final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)
                    iou = calculate_iou(final_out, s2q_y)
                    ious_list.append(iou)

            # Loss
            if self.training:
                pass
            else:
                if if_ft:
                    s2q_miou = sum(ious_list) / len(ious_list)
                    return s2q_miou, final_out, base_out
                else:
                    pass
        # -------------------------------------------Discrimination round--------------------------------------------
        if train_round > 0 and ft_mod == 's2q':
            ious_list = []
            for shot in range(s_x.shape[1]):
                s2q_x = s_x[:, shot, :, :, :]
                s2q_y = s_y[:, shot, :, :]
                s_x_1 = s_x.clone()
                s_y_1 = s_y.clone()
                s_x_1[:, shot, :, :, :] = x
                s_y_1[:, shot, :, :] = y_m

                # query feature
                s2q_x_size = s2q_x.size()
                bs = s2q_x_size[0]
                h = int((s2q_x_size[2] - 1) / 8 * self.zoom_factor + 1)
                w = int((s2q_x_size[3] - 1) / 8 * self.zoom_factor + 1)

                with torch.no_grad():
                    query_feats, query_backbone_layers = self.extract_feats(s2q_x,
                                                                            [self.layer0, self.layer1, self.layer2,
                                                                             self.layer3,
                                                                             self.layer4], self.feat_ids,
                                                                            self.bottleneck_ids, self.lids)

                if self.vgg:
                    query_feat = F.interpolate(query_backbone_layers[2],
                                               size=(
                                                   query_backbone_layers[3].size(2), query_backbone_layers[3].size(3)),
                                               mode='bilinear', align_corners=True)
                    query_feat = torch.cat([query_backbone_layers[3], query_feat], 1)
                else:
                    query_feat = torch.cat([query_backbone_layers[3], query_backbone_layers[2]], 1)

                query_feat = self.down_query(query_feat)

                # same img
                s2q2s_corrs = []
                s2q2s_final_supp_list = []
                s2q2s_mask_list = []
                s2q2s_x = s2q_x.clone().unsqueeze(dim=1)
                s2q2s_y = s2q_y.clone().unsqueeze(dim=1)  # [1, 1, h, w]

                mask = (s2q2s_y[:, 0, :, :] == 1).float().unsqueeze(1)
                s2q2s_mask_list.append(mask)
                with torch.no_grad():
                    support_feats, support_backbone_layers = self.extract_feats(s2q2s_x[:, 0, :, :, :],
                                                                    [self.layer0, self.layer1, self.layer2,
                                                                     self.layer3, self.layer4],
                                                                    self.feat_ids, self.bottleneck_ids,
                                                                    self.lids)
                    s2q2s_final_supp_list.append(support_backbone_layers[4])
                if self.vgg:
                    supp_feat = F.interpolate(support_backbone_layers[2],
                                              size=(
                                                  support_backbone_layers[3].size(2),
                                                  support_backbone_layers[3].size(3)),
                                              mode='bilinear', align_corners=True)
                    supp_feat = torch.cat([support_backbone_layers[3], supp_feat], 1)
                else:
                    supp_feat = torch.cat([support_backbone_layers[3], support_backbone_layers[2]], 1)

                mask_down = F.interpolate(mask,
                                          size=(support_backbone_layers[3].size(2), support_backbone_layers[3].size(3)),
                                          mode='bilinear', align_corners=True)
                supp_feat = self.down_supp(supp_feat)
                supp_pro = Weighted_GAP(supp_feat, mask_down)
                s2q2s_gam = self.gam(supp_feat, mask)
                s2q2s_concat_feat = supp_pro.expand_as(query_feat)
                support_feats_1 = self.mask_feature(support_feats, mask)
                corr = Correlation.multilayer_correlation(query_feats, support_feats_1, self.stack_ids)
                s2q2s_corrs.append(corr)
                corrs_shot = [s2q2s_corrs[0][i] for i in range(len(self.nsimlairy))]
                for ly in range(len(self.nsimlairy)):
                    for s in range(1, 1):
                        corrs_shot[ly] += (s2q2s_corrs[s][ly])

                hyper_4 = corrs_shot[0] / 1
                hyper_3 = corrs_shot[1] / 1
                if self.vgg:
                    hyper_2 = F.interpolate(corr[2], size=(corr[1].size(2), corr[1].size(3)), mode='bilinear',
                                            align_corners=True)
                else:
                    hyper_2 = corrs_shot[2] / 1

                hyper_final = torch.cat([hyper_2, hyper_3, hyper_4], 1)

                s2q2s_hyper_final = self.hyper_final(hyper_final)
                s2q2s_corr_query_mask_list = []
                cosine_eps = 1e-7
                for i, tmp_supp_feat in enumerate(s2q2s_final_supp_list):
                    resize_size = tmp_supp_feat.size(2)
                    tmp_mask = F.interpolate(s2q2s_mask_list[i], size=(resize_size, resize_size), mode='bilinear',
                                             align_corners=True)

                    tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
                    q = query_backbone_layers[4]
                    s = tmp_supp_feat_4
                    bsize, ch_sz, sp_sz, _ = q.size()[:]

                    tmp_query = q
                    tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
                    tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

                    tmp_supp = s
                    tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1)
                    tmp_supp = tmp_supp.permute(0, 2, 1)
                    tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

                    similarity = torch.bmm(tmp_supp, tmp_query) / (
                                torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
                    similarity = similarity.max(1)[0].reshape(bsize, sp_sz * sp_sz)
                    similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                            similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
                    corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
                    corr_query = F.interpolate(corr_query,size=(query_backbone_layers[3].size()[2], query_backbone_layers[3].size()[3]),
                                               mode='bilinear', align_corners=True)
                    s2q2s_corr_query_mask_list.append(corr_query)
                s2q2s_corr_query_mask = torch.cat(s2q2s_corr_query_mask_list, 1)
                weight_soft = torch.ones(bs, 1, 1, 1).cuda()
                s2q2s_corr_query_mask = (weight_soft * s2q2s_corr_query_mask).sum(1, True)

                # diff imgs
                supp_pro_list = []
                final_supp_list = []
                mask_list = []
                supp_feat_list = []
                corrs = []
                gams = 0
                for i in range(self.shot):
                    mask = (s_y_1[:, i, :, :] == 1).float().unsqueeze(1)
                    mask_list.append(mask)
                    with torch.no_grad():
                        support_feats, support_backbone_layers = self.extract_feats(s_x_1[:, i, :, :, :],
                                                                                    [self.layer0, self.layer1,
                                                                                     self.layer2, self.layer3, self.layer4],
                                                                                    self.feat_ids, self.bottleneck_ids,
                                                                                    self.lids)
                        final_supp_list.append(support_backbone_layers[4])

                    if self.vgg:
                        supp_feat = F.interpolate(support_backbone_layers[2],
                                                  size=(
                                                      support_backbone_layers[3].size(2),
                                                      support_backbone_layers[3].size(3)),
                                                  mode='bilinear', align_corners=True)
                        supp_feat = torch.cat([support_backbone_layers[3], supp_feat], 1)
                    else:
                        supp_feat = torch.cat([support_backbone_layers[3], support_backbone_layers[2]], 1)

                    mask_down = F.interpolate(mask,
                                              size=(
                                                  support_backbone_layers[3].size(2),
                                                  support_backbone_layers[3].size(3)),
                                              mode='bilinear', align_corners=True)
                    supp_feat = self.down_supp(supp_feat)
                    supp_pro = Weighted_GAP(supp_feat, mask_down)
                    supp_pro_list.append(supp_pro)
                    supp_feat_list.append(support_backbone_layers[2])

                    support_feats_1 = self.mask_feature(support_feats, mask)
                    corr = Correlation.multilayer_correlation(query_feats, support_feats_1, self.stack_ids)
                    corrs.append(corr)

                    gams += self.gam(supp_feat, mask)

                gam = gams / self.shot
                corrs_shot = [corrs[0][i] for i in range(len(self.nsimlairy))]
                for ly in range(len(self.nsimlairy)):
                    for s in range(1, self.shot):
                        corrs_shot[ly] += (corrs[s][ly])

                hyper_4 = corrs_shot[0] / self.shot
                hyper_3 = corrs_shot[1] / self.shot
                if self.vgg:
                    hyper_2 = F.interpolate(corr[2], size=(corr[1].size(2), corr[1].size(3)), mode='bilinear', align_corners=True)
                else:
                    hyper_2 = corrs_shot[2] / self.shot

                hyper_final = torch.cat([hyper_2, hyper_3, hyper_4], 1)

                hyper_final = self.hyper_final(hyper_final)

                # K-Shot Reweighting
                que_gram = get_gram_matrix(query_backbone_layers[2])  # [bs, C, C] in (0,1)
                norm_max = torch.ones_like(que_gram).norm(dim=(1, 2))
                est_val_list = []
                for supp_item in supp_feat_list:
                    supp_gram = get_gram_matrix(supp_item)
                    gram_diff = que_gram - supp_gram
                    est_val_list.append((gram_diff.norm(dim=(1, 2)) / norm_max).reshape(bs, 1, 1, 1))  # norm2
                est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
                if self.shot > 1:
                    val1, idx1 = est_val_total.sort(1)
                    val2, idx2 = idx1.sort(1)
                    weight = self.kshot_rw(val1)
                    idx3 = idx1.gather(1, idx2)
                    weight = weight.gather(1, idx3)
                    weight_soft = torch.softmax(weight, 1)
                else:
                    weight_soft = torch.ones_like(est_val_total)
                est_val = (weight_soft * est_val_total).sum(1, True)  # [bs, 1, 1, 1]

                corr_query_mask_list = []
                cosine_eps = 1e-7
                for i, tmp_supp_feat in enumerate(final_supp_list):
                    resize_size = tmp_supp_feat.size(2)
                    tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

                    tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
                    q = query_backbone_layers[4]
                    s = tmp_supp_feat_4
                    bsize, ch_sz, sp_sz, _ = q.size()[:]

                    tmp_query = q
                    tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
                    tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

                    tmp_supp = s
                    tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1)
                    tmp_supp = tmp_supp.permute(0, 2, 1)
                    tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

                    similarity = torch.bmm(tmp_supp, tmp_query) / (
                            torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
                    similarity = similarity.max(1)[0].reshape(bsize, sp_sz * sp_sz)
                    similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                            similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
                    corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
                    corr_query = F.interpolate(corr_query, size=(query_backbone_layers[3].size()[2], query_backbone_layers[3].size()[3]),
                                               mode='bilinear', align_corners=True)
                    corr_query_mask_list.append(corr_query)
                corr_query_mask = torch.cat(corr_query_mask_list, 1)
                corr_query_mask = (weight_soft * corr_query_mask).sum(1, True)

                # Support Prototype
                supp_pro = torch.cat(supp_pro_list, 2)  # [bs, 256, shot, 1]
                supp_pro = (weight_soft.permute(0, 2, 1, 3) * supp_pro).sum(2, True)

                # Tile & Cat
                concat_feat = supp_pro.expand_as(query_feat)
                concat_feat = concat_feat * self.change_weighting[0] + s2q2s_concat_feat * self.change_weighting[1]
                gam = gam * self.change_weighting[0] + s2q2s_gam * self.change_weighting[1]
                hyper_final = hyper_final * self.change_weighting[0] + s2q2s_hyper_final * self.change_weighting[1]
                corr_query_mask = corr_query_mask * self.change_weighting[0] + s2q2s_corr_query_mask * self.change_weighting[1]
                merge_feat = torch.cat([query_feat, concat_feat, hyper_final, corr_query_mask, gam], 1)  # 256+256+64+1+256

                merge_feat = self.init_merge(merge_feat)

                # Base and Meta
                sr_base_out = self.learner_base(query_backbone_layers[4])

                query_meta = self.ASPP_meta(merge_feat)
                query_meta = self.res1_meta(query_meta)  # 1080->256
                query_meta = self.res2_meta(query_meta) + query_meta
                sr_meta_out = self.cls_meta(query_meta)

                meta_out_soft = sr_meta_out.softmax(1)
                base_out_soft = sr_base_out.softmax(1)

                # Classifier Ensemble
                meta_map_bg = meta_out_soft[:, 0:1, :, :]  # [bs, 1, 60, 60]
                meta_map_fg = meta_out_soft[:, 1:, :, :]  # [bs, 1, 60, 60]
                if self.training and self.cls_type == 'Base':
                    c_id_array = torch.arange(self.base_classes + 1, device='cuda')
                    base_map_list = []
                    for b_id in range(bs):
                        c_id = cat_idx[0][b_id] + 1
                        c_mask = (c_id_array != 0) & (c_id_array != c_id)
                        base_map_list.append(base_out_soft[b_id, c_mask, :, :].unsqueeze(0).sum(1, True))
                    base_map = torch.cat(base_map_list, 0)

                else:
                    base_map = base_out_soft[:, 1:, :, :].sum(1, True)

                est_map = est_val.expand_as(meta_map_fg)

                meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))
                meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))

                merge_map = torch.cat([meta_map_bg, base_map], 1)
                merge_bg = self.cls_merge(merge_map)  # [bs, 1, 60, 60]

                sr_final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

                # Output Part
                if self.zoom_factor != 1:
                    sr_meta_out = F.interpolate(sr_meta_out, size=(h, w), mode='bilinear', align_corners=True)
                    sr_base_out = F.interpolate(sr_base_out, size=(h, w), mode='bilinear', align_corners=True)
                    sr_final_out = F.interpolate(sr_final_out, size=(h, w), mode='bilinear', align_corners=True)
                iou = calculate_iou(sr_final_out, s2q_y)
                ious_list.append(iou)
            # Loss
            if self.training:
                pass
            else:
                if if_ft:
                    s2q_miou = sum(ious_list) / len(ious_list)
                    return s2q_miou, sr_final_out, sr_base_out, sr_meta_out
                else:
                    pass

    def mask_feature(self, features, support_mask):  # bchw
        bs = features[0].shape[0]
        initSize = ((features[0].shape[-1]) * 2,) * 2
        support_mask = (support_mask).float()
        support_mask = F.interpolate(support_mask, initSize, mode='bilinear', align_corners=True)
        for idx, feature in enumerate(features):
            feat = []
            if support_mask.shape[-1] != feature.shape[-1]:
                support_mask = F.interpolate(support_mask, feature.size()[2:], mode='bilinear', align_corners=True)
            for i in range(bs):
                featI = feature[i].flatten(start_dim=1)  # c,hw
                maskI = support_mask[i].flatten(start_dim=1)  # hw
                featI = featI * maskI
                maskI = maskI.squeeze()
                meanVal = maskI[maskI > 0].mean()
                realSupI = featI[:, maskI >= meanVal]
                if maskI.sum() == 0:
                    realSupI = torch.zeros(featI.shape[0], 1).cuda()
                feat.append(realSupI)  # [b,]ch,w
            features[idx] = feat  # nfeatures ,bs,ch,w
        return features
