import os
import datetime
import random
import time
import cv2
import numpy as np
import logging
import argparse
import math
import copy
from visdom import Visdom
import os.path as osp
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.distributed as dist

from PIL import Image

from tensorboardX import SummaryWriter
from model import EFTNet

from utils import dataset, transform_tri_test
from utils import transform_test, config
from utils.util import AverageMeter, intersectionAndUnionGPU, get_model_para_number, setup_seed, \
    get_logger, check_makedirs, ft_learning_rate

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='EFTNet')
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='config/pascal/pascal_split0_vgg.yaml',
                        help='config file')  # coco/coco_split0_resnet50.yaml
    parser.add_argument('--opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_model(args):
    model = eval(args.arch).OneModel(args, cls_type='Base')
    optimizer = model.get_optim(model, args, LR=args.base_lr)

    if hasattr(model, 'freeze_modules'):
        model.freeze_modules(model)

    model = model.cuda()

    # Resume
    backbone_str = 'vgg' if args.vgg else 'resnet' + str(args.layers)
    args.snapshot_path = 'weights/{}/split{}/{}'.format(args.data_set, args.split, backbone_str)
    exp_dir = 'exp_ft_EFT/{}/{}/split{}/{}'.format(args.data_set, args.arch, args.split, backbone_str)
    args.result_path = os.path.join(exp_dir, 'result')
    args.show_path = os.path.join(exp_dir, 'show', args.show)
    args.show_ft_path = os.path.join(exp_dir, 'show_ft', args.show_ft)
    # print(args.snapshot_path)
    # print(args.result_path)
    # print(args.show_path)
    # print(args.show_ft_path)

    # Show path
    if args.show:
        check_makedirs(args.show_path)
    if args.show_ft:
        check_makedirs(args.show_ft_path)
        check_makedirs(os.path.join(args.show_ft_path, 'normal'))
        check_makedirs(os.path.join(args.show_ft_path, 'best'))

    if args.weight:
        weight_path = osp.join(args.snapshot_path, args.weight)
        if os.path.isfile(weight_path):
            logger.info("=> loading checkpoint '{}'".format(weight_path))
            checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try:
                model.load_state_dict(new_param)
            except RuntimeError:  # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    if key[7:10] == 'sr_':
                        new_param.pop(key)
                    else:
                        new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(weight_path, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(weight_path))

    # Get model para.
    total_number, learnable_number = get_model_para_number(model)
    print('Number of Parameters: %d' % (total_number))
    print('Number of Learnable Parameters: %d' % (learnable_number))

    time.sleep(5)
    return model, optimizer


def main():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    args.distributed = True if torch.cuda.device_count() > 1 else False
    print(args)
    val_manual_seed = args.manual_seed
    val_num = 5
    setup_seed(val_manual_seed, False)
    seed_array = np.random.randint(0, 1000, val_num)  # seed->[0,999]
    seed_array = np.append(seed_array, 321)
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0

    logger.info("=> creating model ...")
    model, optimizer = get_model(args)
    # logger.info(model)

    # ----------------------  DATASET  ----------------------
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    # Val
    if args.evaluate:
        if args.resized_val:
            val_transform = transform_test.Compose([
                transform_test.Resize(size=args.val_size),
                transform_test.ToTensor(),
                transform_test.Normalize(mean=mean, std=std)])
            val_transform_tri = transform_tri_test.Compose([
                transform_tri_test.Resize(size=args.val_size),
                transform_tri_test.ToTensor(),
                transform_tri_test.Normalize(mean=mean, std=std)])
        else:
            val_transform = transform_test.Compose([
                transform_test.test_Resize(size=args.val_size),
                transform_test.ToTensor(),
                transform_test.Normalize(mean=mean, std=std)])
            val_transform_tri = transform_tri_test.Compose([
                transform_tri_test.test_Resize(size=args.val_size),
                transform_tri_test.ToTensor(),
                transform_tri_test.Normalize(mean=mean, std=std)])
        if args.data_set == 'pascal' or args.data_set == 'coco':
            val_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root,
                                       base_data_root=args.base_data_root, data_list=args.val_list,
                                       transform=val_transform, transform_tri=val_transform_tri, mode='val',
                                       data_set=args.data_set, use_split_coco=args.use_split_coco)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                 num_workers=args.workers, pin_memory=False, sampler=None)

    # ----------------------  VAL  ----------------------
    start_time = time.time()
    FBIoU_array = np.zeros(val_num+1)
    FBIoU_array_m = np.zeros(val_num+1)
    mIoU_array = np.zeros(val_num+1)
    mIoU_array_m = np.zeros(val_num+1)
    pIoU_array = np.zeros(val_num+1)
    for val_id in range(val_num+1):
        val_seed = seed_array[val_id]
        print('Val: [{}/{}] \t Seed: {}'.format(val_id + 1, val_num+1, val_seed))
        fb_iou, fb_iou_m, miou, miou_m, miou_b, piou = validate(val_loader, model, val_seed, val_id)
        FBIoU_array[val_id], FBIoU_array_m[val_id], mIoU_array[val_id], mIoU_array_m[val_id], pIoU_array[val_id] = \
            fb_iou, fb_iou_m, miou, miou_m, piou

    total_time = time.time() - start_time
    t_m, t_s = divmod(total_time, 60)
    t_h, t_m = divmod(t_m, 60)
    total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))

    print('\nTotal running time: {}'.format(total_time))
    print('Seed0: {}'.format(val_manual_seed))
    print('Seed:  {}'.format(seed_array))
    print('mIoU:  {}'.format(np.round(mIoU_array, 4)))
    print('mIoU_m:  {}'.format(np.round(mIoU_array_m, 4)))
    print('FBIoU: {}'.format(np.round(FBIoU_array, 4)))
    print('FBIoU_m: {}'.format(np.round(FBIoU_array_m, 4)))
    print('pIoU:  {}'.format(np.round(pIoU_array, 4)))
    print('-' * 43)
    print('Best_Seed_m: {} \t Best_Seed_F: {} \t Best_Seed_p: {}'.format(seed_array[mIoU_array.argmax()],
                                                                         seed_array[FBIoU_array.argmax()],
                                                                         seed_array[pIoU_array.argmax()]))
    print(
        'Best_mIoU: {:.4f} \t Best_mIoU_m: {:.4f} \t Best_FBIoU: {:.4f} \t Best_FBIoU_m: {:.4f} \t Best_pIoU: {:.4f}'.format(
            mIoU_array.max(), mIoU_array_m.max(), FBIoU_array.max(), FBIoU_array_m.max(), pIoU_array.max()))
    print(
        'Mean_mIoU: {:.4f} \t Mean_mIoU_m: {:.4f} \t Mean_FBIoU: {:.4f} \t Mean_FBIoU_m: {:.4f} \t Mean_pIoU: {:.4f}'.format(
            mIoU_array.mean(), mIoU_array_m.mean(), FBIoU_array.mean(), FBIoU_array_m.mean(), pIoU_array.mean()))


def validate(val_loader, model, val_seed, val_id):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    intersection_meter = AverageMeter()  # final
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    intersection_meter_m = AverageMeter()  # meta
    union_meter_m = AverageMeter()
    target_meter_m = AverageMeter()

    if args.data_set == 'pascal':
        test_num = 1000
        split_gap = 5
    elif args.data_set == 'coco':
        test_num = 1000
        split_gap = 20

    class_intersection_meter = [0] * split_gap  # [0, 0, 0, 0, 0]
    class_union_meter = [0] * split_gap
    class_intersection_meter_m = [0] * split_gap
    class_union_meter_m = [0] * split_gap
    class_intersection_meter_b = [0] * split_gap * 3
    class_union_meter_b = [0] * split_gap * 3
    class_target_meter_b = [0] * split_gap * 3

    setup_seed(val_seed, args.seed_deterministic)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    end = time.time()
    val_start = end  #

    assert test_num % args.batch_size_val == 0
    db_epoch = math.ceil(test_num / (len(val_loader) * args.batch_size_val))
    iter_num = 0

    for e in range(db_epoch):
        for i, (input, target, target_b, s_input, s_mask, subcls, ori_label, ori_label_b, black_record) in enumerate(
                val_loader):
            if iter_num * args.batch_size_val >= test_num:
                break

            iter_num += 1
            data_time.update(time.time() - end)

            s_input = s_input.cuda(non_blocking=True)
            s_mask = s_mask.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            target_b = target_b.cuda(non_blocking=True)
            ori_label = ori_label.cuda(non_blocking=True)
            ori_label_b = ori_label_b.cuda(non_blocking=True)
            black_record = black_record.cuda(non_blocking=True)
            start_time = time.time()
            if args.loop_ft:
                model_ft = copy.deepcopy(model)

                output, meta_out, base_out, iou = finetune(model_ft, input,
                                                           target,
                                                           target_b,
                                                           s_input,
                                                           s_mask,
                                                           subcls,
                                                           black_record,
                                                           ft_num=args.ft_number)

            if not args.loop_ft:
                model.eval()

                output, meta_out, base_out = model(s_x=s_input,
                                                   s_y=s_mask,
                                                   x=input, y_m=target,
                                                   y_b=target_b,
                                                   train_round=0,
                                                   black_record=black_record,
                                                   cat_idx=subcls)

            model_time.update(time.time() - start_time)
            if args.ori_resize:
                longerside = max(ori_label.size(1), ori_label.size(2))
                backmask = torch.ones(ori_label.size(0), longerside, longerside, device='cuda') * 255
                backmask_b = torch.ones(ori_label.size(0), longerside, longerside, device='cuda') * 255
                backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
                backmask_b[0, :ori_label.size(1), :ori_label.size(2)] = ori_label_b
                target = backmask.clone().long()
                target_b = backmask_b.clone().long()
            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
            meta_out = F.interpolate(meta_out, size=target.size()[1:], mode='bilinear', align_corners=True)
            base_out = F.interpolate(base_out, size=target.size()[1:], mode='bilinear', align_corners=True)

            loss = criterion(output, target)  # loss

            output = output.max(1)[1]
            meta_out = meta_out.max(1)[1]
            base_out = base_out.max(1)[1]

            subcls = subcls[0].cpu().numpy()[0]

            intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)
            class_intersection_meter[subcls] += intersection[1]
            class_union_meter[subcls] += union[1]

            intersection, union, new_target = intersectionAndUnionGPU(meta_out, target, args.classes, args.ignore_label)
            intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter_m.update(intersection), union_meter_m.update(union), target_meter_m.update(new_target)
            class_intersection_meter_m[subcls] += intersection[1]
            class_union_meter_m[subcls] += union[1]

            intersection, union, new_target = intersectionAndUnionGPU(base_out, target_b, split_gap * 3 + 1,
                                                                      args.ignore_label)
            intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
            for idx in range(1, len(intersection)):
                class_intersection_meter_b[idx - 1] += intersection[idx]
                class_union_meter_b[idx - 1] += union[idx]
                class_target_meter_b[idx - 1] += new_target[idx]

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            remain_iter = test_num / args.batch_size_val - iter_num
            remain_time = remain_iter * batch_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

            if (i + 1) % round((test_num / 100)) == 0:
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(iter_num * args.batch_size_val, test_num,
                                                              data_time=data_time,
                                                              batch_time=batch_time,
                                                              remain_time=remain_time,
                                                              loss_meter=loss_meter,
                                                              accuracy=accuracy))
    val_time = time.time() - val_start

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    iou_class_m = intersection_meter_m.sum / (union_meter_m.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mIoU_m = np.mean(iou_class_m)

    class_iou_class = []
    class_iou_class_m = []
    class_iou_class_b = []
    class_miou = 0
    class_miou_m = 0
    class_miou_s2q = 0
    class_miou_b = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i] / (class_union_meter[i] + 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
        class_iou = class_intersection_meter_m[i] / (class_union_meter_m[i] + 1e-10)
        class_iou_class_m.append(class_iou)
        class_miou_m += class_iou
        class_iou_class_m.append(class_iou)
        class_miou_s2q += class_iou

    for i in range(len(class_intersection_meter_b)):
        class_iou = class_intersection_meter_b[i] / (class_union_meter_b[i] + 1e-10)
        class_iou_class_b.append(class_iou)
        class_miou_b += class_iou

    target_b = np.array(class_target_meter_b)

    class_miou = class_miou * 1.0 / len(class_intersection_meter)
    class_miou_m = class_miou_m * 1.0 / len(class_intersection_meter)
    class_miou_b = class_miou_b * 1.0 / (
            len(class_intersection_meter_b) - len(target_b[target_b == 0]))  # filter the results with GT mIoU=0

    logger.info('meanIoU---Val result: mIoU_f {:.4f}.'.format(class_miou))  # final
    logger.info('meanIoU---Val result: mIoU_m {:.4f}.'.format(class_miou_m))  # meta
    logger.info('meanIoU---Val result: mIoU_b {:.4f}.'.format(class_miou_b))  # base

    logger.info('<<<<<<< Novel Results <<<<<<<')
    for i in range(split_gap):
        logger.info('Class_{} Result: iou_f {:.4f}.'.format(i + 1, class_iou_class[i]))
        logger.info('Class_{} Result: iou_m {:.4f}.'.format(i + 1, class_iou_class_m[i]))
    logger.info('<<<<<<< Base Results <<<<<<<')
    for i in range(split_gap * 3):
        if class_target_meter_b[i] == 0:
            logger.info('Class_{} Result: iou_b None.'.format(i + 1 + split_gap))
        else:
            logger.info('Class_{} Result: iou_b {:.4f}.'.format(i + 1 + split_gap, class_iou_class_b[i]))

    logger.info('FBIoU---Val result: FBIoU_f {:.4f}.'.format(mIoU))
    logger.info('FBIoU---Val result: FBIoU_m {:.4f}.'.format(mIoU_m))
    for i in range(args.classes):
        logger.info('Class_{} Result: iou_f {:.4f}.'.format(i, iou_class[i]))
        logger.info('Class_{} Result: iou_m {:.4f}.'.format(i, iou_class_m[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    print('total time: {:.4f}, avg inference time: {:.4f}, count: {}'.format(val_time, model_time.avg, test_num))

    return mIoU, mIoU_m, class_miou, class_miou_m, class_miou_b, iou_class[1]


def finetune(model, input, target, target_b, s_input, s_mask, subcls, black_record, ft_num):
    if hasattr(model, 'freeze_modules'):
        model.freeze_modules(model)

    opt = model.get_optim(model, args, LR=args.ft_base_lr)

    model.eval()
    # zero round
    fr_output, fr_meta_out, fr_base_out, main_loss, aux_loss1 = model(s_x=s_input,
                                                                      s_y=s_mask,
                                                                      x=input, y_m=target,
                                                                      y_b=target_b,
                                                                      train_round=0,
                                                                      black_record=black_record,
                                                                      cat_idx=subcls,
                                                                      if_ft=True)

    best_iou = 0
    best_final_out = fr_output
    # fine-tuning round
    for i in range(ft_num):
        ft_learning_rate(opt, args.ft_base_lr, ft_num, i, power=args.ft_power)
        iou2, s2q_final_out, s2q_base_out, s2q_meta_out = model(s_x=s_input, s_y=s_mask,
                                                                x=input, y_m=target, y_b=target_b,
                                                                train_round=i + 1,
                                                                cat_idx=subcls,
                                                                black_record=black_record,
                                                                if_ft=True,
                                                                ft_mod='s2q')

        output, meta_out, base_out, sr_pseudo_label, sr_main_loss, sr_aux_loss1, _ = model(s_x=s_input, s_y=s_mask,
                                                                                           x=input, y_m=target, y_b=target_b,
                                                                                           train_round=i + 1,
                                                                                           cat_idx=subcls,
                                                                                           black_record=black_record,
                                                                                           if_ft=True,
                                                                                           fr_output=fr_output)

        loss2 = sr_main_loss + args.sr_aux_weight1 * sr_aux_loss1
        opt.zero_grad()
        loss2.backward()
        opt.step()

        # Discrimination
        if iou2 >= best_iou:
            best_iou = iou2
            best_final_out = output
            best_meta_out = meta_out
            best_base_out = base_out
        fr_output = output

    return best_final_out, best_meta_out, best_base_out, best_iou


if __name__ == '__main__':
    main()
