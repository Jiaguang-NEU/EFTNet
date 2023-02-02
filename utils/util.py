import os
import numpy as np
from PIL import Image
import random
import logging
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from matplotlib import font_manager
from matplotlib import rcParams
import seaborn as sns
import pandas as pd
import math
from seaborn.distributions import distplot
from tqdm import tqdm
from scipy import ndimage
from utils.get_weak_anns import find_bbox

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.init as initer


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


def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9, index_split=-1, scale_lr=10., warmup=False,
                       warmup_step=500):
    """poly learning rate policy"""
    if warmup and curr_iter < warmup_step:
        lr = base_lr * (0.1 + 0.9 * (curr_iter / warmup_step))
    else:
        lr = base_lr * (1 - float(curr_iter) / max_iter) ** power

    # if curr_iter % 50 == 0:   
    #     print('Base LR: {:.4f}, Curr LR: {:.4f}, Warmup: {}.'.format(base_lr, lr, (warmup and curr_iter < warmup_step)))     

    for index, param_group in enumerate(optimizer.param_groups):
        if index <= index_split:
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr * scale_lr  # 10x LR


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)


def init_weights(model, conv='xavier', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if conv == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m,
                        (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):  # , BatchNorm1d, BatchNorm2d, BatchNorm3d)):
            if batchnorm == 'normal':
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant_(param, 0)


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


# ------------------------------------------------------
def get_model_para_number(model):
    total_number = 0
    learnable_number = 0
    for para in model.parameters():
        total_number += torch.numel(para)
        if para.requires_grad == True:
            learnable_number += torch.numel(para)
    return total_number, learnable_number


def setup_seed(seed=2021, deterministic=False):
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def get_save_path(args):
    backbone_str = 'vgg' if args.vgg else 'resnet' + str(args.layers)
    args.backbone_path = 'exp_new/{}/{}/split{}/{}'.format(args.data_set, args.arch, args.split, backbone_str)
    checkpoint_path = args.time
    args.time_path = os.path.join(args.backbone_path, checkpoint_path)
    args.snapshot_path = os.path.join(args.time_path, 'snapshot')
    args.result_path = os.path.join(args.time_path, 'result')
    args.show_path = os.path.join(args.time_path, 'show', args.show)
    args.show_ft_path = os.path.join(args.time_path, 'show_ft', args.show_ft)


def get_train_val_set(args):
    if args.data_set == 'pascal':
        class_list = list(range(1, 21))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        if args.split == 3:
            sub_list = list(range(1, 16))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            sub_val_list = list(range(16, 21))  # [16,17,18,19,20]
        elif args.split == 2:
            sub_list = list(range(1, 11)) + list(range(16, 21))  # [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
            sub_val_list = list(range(11, 16))  # [11,12,13,14,15]
        elif args.split == 1:
            sub_list = list(range(1, 6)) + list(range(11, 21))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
            sub_val_list = list(range(6, 11))  # [6,7,8,9,10]
        elif args.split == 0:
            sub_list = list(range(6, 21))  # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            sub_val_list = list(range(1, 6))  # [1,2,3,4,5]

    elif args.data_set == 'coco':
        if args.use_split_coco:
            print('INFO: using SPLIT COCO (FWB)')
            class_list = list(range(1, 81))
            if args.split == 3:
                sub_val_list = list(range(4, 81, 4))
                sub_list = list(set(class_list) - set(sub_val_list))
            elif args.split == 2:
                sub_val_list = list(range(3, 80, 4))
                sub_list = list(set(class_list) - set(sub_val_list))
            elif args.split == 1:
                sub_val_list = list(range(2, 79, 4))
                sub_list = list(set(class_list) - set(sub_val_list))
            elif args.split == 0:
                sub_val_list = list(range(1, 78, 4))
                sub_list = list(set(class_list) - set(sub_val_list))
        else:
            print('INFO: using COCO (PANet)')
            class_list = list(range(1, 81))
            if args.split == 3:
                sub_list = list(range(1, 61))
                sub_val_list = list(range(61, 81))
            elif args.split == 2:
                sub_list = list(range(1, 41)) + list(range(61, 81))
                sub_val_list = list(range(41, 61))
            elif args.split == 1:
                sub_list = list(range(1, 21)) + list(range(41, 81))
                sub_val_list = list(range(21, 41))
            elif args.split == 0:
                sub_list = list(range(21, 81))
                sub_val_list = list(range(1, 21))

    return sub_list, sub_val_list


def is_same_model(model1, model2):
    flag = 0
    count = 0
    for k, v in model1.state_dict().items():
        model1_val = v
        model2_val = model2.state_dict()[k]
        if (model1_val == model2_val).all():
            pass
        else:
            flag += 1
            print('value of key <{}> mismatch'.format(k))
        count += 1

    return True if flag == 0 else False


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def sum_list(list):
    sum = 0
    for item in list:
        sum += item
    return sum


def ft_learning_rate(optimizer, base_lr, ft_num, ft_now, power=1):
    """poly learning rate policy"""
    lr = base_lr * (1 - ft_now / (ft_num + 1)) ** power + 1e-10
    # if curr_iter % 50 == 0:
    #     print('Base LR: {:.4f}, Curr LR: {:.4f}, Warmup: {}.'.format(base_lr, lr, (warmup and curr_iter < warmup_step)))

    for index, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr


def show_img(base, label, mode, path, color):
    """
    :param color:
    :param base: base img[bs, 3, h, w]
    :param label: label[1, h, w]
    :param mode: 'cat' 'sin' 'mask'
    :param path: save path
    :return:
    """
    assert len(base.shape) == 4, len(label.shape) == 3
    if mode == 'cat':
        assert base.shape[-2:] == label.shape[-2:]
        base_show = base.clone()
        base_show = ((base_show - torch.min(base_show)) / (torch.max(base_show) - torch.min(base_show)) * 255).type(torch.int)
        input_np = base_show[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        input_img = Image.fromarray(np.uint8(input_np))
        target_tensor = label.clone()
        targetr = torch.zeros_like(target_tensor)
        targetg = torch.zeros_like(target_tensor)
        targetb = torch.zeros_like(target_tensor)
        if color == 'r':
            targetr[target_tensor >= 1] = 255
            targetr[target_tensor == 255] = 0
        if color == 'g':
            targetg[target_tensor >= 1] = 255
            targetg[target_tensor == 255] = 0
        if color == 'b':
            targetb[target_tensor >= 1] = 255
            targetb[target_tensor == 255] = 0
        target_tensor = torch.cat([targetr, targetg, targetb], dim=0)
        target_np = target_tensor.permute(1, 2, 0).detach().cpu().numpy()
        target_img = Image.fromarray(np.uint8(target_np))
        mask_img_show = Image.blend(input_img.convert('RGBA'), target_img.convert('RGBA'), 0.3)
        plt.imshow(mask_img_show)
        plt.savefig(path)
    if mode == 'sin':
        base_show = ((base - torch.min(base)) / (
                torch.max(base) - torch.min(base)) * 255).type(torch.int)
        input_np = base_show[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        input_img = Image.fromarray(np.uint8(input_np))
        plt.imshow(input_img)
        plt.savefig(path)
    if mode == 'mask':
        target_tensor = label.clone()
        targetr = torch.zeros_like(target_tensor)
        targetg = torch.zeros_like(target_tensor)
        targetb = torch.zeros_like(target_tensor)
        targetr[target_tensor >= 1] = 255
        targetr[target_tensor == 255] = 0
        targetg[target_tensor >= 1] = 255
        targetg[target_tensor == 255] = 0
        targetb[target_tensor >= 1] = 255
        targetb[target_tensor == 255] = 0
        target_tensor = torch.cat([targetr, targetg, targetb], dim=0)
        print(target_tensor.shape)
        target_np = target_tensor.permute(1, 2, 0).detach().cpu().numpy()
        target_img = Image.fromarray(np.uint8(target_np))
        plt.imshow(target_img)
        plt.savefig(path)


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x, s_x, s_y, y_m, y_b, train_round, black_record, if_ft):
        self.gradients = []
        self.activations = []
        return self.model(x, s_x, s_y, y_m, y_b, train_round, black_record, if_ft)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(final_out, meta_out, y_m):
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        main_loss = criterion(final_out, y_m.long())
        aux_loss1 = criterion(meta_out, y_m.long())
        loss = main_loss + 0.9 * aux_loss1
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, x, s_x, s_y, y_m, y_b, train_round, black_record, if_ft):

        final_out, meta_out, base_out = self.activations_and_grads(x, s_x, s_y, y_m, y_b, train_round, black_record,
                                                                   if_ft=if_ft)

        self.model.zero_grad()
        loss = self.get_loss(final_out, meta_out, y_m)
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(x)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray,mask: np.ndarray,use_rgb: bool = False,colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def show_cam_img(model, target_layers, s_input, s_mask, input, target, target_b, train_round, black_record):
    # load image
    base_show = input.clone()
    base_show = ((base_show - torch.min(base_show)) / (torch.max(base_show) - torch.min(base_show)) * 255).type(torch.int)
    input_np = base_show[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()

    # [N, C, H, W]
    input_tensor = input.clone()

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    grayscale_cam = cam(input_tensor, s_input, s_mask, target, target_b, train_round, black_record, if_ft=True)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(input_np.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)

    return visualization
