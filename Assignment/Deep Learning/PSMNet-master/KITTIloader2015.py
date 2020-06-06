import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import sys
sys.path.insert(0, '/utils_1/auxiliary/')
import cv2
# from utils_1.auxiliary import generate_mask
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath):
    left_fold = 'image_2/'
    right_fold = 'image_3/'
    disp_L = 'disp_occ_0/'
    disp_R = 'disp_occ_1/'
    left_test = []
    rigth_test = []
    disp_L_test = []

    image = [img for img in os.listdir(filepath + left_fold) if img.find('_10') > -1]

    all_index = np.arange(200)
    # if split_file is None:
    #     np.random.shuffle(all_index)
    #     vallist = all_index[:40]
    # else:
    #     with open(split_file) as f:
    #         vallist = sorted([int(x.strip()) for x in f.readlines() if len(x) > 0])
    # log.info(vallist)
    # val = ['{:06d}_10.png'.format(x) for x in vallist]
    # test = [x for x in image if x not in val]
    for img in image:
        left_path = os.path.join(filepath, left_fold, img)
        right_path = os.path.join(filepath, right_fold, img)
        disp_path = os.path.join(filepath, disp_L, img)
        # im = Image.open(left_path)
        # # im = im.convert('LA')
        # im = color.rgb2gray(im)
        im = cv2.imread(left_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        left_test.append(np.array(im))
        # im = Image.open(right_path)
        # im = color.rgb2gray(im)
        # im = im.convert('LA')
        im = cv2.imread(right_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        rigth_test.append(np.array(im))
        # im = Image.open(disp_path)
        # im = color.rgb2gray(im)
        # im = im.convert('LA')
        im = cv2.imread(disp_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        disp_L_test.append(np.array(im))
    # left_train = [ for img in image]
    # right_train = [os.path.join(filepath, right_fold, img) for img in image]
    # disp_train_L = [os.path.join(filepath, disp_L, img) for img in image]
    # disp_train_R = [filepath+disp_R+img for img in train]

    # left_val = [os.path.join(filepath, left_fold, img) for img in val]
    # right_val = [os.path.join(filepath, right_fold, img) for img in val]
    # disp_val_L = [os.path.join(filepath, disp_L, img) for img in val]
    # disp_val_R = [filepath+disp_R+img for img in val]

    return left_test, rigth_test, disp_L_test


# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count

# def test(output, dispL):
#
#     EPEs = [AverageMeter() for _ in range(len(output))]
#     length_loader = len(output)
#
#     # model.eval()
#
#     for batch_idx, (imgL, imgR, disp_L) in
#
#         mask = disp_L < args.maxdisp
#         with torch.no_grad():
#             outputs = model(imgL, imgR)
#             for x in range(stages):
#                 if len(disp_L[mask]) == 0:
#                     EPEs[x].update(0)
#                     continue
#                 output = torch.squeeze(outputs[x], 1)
#                 output = output[:, 4:, :]
#                 EPEs[x].update((output[mask] - disp_L[mask]).abs().mean())
#
#         info_str = '\t'.join(['Stage {} = {:.2f}({:.2f})'.format(x, EPEs[x].val, EPEs[x].avg) for x in range(stages)])
#
#         log.info('[{}/{}] {}'.format(
#             batch_idx, length_loader, info_str))
#
#     info_str = ', '.join(['Stage {}={:.2f}'.format(x, EPEs[x].avg) for x in range(stages)])
#     log.info('Average test EPE = ' + info_str)


# https://github.com/victoriamazo/depth_regression/blob/master/metrics/depth_metrics.py
def compute_depth_metrics_i(pred_depth, gt_depth, min_depth=1e-3, max_depth=80):
    '''Computes depth metrics for frame i:
         - a1
         - a2
         - a3
         - RMSE
         - RMSE_log
         - Abs_rel
         - Sq_rel
        '''
    pred_depth_zoomed = zoom(pred_depth, (gt_depth.shape[0] / pred_depth.shape[0],
                                          gt_depth.shape[1] / pred_depth.shape[1])).clip(min_depth, max_depth)
    mask = generate_mask(gt_depth)
    pred_depth_zoomed = pred_depth_zoomed[mask]
    gt_depth = gt_depth[mask]

    # scale factor determined by GT/prediction ratio (like the original paper)
    scale_factor = np.median(gt_depth) / np.median(pred_depth_zoomed)
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_errors(gt_depth, pred_depth_zoomed * scale_factor)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3