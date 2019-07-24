#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image
import scipy.misc

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i)  for id in ids for i in range(n))


def to_cropped_imgs(ids, dir, suffix, height, scale):

    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        #print(dir + id + suffix, dir.split('/'))
        img=Image.open(dir + id + suffix)
        if img.mode=='L' and dir.split('/')[-2] == 'train_images':
            print(dir + id + suffix)
            img=img.convert('RGB')
        im = resize_and_crop(img, height=height,scale=scale)
        #print(im.shape, id+suffix)
        yield get_square(im, pos)

def get_imgs_and_masks(ids, dir_img, dir_mask, height, scale):
    """Return all the couples (img, mask)"""

    try:
        imgs = to_cropped_imgs(ids, dir_img, '.jpg', height, scale)

        # need to transform from HWC to CHW
        imgs_switched = map(hwc_to_chw, imgs)
        imgs_normalized = map(normalize, imgs_switched)

        masks = to_cropped_imgs(ids, dir_mask, '.jpg', height, scale)
        masks_normalized = map(normalize, masks)   
    except Exception as e:
        print("Unexpected Error :{}".format(e))
    #masks_normalized = masks
    return zip(imgs_normalized, masks_normalized)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '.jpg')
    return np.array(im), np.array(mask)
