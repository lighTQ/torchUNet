import random
import numpy as np
from PIL import Image

def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]

def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)

def hwc_to_chw(img):
    """
    try:
        arr = np.transpose(img, axes=[2, 0, 1])
        return arr 
    except Exception as e:
        print(Exception)
        print(img.shape)
    """
    return np.transpose(img, axes=[2, 0, 1])

def resize_and_crop(pilimg, scale=0.5, height=512, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    if h >= w:
        rotate_img = pilimg.rotate(90,expand=True)
        w = rotate_img.size[0]
        h = rotate_img.size[1]
        #print('rotate h %s w %s'%(h,w))
    else:
        rotate_img = pilimg
    width = int(height*w/float(h))

    newW = int(width * scale)
    newH = int(height * scale)
    
   # print("newH {}  newW {}".format(newW, newH))

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = rotate_img.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    img_arr = np.array(img, dtype=np.float32)
    #print('resize_and_crop diff %s w %s h %s nwew %s newh %s'%(diff, w, h, newW, newH), img_arr.shape)
    #print(pilimg.format, pilimg.size, pilimg.mode, img_arr.shape)
    return img_arr

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255

def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs
