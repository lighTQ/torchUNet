import cv2
import time
import numpy as np
import datetime
import json
import os
import  cv2 as cv
from multiprocessing import Pool
import pycocotools.mask as maskUtils
from PIL import Image

def annToMask(ann, h=2048, w=2048):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to b
inary mask.
    :return: binary mask (numpy 2D array)
    """
    rles = maskUtils.frPyObjects(ann, h, w)
    rle = maskUtils.merge(rles)
    m = maskUtils.decode(rle)
    return m


def instance_vector_field_2(anno, h=None, w=None, vf=False):
    """

    :param anno:
    :param h:
    :param w:
    :param vf:
    :return: Float32, Int8, Float32, Float32
    """
    # 不同标注改一下
    heatmap = annToMask(anno,h,w).astype(np.uint8)

    boundary = np.zeros((h, w), dtype=np.uint8).copy()
    interior = np.zeros((h, w), dtype=np.float32).copy()
    ps = np.array(anno).squeeze().reshape(-1, 2).astype(np.int32)
    boundary = cv2.polylines(boundary, [ps], True, 1, 2, cv2.LINE_4)
    interior = cv2.fillPoly(interior, [ps], 1.0, cv2.LINE_4)


    ctrs = np.nonzero(boundary)
    # x, y
    p = np.nonzero(heatmap)
    _c = np.array((np.mean(p[1]), np.mean(p[0])), dtype=np.int)

    lines_heatmap = np.zeros((h, w), dtype=np.float32)

    for i, (y, x) in enumerate(zip(*ctrs)):
        crop_h, crop_w = abs(_c[1] - y) + 7, abs(_c[0] - x) + 7
        min_h, min_w = min(y, _c[1]), min(x, _c[0])
        if min_h - 3 < 0 or min_w - 3 < 0 or min_w + crop_w - 3 > w or min_h + crop_h - 3 > h:
            line_heatmap = np.zeros((h, w), dtype=np.uint8)
            cv2.line(line_heatmap, (x, y), tuple(_c), (1,), (1))
            lines_heatmap = np.where(lines_heatmap < line_heatmap, np.sqrt((y - _c[1]) ** 2 + (x - _c[0]) ** 2),
                                     lines_heatmap)
        else:
            line_heatmap = np.zeros((crop_h, crop_w), dtype=np.uint8)
            cv2.line(line_heatmap, (x - min_w + 3, y - min_h + 3), (_c[0] - min_w + 3, _c[1] - min_h + 3), (1,), (1))
            lines_heatmap[min_h - 3: min_h + crop_h - 3, min_w - 3: min_w + crop_w - 3] = np.where(
                lines_heatmap[min_h - 3: min_h + crop_h - 3, min_w - 3: min_w + crop_w - 3] < line_heatmap,
                np.sqrt((y - _c[1]) ** 2 + (x - _c[0]) ** 2),
                lines_heatmap[min_h - 3: min_h + crop_h - 3, min_w - 3: min_w + crop_w - 3])

    if vf:
        xs, ys = np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)
        for y, x in zip(*np.nonzero(lines_heatmap)):
            _h = (np.sqrt((y - _c[1]) ** 2 + (x - _c[0]) ** 2)) / lines_heatmap[y, x]
            lines_heatmap[y, x] = _h
            xs[y, x] = abs(x - _c[0]) * _h * interior[y, x]
            ys[y, x] = abs(y - _c[1]) * _h * interior[y, x]

            return [lines_heatmap, boundary], [xs, ys]
    else:
        for y, x in zip(*np.nonzero(lines_heatmap)):
            _h = (np.sqrt((y - _c[1]) ** 2 + (x - _c[0]) ** 2)) / lines_heatmap[y, x]
            lines_heatmap[y, x] = _h * interior[y, x]
        return lines_heatmap, boundary


def RotateClockWise90(img):
    trans_img = cv.transpose(img)
    new_img = cv.flip(trans_img, 1)
    return new_img



"""
使用时只需要修改下面的四个路径即可
"""
gt_file = '/root/ctw/CWT/annos/val.jsonl'
image_path = '/root/ctw/CWT/images-trainval/'

dest_train_images = './val_images_preprocess_test/'
dest_train_masks = './val_images_mask_preprocess_test/'

if not os.path.exists(dest_train_images):
    os.makedirs(dest_train_images)

if not os.path.exists(dest_train_masks):
    os.makedirs(dest_train_masks)



def singleMask(labelContent):
    try:
        s1=time.time()
        img_name = labelContent['file_name']
        img_file = os.path.join(image_path, img_name)
        if os.path.exists(img_file):
            img = Image.open(img_file)
            if img.mode!='RGB':
                img = img.convert('RGB')
            img = np.array(img)
            # print(img)
            print('%s' % datetime.datetime.now(), img_name, img.shape)
            h = img.shape[0]
            w = img.shape[1]
            h1 = h+180
            w1 = w+180
            # mask = np.zeros((h0, w0), dtype=np.uint8)
            annos = labelContent['annotations']
            heatmap = np.zeros((h1, w1), dtype=np.float32)
            for j in range(len(annos)):
                for i in annos[j]:
                    points = i['polygon']
                    anno = np.reshape(points, [1, -1]).tolist()
                    result = instance_vector_field_2(anno,  h1, w1, False)
#                    print(" instance_vector_field_2 Func cost time {}".format(t2-t1))
                    heatmap = np.maximum(heatmap, result[0])
#                    heatmap[..., 1] = np.where(result[1].astype(np.float32) == 1.0, 1.0,
#                                               heatmap[..., 1])
            heatmap=cv.normalize(heatmap,None,255,0,cv.NORM_MINMAX,cv.CV_8UC1)
            heatmap=cv.resize(heatmap,(h,w),cv2.INTER_CUBIC)
            mask_file = os.path.join(dest_train_masks, '%s' % (img_name))
            cv.imwrite(mask_file, heatmap)
            wrap_default = cv2.copyMakeBorder(img,80,80,80,80,cv2.BORDER_DEFAULT)
            img=cv.resize(img,(h,w),cv2.INTER_CUBIC)
            train_file = os.path.join(dest_train_images, '%s' % (img_name))
            cv.imwrite(train_file, img)
            print("deal {} cost time {} s".format(img_name,(time.time()-s1)))
        else:
            print("path not exists!")
    except Exception as e:
        print("Unexpected Error {}".format(e))

if __name__ == '__main__':
    
    lines = [json.loads(line) for line in open(gt_file, 'r')]
    with Pool(100) as p:
        p.map(singleMask, (lines))





