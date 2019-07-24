import sys
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch

import datetime

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_height=512,
              img_scale=0.5):

    #dir_img = 'carvana-image-masking-challenge/train/'
    #dir_mask = 'carvana-image-masking-challenge/train_masks/'

    dir_img = '/root/ctw/train_images_preprocess_other/'
    dir_mask = '/root/ctw/train_images_mask_preprocess_other/'


    #dir_img = '/root/ctw/val_images_preprocess_test/'
    #dir_mask = '/root/ctw/val_images_mask_preprocess_test/'
    dir_checkpoint = 'checkpoints/'

    ids = list(get_ids(dir_img))
    ids = split_ids(ids)

    iddataset = split_train_val(ids, val_percent)


    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])

    optimizer = optim.Adam(net.parameters(),lr=lr)

   # optimizer = optim.SGD(net.parameters(),
   #                       lr=lr,
   #                       momentum=0.92,
   #                       weight_decay=0.0005)

    criterion = nn.BCELoss()
    #criterion = nn.MSELoss()
 
    #import scipy.misc
    iteration = 0
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_height, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_height, img_scale)

        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
            #print(i, len(b))
            """
            for j in b:
                #print(j[0].shape, j[1].shape)
                #print(j[1])
                #scipy.misc.toimage(j[0], cmin=0.0, cmax=1.0).save('%s_outfile.jpg'%count)
                #scipy.misc.toimage(j[1], cmin=0.0, cmax=1.0).save('%s_outmask.jpg'%count)
                count += 1
            """
            iteration += 1 
            try:            
                imgs = np.array([i[0] for i in b]).astype(np.float32)
                true_masks = np.array([i[1] for i in b])
              
    #            print("\nImgs :  \n{}".format(np.unique(imgs)))
    #            print("\ntrue mask \n {} ".format(np.unique(true_masks)))
            #print('%s'%(datetime.datetime.now()), '{0:.4f}'.format(i * batch_size))
            except Exception as e:
                print(Exception)
                continue

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()
            if iteration%100==0:
                print('iter %s'%iteration, '%s'%(datetime.datetime.now()), '{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        if 1:
            val_dice = eval_net(net, val, gpu)
            val_iou=val_dice/(2-val_dice)
            print('Validation Dice Coeff: {}'.format(val_dice))
            print('Validation iouScore : {}'.format(val_iou))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    parser.add_option('-t', '--height', dest='height', type='int',
                      default=1024, help='rescale images to height')


    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    
    print("Let's record it \n")
    args = get_args()

    net = UNet(n_channels=3, n_classes=1)
    net = torch.nn.DataParallel(net)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_height=args.height,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED2.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

