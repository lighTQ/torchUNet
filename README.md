1. nl_pre3.py 为数据预处理脚本
   将heatmap[...,0]的输出之scale到[0,255]输出。

2. Pytorch-UNet 为训练网络
    已修改地方:
       将训练时的优化器从SGD换成了Adam
       添加IOU score：根据dice score
       添加了单机多卡支持

    训练时使用的而参数：

      [  python train.py -l  1e-4  -b 2    -e 20 -s 1 -t 1500 ]

      -e EPOCHS, --epochs=EPOCHS
                            number of epochs
      
      -b BATCHSIZE, --batch-size=BATCHSIZE
                            batch size
      
      -l LR, --learning-rate=LR
                            learning rate
      
      -g, --gpu             use cuda
      
      -c LOAD, --load=LOAD  load file model
      
      -s SCALE, --scale=SCALE   downscaling factor of the images
      
      -t HEIGHT, --height=HEIGHT
                            rescale images to height

                        
