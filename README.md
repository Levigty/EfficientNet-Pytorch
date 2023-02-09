# EfficientNet-Pytorch
A demo for train your own dataset on EfficientNet

Thanks for the >[A PyTorch implementation of EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch), I just simply demonstrate how to train your own dataset based on the EfficientNet-Pytorch.

## Step 1：Prepare your own classification dataset
---
Then the data directory should looks like:   
```
-dataset\
    -model\
    -train\
        -1\
        -2\
        ...
    -test\
        -1\
        -2\
        ...
```

## Step 2: train and test 

```python efficientnet_sample.py```  

```--data-dir``` : (str) Path of ```/dataset``` folder. Default: ```None```

```--num-epochs``` : (int) Number of epochs for training. Default: ```40```

```--batch-size``` : (int) Batch size. Default: ```4```

```--img-size``` : (int) Selected size for image to be resized. Default: ```[1024,1024]```

```--class-num``` : (int) Number of classes in dataset. Default: ```3```

```--weights-loc``` : (str) Path of weights to be loaded. If None, pretrained weights will automatically be downloaded & loaded. Default: ```None``` Example: ```"...//weights.pth//"``` 

```--lr``` : (float) Learning rate. Default: ```0.01```

```--net-name``` : (str) States which efficientnet model will be used. Used for downloading pretrained weights as well.

```--resume-epoch``` : (int) Defines starting epoch. Default: ```0```

```--momentum``` : (float) Sets momentum. Default: ```0.9```

 Example usage: ```python ".\efficientnet_sample.py" --data-dir "D:\\ml_data\\dataset" --num-epochs 80 --batch-size 4 --img-size 896  --class-num 3 --weights-loc "D:\\ML\\efficientnet-b3-birads.pth" --lr 0.01 --net-name "efficientnet-b3" --resume-epoch 40```

The pre-trained model is available on >[release](https://github.com/lukemelas/EfficientNet-PyTorch/releases). 

You can download them under the folder ```eff_weights```.

(3)You can get the final results and the best model on ```dataset/model/```.
