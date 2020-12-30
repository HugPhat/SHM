A pytorch implementation of paper *Semantic Human Matting (ACM Multimedia 2018)*. <br>
upgrade code from [repo](https://github.com/bluestyle97/semantic-human-matting)

**Usage** <br>
* Training: follow the config in [pretrain_tnet.yaml](configs/pretrain_tnet.yaml) <br>
*Explaination*
```
mode: [pretrain_tnet, pretrained_mnet, end_to_end]
cuda: true # use cuda
ngpu: 1
tnet_checkpoint: Tnet18.pth #
checkpoint_name : Tnet18.pth # set None to save by period
savedir : path_to_saving_folder
eval : True # eval or train only
backbone: resnet18 # support resnet18->101
data_root: data/dataset #path to root of dataset folder 
loss: dice # for dice loss or 'ce' for  cross entropy 
batch_size: 10 # number of bactch
lr: 0.0001
weight_decay: 0.001
max_epoch: 50
loss_lambda: 0.01    # (for end to end) trade-off between prediction loss and classification loss
sample_period: 1 # freq of checking result by image in 'sample' folder
checkpoint_period: 1 # freq of saving model
```

* Play with [colab](https://colab.research.google.com/drive/1ah-l6lmEMAJSRx--d97IYmBd0Od_Vmhd?usp=sharing)

* Dataset structure:
```bash
└───dataset
    ├─── input
    ├─── mask
    ├─── trimap
    ├─── train.txt
    └─── val.txt

```

* prepare data

> python .\data\prepare_data.py --help
```bash
usage: prepare_data.py [-h] --savedir SAVEDIR [--pubdir PUBDIR] [--trimapdir]
                       --prefix PREFIX [--vocroot VOCROOT] [--inimg INIMG]
                       [--inalp INALP] [--intri INTRI] [--numbg NUMBG]
                       --trainsplit TRAINSPLIT

gen data folder

optional arguments:
  -h, --help                show this help message and exit
  --savedir SAVEDIR         where to save
  --pubdir PUBDIR           root to public data
  --trimapdir               if set, ignore trimap
  --prefix PREFIX           pre/post fix of mask file in folder
  --vocroot VOCROOT         root to pascal voc 2012
  --inimg INIMG             path to input image folder
  --inalp INALP             path to input alpha folder
  --intri INTRI             path to input trimap folder
  --numbg NUMBG             number of random bg
  --trainsplit TRAINSPLIT   train test split ratio

```

* Dataset: <br>
    * Any public image data (VOC, COCO) combine with [Matting data](http://www.alphamatting.com/datasets.php)
    * With other matting dataset, you **should** use script to *create trimap*