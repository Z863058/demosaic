# Demosaic
Use GAN to remove mosaic.

**Major innovations:**

**1. Automatic video processing;**

**2. The mosaic is generated randomly in the training process.**

<div align="center">
  <img src='./imgs/f_mosaic.jpg' width='400'/>
  <img src='./imgs/f_demosaic.jpg' width='400'/>
</div>

## Getting Started
#### Prerequisites
  - Linux, Mac OS
  - Python 3.5+
  - ffmpeg 3.4.6
  - Pytorch 1.0+
  - NVIDIA GPU (11G memory or larger) + CUDA cuDNN
#### Dependencies
This code depends on opencv-python, torchvision, Matplotlib, dominate, and so on, available via pip install.
#### Clone this repo
```bash
git clone https://github.com/Z863058/demosaic
cd demosaic
```
## Make training datasets
Use addmosaic model to make video datasets(Require addmosaic pre-trained model. This is better for processing video mosaics).

Train addmosaic pre-trained model, please refer to [README_addmosaic.md](./README_addmosaic.md).

```bash
cd z_make_datasets
python z_make_video_dataset.py --datadir x0.mp4 --savedir ../datasets/demosaic
```
## Set training parameters

**Modify the method get_random_parameter(img, mask) in z_util.mosaic.py.**

**Let the method produce the appropriate parameters of mosaic_size, mod and rect_rat to fit the mosaic video.**

## Training
```bash
python z_train.py --dataroot ./datasets/demosaic_20200501_286to256 --name demosaic_20200501_286to256_random --loadSize 286 --fineSize 256 --resize_or_crop crop --label_nc 0 --no_instance --niter 100 --niter_decay 100 --tf_log --gpu_ids 1 --continue_train
```
## Testing
```bash
# for images
python z_test.py --dataroot ./datasets/demosaic_20200501_286to256 --name demosaic_20200501_286to256_random --loadSize 256 --fineSize 256 --label_nc 0 --no_instance --gpu_ids 0
# for video
python z_demosaic.py --name demosaic_20200501_286to256_random --media_path x.mp4
```
## Acknowledgments
This code borrows heavily from [[pix2pixHD]](https://github.com/NVIDIA/pix2pixHD) [[DeepMosaics]](https://github.com/HypoX64/DeepMosaics.git) [[Pytorch-UNet]](https://github.com/milesial/Pytorch-UNet) [[BiSeNet]](https://github.com/ooooverflow/BiSeNet).

## More
更多（不可描述）信息，敬请微信搜索或扫码关注“天天P图AI”公众号。

<div align="center">
  <img src='./imgs/tswt.png' width='400'/>
</div>


