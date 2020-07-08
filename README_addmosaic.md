### Add mosaic dataset
Please generate mask from images which you want to add mosaic(number of images should be above 1000). And then put the images in ```face/origin_image```, and masks in ```face/mask```.<br>
* You can use ```draw_mask.py```to generate them.
```bash
python draw_mask.py --datadir 'dir for your pictures' --savedir ../datasets/draw/face
#Press the left mouse button to draw the mask .  Press 'S' to save mask, 'A' to reduce  brush size, 'D' to increase brush size, 'W' to cancel drawing.
```
* If you want to get images from videos, you can use ```get_image_from_video.py```
```bash
python get_image_from_video.py --datadir 'dir for your videos' --savedir ../datasets/video2image --fps 1
```
## Training
```bash
cd train/add
python train.py --gpu_id 0 --dataset ../../datasets/draw/face --savename face --loadsize 512 --finesize 360 --batchsize 16
