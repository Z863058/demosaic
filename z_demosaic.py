import os
from options.test_options import TestOptions
from models.models import create_model
import util.util as util
import torch

from z_util import util as z_util
from z_models import loadmodel
from z_cores import core

opt = TestOptions().parse(save=False)
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

opt.loadSize = 256
opt.fineSize = 256
opt.label_nc = 0
opt.no_instance = True

opt.mode = "clean"
opt.use_gpu = True

opt.fps = 0
opt.tempimage_type = 'jpg'
opt.mask_threshold = 96

opt.mosaic_position_model_path = "./z_models/add_youknow.pth"
opt.traditional = False
opt.tr_blur = 10
opt.tr_down = 10
opt.no_feather = False
opt.all_mosaic_area = False
opt.medfilt_num = 11
opt.ex_mult = 1.5

# opt.name = 'demosaic_20200501_286to256_random'
# opt.media_path = 'x.mp4'

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)

    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx

if not os.path.isdir(opt.results_dir):
    os.makedirs(opt.result_dir)
    print('makedir:', opt.results_dir)
z_util.clean_tempfiles(True)


def main():
    if os.path.isdir(opt.media_path):
        files = util.Traversal(opt.media_path)
    else:
        files = [opt.media_path]

    if opt.mode == 'clean':
        netM = loadmodel.bisenet(opt, 'mosaic')
        if opt.traditional:
            netG = None
        else:
            netG = model.netG

        for file in files:
            opt.media_path = file
            if z_util.is_img(file):
                core.cleanmosaic_img(opt, netG, netM)
            elif z_util.is_video(file):
                core.cleanmosaic_video_byframe(opt, netG, netM)
            else:
                print('This type of file is not supported')

    z_util.clean_tempfiles(tmp_init=False)


if __name__ == '__main__':
    main()


