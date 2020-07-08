import os
import sys

sys.path.append("..")
from z_cores.options import Options
from z_util import util, ffmpeg

opt = Options()
opt.parser.add_argument('--datadir', type=str, default='', help='your video dir')
opt.parser.add_argument('--savedir', type=str, default='../datasets/demosaic/video2images', help='')
opt = opt.getparse()

if os.path.isfile(opt.datadir):
    files = [opt.datadir]
else:
    files = util.Traversal(opt.datadir)
videos = util.is_videos(files)

util.makedirs(opt.savedir)
for video in videos:
    ffmpeg.continuous_screenshot(video, opt.savedir, opt.fps)
