import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_filter", default="./pretrained_weights/neural_filter.pth",type=str, help="the ckpt of neural filter network")
parser.add_argument("--ckpt_local", default="./pretrained_weights/local_refinement_net.pth", type=str, help="the ckpt of local refinement network")
parser.add_argument("--video_name", default=None, type=str, help="the name of input video")
parser.add_argument("--video_frame_folder", default=None, type=str, help="the name of input video frame folders")
parser.add_argument("--fps", default=10, type=int, help="frame per second")
parser.add_argument('--gpu',             type=int,     default=0,                help='gpu device id')


# process arguments
opts = parser.parse_args()
print(opts)

### Preprocessing
if opts.video_name is not None:
    video_base_name = os.path.basename(opts.video_name)[:-4]
    input_video_frames_folder = "./data/test/{}".format(video_base_name)
    os.makedirs(input_video_frames_folder, exist_ok=True)
    video_to_frames_cmd = "ffmpeg -i {} -vf fps={} -start_number 0 {}/%05d.png".format(opts.video_name, opts.fps, input_video_frames_folder)
    print(video_to_frames_cmd)
    os.system(video_to_frames_cmd)
else:
    video_base_name = os.path.basename(opts.video_frame_folder)
    input_video_frames_folder = "./data/test/{}".format(video_base_name)    
    if os.path.isdir(input_video_frames_folder):
        print("input folder {} exist".format(input_video_frames_folder))
    else:
        cmd = "mv {} {}".format(video_base_name, input_video_frames_folder)
        print(cmd)
        os.system(cmd)

atlas_generation_cmd = "python src/stage1_neural_atlas.py --vid_name {}".format(video_base_name)
os.system(atlas_generation_cmd)

neural_filter_and_refinement_cmd = "python src/neural_filter_and_refinement.py --video_name {}".format(video_base_name)
os.system(neural_filter_and_refinement_cmd)