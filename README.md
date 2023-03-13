# Blind Video Deflickering by Neural Filtering with a Flawed Atlas


> **Blind Video Deflickering by Neural Filtering with a Flawed Atlas** <br>
> Chenyang Lei*, Xuanchi Ren*, Zhaoxiang Zhang and Qifeng Chen <br>
> *CVPR 2023*<br>
> \* indicates equal contribution 

[[Paper](https://chenyanglei.github.io/deflicker/CVPR2023_deflicker_lowres.pdf)]
[[ArXiv (Coming soon)]()]
[[Project Website](https://chenyanglei.github.io/deflicker/)]
<!-- [[Appendix]()] -->


<div align="center">
  <br><br>
  <img src="demo.gif" alt="this slowpoke moves"  width="700" />
</div>

## News！
- Mar 12, 2023: Inference code and paper are released! Collected dataset will release soon.
- Mar 1, 2023: Paper will be public in one week. 
- Feb 28, 2023: Our paper is accepted by CVPR 2023, code will be released in two weeks. 

## Environment & Dependency

Install environment:
```
conda env create -f environment.yml 
conda activate deflicker
```

Download pretrained ckpt:
```
git clone https://github.com/ChenyangLEI/cvpr2023_deflicker_public_folder
mv cvpr2023_deflicker_public_folder/pretrained_weights ./ && rm -r cvpr2023_deflicker_public_folder
```

## Inference
Put your video or image foler under ``data/test``. For example:
```
export PYTHONPATH=$PWD
python test.py --video_name data/test/Winter_Scenes_in_Holland.mp4 # for video input
python test.py --video_frame_folder data/test/Winter_Scenes_in_Holland # for image folder input
```
Find the results under ``results/$YOUR_DATA_NAME/final/output.mp4``. 

**Note**: our inference code only takes about ``3000M`` GPU memory.

## All evaluated types of flickering videos:

- Synthesized videos from text-to-video algorithms

- Old movies

- Old cartoons 

- Time-lapse videos

- Slow-motion videos

- Processed videos by the following image processing algorithms:
  -  [Style transfer](https://github.com/Yijunmaverick/UniversalStyleTransfer)
  -  [Image Enhancement](https://groups.csail.mit.edu/graphics/hdrnet/)
  -  [Intrinsic Image Decomposition](http://opensurfaces.cs.cornell.edu/publications/intrinsic/)
  -  [Image-to-Image Translation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
  -  [Colorization](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## Suggestions for Choosing the Hyperparameters
If you want to find the best setting for getting an atlas for deflickering, we provide a reference guide here:

1. [Iteration number](): Please change this according to the total frame number of your video and the [downsample rate]() of the image size. For example, we adopt ``10000`` iteration number for the example video with ``80`` frames and downsample rate ``4``. If you find the results are not as expected, you can try to increase the ``iters_num``.

2. [Optical flow loss weight](): Please change ``optical_flow_coeff`` according the intensity of flicker in your video. For example, we adopt ``500.0`` for the sample video. If the video has minor flickering, you can use ``5.0`` as the ``optical_flow_coeff``.

3. [Downsample rate](): We find that downsampling the resolution of the neural atlas by ``4`` times make the convergence much faster and slightly influences the quality. You can define your own downsample rate.

4. [Maximum number of frames](): We set the ``maximum_number_of_frames`` to 200. The performance for longer videos is not evaluated. It would be better to split the long video into several shorter sequences. 

5. Useness of segmentation masks: Perfect segmentation masks will increase the quality of the neural atlas, especially for objects with significant motion. However, in most cases,  the improvement brought by segmentation on the final prediction is not obvious since neural filtering can filter the flaws in the atlas. If you want to use segmentation for better results, refer to [layered-neural-atlases](https://github.com/ykasten/layered-neural-atlases) and use our ``src/neural_filter_and_refinement.py`` based on it. Note that layered-neural-atlases use Mask-RCNN, you can also try [lang-seg](https://github.com/isl-org/lang-seg) or [ODISE](https://jerryxu.net/ODISE/).

<!-- 
```
## Training
Coming soon...
list_filename = os.path.join(opts.list_dir, "%s_%s.txt" % (dataset, "test"))

python src/
``` -->

## Discussion and Related work
**Potential applications**: Our model can be applied to all evaluated types of flickering videos. Besides, while our approach is designed for videos, it is possible to apply *Blind Deflickering* for other tasks (e.g., novel view synthesis) where flickering artifacts exist. 

**Temporal consistency beyond our scope**: Solving the temporal inconsistency of video content is beyond the scope of deflickering. For example, the contents obtained by video generation algorithms can be very different. Large scratches in old films can destroy the contents and result in unstable videos, which requires extra restoration technique. We leave the study for a general framework to solve these temporally inconsistent artifacts for future work.

## Credit

Our code is heavily relied on [layered-neural-atlases](https://github.com/ykasten/layered-neural-atlases), [fast_blind_video_consistency](https://github.com/phoenix104104/fast_blind_video_consistency), and [pytorch-deep-video-prior](https://github.com/yzxing87/pytorch-deep-video-prior).

## Others

While we do not work on this project full-time, please feel free to provide any suggestions. We would also appreciate it if anyone could help us improve the engineering part of this project.

## Citation

If you find our work useful in your research, please consider citing:

```
@InProceedings{Lei_2023_CVPR,
      author    = {Lei, Chenyang and Ren, Xuanchi and Zhang, Zhaoxiang and Chen, Qifeng},
      title     = {Blind Video Deflickering by Neural Filtering with a Flawed Atlas},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month     = {June},
      year      = {2023},
  }
```
