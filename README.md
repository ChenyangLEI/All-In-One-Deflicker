# Blind Video Deflickering by Neural Filtering with a Flawed Atlas


> **Blind Video Deflickering by Neural Filtering with a Flawed Atlas** <br>
> Chenyang Lei*, Xuanchi Ren*, Zhaoxiang Zhang and Qifeng Chen <br>
> *CVPR 2023*<br>
> \* indicates equal contribution 

[[Paper](https://chenyanglei.github.io/deflicker/CVPR2023_deflicker_lowres.pdf)]
[[ArXiv](https://arxiv.org/pdf/2303.08120.pdf)]
[[Project Website](https://chenyanglei.github.io/deflicker/)]
<!-- [[Appendix]()] -->


<div align="center">
  <br><br>
  <img src="demo.gif" alt="this slowpoke moves"  width="700" />
</div>

## News！
- Apr 10, 2023: Our code can work with segmentations masks for a foreground object.
- Mar 12, 2023: Inference code and paper are released! Collected dataset will release soon.
- Feb 28, 2023: Our paper is accepted by CVPR 2023, code will be released in two weeks. 

## Contents

1. [Environment & Dependency](#environment--dependency)
2. [Inference](#inference)
3. [All Evaluated Types of Flickering Videos](#all-evaluated-types-of-flickering-videos)
4. [Advanced Features](#advanced-features)
   - [Using segmentation masks](#using-segmentation-masks)
5. [Suggestions for Choosing the Hyperparameters](#suggestions-for-choosing-the-hyperparameters)
6. [Discussion and Related work](#discussion-and-related-work)

## Environment & Dependency

We provide an environment with python ``3.10`` & torch ``1.12`` with CUDA 11. If you want a torch ``1.6`` with CUDA 10, please check this [env file](https://github.com/ChenyangLEI/All-In-One-Deflicker/blob/53bf1d65e71bde2866d287e2b5e59ac0431c5a15/environment.yml#L1). 

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
Put your video or image folder under ``data/test``. For example:
```
export PYTHONPATH=$PWD
python test.py --video_name data/test/Winter_Scenes_in_Holland.mp4 # for video input
python test.py --video_frame_folder data/test/Winter_Scenes_in_Holland # for image folder input
```
Find the results under ``results/$YOUR_DATA_NAME/final/output.mp4``. 

**Note**: our inference code only takes about ``3000M`` GPU memory. For the ``video_frame_folder``, the default format is ``png``.

## All Evaluated Types of Flickering Videos:

- Synthesized videos from text-to-video algorithms
  -  [Magic Video](https://magicvideo.github.io/)
  -  [Make-A-Video](https://makeavideo.studio/)

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

## Advanced Features
### Using segmentation masks:

Currently, we support to process video with [Carvekit](https://github.com/OPHoperHPO/image-background-remove-tool) or [Mask-RCNN](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). This support can help improve the atlas, particularly for videos featuring a salient object or human. Please note that the current implementation supports only one foreground object with a background.

- To use [Carvekit](https://github.com/OPHoperHPO/image-background-remove-tool), which is for background removal:
```
git clone https://github.com/OPHoperHPO/image-background-remove-tool.git
export PYTHONPATH=$PWD
python test.py --video_name data/test/Winter_Scenes_in_Holland.mp4 --class_name portrait # portrait triggers Carvekit
```
- To use [Mask-RCNN](https://detectron2.readthedocs.io/en/latest/tutorials/install.html): 
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
export PYTHONPATH=$PWD
python test.py --video_name data/test/Winter_Scenes_in_Holland.mp4 --class_name anything # actually not work for this video 
```
where --class_name determines the COCO class name of the sought foreground object. It is also possible to choose the first instance retrieved by Mask-RCNN by using ``--class_name anything``. 

In both two settings, we suggest you to check the generated masks under ``data/test/{vid_name}_seg``. If the images are all black, you can only use the non-segmentation implementation above. 

## Suggestions for Choosing the Hyperparameters
If you want to find the best setting to get an atlas for deflickering, we provide a reference guide here:

1. (**Important**) [Iteration number](https://github.com/ChenyangLEI/All-In-One-Deflicker/blob/53bf1d65e71bde2866d287e2b5e59ac0431c5a15/src/config/config_flow_100.json#L6): Please change this according to the total frame number of your video and the [downsample rate](https://github.com/ChenyangLEI/All-In-One-Deflicker/blob/53bf1d65e71bde2866d287e2b5e59ac0431c5a15/src/stage1_neural_atlas.py#L265) of the image size. For example, we adopt ``10000`` iteration number for the example video with ``80`` frames and a downsample rate of ``4``. If you find the results are not as expected, you can try to increase the ``iters_num`` (for example: ``100000``). If you want to use the implementation with segmentation masks, it is suggested to increase the ``iters_num``.

2. (**Important**) [Optical flow loss weight](https://github.com/ChenyangLEI/All-In-One-Deflicker/blob/53bf1d65e71bde2866d287e2b5e59ac0431c5a15/src/config/config_flow_100.json#L8): Please change ``optical_flow_coeff`` and ``alpha_flow_factor`` (**Note**: ``alpha_flow_factor`` only used in the advanced features with segmentation masks) according the intensity of flicker in your video. For example, we adopt ``500.0``  for the ``optical_flow_coeff`` and ``4900.0`` for the ``alpha_flow_factor`` for the sample video. If the video has minor flickering, you can use ``5.0`` for the ``optical_flow_coeff`` and ``49.0`` for the ``alpha_flow_factor``. 

3. [Downsample rate](https://github.com/ChenyangLEI/All-In-One-Deflicker/blob/53bf1d65e71bde2866d287e2b5e59ac0431c5a15/src/stage1_neural_atlas.py#L265): We find that downsampling the resolution of the neural atlas by ``4`` times make the convergence much faster and slightly influences the quality. You can choose your own downsample rate.

4. [Maximum number of frames](https://github.com/ChenyangLEI/All-In-One-Deflicker/blob/53bf1d65e71bde2866d287e2b5e59ac0431c5a15/src/config/config_flow_100.json#L3): We set the ``maximum_number_of_frames`` to 200. The performance for longer videos is not evaluated. It is recommended to split long videos into several shorter sequences. 

5. Useness of segmentation masks: Perfect segmentation masks will increase the quality of the neural atlas, especially for objects with significant motion. However, in most cases, the improvement brought by segmentation on the final prediction is not significant since neural filtering can filter the flaws in the atlas. For now, we provide a naive version for segmentation masks support above.

<!-- If you want to use segmentation for better results, refer to [layered-neural-atlases](https://github.com/ykasten/layered-neural-atlases) and use our ``src/neural_filter_and_refinement.py`` based on it. Note that layered-neural-atlases use Mask-RCNN, you can also try [lang-seg](https://github.com/isl-org/lang-seg) or [ODISE](https://jerryxu.net/ODISE/). -->

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


or 

```
@article{lei2023blind,
  title={Blind Video Deflickering by Neural Filtering with a Flawed Atlas},
  author={Lei, Chenyang and Ren, Xuanchi and Zhang, Zhaoxiang and Chen, Qifeng},
  journal={arXiv preprint arXiv:2303.08120},
  year={2023}
}
```
