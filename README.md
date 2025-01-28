# GI-GS: Global Illumination Decomposition on Gaussian Splatting for Inverse Rendering

##### Hongze Chen, [Zehong Lin](https://zhlinup.github.io/), [Jun Zhang](https://eejzhang.people.ust.hk/).

#### [**Paper**](https://arxiv.org/abs/2410.02619) | [**Project Page**](https://stopaimme.github.io/GI-GS/) 

## Overview

<p align="center">
  <img width="100%" src="assets/pipeline.png"/>
</p>

https://github.com/user-attachments/assets/632ebe65-57c3-4757-97d5-6f373034b5da

We present GI-GS, a novel inverse rendering framework that leverages 3D Gaussian Splatting (3DGS) and deferred shading to achieve photo-realistic novel view synthesis and relighting. In our framework, we first render a G-buffer to capture the detailed geometry and material properties of the scene. Then, we perform physically-based rendering (PBR) only for direct lighting. With the G-buffer and previous rendering results, the indirect lighting can be calculated through a lightweight path tracing. Our method effectively models indirect lighting under any given lighting conditions, thereby achieving better novel view synthesis and relighting. Quantitative and qualitative results show that our GI-GS outperforms existing baselines in both rendering quality and efficiency.

## Installation

1. Clone the repo

```sh
git clone https://github.com/stopaimme/GI-GS.git --recursive
cd GI-GS
```

2. Create the environment

```sh
conda env create --file environment.yml
conda activate GI-GS

cd submodules

git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast

pip install ./simple-knn
cd ./diff-gaussian-rasterization && python setup.py develop && cd ../..
```

## Dataset

You can find the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) and [TensoIR-Synthetic](https://zenodo.org/records/7880113#.ZE68FHZBz18) datasets from the link provided by the paper authors. And the authors of TensoIR have provided the environment maps in this [link](https://drive.google.com/file/d/10WLc4zk2idf4xGb6nPL43OXTTHvAXSR3/view).

## Running

The training scripts heavily rely on the code provided by [GS-IR](https://github.com/lzhnb/GS-IR). Thanks for its great work!

### TensoIR-Synthetic

Take the `lego` case as an example.

**Training**

```sh
python train.py \
-m outputs/lego/ \
-s datasets/TensoIR/lego/ \
--iterations 35000 \
--eval \
--gamma \
--radius 0.8 \
--bias 0.01 \
--thick 0.05 \
--delta 0.0625 \
--step 16 \
--start 64 \
--indirect
```
> set `--gamma` to enable **linear_to_sRGB** will cause *better relighting results* but *worse novel view synthesis results*  
> set `--indirect` to enable indirect illumination modelling 

**Global illumination settings**

- radius →  path tracing range
- bias → ensure ray hit the surface
- thick → thickness of the surface
- delta → angle interval to control the num-sample
- step → path tracing steps
- start → path tracing starting point

You can change the radius, bias, thick, delta, step, step ,start to achieve different indirect illumination and occlusion. For the precise meanings, please refer to [forward.cu (line 635-910)](https://github.com/stopaimme/GI-GS/blob/master/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu).



**Evaluation (Novel View Synthesis)**

```sh
python render.py \
-m outputs/lego \
-s datasets/TensoIR/lego/ \
--checkpoint outputs/lego/chkpnt35000.pth \
--eval \
--skip_train \
--pbr \
--gamma \
--indirect
```

**Evaluation (Normal)**

```sh
python normal_eval.py \
--gt_dir datasets/TensoIR/lego/ \
--output_dir outputs/lego/test/ours_None
```

**Evaluation (Albedo)**

```sh
python render.py \
-m outputs/lego \
-s datasets/TensoIR/lego/ \
--checkpoint outputs/lego/chkpnt35000.pth \
--eval \
--skip_train \
--brdf_eval
```

**Relighting**

```sh
python relight.py \
-m outputs/lego \
-s datasets/TensoIR/lego/ \
--checkpoint outputs/lego/chkpnt35000.pth \
--hdri datasets/TensoIR/Environment_Maps/high_res_envmaps_2k/bridge.hdr \
--eval \
--gamma
```

> set `--gamma` to enable **linear_to_sRGB** will cause better relighting results but worse novel view synthesis results

**Relighting Evaluation**

```sh
python relight_eval.py \
--output_dir outputs/lego/test/ours_None/relight/ \
--gt_dir datasets/TensoIR/lego/
```

### Mip-NeRF 360

Take the `bicycle` case as an example.

**Training**

```sh
python train.py \
-m outputs/bicycle \
-s datasets/nerf_real_360/bicycle/ \
--iterations 40000 \
-i images_4 \
-r 1 \
--eval \
--metallic \
--radius 0.8 \
--bias 0.01 \
--thick 0.05 \
--delta 0.0625 \
--step 16 \
--start 64 \
--degree 3 \
--indirect
```

> `-i images_4` for outdoor scenes and `-i images_2` for indoor scenes  
> `-r 1` for resolution scaling (not rescale)  
> `-degree 3` for bicycle, flowers, stump and `-degree 1` for garden, treehill and indoor scenes. We find that set SH_degree to 1 can achieve better geometry.  
> set `--metallic` choose to reconstruct metallicness  
> set `--gamma` to enable **linear_to_sRGB** will cause better relighting results but worse novel view synthesis results  
> set `--indirect` to enable indirect illumination modelling  


**Evaluation**

```sh
python render.py \
-m outputs/bicycle \
-s datasets/nerf_real_360/bicycle/ \
--checkpoint outputs/bicycle/chkpnt40000.pth \
-i images_4 \
-r 1 \
--eval \
--skip_train \
--pbr \
--metallic \
--indirect
```

> set `--gamma` to enable **linear_to_sRGB** will cause better relighting results but worse novel view synthesis results

**Relighting**

```sh
python relight.py \
-m outputs/bicycle \
-s datasets/nerf_real_360/bicycle/ \
--checkpoint outputs/bicycle/chkpnt40000.pth \
--hdri datasets/TensoIR/Environment_Maps/high_res_envmaps_2k/bridge.hdr \
--eval \
--gamma
```

> set `--gamma` to enable **linear_to_sRGB** will cause better relighting results but worse novel view synthesis results

## Acknowledge

- [GS-IR](https://github.com/lzhnb/GS-IR) (Provides framework)⭐
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [nvdiffrec](https://github.com/NVlabs/nvdiffrec)



## Citation

```bibtex
@inproceedings{chen2025gigs,
      title={GI-GS: Global Illumination Decomposition on Gaussian Splatting for Inverse Rendering}, 
      author={Hongze Chen and Zehong Lin and Jun Zhang},
      booktitle={ICLR},
      year={2025},
}
```
