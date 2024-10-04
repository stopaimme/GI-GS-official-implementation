# GI-GS: Global Illumination Decomposition on Gaussian Splatting for Inverse Rendering

[**Paper**](https://arxiv.org/abs/2410.02619) | [**Project Page**](https://stopaimme.github.io/GI-GS/) 

Official implementation of GI-GS: Global Illumination Decomposition on Gaussian Splatting for Inverse Rendering

Hongze Chen, Zehong Lin, Jun Zhang.

<p align="center"> All Code will be released soon... 🏗️ 🚧 🔨</p>

Abstract: *We present GI-GS, a novel inverse rendering framework that leverages 3D Gaussian Splatting (3DGS) and deferred shading to achieve photo-realistic novel view synthesis and relighting. In inverse rendering, accurately modeling the shading processes of objects is essential for achieving high-fidelity results. Therefore, it is critical to incorporate global illumination to account for indirect lighting that reaches an object after multiple bounces across the scene. Previous 3DGS-based methods have attempted to model indirect lighting by characterizing indirect illumination as learnable lighting volumes or additional attributes of each Gaussian, while using baked occlusion to represent shadow effects. These methods, however, fail to accurately model the complex physical interactions between light and objects, making it impossible to construct realistic indirect illumination during relighting. To address this limitation, we propose to calculate indirect lighting using efficient path tracing with deferred shading. In our framework, we first render a G-buffer to capture the detailed geometry and material properties of the scene. Then, we perform physically-based rendering (PBR) only for direct lighting. With the G-buffer and previous rendering results, the indirect lighting can be calculated through a lightweight path tracing. Our method effectively models indirect lighting under any given lighting conditions, thereby achieving better novel view synthesis and relighting. Quantitative and qualitative results show that our GI-GS outperforms existing baselines in both rendering quality and efficiency.*

## Decompositon Results

<video id="teaser" autoplay="" muted="" loop="" playsinline="" height="100%">
        <source src="./static/teaser.mp4" type="video/mp4">
</video>
<video id="teaser" autoplay="" muted="" loop="" playsinline="" height="100%">
        <source src="./static/garden.mp4" type="video/mp4">
</video>
<video id="teaser" autoplay="" muted="" loop="" playsinline="" height="100%">
        <source src="./static/lego.mp4" type="video/mp4">
</video>


## Pipeline

Overview of GI-GS. GI-GS takes input a set of pretrianed 3D Gaussians, each with a normal attribute. It first rasterizes the scene geometry and materials into a G-buffer. Next, it incorporatesa differentiable PBR pipeline to obtain the rendering result under direct lighting and performs path tracing to model the occlusion. Finally, it employs differentiable ray tracing to calculate indirect lighting from the scene geometry and the previous rendering result. The final rendered image is a fusion of the first-pass and second-pass results and uses the ground truth image for supervision.
<p align="center">
    <img src="static/pipeline.png">
</p>






## BibTeX

```bibtex
@misc{chen2024gigsglobalilluminationdecomposition,
      title={GI-GS: Global Illumination Decomposition on Gaussian Splatting for Inverse Rendering}, 
      author={Hongze Chen and Zehong Lin and Jun Zhang},
      year={2024},
      eprint={2410.02619},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.02619}, 
}
```
