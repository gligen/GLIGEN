---
title: GLIGen
emoji: 👁
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: 3.15.0
app_file: app.py
pinned: false
---

# Gradio App Demo for GLIGEN

## :notes: Introduction

This folder includes the source code of our [Gradio app demo](https://huggingface.co/spaces/gligen/demo) for GLIGEN. It automatically downloads and loads our checkpoints hosted on Huggingface.

NOTE: You may notice slight implementation differences of the pipeline between this code base and main GLIGEN repo, although the functionality and the checkpoints are the same. We'll replace the implementation pipeline to Diffusers after we finish the integration.

## :toolbox: Installation

To install GLIGEN demo with CUDA support, create an environment.

```Shell
conda env create -f environment.yaml
```

In case you don't have a CUDA-enabled GPU, you can run it on a CPU - though, it will be very slow.
For some speedup on Macbooks with M1 Apple Silicon, there is support with [MPS](https://pytorch.org/docs/stable/notes/mps.html) (much faster than CPU, slower than CUDA). To use Macbook GPUs, make sure that you install [conda miniforge for the arm64 architecture (recommended: mambaforge)](https://github.com/conda-forge/miniforge).

```Shell
mamba env create -f environment_cpu_mps.yaml
```

## :notebook: Usage

Activate the environment with

```Shell
conda activate gligen_demo
```

By default, it only loads the base text-box generation pipeline to save memory. You'll see error in the UI interface if attempting to run pipelines that are not loaded. Modify command line arguments to enable/disable specific pipelines.

```Shell
python app.py \
    --load-text-box-generation=True \
    --load-text-box-inpainting=False \
    --load-text-image-box-generation=False
```

## :question: How do you draw bounding boxes using Gradio sketchpad?

Gradio does not natively support drawing bounding boxes in its sketchpad. In this repo, we use a simple workaround where users draw their boxes using freeform brush, and the backend calculates the min/max point along x/y axis, and "guesses" a bounding box. The interpreted boxes are visualized on the side for better user experience.

Hope that we'll have native support for drawing bounding boxes with Gradio soon! :partying_face:

## :snowflake: TODO

- [ ] Use diffusers as the inference pipeline
- [ ] Refactor code base

## :book: Citation

```
@article{li2023gligen,
  title={GLIGEN: Open-Set Grounded Text-to-Image Generation},
  author={Li, Yuheng and Liu, Haotian and Wu, Qingyang and Mu, Fangzhou and Yang, Jianwei and Gao, Jianfeng and Li, Chunyuan and Lee, Yong Jae},
  journal={CVPR},
  year={2023}
}
```

## Disclaimer

The original GLIGEN was partly implemented and trained during an internship at Microsoft. This repo re-implements GLIGEN in PyTorch with university GPUs after the internship. Despite the minor implementation differences, this repo aims to reproduce the results and observations in the paper for research purposes.
