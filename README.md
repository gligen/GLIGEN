
# GLIGEN: Open-Set Grounded Text-to-Image Generation (CVPR 2023)

[Yuheng Li](https://yuheng-li.github.io/), [Haotian Liu](https://hliu.cc), [Qingyang Wu](https://scholar.google.ca/citations?user=HDiw-TsAAAAJ&hl=en/), [Fangzhou Mu](https://pages.cs.wisc.edu/~fmu/), [Jianwei Yang](https://jwyang.github.io/), [Jianfeng Gao](https://www.microsoft.com/en-us/research/people/jfgao/), [Chunyuan Li*](https://chunyuan.li/), [Yong Jae Lee*](https://pages.cs.wisc.edu/~yongjaelee/) (*Co-senior authors)

[[Project Page](https://gligen.github.io/)] [[Paper](https://arxiv.org/abs/2301.07093)] [[Demo](https://huggingface.co/spaces/gligen/demo)] [[YouTube Video](https://youtu.be/-MCkU7IAGKs)]
![Teaser figure](figures/concept.gif)

[![IMAGE ALT TEXT HERE](https://github.com/gligen/GLIGEN/blob/master/figures/teaser_v4.png)](https://youtu.be/-MCkU7IAGKs)

- Go beyond text prompt with GLIGEN: enable new capabilities on frozen text-to-image generation models to ground on various prompts, including box, keypoints and images.
- GLIGEN’s zero-shot performance on COCO and LVIS outperforms that of existing supervised layout-to-image baselines by a large margin.


## :fire: News


* **[2023.11.2]**  GLIGEN is integreated into [LLaVA-Interactive](https://llava-vl.github.io/llava-interactive/): an all-in-one demo for Image Chat, Segmentation, Generation and Editing. Experience the future of interactive image editing with visual chat.
[[Project Page](https://llava-vl.github.io/llava-interactive/)] [[Demo](https://6dd3-20-163-117-69.ngrok-free.app/)] [[Code](https://github.com/LLaVA-VL/LLaVA-Interactive-Demo)] [[Paper](https://arxiv.org/abs/2311.00571)]
<center>
<img src="https://github.com/LLaVA-VL/llava-interactive/blob/main/images/llava_interactive_workflow.png" width="30%"> 
</center>


* **[2023.04.18]** We have updated our arxiv paper. We explain the difference between GLIGEN and ControlNet [here](docs/gligen_vs_controlnet.MD) to help researchers to have a better and deeper understanding. 

* **[2023.04.08]** GLIGEN is combined with [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), which free humans from anotating bounding boxes and their concepts. Given a language prompt, Grounding DINO localizes the concepts with boxes:  image $\rightarrow$ (box, concept), then GLIGEN inpaint the image:  (box, concept) $\rightarrow$ image:

<center>
<img src="https://camo.githubusercontent.com/4dabf8128cd4f40eaa97ee45d050ddcd8063356f631d98072fb5a5c19c35fa9c/68747470733a2f2f68756767696e67666163652e636f2f5368696c6f6e674c69752f47726f756e64696e6744494e4f2f7265736f6c76652f6d61696e2f47445f474c4947454e2e706e67" width="600"> 
</center>


* **[2023.03.22]** [Our fork on diffusers](https://github.com/gligen/diffusers/tree/gligen/examples/gligen) with support of text-box-conditioned generation and inpainting is released.  It is now faster, more flexible, and automatically downloads and loads model from Huggingface Hub!  Try it out!
* **[2023.03.20]** Stay up-to-date on the line of research on *grounded image generation* such as GLIGEN, by checking out [`Computer Vision in the Wild (CVinW) Reading List`](https://github.com/Computer-Vision-in-the-Wild/CVinW_Readings#orange_book-grounded-image-generation-in-the-wild).
* **[2023.03.19]** GLIGEN is covered by great Yannic Kilcher in his latest YouTube video on [`The biggest week in AI`](https://www.youtube.com/watch?v=YqPYDWPYXFs&t=2245s).
* **[2023.03.05]** Gradio demo code is released at [`GLIGEN/demo`](https://github.com/gligen/GLIGEN/tree/master/demo).
* **[2023.03.03]** Code base and checkpoints are released.
* **[2023.02.28]** Paper is accepted to CVPR 2023.
* **[2023.01.17]** GLIGEN paper and demo is released.

## Requirements
We provide [dockerfile](env_docker/Dockerfile) to setup environment. 


## Download GLIGEN models

We provide ten checkpoints for different use scenarios. All models here are based on SD-V-1.4.
| Mode       | Modality       | Download                                                                                                       |
|------------|----------------|----------------------------------------------------------------------------------------------------------------|
| Generation | Box+Text       | [HF Hub](https://huggingface.co/gligen/gligen-generation-text-box/blob/main/diffusion_pytorch_model.bin)       |
| Generation | Box+Text+Image | [HF Hub](https://huggingface.co/gligen/gligen-generation-text-image-box/blob/main/diffusion_pytorch_model.bin) |
| Generation | Keypoint       | [HF Hub](https://huggingface.co/gligen/gligen-generation-keypoint/blob/main/diffusion_pytorch_model.bin)       |
| Inpainting | Box+Text       | [HF Hub](https://huggingface.co/gligen/gligen-inpainting-text-box/blob/main/diffusion_pytorch_model.bin)       |
| Inpainting | Box+Text+Image | [HF Hub](https://huggingface.co/gligen/gligen-inpainting-text-image-box/blob/main/diffusion_pytorch_model.bin) |
| Generation | Hed map        | [HF Hub](https://huggingface.co/gligen/gligen-generation-hed/blob/main/diffusion_pytorch_model.bin)      |
| Generation | Canny map      | [HF Hub](https://huggingface.co/gligen/gligen-generation-canny/blob/main/diffusion_pytorch_model.bin)      |
| Generation | Depth map      | [HF Hub](https://huggingface.co/gligen/gligen-generation-depth/blob/main/diffusion_pytorch_model.bin)      |
| Generation | Semantic map   | [HF Hub](https://huggingface.co/gligen/gligen-generation-sem/blob/main/diffusion_pytorch_model.bin)      |
| Generation | Normal map     | [HF Hub](https://huggingface.co/gligen/gligen-generation-normal/blob/main/diffusion_pytorch_model.bin)      |

Note that the provided checkpoint for semantic map is only trained on ADE20K dataset; the checkpoint for normal map is only trained on DIODE dataset.

## Inference: Generate images with GLIGEN

We provide one script to generate images using provided checkpoints. First download models and put them in `gligen_checkpoints`. Then run
```bash
python gligen_inference.py
```
Example samples for each checkpoint will be saved in `generation_samples`. One can check `gligen_inference.py` for more details about interface. 


## Training 

### Grounded generation training

One need to first prepare data for different grounding modality conditions. Refer [data](DATA/README.MD) for the data we used for different GLIGEN models. Once data is ready, the following command is used to train GLIGEN. (We support multi-GPUs training)

```bash
ptyhon main.py --name=your_experiment_name  --yaml_file=path_to_your_yaml_config
```
The `--yaml_file` is the most important argument and below we will use one example to explain key components so that one can be familiar with our code and know how to customize training on their own grounding modalities. The other args are self-explanatory by their names. The experiment will be saved in `OUTPUT_ROOT/name`

One can refer `configs/flicker_text.yaml` as one example. One can see that there are 5 components defining this yaml: **diffusion**, **model**, **autoencoder**, **text_encoder**, **train_dataset_names** and **grounding_tokenizer_input**. Typecially, **diffusion**, **autoencoder** and **text_encoder** should not be changed as they are defined by Stable Diffusion. One should pay attention to following:

 - Within **model** we add new argument **grounding_tokenizer** which defines a network producing grounding tokens. This network will be instantized in the model. One can refer to `ldm/modules/diffusionmodules/grounding_net_example.py` for more details about defining this network.
 - **grounding_tokenizer_input** will define a network taking in batch data from dataloader and produce input for the grounding_tokenizer. In other words, it is an intermediante class between dataloader and grounding_tokenizer. One can refer `grounding_input/__init__.py` for details about defining this class.
 - **train_dataset_names** should be listing a serial of names of datasets (all datasets will be concatenated internally, thus it is useful to combine datasets for training). Each dataset name should be first registered in `dataset/catalog.py`. We have listed all dataset we used; if one needs to train GLIGEN on their own modality dataset, please don't forget first list its name there. 


### Grounded inpainting training

GLIGEN also supports inpainting training. The following command can be used:
```bash
ptyhon main.py --name=your_experiment_name  --yaml_file=path_to_your_yaml_config --inpaint_mode=True  --ckpt=path_to_an_adapted_model
```
Typecially, we first train GLIGEN on generation task (e.g., text grounded generation) and this model has 4 channels for input conv (latent space of Stable Diffusion), then we modify the saved checkpoint to 9 channels with addition 5 channels initilized with 0. This continue training can lead to faster convergence and better results. path_to_an_adapted_model refers to this modified checkpoint, `convert_ckpt.py` can be used for modifying checkpoint. **NOTE:** yaml file is the same for generation and inpainting training, one only need to change `--inpaint_mode`

## Citation
```
@article{li2023gligen,
  title={GLIGEN: Open-Set Grounded Text-to-Image Generation},
  author={Li, Yuheng and Liu, Haotian and Wu, Qingyang and Mu, Fangzhou and Yang, Jianwei and Gao, Jianfeng and Li, Chunyuan and Lee, Yong Jae},
  journal={CVPR},
  year={2023}
}
```

## Disclaimer

The original GLIGEN was partly implemented during a part-time internship at Microsoft while the first author was working at The University of Wisconsin-Madison. This repo re-implements GLIGEN in PyTorch with university GPUs. Despite the minor implementation differences, this repo aims to reproduce the results and observations in the paper for research purposes.

## Terms and Conditions

We have strict terms and conditions for using the model checkpoints and the demo; it is restricted to uses that follow the license agreement of <a href="https://github.com/CompVis/latent-diffusion">Latent Diffusion Model</a> and <a href="https://github.com/Stability-AI/StableDiffusion">Stable Diffusion</a>.


## Broader Impact

It is important to note that our model GLIGEN is designed for open-world grounded text-to-image generation with caption and various condition inputs (e.g. bounding box). However, we also recognize the importance of responsible AI considerations and the need to clearly communicate the capabilities and limitations of our research. While the grounding ability generalizes well to novel spatial configuration and concepts, our model may not perform well in scenarios that are out of scope or beyond the intended use case. We strongly discourage the misuse of our model in scenarios, where our technology could be used to generate misleading or malicious images. We also acknowledge the potential biases that may be present in the data used to train our model, and the need for ongoing evaluation and improvement to address these concerns. To ensure transparency and accountability, we have included a model card that describes the intended use cases, limitations, and potential biases of our model. We encourage users to refer to this model card and exercise caution when applying our technology in new contexts. We hope that our work will inspire further research and discussion on the ethical implications of AI and the importance of transparency and accountability in the development of new technologies.
