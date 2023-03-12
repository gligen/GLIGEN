import gradio as gr
import torch
import argparse
from omegaconf import OmegaConf
from gligen.task_grounded_generation import grounded_generation_box, load_ckpt
from ldm.util import default_device

import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from functools import partial
import math
from contextlib import nullcontext

from gradio import processing_utils
from typing import Optional

from huggingface_hub import hf_hub_download
hf_hub_download = partial(hf_hub_download, library_name="gligen_demo")

arg_bool = lambda x: x.lower() == 'true'
device = default_device()

print(f"GLIGEN uses {device.upper()} device.")
if device == "cpu":
    print("It will be sloooow. Consider using GPU support with CUDA or (in case of M1/M2 Apple Silicon) MPS.")
elif device == "mps":
    print("The fastest you can get on M1/2 Apple Silicon. Yet, still many opimizations are switched off and it will is much slower than CUDA.")

def parse_option():
    parser = argparse.ArgumentParser('GLIGen Demo', add_help=False)
    parser.add_argument("--folder", type=str,  default="create_samples", help="path to OUTPUT")
    parser.add_argument("--official_ckpt", type=str,  default='ckpts/sd-v1-4.ckpt', help="")
    parser.add_argument("--guidance_scale", type=float,  default=5, help="")
    parser.add_argument("--alpha_scale", type=float,  default=1, help="scale tanh(alpha). If 0, the behaviour is same as original model")
    parser.add_argument("--load-text-box-generation", type=arg_bool, default=True, help="Load text-box generation pipeline.")
    parser.add_argument("--load-text-box-inpainting", type=arg_bool, default=False, help="Load text-box inpainting pipeline.")
    parser.add_argument("--load-text-image-box-generation", type=arg_bool, default=False, help="Load text-image-box generation pipeline.")
    args = parser.parse_args()
    return args
args = parse_option()


def load_from_hf(repo_id, filename='diffusion_pytorch_model.bin'):
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    return torch.load(cache_file, map_location='cpu')

def load_ckpt_config_from_hf(modality):
    ckpt = load_from_hf(f'gligen/{modality}')
    config = load_from_hf('gligen/demo_config_legacy', filename=f'{modality}.pth')
    return ckpt, config


if args.load_text_box_generation:
    pretrained_ckpt_gligen, config = load_ckpt_config_from_hf('gligen-generation-text-box')
    config = OmegaConf.create( config["_content"] ) # config used in training
    config.update( vars(args) )
    config.model['params']['is_inpaint'] = False
    config.model['params']['is_style'] = False
    loaded_model_list = load_ckpt(config, pretrained_ckpt_gligen) 


if args.load_text_box_inpainting:
    pretrained_ckpt_gligen_inpaint, config = load_ckpt_config_from_hf('gligen-inpainting-text-box')
    config = OmegaConf.create( config["_content"] ) # config used in training
    config.update( vars(args) )
    config.model['params']['is_inpaint'] = True 
    config.model['params']['is_style'] = False
    loaded_model_list_inpaint = load_ckpt(config, pretrained_ckpt_gligen_inpaint)


if args.load_text_image_box_generation:
    pretrained_ckpt_gligen_style, config = load_ckpt_config_from_hf('gligen-generation-text-image-box')
    config = OmegaConf.create( config["_content"] ) # config used in training
    config.update( vars(args) )
    config.model['params']['is_inpaint'] = False 
    config.model['params']['is_style'] = True
    loaded_model_list_style = load_ckpt(config, pretrained_ckpt_gligen_style)


def load_clip_model():
    from transformers import CLIPProcessor, CLIPModel
    version = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(version).to(device)
    processor = CLIPProcessor.from_pretrained(version)

    return {
        'version': version,
        'model': model,
        'processor': processor,
    }

clip_model = load_clip_model()


class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        if x is None:
            return x
        if self.tool == "sketch" and self.source in ["upload", "webcam"] and type(x) != dict:
            decode_image = processing_utils.decode_base64_to_image(x)
            width, height = decode_image.size
            mask = np.zeros((height, width, 4), dtype=np.uint8)
            mask[..., -1] = 255
            mask = self.postprocess(mask)
            x = {'image': x, 'mask': mask}
        return super().preprocess(x)


class Blocks(gr.Blocks):

    def __init__(
        self,
        theme: str = "default",
        analytics_enabled: Optional[bool] = None,
        mode: str = "blocks",
        title: str = "Gradio",
        css: Optional[str] = None,
        **kwargs,
    ):

        self.extra_configs = {
            'thumbnail': kwargs.pop('thumbnail', ''),
            'url': kwargs.pop('url', 'https://gradio.app/'),
            'creator': kwargs.pop('creator', '@teamGradio'),
        }

        super(Blocks, self).__init__(theme, analytics_enabled, mode, title, css, **kwargs)

    def get_config_file(self):
        config = super(Blocks, self).get_config_file()

        for k, v in self.extra_configs.items():
            config[k] = v
        
        return config

'''
inference model
'''

@torch.no_grad()
def inference(task, language_instruction, grounding_instruction, inpainting_boxes_nodrop, image,
              alpha_sample, guidance_scale, batch_size,
              fix_seed, rand_seed, actual_mask, style_image,
              *args, **kwargs):
    grounding_instruction = json.loads(grounding_instruction)
    phrase_list, location_list = [], []
    for k, v  in grounding_instruction.items():
        phrase_list.append(k)
        location_list.append(v)

    placeholder_image = Image.open('images/teddy.jpg').convert("RGB")    
    image_list = [placeholder_image] * len(phrase_list) # placeholder input for visual prompt, which is disabled

    batch_size = int(batch_size)
    if not 1 <= batch_size <= 4:
        batch_size = 2

    if style_image == None:
        has_text_mask = 1 
        has_image_mask = 0 # then we hack above 'image_list' 
    else:
        valid_phrase_len = len(phrase_list)

        phrase_list += ['placeholder']
        has_text_mask = [1]*valid_phrase_len + [0]

        image_list = [placeholder_image]*valid_phrase_len + [style_image]
        has_image_mask = [0]*valid_phrase_len + [1]
        
        location_list += [ [0.0, 0.0, 1, 0.01]  ] # style image grounding location

    if task == 'Grounded Inpainting':
        alpha_sample = 1.0

    instruction = dict(
        prompt = language_instruction,
        phrases = phrase_list,
        images = image_list,
        locations = location_list,
        alpha_type = [alpha_sample, 0, 1.0 - alpha_sample], 
        has_text_mask = has_text_mask,
        has_image_mask = has_image_mask,
        save_folder_name = language_instruction,
        guidance_scale = guidance_scale,
        batch_size = batch_size,
        fix_seed = bool(fix_seed),
        rand_seed = int(rand_seed),
        actual_mask = actual_mask,
        inpainting_boxes_nodrop = inpainting_boxes_nodrop,
    )

    # float16 autocasting only CUDA device
    with torch.autocast(device_type='cuda', dtype=torch.float16) if device == "cuda" else nullcontext():
        if task == 'Grounded Generation':
            if style_image == None:
                return grounded_generation_box(loaded_model_list, instruction, *args, **kwargs)
            else:
                return grounded_generation_box(loaded_model_list_style, instruction, *args, **kwargs)
        elif task == 'Grounded Inpainting':
            assert image is not None
            instruction['input_image'] = image.convert("RGB")
            return grounded_generation_box(loaded_model_list_inpaint, instruction, *args, **kwargs)


def draw_box(boxes=[], texts=[], img=None):
    if len(boxes) == 0 and img is None:
        return None

    if img is None:
        img = Image.new('RGB', (512, 512), (255, 255, 255))
    colors = ["red", "olive", "blue", "green", "orange", "brown", "cyan", "purple"]
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("DejaVuSansMono.ttf", size=18)
    for bid, box in enumerate(boxes):
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=colors[bid % len(colors)], width=4)
        anno_text = texts[bid]
        draw.rectangle([box[0], box[3] - int(font.size * 1.2), box[0] + int((len(anno_text) + 0.8) * font.size * 0.6), box[3]], outline=colors[bid % len(colors)], fill=colors[bid % len(colors)], width=4)
        draw.text([box[0] + int(font.size * 0.2), box[3] - int(font.size*1.2)], anno_text, font=font, fill=(255,255,255))
    return img

def get_concat(ims):
    if len(ims) == 1:
        n_col = 1
    else:
        n_col = 2
    n_row = math.ceil(len(ims) / 2)
    dst = Image.new('RGB', (ims[0].width * n_col, ims[0].height * n_row), color="white")
    for i, im in enumerate(ims):
        row_id = i // n_col
        col_id = i % n_col
        dst.paste(im, (im.width * col_id, im.height * row_id))
    return dst


def auto_append_grounding(language_instruction, grounding_texts):
    for grounding_text in grounding_texts:
        if grounding_text not in language_instruction and grounding_text != 'auto':
            language_instruction += "; " + grounding_text
    print(language_instruction)
    return language_instruction




def generate(task, language_instruction, grounding_texts, sketch_pad,
             alpha_sample, guidance_scale, batch_size,
             fix_seed, rand_seed, use_actual_mask, append_grounding, style_cond_image,
             state):
    if 'boxes' not in state:
        state['boxes'] = []

    boxes = state['boxes']
    grounding_texts = [x.strip() for x in grounding_texts.split(';')]
    assert len(boxes) == len(grounding_texts)
    boxes = (np.asarray(boxes) / 512).tolist()
    grounding_instruction = json.dumps({obj: box for obj,box in zip(grounding_texts, boxes)})

    image = None
    actual_mask = None
    if task == 'Grounded Inpainting':
        image = state.get('original_image', sketch_pad['image']).copy()
        image = center_crop(image)
        image = Image.fromarray(image)

        if use_actual_mask:
            actual_mask = sketch_pad['mask'].copy()
            if actual_mask.ndim == 3:
                actual_mask = actual_mask[..., 0]
            actual_mask = center_crop(actual_mask, tgt_size=(64, 64))
            actual_mask = torch.from_numpy(actual_mask == 0).float()

        if state.get('inpaint_hw', None):
            boxes = np.asarray(boxes) * 0.9 + 0.05
            boxes = boxes.tolist()
            grounding_instruction = json.dumps({obj: box for obj,box in zip(grounding_texts, boxes) if obj != 'auto'})
    
    if append_grounding:
        language_instruction = auto_append_grounding(language_instruction, grounding_texts)

    gen_images, gen_overlays = inference(
        task, language_instruction, grounding_instruction, boxes, image,
        alpha_sample, guidance_scale, batch_size,
        fix_seed, rand_seed, actual_mask, style_cond_image, clip_model=clip_model,
    )

    for idx, gen_image in enumerate(gen_images):

        if task == 'Grounded Inpainting' and state.get('inpaint_hw', None):
            hw = min(*state['original_image'].shape[:2])
            gen_image = sized_center_fill(state['original_image'].copy(), np.array(gen_image.resize((hw, hw))), hw, hw)
            gen_image = Image.fromarray(gen_image)
        
        gen_images[idx] = gen_image

    blank_samples = batch_size % 2 if batch_size > 1 else 0
    gen_images = [gr.Image.update(value=x, visible=True) for i,x in enumerate(gen_images)] \
                    + [gr.Image.update(value=None, visible=True) for _ in range(blank_samples)] \
                    + [gr.Image.update(value=None, visible=False) for _ in range(4 - batch_size - blank_samples)]

    return gen_images + [state]


def binarize(x):
    return (x != 0).astype('uint8') * 255

def sized_center_crop(img, cropx, cropy):
    y, x = img.shape[:2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    return img[starty:starty+cropy, startx:startx+cropx]

def sized_center_fill(img, fill, cropx, cropy):
    y, x = img.shape[:2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    img[starty:starty+cropy, startx:startx+cropx] = fill
    return img

def sized_center_mask(img, cropx, cropy):
    y, x = img.shape[:2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    center_region = img[starty:starty+cropy, startx:startx+cropx].copy()
    img = (img * 0.2).astype('uint8')
    img[starty:starty+cropy, startx:startx+cropx] = center_region
    return img

def center_crop(img, HW=None, tgt_size=(512, 512)):
    if HW is None:
        H, W = img.shape[:2]
        HW = min(H, W)
    img = sized_center_crop(img, HW, HW)
    img = Image.fromarray(img)
    img = img.resize(tgt_size)
    return np.array(img)

def draw(task, input, grounding_texts, new_image_trigger, state):
    if type(input) == dict:
        image = input['image']
        mask = input['mask']
    else:
        mask = input

    if mask.ndim == 3:
        mask = mask[..., 0]

    image_scale = 1.0

    # resize trigger
    if task == "Grounded Inpainting":
        mask_cond = mask.sum() == 0
        # size_cond = mask.shape != (512, 512)
        if mask_cond and 'original_image' not in state:
            image = Image.fromarray(image)
            width, height = image.size
            scale = 600 / min(width, height)
            image = image.resize((int(width * scale), int(height * scale)))
            state['original_image'] = np.array(image).copy()
            image_scale = float(height / width)
            return [None, new_image_trigger + 1, image_scale, state]
        else:
            original_image = state['original_image']
            H, W = original_image.shape[:2]
            image_scale = float(H / W)

    mask = binarize(mask)
    if mask.shape != (512, 512):
        # assert False, "should not receive any non- 512x512 masks."
        if 'original_image' in state and state['original_image'].shape[:2] == mask.shape:
            mask = center_crop(mask, state['inpaint_hw'])
            image = center_crop(state['original_image'], state['inpaint_hw'])
        else:
            mask = np.zeros((512, 512), dtype=np.uint8)
    # mask = center_crop(mask)
    mask = binarize(mask)

    if type(mask) != np.ndarray:
        mask = np.array(mask)

    if mask.sum() == 0 and task != "Grounded Inpainting":
        state = {}

    if task != 'Grounded Inpainting':
        image = None
    else:
        image = Image.fromarray(image)

    if 'boxes' not in state:
        state['boxes'] = []

    if 'masks' not in state or len(state['masks']) == 0:
        state['masks'] = []
        last_mask = np.zeros_like(mask)
    else:
        last_mask = state['masks'][-1]

    if type(mask) == np.ndarray and mask.size > 1:
        diff_mask = mask - last_mask
    else:
        diff_mask = np.zeros([])

    if diff_mask.sum() > 0:
        x1x2 = np.where(diff_mask.max(0) != 0)[0]
        y1y2 = np.where(diff_mask.max(1) != 0)[0]
        y1, y2 = y1y2.min(), y1y2.max()
        x1, x2 = x1x2.min(), x1x2.max()

        if (x2 - x1 > 5) and (y2 - y1 > 5):
            state['masks'].append(mask.copy())
            state['boxes'].append((x1, y1, x2, y2))

    grounding_texts = [x.strip() for x in grounding_texts.split(';')]
    grounding_texts = [x for x in grounding_texts if len(x) > 0]
    if len(grounding_texts) < len(state['boxes']):
        grounding_texts += [f'Obj. {bid+1}' for bid in range(len(grounding_texts), len(state['boxes']))]

    box_image = draw_box(state['boxes'], grounding_texts, image)

    if box_image is not None and state.get('inpaint_hw', None):
        inpaint_hw = state['inpaint_hw']
        box_image_resize = np.array(box_image.resize((inpaint_hw, inpaint_hw)))
        original_image = state['original_image'].copy()
        box_image = sized_center_fill(original_image, box_image_resize, inpaint_hw, inpaint_hw)

    return [box_image, new_image_trigger, image_scale, state]

def clear(task, sketch_pad_trigger, batch_size, state, switch_task=False):
    if task != 'Grounded Inpainting':
        sketch_pad_trigger = sketch_pad_trigger + 1
    blank_samples = batch_size % 2 if batch_size > 1 else 0
    out_images = [gr.Image.update(value=None, visible=True) for i in range(batch_size)] \
                    + [gr.Image.update(value=None, visible=True) for _ in range(blank_samples)] \
                    + [gr.Image.update(value=None, visible=False) for _ in range(4 - batch_size - blank_samples)]
    state = {}
    return [None, sketch_pad_trigger, None, 1.0] + out_images + [state]

css = """
#generate-btn {
    --tw-border-opacity: 1;
    border-color: rgb(255 216 180 / var(--tw-border-opacity));
    --tw-gradient-from: rgb(255 216 180 / .7);
    --tw-gradient-to: rgb(255 216 180 / 0);
    --tw-gradient-stops: var(--tw-gradient-from), var(--tw-gradient-to);
    --tw-gradient-to: rgb(255 176 102 / .8);
    --tw-text-opacity: 1;
    color: rgb(238 116 0 / var(--tw-text-opacity));
}
#img2img_image, #img2img_image > .h-60, #img2img_image > .h-60 > div, #img2img_image > .h-60 > div > img
{
    height: var(--height) !important;
    max-height: var(--height) !important;
    min-height: var(--height) !important;
}
#mirrors a:hover {
    cursor:pointer;
}
#paper-info a {
    color:#008AD7;
}
#paper-info a:hover {
    cursor: pointer;
}
"""

rescale_js = """
function(x) {
    const root = document.querySelector('gradio-app').shadowRoot || document.querySelector('gradio-app');
    let image_scale = parseFloat(root.querySelector('#image_scale input').value) || 1.0;
    const image_width = root.querySelector('#img2img_image').clientWidth;
    const target_height = parseInt(image_width * image_scale);
    document.body.style.setProperty('--height', `${target_height}px`);
    root.querySelectorAll('button.justify-center.rounded')[0].style.display='none';
    root.querySelectorAll('button.justify-center.rounded')[1].style.display='none';
    return x;
}
"""

mirror_js = """
function () {
    const root = document.querySelector('gradio-app').shadowRoot || document.querySelector('gradio-app');
    const mirrors_div = root.querySelector('#mirrors');
    const current_url = window.location.href;
    const mirrors = [
        'https://dev.hliu.cc/gligen_mirror1/',
        'https://dev.hliu.cc/gligen_mirror2/',
    ];

    let mirror_html = '';
    mirror_html += '[<a href="https://gligen.github.io" target="_blank" style="">Project Page</a>]';
    mirror_html += '[<a href="https://arxiv.org/abs/2301.07093" target="_blank" style="">Paper</a>]';
    mirror_html += '[<a href="https://github.com/gligen/GLIGEN" target="_blank" style="">GitHub Repo</a>]';
    mirror_html += '&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;';
    mirror_html += 'Mirrors: ';

    mirrors.forEach((e, index) => {
        let cur_index = index + 1;
        if (current_url.includes(e)) {
            mirror_html += `[Mirror ${cur_index}] `;
        } else {
            mirror_html += `[<a onclick="window.location.href = '${e}'">Mirror ${cur_index}</a>] `;
        }
    });

    mirror_html = `<div class="output-markdown gr-prose" style="max-width: 100%;"><h3 style="text-align: center" id="paper-info">${mirror_html}</h3></div>`;

    mirrors_div.innerHTML = mirror_html;
}
"""

with Blocks(
    css=css,
    analytics_enabled=False,
    title="GLIGen demo",
) as main:
    gr.Markdown('<h1 style="text-align: center;">GLIGen: Open-Set Grounded Text-to-Image Generation</h1>')
    gr.Markdown("""<h3 style="text-align: center" id="paper-info">
    [<a href="https://gligen.github.io" target="_blank" style="">Project Page</a>]
    [<a href="https://arxiv.org/abs/2301.07093" target="_blank" style="">Paper</a>]
    [<a href="https://github.com/gligen/GLIGEN" target="_blank" style="">GitHub Repo</a>]
    </h3>""")
    # gr.HTML("", elem_id="mirrors")
    gr.Markdown("To ground concepts of interest with desired spatial specification, please (1) &#9000;&#65039; enter the concept names in <em> Grounding Instruction</em>, and (2) &#128433;&#65039; draw their corresponding bounding boxes one by one using <em> Sketch Pad</em> -- the parsed boxes will be displayed automatically.")
    with gr.Row():
        with gr.Column(scale=4):
            sketch_pad_trigger = gr.Number(value=0, visible=False)
            sketch_pad_resize_trigger = gr.Number(value=0, visible=False)
            init_white_trigger = gr.Number(value=0, visible=False)
            image_scale = gr.Number(value=0, elem_id="image_scale", visible=False)
            new_image_trigger = gr.Number(value=0, visible=False)

            task = gr.Radio(
                choices=["Grounded Generation", 'Grounded Inpainting'],
                type="value",
                value="Grounded Generation",
                label="Task",
            )
            language_instruction = gr.Textbox(
                label="Language instruction",
            )
            grounding_instruction = gr.Textbox(
                label="Grounding instruction (Separated by semicolon)",
            )
            with gr.Row():
                sketch_pad = ImageMask(label="Sketch Pad", elem_id="img2img_image")
                out_imagebox = gr.Image(type="pil", label="Parsed Sketch Pad")
            with gr.Row():
                clear_btn = gr.Button(value='Clear')
                gen_btn = gr.Button(value='Generate', elem_id="generate-btn")
            with gr.Accordion("Advanced Options", open=False):
                with gr.Column():
                    alpha_sample = gr.Slider(minimum=0, maximum=1.0, step=0.1, value=0.3, label="Scheduled Sampling (τ)")
                    guidance_scale = gr.Slider(minimum=0, maximum=50, step=0.5, value=7.5, label="Guidance Scale")
                    batch_size = gr.Slider(minimum=1, maximum=4, step=1, value=2, label="Number of Samples")
                    append_grounding = gr.Checkbox(value=True, label="Append grounding instructions to the caption")
                    use_actual_mask = gr.Checkbox(value=False, label="Use actual mask for inpainting", visible=False)
                    with gr.Row():
                        fix_seed = gr.Checkbox(value=True, label="Fixed seed")
                        rand_seed = gr.Slider(minimum=0, maximum=1000, step=1, value=0, label="Seed")
                    with gr.Row():
                        use_style_cond = gr.Checkbox(value=False, label="Enable Style Condition")
                        style_cond_image = gr.Image(type="pil", label="Style Condition", visible=False, interactive=True)
        with gr.Column(scale=4):
            gr.Markdown("### Generated Images")
            with gr.Row():
                out_gen_1 = gr.Image(type="pil", visible=True, show_label=False)
                out_gen_2 = gr.Image(type="pil", visible=True, show_label=False)
            with gr.Row():
                out_gen_3 = gr.Image(type="pil", visible=False, show_label=False)
                out_gen_4 = gr.Image(type="pil", visible=False, show_label=False)

        state = gr.State({})

        class Controller:
            def __init__(self):
                self.calls = 0
                self.tracks = 0
                self.resizes = 0
                self.scales = 0

            def init_white(self, init_white_trigger):
                self.calls += 1
                return np.ones((512, 512), dtype='uint8') * 255, 1.0, init_white_trigger+1

            def change_n_samples(self, n_samples):
                blank_samples = n_samples % 2 if n_samples > 1 else 0
                return [gr.Image.update(visible=True) for _ in range(n_samples + blank_samples)] \
                    + [gr.Image.update(visible=False) for _ in range(4 - n_samples - blank_samples)]

            def resize_centercrop(self, state):
                self.resizes += 1
                image = state['original_image'].copy()
                inpaint_hw = int(0.9 * min(*image.shape[:2]))
                state['inpaint_hw'] = inpaint_hw
                image_cc = center_crop(image, inpaint_hw)
                # print(f'resize triggered {self.resizes}', image.shape, '->', image_cc.shape)
                return image_cc, state

            def resize_masked(self, state):
                self.resizes += 1
                image = state['original_image'].copy()
                inpaint_hw = int(0.9 * min(*image.shape[:2]))
                state['inpaint_hw'] = inpaint_hw
                image_mask = sized_center_mask(image, inpaint_hw, inpaint_hw)
                state['masked_image'] = image_mask.copy()
                # print(f'mask triggered {self.resizes}')
                return image_mask, state
            
            def switch_task_hide_cond(self, task):
                cond = False
                if task == "Grounded Generation":
                    cond = True

                return gr.Checkbox.update(visible=cond, value=False), gr.Image.update(value=None, visible=False), gr.Slider.update(visible=cond), gr.Checkbox.update(visible=(not cond), value=False)

        controller = Controller()
        main.load(
            lambda x:x+1,
            inputs=sketch_pad_trigger,
            outputs=sketch_pad_trigger,
            queue=False)
        sketch_pad.edit(
            draw,
            inputs=[task, sketch_pad, grounding_instruction, sketch_pad_resize_trigger, state],
            outputs=[out_imagebox, sketch_pad_resize_trigger, image_scale, state],
            queue=False,
        )
        grounding_instruction.change(
            draw,
            inputs=[task, sketch_pad, grounding_instruction, sketch_pad_resize_trigger, state],
            outputs=[out_imagebox, sketch_pad_resize_trigger, image_scale, state],
            queue=False,
        )
        clear_btn.click(
            clear,
            inputs=[task, sketch_pad_trigger, batch_size, state],
            outputs=[sketch_pad, sketch_pad_trigger, out_imagebox, image_scale, out_gen_1, out_gen_2, out_gen_3, out_gen_4, state],
            queue=False)
        task.change(
            partial(clear, switch_task=True),
            inputs=[task, sketch_pad_trigger, batch_size, state],
            outputs=[sketch_pad, sketch_pad_trigger, out_imagebox, image_scale, out_gen_1, out_gen_2, out_gen_3, out_gen_4, state],
            queue=False)
        sketch_pad_trigger.change(
            controller.init_white,
            inputs=[init_white_trigger],
            outputs=[sketch_pad, image_scale, init_white_trigger],
            queue=False)
        sketch_pad_resize_trigger.change(
            controller.resize_masked,
            inputs=[state],
            outputs=[sketch_pad, state],
            queue=False)
        batch_size.change(
            controller.change_n_samples,
            inputs=[batch_size],
            outputs=[out_gen_1, out_gen_2, out_gen_3, out_gen_4],
            queue=False)
        gen_btn.click(
            generate,
            inputs=[
                task, language_instruction, grounding_instruction, sketch_pad,
                alpha_sample, guidance_scale, batch_size,
                fix_seed, rand_seed,
                use_actual_mask,
                append_grounding, style_cond_image,
                state,
            ],
            outputs=[out_gen_1, out_gen_2, out_gen_3, out_gen_4, state],
            queue=True
        )
        sketch_pad_resize_trigger.change(
            None,
            None,
            sketch_pad_resize_trigger,
            _js=rescale_js,
            queue=False)
        init_white_trigger.change(
            None,
            None,
            init_white_trigger,
            _js=rescale_js,
            queue=False)
        use_style_cond.change(
            lambda cond: gr.Image.update(visible=cond),
            use_style_cond,
            style_cond_image,
            queue=False)
        task.change(
            controller.switch_task_hide_cond,
            inputs=task,
            outputs=[use_style_cond, style_cond_image, alpha_sample, use_actual_mask],
            queue=False)

    with gr.Column():
        gr.Examples(
            examples=[
                [
                    "images/blank.png",
                    "Grounded Generation",
                    "a dog and an apple",
                    "a dog;an apple",
                ],
                [
                    "images/blank.png",
                    "Grounded Generation",
                    "John Lennon is using a pc",
                    "John Lennon;a pc",
                [
                    "images/blank.png",
                    "Grounded Generation",
                    "a painting of a fox sitting in a field at sunrise in the style of Claude Mone",
                    "fox;sunrise",
                ],
                ],
                [
                    "images/blank.png",
                    "Grounded Generation",
                    "a beautiful painting of hot dog by studio ghibli, octane render, brilliantly coloured",
                    "hot dog",
                ],
                [
                    "images/blank.png",
                    "Grounded Generation",
                    "a sport car, unreal engine, global illumination, ray tracing",
                    "a sport car",
                ],
                [
                    "images/flower_beach.jpg",
                    "Grounded Inpainting",
                    "a squirrel and the space needle",
                    "a squirrel;the space needle",
                ],
                [
                    "images/arg_corgis.jpeg",
                    "Grounded Inpainting",
                    "a dog and a birthday cake",
                    "a dog; a birthday cake",
                ],
                [
                    "images/teddy.jpg",
                    "Grounded Inpainting",
                    "a teddy bear wearing a santa claus red shirt; holding a Christmas gift box on hand",
                    "a santa claus shirt; a Christmas gift box",
                ],
            ],
            inputs=[sketch_pad, task, language_instruction, grounding_instruction],
            outputs=None,
            fn=None,
            cache_examples=False,
        )

main.queue(concurrency_count=1, api_open=False)
main.launch(share=False, show_api=False)


