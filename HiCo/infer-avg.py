import json
import os
import sys
import argparse

# The local diffusers fork used by HiCo only needs the PyTorch pipeline for
# this script.  Newer Colab/Python images may have JAX/Flax installed while
# using a transformers build that no longer exposes FlaxCLIPTextModel, which
# makes diffusers import optional Flax Stable Diffusion modules and fail before
# inference starts. Disable Flax auto-detection unless the caller explicitly
# opts back in.
os.environ.setdefault("USE_FLAX", "0")
PLACEHOLDER_IMAGE_PATH = "The local path of your own image."
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run HiCo inference. By default this reads results/examples/json_1.json; "
            "the image path is optional because generation uses prompt and layout boxes."
        )
    )
    parser.add_argument("--json", default="results/examples/json_1.json", help="Path to the inference JSON file.")
    parser.add_argument("--controlnet-path", default="models/controlnet", help="Path to the HiCo ControlNet checkpoint.")
    parser.add_argument(
        "--base-model-path",
        default="models/realisticVisionV51_v51VAE",
        help="Path to the Stable Diffusion 1.5-compatible base model.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help=(
            "Optional path to an input image. If omitted, the script builds a blank "
            "layout canvas from the JSON image size."
        ),
    )
    parser.add_argument("--save-dir", default="./results", help="Directory where generated images are saved.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument(
        "--scheduler",
        choices=("unipc", "dpm"),
        default="unipc",
        help="Scheduler to use for denoising.",
    )
    parser.add_argument(
        "--fuse-type",
        choices=("avg", "sum"),
        default="avg",
        help="How to fuse per-layout ControlNet residuals. `mask` exists in the pipeline but is not implemented.",
    )
    parser.add_argument(
        "--infer-mode",
        choices=("batch", "single"),
        default="batch",
        help="Run layout conditions in one ControlNet batch or sequentially.",
    )
    parser.add_argument(
        "--use-unet-prompt",
        action="store_true",
        help="Pass the global caption into the base UNet prompt instead of using an empty prompt.",
    )
    parser.add_argument(
        "--prompt-prefix",
        default="",
        help="Optional text prepended to the global prompt, e.g. 'photorealistic, high quality, '.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Negative prompt used for classifier-free guidance.",
    )
    parser.add_argument(
        "--controlnet-conditioning-scale",
        type=float,
        default=1.0,
        help="Scale applied to HiCo ControlNet residuals before UNet injection.",
    )
    parser.add_argument(
        "--control-guidance-start",
        type=float,
        default=0.0,
        help="Fraction of denoising steps when ControlNet guidance starts.",
    )
    parser.add_argument(
        "--control-guidance-end",
        type=float,
        default=1.0,
        help="Fraction of denoising steps when ControlNet guidance ends.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed. Use -1 to sample a new random seed for each image.",
    )
    parser.add_argument(
        "--background-mask",
        choices=("blank", "full"),
        default="blank",
        help="Use a blank or full-white mask for the global/background caption.",
    )
    parser.add_argument(
        "--bbox-padding",
        type=int,
        default=0,
        help="Expand each object box by this many pixels before building masks.",
    )
    parser.add_argument(
        "--enable-xformers",
        action="store_true",
        help="Enable xFormers memory-efficient attention if xFormers is installed.",
    )
    parser.add_argument(
        "--enable-model-cpu-offload",
        action="store_true",
        help="Enable model CPU offload to reduce VRAM usage.",
    )
    return parser.parse_args()


args = parse_args()

fuse_type = args.fuse_type
mode = args.infer_mode      # "batch" for parallel processing, "single" for sequential processing
unet_flag = args.use_unet_prompt
cfg = args.guidance_scale
controlnet_path = args.controlnet_path
base_model_path = args.base_model_path
schd = args.scheduler
save_dir_base = args.save_dir

def optional_image_path(image_path, base_info=None):
    if args.image:
        return args.image
    if image_path == PLACEHOLDER_IMAGE_PATH and base_info:
            image_path = base_info.get("f_path", image_path)
    if isinstance(image_path, str) and image_path.startswith(PLACEHOLDER_IMAGE_PATH):
        return None

    if image_path and not os.path.isfile(image_path):
        print(f"Input image not found: {image_path!r}; using a blank layout canvas instead.")
        return None

    return image_path


def load_image(image_path, img_size, base_info=None):
    image_path = optional_image_path(image_path, base_info)

    if not image_path:
        width = int(img_size.get("W", img_size.get("width", 512)))
        height = int(img_size.get("H", img_size.get("height", 512)))
        return PIL.Image.new("RGB", (width, height), color=(0, 0, 0))
    with open(image_path, 'rb') as f:
        with PIL.Image.open(f) as image:
            image = image.convert('RGB')
    return image



save_dir = "%s/" % save_dir_base
save_dir_bbox = "%s/" % save_dir_base

os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_dir_bbox, exist_ok=True)
file_json = args.json   #  your own test samples path
 

with open(file_json, encoding='utf-8') as f:
    json_data = json.load(f)

for entry in json_data:
    if len(entry) < 6:
        raise ValueError(f"Expected at least 6 fields in dataset entry, got {len(entry)}: {entry}")

if args.image and not os.path.isfile(args.image):
    raise FileNotFoundError(f"Input image override not found: {args.image!r}.")


import random
import torch
import PIL
import numpy as np
from utils.demo_visiual_bbox import draw_image

LOCAL_DIFFUSERS_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diffusers', 'src')
if LOCAL_DIFFUSERS_SRC not in sys.path:
    sys.path.insert(0, LOCAL_DIFFUSERS_SRC)

from diffusers import  ControlNetModel, UniPCMultistepScheduler, DPMSolverMultistepScheduler, StableDiffusionHicoNetLayoutPipeline


HiCoNet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float32)

#pipe = StableDiffusionControlNetMultiLayoutPipeline.from_pretrained(
pipe = StableDiffusionHicoNetLayoutPipeline.from_pretrained(
    base_model_path, controlnet=[HiCoNet], torch_dtype=torch.float32
)
pipe.enable_attention_slicing()

#
# speed up diffusion process with faster scheduler and memory optimization
if schd == "unipc":
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
elif schd == "dpm":
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
else:
    raise ValueError("Scheduler setup error.")

# remove following line if xformers is not installed or when using Torch 2.0.
if args.enable_xformers:
    pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
if args.enable_model_cpu_offload:
    pipe.enable_model_cpu_offload()
else:
    pipe.to('cuda')


def pad_bbox(bbox, padding, width=512, height=512):
    x1, y1, x2, y2 = [int(xx) for xx in bbox]
    if padding:
        x1 -= padding
        y1 -= padding
        x2 += padding
        y2 += padding
    return [
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    ]



for v in json_data:

    # Dataset entries may include optional fields after list_bbox_info, such as
    # crop_location (documented in README.md).  Only the first six fields are
    # needed for inference, so keep the loader compatible with both the older
    # 6-field example format and the documented 7-field dataset format.
    if len(v) < 6:
        raise ValueError(f"Expected at least 6 fields in dataset entry, got {len(v)}: {v}")
    base_info, caption, obj_nums, img_size, path_img, list_bbox_info = v[:6]

    img_id = base_info["id"]
    image = load_image(path_img, img_size, base_info)

    obj_bbox = [pad_bbox(obj[1], args.bbox_padding) for obj in list_bbox_info]
    obj_bbox = np.array(obj_bbox)
    obj_bbox = np.clip(obj_bbox, 0, 512)

    obj_class = [obj[0] for obj in list_bbox_info]


    W, H = image.size

    r_image = image
    r_obj_bbox = obj_bbox.copy()
    r_obj_class = obj_class.copy()

    if W != 512 and H != 512:
        print ("image size is not 512." % img_id)
        continue

    r_obj_class.insert(0, caption)
    r_obj_bbox = np.insert(r_obj_bbox, obj=0, values=[0,0,512,512], axis=0)
    list_cond_image = []
    cond_image = np.zeros((H, W, 3), dtype=np.uint8)
    if args.background_mask == "full":
        cond_image[:, :] = 255
    list_cond_image.append(cond_image)
    for iit in range(1, len(r_obj_bbox)):
        dot_bbox = r_obj_bbox[iit]
        dx1, dy1, dx2, dy2 = [int(xx) for xx in dot_bbox]
        cond_image = np.zeros((H, W, 3), dtype=np.uint8)
        cond_image[dy1:dy2, dx1:dx2] = 255

        list_cond_image.append(cond_image)
    obj_cond_image = np.stack(list_cond_image, axis=0)


    layo_prompt = r_obj_class


    if unet_flag:
        prompt = caption
    else:
        prompt = ""



    if True:
        seed = args.seed
        if seed == -1: 
            seed = int(random.randrange(4294967294))
        generator = torch.manual_seed(seed)

        if seed != args.seed:
            print(f"Using random seed {seed} for image {img_id}.")
        else:
            print(f"Using seed {seed} for image {img_id}.")

        prompt = args.prompt_prefix + prompt if prompt else prompt
        list_cond_image_pil = [PIL.Image.fromarray(dot_cond).convert('RGB') for dot_cond in list_cond_image]
        image = pipe(
            prompt, layo_prompt, guidance_scale=cfg, infer_mode=mode,
            num_inference_steps=args.num_inference_steps, image=list_cond_image_pil, fuse_type=fuse_type,
            width=512, height=512, generator=generator,
            negative_prompt=args.negative_prompt,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            control_guidance_start=args.control_guidance_start,
            control_guidance_end=args.control_guidance_end,
        ).images[0]
        img_name = "%s" % (img_id)
        image.save("%s/%s_%s_%s.png" % (save_dir, mode, fuse_type, img_name))

        cond_image = np.array(image) / 255
        draw_image(cond_image, r_obj_bbox, r_obj_class, "%s/%s_%s_%s_bbox.png" % (save_dir_bbox, mode, fuse_type, img_name))
