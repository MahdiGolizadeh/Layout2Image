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
    parser.add_argument(
        "--image",
        default=None,
        help=(
            "Optional path to an input image. If omitted, the script builds a blank "
            "layout canvas from the JSON image size."
        ),
    )
    parser.add_argument("--save-dir", default="./results", help="Directory where generated images are saved.")
    return parser.parse_args()


args = parse_args()

fuse_type = "avg"   # "avg", "sum"
mode = "batch"      # "batch", "single"   "batch" for parallel processing  "single" for sequential processing
unet_flag = 0
cfg = 7.5
controlnet_path= "" #HiCo checkpoints
base_model = ""     #SD 1.5 checkpoints
schd = "UniPCM"
save_dir_base = args.save_dir
# avg
controlnet_path = "models/controlnet"

base_model_path = "models/realisticVisionV51_v51VAE" 

# sum
# controlnet_path = "models/controlnet"

# base_model_path = "models/realisticVisionV51_v51VAE" 

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

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(save_dir_bbox):
    os.mkdir(save_dir_bbox)
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
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
if schd == "UniPCM":
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
elif schd == "DPM":
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
else:
    raise ValueError("Scheduler setup error.")

# remove following line if xformers is not installed or when using Torch 2.0.
#pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
#pipe.enable_model_cpu_offload()
pipe.to('cuda')



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

    obj_bbox = [obj[1] for obj in list_bbox_info]
    obj_bbox = np.array(obj_bbox)
    obj_bbox = np.clip(obj_bbox, 0, 512)

    obj_class = [obj[0] for obj in list_bbox_info]


    W, H = image.size

    r_image = image
    r_obj_bbox = obj_bbox
    r_obj_class = obj_class

    if W != 512 and H != 512:
        print ("image size is not 512." % img_id)
        continue

    r_obj_class.insert(0, caption)
    r_obj_bbox = np.insert(r_obj_bbox, obj=0, values=[0,0,512,512], axis=0)
    list_cond_image = []
    cond_image = np.zeros((H, W, 3), dtype=np.uint8)
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
        seed = -1
        if seed == -1: 
            seed = int(random.randrange(4294967294))
        generator = torch.manual_seed(seed)

        list_cond_image_pil = [PIL.Image.fromarray(dot_cond).convert('RGB') for dot_cond in list_cond_image]
        image = pipe(
            prompt, layo_prompt, guidance_scale=cfg, infer_mode=mode,
            num_inference_steps=50, image=list_cond_image_pil, fuse_type=fuse_type,
            width=512, height=512, generator=generator,
        ).images[0]
        img_name = "%s" % (img_id)
        image.save("%s/%s_%s_%s.png" % (save_dir, mode, fuse_type, img_name))

        cond_image = np.array(image) / 255
        draw_image(cond_image, r_obj_bbox, r_obj_class, "%s/%s_%s_%s_bbox.png" % (save_dir_bbox, mode, fuse_type, img_name))

