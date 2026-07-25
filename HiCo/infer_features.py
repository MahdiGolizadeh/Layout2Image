import argparse
import json
import os
import random
import sys

os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import PIL.Image
import torch

LOCAL_DIFFUSERS_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diffusers", "src")
if LOCAL_DIFFUSERS_SRC not in sys.path:
    sys.path.insert(0, LOCAL_DIFFUSERS_SRC)

from utils.demo_visiual_bbox import draw_image


PLACEHOLDER_IMAGE_PATH = "The local path of your own image."


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run HiCo inference and save UNet encoder/decoder activations plus HiCo "
            "ControlNet residual features before and after fusion."
        )
    )
    parser.add_argument("--json", default="results/examples/json_1.json", help="Path to the inference JSON file.")
    parser.add_argument("--controlnet-path", default="models/controlnet", help="Path to the HiCo ControlNet checkpoint.")
    parser.add_argument(
        "--base-model-path",
        default="models/realisticVisionV51_v51VAE",
        help="Path to the Stable Diffusion 1.5-compatible base model.",
    )
    parser.add_argument("--image", default=None, help="Optional path to an input image.")
    parser.add_argument("--save-dir", default="./results", help="Directory where generated images are saved.")
    parser.add_argument(
        "--feature-dir",
        default="./results/features",
        help="Directory where activation maps and feature tensors are saved.",
    )
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--scheduler", choices=("unipc", "dpm"), default="unipc", help="Denoising scheduler.")
    parser.add_argument("--fuse-type", choices=("avg", "sum"), default="avg", help="HiCo feature fusion mode.")
    parser.add_argument("--infer-mode", choices=("batch", "single"), default="batch", help="HiCo inference mode.")
    parser.add_argument(
        "--use-unet-prompt",
        action="store_true",
        help="Pass the global caption into the base UNet prompt instead of using an empty prompt.",
    )
    parser.add_argument("--prompt-prefix", default="", help="Optional text prepended to the global prompt.")
    parser.add_argument("--negative-prompt", default=None, help="Negative prompt for classifier-free guidance.")
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0, help="HiCo ControlNet scale.")
    parser.add_argument("--control-guidance-start", type=float, default=0.0, help="ControlNet guidance start fraction.")
    parser.add_argument("--control-guidance-end", type=float, default=1.0, help="ControlNet guidance end fraction.")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed. Use -1 to sample a new seed per image.")
    parser.add_argument(
        "--background-mask",
        choices=("blank", "full"),
        default="blank",
        help="Use a blank or full-white mask for the global/background caption.",
    )
    parser.add_argument("--bbox-padding", type=int, default=0, help="Expand object boxes before building masks.")
    parser.add_argument(
        "--feature-mode",
        choices=("activation", "tensor", "both"),
        default="activation",
        help=(
            "`activation` saves channel-averaged absolute maps; `tensor` saves raw tensors; "
            "`both` saves both. Raw tensors can be very large."
        ),
    )
    parser.add_argument(
        "--feature-every",
        type=int,
        default=1,
        help="Save features every N denoising steps. Use values >1 to reduce output size.",
    )
    parser.add_argument(
        "--feature-batch-index",
        type=int,
        default=None,
        help="Optionally save only one batch element from feature tensors, e.g. 1 for the conditional CFG branch.",
    )
    parser.add_argument("--enable-xformers", action="store_true", help="Enable xFormers attention if installed.")
    parser.add_argument("--enable-model-cpu-offload", action="store_true", help="Enable model CPU offload.")
    return parser.parse_args()


def optional_image_path(args, image_path, base_info=None):
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


def load_image(args, image_path, img_size, base_info=None):
    image_path = optional_image_path(args, image_path, base_info)
    if not image_path:
        width = int(img_size.get("W", img_size.get("width", 512)))
        height = int(img_size.get("H", img_size.get("height", 512)))
        return PIL.Image.new("RGB", (width, height), color=(0, 0, 0))
    with open(image_path, "rb") as f:
        with PIL.Image.open(f) as image:
            return image.convert("RGB")


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


def first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


class FeatureRecorder:
    def __init__(self, root_dir, feature_mode="activation", feature_every=1, batch_index=None):
        self.root_dir = root_dir
        self.feature_mode = feature_mode
        self.feature_every = feature_every
        self.batch_index = batch_index
        self.current_step = -1
        self.current_timestep = None
        self.handles = []
        self.manifest = []
        os.makedirs(self.root_dir, exist_ok=True)

    def attach_unet(self, unet):
        self.handles.append(unet.register_forward_pre_hook(self._unet_pre_hook))
        for idx, block in enumerate(unet.down_blocks):
            self.handles.append(block.register_forward_hook(self._make_unet_hook(f"encoder_down_{idx}")))
        self.handles.append(unet.mid_block.register_forward_hook(self._make_unet_hook("bottleneck_mid")))
        for idx, block in enumerate(unet.up_blocks):
            self.handles.append(block.register_forward_hook(self._make_unet_hook(f"decoder_up_{idx}")))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _unet_pre_hook(self, module, inputs):
        self.current_step += 1
        self.current_timestep = inputs[1].detach().cpu() if len(inputs) > 1 and torch.is_tensor(inputs[1]) else inputs[1]

    def _make_unet_hook(self, name):
        def hook(module, inputs, output):
            tensor = first_tensor(output)
            if tensor is not None:
                self.save_tensor("unet", name, tensor, self.current_step, self.current_timestep)

        return hook

    def hico_callback(self, payload):
        step = payload["timestep_index"]
        timestep = payload["timestep"]
        stage = payload["stage"]
        down_samples = payload["down_block_res_samples"]
        mid_sample = payload["mid_block_res_sample"]

        for idx, tensor in enumerate(down_samples):
            self.save_tensor(stage, f"down_residual_{idx}", tensor, step, timestep)
        self.save_tensor(stage, "mid_residual", mid_sample, step, timestep)

    def save_tensor(self, source, name, tensor, step, timestep):
        if step % self.feature_every != 0:
            return

        tensor_to_save = tensor.detach().float().cpu()
        if self.batch_index is not None and tensor_to_save.ndim > 0:
            tensor_to_save = tensor_to_save[self.batch_index : self.batch_index + 1]

        payload = {
            "source": source,
            "name": name,
            "step": int(step),
            "timestep": timestep.detach().cpu() if torch.is_tensor(timestep) else timestep,
            "shape": tuple(tensor_to_save.shape),
        }

        if self.feature_mode in ("tensor", "both"):
            payload["tensor"] = tensor_to_save
        if self.feature_mode in ("activation", "both"):
            payload["activation_map"] = self.to_activation_map(tensor_to_save)

        filename = f"step_{step:03d}_{source}_{name}.pt"
        path = os.path.join(self.root_dir, filename)
        torch.save(payload, path)
        self.manifest.append({"path": path, "source": source, "name": name, "step": int(step), "shape": payload["shape"]})

    @staticmethod
    def to_activation_map(tensor):
        if tensor.ndim == 4:
            return tensor.abs().mean(dim=1)
        if tensor.ndim == 3:
            return tensor.abs().mean(dim=0, keepdim=True)
        return tensor.abs()

    def save_manifest(self):
        manifest_path = os.path.join(self.root_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)
        return manifest_path


def prepare_layout(args, image, caption, list_bbox_info):
    obj_bbox = [pad_bbox(obj[1], args.bbox_padding) for obj in list_bbox_info]
    obj_bbox = np.clip(np.array(obj_bbox), 0, 512)
    obj_class = [obj[0] for obj in list_bbox_info]

    width, height = image.size
    if width != 512 or height != 512:
        raise ValueError(f"HiCo inference expects 512x512 images, got {width}x{height}.")

    layout_classes = obj_class.copy()
    layout_boxes = obj_bbox.copy()
    layout_classes.insert(0, caption)
    layout_boxes = np.insert(layout_boxes, obj=0, values=[0, 0, 512, 512], axis=0)

    cond_images = []
    background = np.zeros((height, width, 3), dtype=np.uint8)
    if args.background_mask == "full":
        background[:, :] = 255
    cond_images.append(background)

    for box in layout_boxes[1:]:
        x1, y1, x2, y2 = [int(xx) for xx in box]
        cond_image = np.zeros((height, width, 3), dtype=np.uint8)
        cond_image[y1:y2, x1:x2] = 255
        cond_images.append(cond_image)

    cond_images_pil = [PIL.Image.fromarray(cond).convert("RGB") for cond in cond_images]
    return layout_classes, layout_boxes, cond_images_pil


def main():
    args = parse_args()

    from diffusers import (
        ControlNetModel,
        DPMSolverMultistepScheduler,
        StableDiffusionHicoNetLayoutPipeline,
        UniPCMultistepScheduler,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.feature_dir, exist_ok=True)

    with open(args.json, encoding="utf-8") as f:
        json_data = json.load(f)

    for entry in json_data:
        if len(entry) < 6:
            raise ValueError(f"Expected at least 6 fields in dataset entry, got {len(entry)}: {entry}")

    if args.image and not os.path.isfile(args.image):
        raise FileNotFoundError(f"Input image override not found: {args.image!r}.")

    hico_net = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float32)
    pipe = StableDiffusionHicoNetLayoutPipeline.from_pretrained(
        args.base_model_path, controlnet=[hico_net], torch_dtype=torch.float32
    )
    pipe.enable_attention_slicing()

    if args.scheduler == "unipc":
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if args.enable_xformers:
        pipe.enable_xformers_memory_efficient_attention()
    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    for entry in json_data:
        base_info, caption, obj_nums, img_size, path_img, list_bbox_info = entry[:6]
        img_id = base_info["id"]
        input_image = load_image(args, path_img, img_size, base_info)
        layo_prompt, layout_boxes, cond_images_pil = prepare_layout(args, input_image, caption, list_bbox_info)

        prompt = args.prompt_prefix + caption if args.use_unet_prompt else ""
        seed = args.seed if args.seed != -1 else int(random.randrange(4294967294))
        generator = torch.manual_seed(seed)
        print(f"Using seed {seed} for image {img_id}.")

        image_feature_dir = os.path.join(args.feature_dir, str(img_id))
        recorder = FeatureRecorder(
            image_feature_dir,
            feature_mode=args.feature_mode,
            feature_every=args.feature_every,
            batch_index=args.feature_batch_index,
        )
        recorder.attach_unet(pipe.unet)

        output = pipe(
            prompt,
            layo_prompt,
            guidance_scale=args.guidance_scale,
            infer_mode=args.infer_mode,
            num_inference_steps=args.num_inference_steps,
            image=cond_images_pil,
            fuse_type=args.fuse_type,
            width=512,
            height=512,
            generator=generator,
            negative_prompt=args.negative_prompt,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            control_guidance_start=args.control_guidance_start,
            control_guidance_end=args.control_guidance_end,
            debug_feature_callback=recorder.hico_callback,
        ).images[0]

        recorder.close()
        manifest_path = recorder.save_manifest()

        image_path = os.path.join(args.save_dir, f"{args.infer_mode}_{args.fuse_type}_{img_id}.png")
        output.save(image_path)
        bbox_path = os.path.join(args.save_dir, f"{args.infer_mode}_{args.fuse_type}_{img_id}_bbox.png")
        draw_image(np.array(output) / 255, layout_boxes, layo_prompt, bbox_path)
        print(f"Saved image to {image_path}.")
        print(f"Saved feature manifest to {manifest_path}.")


if __name__ == "__main__":
    main()
