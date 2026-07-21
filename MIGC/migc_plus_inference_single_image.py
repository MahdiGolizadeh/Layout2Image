import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

from diffusers import EulerDiscreteScheduler
from migc.migc_utils import seed_everything
from migc_plus.migc_plus_utils import change_bbox_to_mask
from migc_plus.migc_plus_pipeline import StableDiffusionMIGCPlusPipeline, MIGCPlusProcessor, AttentionStore


def _record_key(record, index):
    metadata = record[0] if record and isinstance(record[0], dict) else {}
    return str(metadata.get("key") or metadata.get("id") or f"sample_{index:06d}")


def _record_size(record):
    metadata = record[0] if record and isinstance(record[0], dict) else {}
    size = record[3] if len(record) > 3 and isinstance(record[3], dict) else {}
    width = int(size.get("W") or metadata.get("width") or 512)
    height = int(size.get("H") or metadata.get("height") or 512)
    return width, height


def _normalize_bbox(bbox, width, height):
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got {bbox}")
    x1, y1, x2, y2 = [float(value) for value in bbox]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        return [x1, y1, x2, y2]
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def load_prompt_boxes_and_masks(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError("Input JSON must contain a list of records.")

    samples = []
    for index, record in enumerate(records):
        if not isinstance(record, list) or len(record) < 6:
            raise ValueError(f"Record {index} must be a list with at least 6 entries.")
        caption = record[1]
        instances = record[5]
        width, height = _record_size(record)
        prompt = [caption]
        bboxes = []
        masks = []
        for instance in instances:
            if not isinstance(instance, list) or len(instance) != 2:
                raise ValueError(f"Invalid instance in record {index}: {instance}")
            instance_prompt, bbox_values = instance
            bbox = _normalize_bbox(bbox_values, width, height)
            prompt.append(instance_prompt)
            bboxes.append(bbox)
            masks.append(change_bbox_to_mask(bbox, height=height, width=width))
        samples.append({"key": _record_key(record, index), "prompt": [prompt], "bboxes": [bboxes], "masks": masks})
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Run MIGC++ inference from layout JSON records.")
    parser.add_argument("json_path", help="Path to the layout JSON file.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated images. Defaults to outputs/<json_stem>.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--migc-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--negative-prompt", default="worst quality, low quality, bad anatomy, watermark, text, blurry")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    migc_plus_ckpt_path = 'pretrained_weights/MIGC++_SD14.ckpt'
    assert os.path.isfile(migc_plus_ckpt_path), "Please download the ckpt of migc++ and put it in the pretrained_weights/ folder!"

    sd1x_path = '/mnt/sda/zdw/ckpt/new_sd14' if os.path.isdir('/mnt/sda/zdw/ckpt/new_sd14') else "CompVis/stable-diffusion-v1-4"
    # MIGC is a plug-and-play controller.
    # You can go to https://civitai.com/search/models?baseModel=SD%201.4&baseModel=SD%201.5&sortBy=models_v5 find a base model with better generation ability to achieve better creations.

    pipe = StableDiffusionMIGCPlusPipeline.from_pretrained(sd1x_path)
    pipe.attention_store = AttentionStore()
    from migc_plus.migc_plus_utils import load_migc_plus
    load_migc_plus(pipe.unet, pipe.attention_store, migc_plus_ckpt_path, attn_processor=MIGCPlusProcessor)
    pipe = pipe.to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    samples = load_prompt_boxes_and_masks(args.json_path)
    json_stem = Path(args.json_path).stem
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / json_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    for sample in samples:
        image = pipe(
            deepcopy(sample["prompt"]),
            sample["bboxes"],
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            MIGCsteps=args.migc_steps,
            aug_phase_with_and=False,
            negative_prompt=args.negative_prompt,
            RefinedSteps=args.num_inference_steps,
            masks=sample["masks"],
        ).images[0]
        image.save(output_dir / f'{sample["key"]}_MIGC++.png')
