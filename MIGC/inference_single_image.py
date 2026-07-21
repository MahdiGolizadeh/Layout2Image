import argparse
import json
import os
from pathlib import Path

from diffusers import EulerDiscreteScheduler
from migc.migc_utils import seed_everything
from migc.migc_pipeline import StableDiffusionMIGCPipeline, MIGCProcessor, AttentionStore


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


def load_prompt_and_boxes(json_path):
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
        for instance in instances:
            if not isinstance(instance, list) or len(instance) != 2:
                raise ValueError(f"Invalid instance in record {index}: {instance}")
            instance_prompt, bbox = instance
            prompt.append(instance_prompt)
            bboxes.append(_normalize_bbox(bbox, width, height))
        samples.append({"key": _record_key(record, index), "prompt": [prompt], "bboxes": [bboxes]})
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Run MIGC inference from layout JSON records.")
    parser.add_argument("json_path", help="Path to the layout JSON file.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated images. Defaults to outputs/<json_stem>.")
    parser.add_argument("--seed", type=int, default=7351007268695528845)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--migc-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--negative-prompt", default="worst quality, low quality, bad anatomy, watermark, text, blurry")
    parser.add_argument("--show", action="store_true", help="Display generated and annotated images.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    migc_ckpt_path = 'pretrained_weights/MIGC_SD14.ckpt'
    assert os.path.isfile(migc_ckpt_path), "Please download the ckpt of migc and put it in the pretrained_weights/ folder!"

    sd1x_path = '/sdb/zdw/weights/stable-diffusion-v1-4' if os.path.isdir('/sdb/zdw/weights/stable-diffusion-v1-4') else "CompVis/stable-diffusion-v1-4"
    # MIGC is a plug-and-play controller.
    # You can go to https://civitai.com/search/models?baseModel=SD%201.4&baseModel=SD%201.5&sortBy=models_v5 find a base model with better generation ability to achieve better creations.

    pipe = StableDiffusionMIGCPipeline.from_pretrained(sd1x_path)
    pipe.attention_store = AttentionStore()
    from migc.migc_utils import load_migc
    load_migc(pipe.unet, pipe.attention_store, migc_ckpt_path, attn_processor=MIGCProcessor)
    pipe = pipe.to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    samples = load_prompt_and_boxes(args.json_path)
    json_stem = Path(args.json_path).stem
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / json_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    for sample in samples:
        image = pipe(
            sample["prompt"],
            sample["bboxes"],
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            MIGCsteps=args.migc_steps,
            aug_phase_with_and=False,
            negative_prompt=args.negative_prompt,
        ).images[0]
        image_path = output_dir / f'{sample["key"]}.png'
        image.save(image_path)
        if args.show:
            image.show()

        anno_image = pipe.draw_box_desc(image, sample["bboxes"][0], sample["prompt"][0][1:])
        anno_image_path = output_dir / f'{sample["key"]}_anno.png'
        anno_image.save(anno_image_path)
        if args.show:
            anno_image.show()
