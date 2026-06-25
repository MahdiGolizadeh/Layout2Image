import argparse
import json
import os
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate one image from user-provided LayoutDiffuse bounding boxes "
            "and COCO class names without Gradio, Flask, or a background watcher."
        )
    )
    parser.add_argument("-c", "--config", default="configs/cocostuff_SD2_1.json", help="Path to LayoutDiffuse config JSON.")
    parser.add_argument("--model_path", default=None, help="Checkpoint path. If omitted, uses latest.ckpt from the config experiment folder.")
    parser.add_argument("-e", "--epoch", type=int, default=None, help="Epoch checkpoint to load when --model_path is omitted.")
    parser.add_argument("--openai_api_key", default=None, help="Optional key for prompt expansion. Omit to use class labels as the prompt.")
    parser.add_argument("--additional_caption", default="", help="Optional text appended to the generated/default prompt.")
    parser.add_argument("--image_width", type=int, default=512, help="Output image width in pixels.")
    parser.add_argument("--image_height", type=int, default=512, help="Output image height in pixels.")
    parser.add_argument("--output_dir", default="outputs/layout_generation", help="Directory for generated images.")
    parser.add_argument("--output_name", default="layout_sample", help="Base filename without extension.")
    parser.add_argument(
        "--bbox_format",
        choices=("xywh", "xyxy"),
        default="xywh",
        help="Format for each box: xywh = x,y,width,height; xyxy = x1,y1,x2,y2.",
    )
    parser.add_argument(
        "--box_units",
        choices=("normalized", "pixels"),
        default="normalized",
        help="Use normalized coordinates in [0, 1] or pixel coordinates in the requested output size.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--layout_json",
        help=(
            "JSON string or path to a JSON file. Expected format: "
            "[{\"bbox\":[x,y,w,h],\"class\":\"person\"}, ...]."
        ),
    )
    input_group.add_argument(
        "--layout_txt",
        help="Path to a text file with one object per line: x,y,w,h,class_name.",
    )
    return parser.parse_args()


def load_config(args):
    with open(args.config, "r") as handle:
        config = json.load(handle)
    config.update(vars(args))
    return config


def load_layout_json(layout_json):
    try:
        possible_path = Path(layout_json)
        if possible_path.exists():
            with possible_path.open("r") as handle:
                return json.load(handle)
    except OSError:
        # Treat strings that are not valid filesystem paths as inline JSON.
        pass
    return json.loads(layout_json)


def convert_box(box, *, bbox_format, box_units, image_width, image_height):
    if len(box) != 4:
        raise ValueError(f"Each bbox must contain exactly four numbers, got {box!r}")
    x1, y1, third, fourth = [float(value) for value in box]
    if bbox_format == "xyxy":
        x1, y1, x2, y2 = x1, y1, third, fourth
        width = x2 - x1
        height = y2 - y1
    else:
        width = third
        height = fourth
    if box_units == "pixels":
        x1 /= image_width
        width /= image_width
        y1 /= image_height
        height /= image_height
    normalized = [x1, y1, width, height]
    if any(value < 0 or value > 1 for value in normalized):
        raise ValueError(
            f"Normalized xywh box values must be in [0, 1]; got {normalized}. "
            "Use --box_units pixels for pixel coordinates."
        )
    return normalized


def write_layout_file(args):
    if args.layout_txt:
        return args.layout_txt, False
    objects = load_layout_json(args.layout_json)
    if not isinstance(objects, list) or len(objects) == 0:
        raise ValueError("--layout_json must contain a non-empty list of objects.")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    with tmp:
        for obj in objects:
            class_name = obj.get("class", obj.get("class_name", obj.get("label")))
            if class_name is None:
                raise ValueError(f"Missing class/class_name/label in object: {obj!r}")
            bbox = convert_box(
                obj["bbox"],
                bbox_format=args.bbox_format,
                box_units=args.box_units,
                image_width=args.image_width,
                image_height=args.image_height,
            )
            tmp.write(",".join([str(value) for value in bbox] + [str(class_name)]))
            tmp.write("\n")
    return tmp.name, True


def move_model_to_device(ddpm_model, device):
    ddpm_model = ddpm_model.to(device)
    ddpm_model.text_fn = ddpm_model.text_fn.to(device)
    ddpm_model.text_fn.device = device
    ddpm_model.denoise_fn = ddpm_model.denoise_fn.to(device)
    ddpm_model.vqvae_fn = ddpm_model.vqvae_fn.to(device)
    return ddpm_model


def save_rgb_image(path, image):
    import cv2
    import numpy as np

    cv2.imwrite(str(path), (image[..., ::-1] * 255).astype(np.uint8))


def main():
    args = parse_args()

    import numpy as np
    import torch

    from data.coco_w_stuff import get_coco_id_mapping
    from test_utils import load_model_weights, load_test_models, sample_one_image

    config = load_config(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    coco_id_to_name = get_coco_id_mapping()
    coco_name_to_id = {name: int(idx) for idx, name in coco_id_to_name.items()}

    ddpm_model = load_test_models(config)
    load_model_weights(ddpm_model=ddpm_model, args=config)
    ddpm_model = move_model_to_device(ddpm_model, device)

    layout_path, should_delete_layout = write_layout_file(args)
    try:
        image, image_with_bbox, canvas_with_bbox = sample_one_image(
            layout_path,
            ddpm_model,
            device,
            class_name_to_id=coco_name_to_id,
            class_id_to_name=coco_id_to_name,
            api_key=config.get("openai_api_key"),
            image_size=(args.image_height, args.image_width),
            additional_caption=config.get("additional_caption", ""),
        )
    finally:
        if should_delete_layout:
            os.remove(layout_path)

    if image is None:
        raise RuntimeError("No valid COCO objects were found in the provided layout.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{args.output_name}.jpg"
    image_with_bbox_path = output_dir / f"{args.output_name}_with_bbox.jpg"
    canvas_with_bbox_path = output_dir / f"{args.output_name}_layout.jpg"
    triptych_path = output_dir / f"{args.output_name}_triptych.jpg"

    save_rgb_image(image_path, image)
    save_rgb_image(image_with_bbox_path, image_with_bbox)
    save_rgb_image(canvas_with_bbox_path, canvas_with_bbox)
    save_rgb_image(triptych_path, np.concatenate([image, image_with_bbox, canvas_with_bbox], axis=1))

    print(f"Saved generated image: {image_path}")
    print(f"Saved image with boxes: {image_with_bbox_path}")
    print(f"Saved layout canvas: {canvas_with_bbox_path}")
    print(f"Saved triptych: {triptych_path}")


if __name__ == "__main__":
    main()
