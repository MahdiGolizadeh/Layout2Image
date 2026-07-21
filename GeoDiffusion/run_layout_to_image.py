import os
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from utils.generation_utils import load_checkpoint, bbox_encode, draw_layout

########################
# Set random seed
#########################
from accelerate.utils import set_seed
set_seed(0)

def normalize_json_record(record):
  """Convert one JSON input record to the layout format expected by GeoDiffusion."""
  if not isinstance(record, list) or len(record) < 6:
    raise ValueError("Each input JSON record must be a list with at least 6 entries.")

  image_info = record[0] if isinstance(record[0], dict) else {}
  caption = record[1]
  size_info = record[3] if isinstance(record[3], dict) else {}
  objects = record[5]

  width = int(size_info.get("W", image_info.get("width", 512)))
  height = int(size_info.get("H", image_info.get("height", 512)))
  if width <= 0 or height <= 0:
    raise ValueError(f"Image dimensions must be positive, got W={width}, H={height}.")

  bboxes = []
  for obj in objects:
    if not isinstance(obj, list) or len(obj) != 2:
      raise ValueError(f"Each object must be [label, [x1, y1, x2, y2]], got {obj}.")
    label, box = obj
    if len(box) != 4:
      raise ValueError(f"Bounding boxes must contain 4 values, got {box}.")
    x1, y1, x2, y2 = [float(coord) for coord in box]
    bboxes.append([label, x1 / width, y1 / height, x2 / width, y2 / height])

  layout = {
    "bbox": bboxes,
    "caption": caption,
    "width": width,
    "height": height,
  }
  if isinstance(image_info, dict):
    layout["key"] = image_info.get("key")
    layout["id"] = image_info.get("id")
  return layout


def load_layouts_from_json(input_json):
  with open(input_json, "r", encoding="utf-8") as f:
    records = json.load(f)
  if not isinstance(records, list):
    raise ValueError("Input JSON must contain a top-level list of records.")
  return [normalize_json_record(record) for record in records]


def run_layout_to_image(layout, args, pipe=None, generation_config=None, output_stem=None):
  ########################
  # Build pipeline
  #########################
  if pipe is None or generation_config is None:
    pipe, generation_config = load_checkpoint(args.ckpt_path)
    pipe = pipe.to("cuda")
  else:
    generation_config = generation_config.copy()
  args = {arg: getattr(args, arg) for arg in vars(args) if getattr(args, arg) is not None}
  generation_config.update(args)
  if "width" in layout:
    generation_config["width"] = layout["width"]
  if "height" in layout:
    generation_config["height"] = layout["height"]
  
  # Sometimes the nsfw checker is confused by the Pokémon images, you can disable
  # it at your own risk here
  disable_safety = True
  if disable_safety:
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
  
  ########################
  # Encode layout and build text prompt
  #########################
  # timeofday and weather sanity check  
  assert not generation_config['dataset'] == 'nuimages' or "timeofday" not in layout or layout['timeofday'] in ['daytime', 'night']
  assert not generation_config['dataset'] == 'nuimages' or "weather" not in layout or layout['weather'] in ['sunny', 'rain']
  if "timeofday" in generation_config['prompt_template'] and "timeofday" not in layout.keys():
    layout["timeofday"] = "daytime"
  if "weather" in generation_config['prompt_template'] and "weather" not in layout.keys():
    layout["weather"] = "sunny"

  # camera sanity check
  assert not generation_config['dataset'] == 'nuimages' or ("camera" in layout and layout['camera'] in ['front', 'front left', 'front right', 'back', 'back left', 'back right'])
  bboxes = layout['bbox'].copy()
  layout["bbox"] = bbox_encode(layout['bbox'], generation_config)
  prompt = generation_config['prompt_template'].format(**layout)
  if layout.get("caption") and "caption" not in generation_config['prompt_template']:
    prompt = f"{layout['caption']} {prompt}"
  print(prompt)
  
  ########################
  # Generation
  ########################
  # generation params
  width = generation_config["width"]  
  height = generation_config["height"]
  scale = generation_config["cfg_scale"]
  n_samples = generation_config["nsamples"]
  num_inference_steps = generation_config["num_inference_steps"]
  
  # run generation
  images = pipe(n_samples*[prompt], guidance_scale=scale, num_inference_steps=num_inference_steps, height=int(height), width=int(width)).images
  
  ########################
  # Save results
  #########################
  root = args["output_dir"]
  os.makedirs(root, exist_ok=True)
  layout_canvas = draw_layout(bboxes)
  output_stem = output_stem or generation_config['dataset']
  layout_canvas = Image.fromarray(layout_canvas, mode='RGB').save(os.path.join(root, f'{output_stem}_layout.jpg'))
  for idx, image in enumerate(images):
    image = np.asarray(image)
    image = Image.fromarray(image, mode='RGB')
    suffix = '' if len(images) == 1 else f'_{idx}'
    image.save(os.path.join(root, f'{output_stem}{suffix}.jpg'))

if __name__ == "__main__":
  parser = ArgumentParser(description='Layout-to-image generation script')
  parser.add_argument('ckpt_path', type=str)
  parser.add_argument('--nsamples', type=int, default=1)
  parser.add_argument('--cfg_scale', type=float, default=None)
  parser.add_argument('--num_inference_steps', type=int, default=None)
  parser.add_argument('--output_dir', type=str, default="./results/")
  parser.add_argument('--input_json', type=str, default=None, help='Path to a JSON file containing layout-to-image records.')
  args = parser.parse_args()
  
  if args.input_json:
    pipe, generation_config = load_checkpoint(args.ckpt_path)
    pipe = pipe.to("cuda")
    layouts = load_layouts_from_json(args.input_json)
    input_stem = Path(args.input_json).stem
    for idx, layout in enumerate(layouts):
      output_stem = input_stem if len(layouts) == 1 else f"{input_stem}_{idx:06d}"
      run_layout_to_image(layout, args, pipe=pipe, generation_config=generation_config, output_stem=output_stem)
    raise SystemExit(0)

  ########################
  # Define layouts
  # Note: 
  # 1) "camera": specific for nuimages, and should be selected from [front, front left, front right, back, back left, back right]
  # 2) "bbox": list of bounding boxes, each defined as [category, x1, y1, x2, y2] 
  #   a) category (str):, check dataset2classes in utils.generation_utils
  #   b) x1, y1, x2, y2 (float): in range of [0, 1]
  ########################
  # example layout for nuimages
  layout = {
    "camera": "front",
    "bbox": [
      ["car", 0.756875, 0.4622, 0.90375, 0.5844],
      ["car", 0.47625, 0.4822, 0.691875, 0.8011],
      ["car", 0.0, 0.4933, 0.223125, 0.9267],
      ["car", 0.273125, 0.4511, 0.47375, 0.6444],
      ["car", 0.7125, 0.6689, 0.999375, 1.0],
    ]
  }
  
  # # example layout for nuimages with timeofday and weather
  # layout = {
  #   "camera": "front",
  #   "timeofday": "night",
  #   "weather": "rain",
  #   "bbox": [
  #     ["car", 0.756875, 0.4622, 0.90375, 0.5844],
  #     ["car", 0.47625, 0.4822, 0.691875, 0.8011],
  #     ["car", 0.0, 0.4933, 0.223125, 0.9267],
  #     ["car", 0.273125, 0.4511, 0.47375, 0.6444],
  #     ["car", 0.7125, 0.6689, 0.999375, 1.0],
  #   ]
  # }
  
  # # example layout for coco-stuff
  # layout = {
  #   "bbox": [
  #     ["truck", 0.1321, 0.3914, 0.7247, 0.7849],
  #     ["house", 0.0, 0.06875, 0.9046875, 0.5458],
  #     ["road", 0.0, 0.6, 1.0, 1.0],
  #     ["snow", 0.0, 0.4792, 1.0, 0.89375],
  #     ["tree", 0.0, 0.0, 1.0, 0.56875]
  #   ]
  # }
  
  ########################
  # Run layout-to-image generation
  ########################
  run_layout_to_image(layout, args)