import PIL
from PIL import Image, ImageFont, ImageDraw
import json
import numpy as np
import random
import pdb
import os
import sys

def load_image(image_path):
    with open(image_path, 'rb') as f:
        with PIL.Image.open(f) as image:
            image = image.convert('RGB')
    return image



def draw_image(image, obj_bbox, obj_class, img_save):
    dw_img = PIL.Image.fromarray(np.uint8(image * 255))
    draw = PIL.ImageDraw.Draw(dw_img)
    color = tuple(np.random.randint(0, 255, size=3).tolist())
    for iix in range(len(obj_bbox)):
        rec = obj_bbox[iix]
        d_rec = [int(xx) for xx in rec]
        draw.rectangle(d_rec, outline = color, width = 3)

        text = obj_class[iix]
        font = ImageFont.truetype("./font/msyh.ttf", size=10)
        draw.text((d_rec[0], d_rec[1]), text, font = font, fill="red", align="left")
    dw_img.save(img_save)

# resize and crop to 512-512
def resize_crop(image, obj_bbox, obj_class):

    W, H = image.size

    ##### resize ####
    res_min = 512
    if W < H:
        r_w = res_min
        r_h = int(r_w * H / W)
    else:
        r_h = res_min
        r_w = int(r_h * W / H)

    r_image = image.resize((r_w, r_h))


    r_image = np.array(r_image, dtype=np.float32)

    rescale = r_h / H
    r_obj_bbox = obj_bbox * rescale
    
    #### crop ####
    w, h = res_min, res_min
    left = random.uniform(0, r_w - w)
    top = random.uniform(0, r_h - h)
    rect = np.array([int(left), int(top), int(left + w), int(top + h)])

    c_image = r_image[rect[1]:rect[3], rect[0]:rect[2],:]

    r_obj_bbox[:,:2] = np.maximum(r_obj_bbox[:,:2], rect[:2])
    r_obj_bbox[:, :2] -= rect[:2]
    r_obj_bbox[:,2:] = np.minimum(r_obj_bbox[:,2:], rect[2:])
    r_obj_bbox[:, 2:] -= rect[:2]


    return c_image, r_obj_bbox, obj_class



