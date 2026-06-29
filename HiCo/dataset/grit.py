#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from collections import defaultdict
import random
import PIL
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

#from layout_diffusion.dataset.util import image_normalize
#from layout_diffusion.dataset.augmentations import RandomSampleCrop, RandomMirror
from dataset.util import image_normalize
from dataset.augmentations import RandomMirror, RandomSampleCrop, CenterSampleCrop
import pdb

Image.MAX_IMAGE_PIXELS = None

class GritSceneGraphDataset(Dataset):
    def __init__(self, tokenizers, grit_json, 
                 image_dir, instances_json, stuff_json=None,
                 stuff_only=True, image_size=(64, 64), mask_size=16,
                 max_num_samples=None,
                 include_relationships=True, min_object_size=0.02,
                 min_objects_per_image=3, max_objects_per_image=8, left_right_flip=False,
                 include_other=False, instance_whitelist=None, stuff_whitelist=None, mode='train',
                 use_deprecated_stuff2017=False, deprecated_coco_stuff_ids_txt='', filter_mode='LostGAN',
                 use_MinIoURandomCrop=False,
                 return_origin_image=False, specific_image_ids=None
                 ):
        """
        A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
        them to scene graphs on the fly.

        Inputs:
        - image_dir: Path to a directory where images are held
        - instances_json: Path to a JSON file giving COCO annotations
        - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations
        - stuff_only: (optional, default True) If True then only iterate over
          images which appear in stuff_json; if False then iterate over all images
          in instances_json.
        - image_size: Size (H, W) at which to load images. Default (64, 64).
        - mask_size: Size M for object segmentation masks; default 16.
        - max_num_samples: If None use all images. Other wise only use images in the
          range [0, max_num_samples). Default None.
        - include_relationships: If True then include spatial relationships; if
          False then only include the trivial __in_image__ relationship.
        - min_object_size: Ignore objects whose bounding box takes up less than
          this fraction of the image.
        - min_objects_per_image: Ignore images which have fewer than this many
          object annotations.
        - max_objects_per_image: Ignore images which have more than this many
          object annotations.
        - include_other: If True, include COCO-Stuff annotations which have category
          "other". Default is False, because I found that these were really noisy
          and pretty much impossible for the system to model.
        - instance_whitelist: None means use all instance categories. Otherwise a
          list giving a whitelist of instance category names to use.
        - stuff_whitelist: None means use all stuff categories. Otherwise a list
          giving a whitelist of stuff category names to use.
        """
        super(Dataset, self).__init__()

        self.return_origin_image = return_origin_image
        if self.return_origin_image:
            self.origin_transform = T.Compose([
                T.ToTensor(),
                image_normalize()
            ])

        if stuff_only and stuff_json is None:
            print('WARNING: Got stuff_only=True but stuff_json=None.')
            print('Falling back to stuff_only=False.')

        self.use_deprecated_stuff2017 = use_deprecated_stuff2017
        self.deprecated_coco_stuff_ids_txt = deprecated_coco_stuff_ids_txt
        self.mode = mode
        self.max_objects_per_image = max_objects_per_image
        self.image_dir = image_dir
        self.mask_size = mask_size
        self.max_num_samples = max_num_samples
        self.include_relationships = include_relationships
        self.filter_mode = filter_mode
        self.image_size = image_size
        self.min_image_size = min(self.image_size)
        self.min_object_size = min_object_size
        self.left_right_flip = left_right_flip
        if left_right_flip:
            self.random_flip = RandomMirror()

        self.layout_length = self.max_objects_per_image + 2

        self.use_MinIoURandomCrop = use_MinIoURandomCrop
        if use_MinIoURandomCrop:
            self.MinIoURandomCrop = RandomSampleCrop()
            self.MinIoUCenterCrop = CenterSampleCrop()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(size=image_size, antialias=True),
            image_normalize()
        ])

        self.transform_cond = T.Compose([
            T.ToTensor(),
            #T.Resize(size=image_size, antialias=True),
            #image_normalize()
        ])

        self.total_num_bbox = 0
        self.total_num_invalid_bbox = 0

        self.tokenizers = tokenizers

        # read grit-20m data
        with open(grit_json, 'r') as f:
            grit_data = json.load(f)

        self.image_ids = []
        self.image_id_to_objects = {}
        for idx, obj_data in grit_data.items():
            f_img_path = obj_data["f_path"]
            list_exps = obj_data["ref_exps"]
            image_w = obj_data["width"]
            image_h = obj_data["height"]
            caption = obj_data["caption"]
            url = obj_data["url"]

            obj_nums = len(list_exps)
            # get sub-caption
            list_bbox_info = []
            for box_info in list_exps:
                phrase_s, phrase_e, x1_norm, y1_norm, x2_norm, y2_norm, score = box_info
                phrase_s = int(phrase_s)
                phrase_e = int(phrase_e)
                phrase = caption[phrase_s:phrase_e]
                x1, y1, x2, y2 = int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)

                x1, y1 = min(x1, image_w), min(y1, image_h)
                x2, y2 = min(x2, image_w), min(y2, image_h)
                if int(x2 - x1) < 0.05 * image_w or int(y2 - y1) < 0.05 * image_h:
                    continue
                
                #list_bbox_info.append([phrase, [x1, y1, x2, y2]])
                list_bbox_info.append([phrase, [x1, y1, int(x2 - x1), int(y2 - y1)]])
                if len(list_bbox_info) >= self.max_objects_per_image:
                    break
            if len(list_bbox_info) == 0:
                continue

            self.image_ids.append([idx, f_img_path, obj_nums])
            self.image_id_to_objects.setdefault(idx, [caption, image_w, image_h, list_bbox_info, url])

        print ("data nums : %s." % len(self.image_id_to_objects))

    def filter_invalid_bbox(self, H, W, bbox, is_valid_bbox, verbose=False):
        #pdb.set_trace()
        for idx, obj_bbox in enumerate(bbox):
            if not is_valid_bbox[idx]:
                continue
            self.total_num_bbox += 1

            x, y, w, h = obj_bbox

            if (x >= W) or (y >= H):
                is_valid_bbox[idx] = False
                self.total_num_invalid_bbox += 1
                if verbose:
                    print(
                        'total_num = {}, invalid_num = {}, x = {}, y={}, w={}, h={}, W={}, H={}'.format(
                            self.total_num_bbox, self.total_num_invalid_bbox, x, y, w, h, W, H,
                        )
                    )
                continue

            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = np.clip(x + w, 1, W)
            y1 = np.clip(y + h, 1, H)

            if (y1 - y0 < self.min_object_size * H) or (x1 - x0 < self.min_object_size * W):
                is_valid_bbox[idx] = False
                self.total_num_invalid_bbox += 1
                if verbose:
                    print(
                        'total_num = {}, invalid_num = {}, x = {}, y={}, w={}, h={}, W={}, H={}'.format(
                            self.total_num_bbox, self.total_num_invalid_bbox, x, y, w, h, W, H,
                        )
                    )
                continue
            bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3] = x0, y0, x1, y1

        return bbox, is_valid_bbox

    def total_objects(self):
        total_objs = 0
        for i, image_info in enumerate(self.image_ids):
            total_objs += image_info[2]
        return total_objs

    def get_init_meta_data(self, image_id, caption):
        #self.layout_length = self.max_objects_per_image + 2
        layout_length = self.layout_length
        clip_text_ids = self.tokenize_caption("")
        meta_data = {
            'obj_bbox': torch.zeros([layout_length, 4]),
            'obj_class': [""] * layout_length,
            'is_valid_obj': torch.zeros([layout_length]),
            'obj_class_text_ids': clip_text_ids.repeat(layout_length, 1),
            #'obj_class': torch.LongTensor(layout_length).fill_(self.vocab['object_name_to_idx']['__null__']),
            #'filename': self.image_id_to_filename[image_id].replace('/', '_').split('.')[0]
        }

        # The first object will be the special __image__ object
        meta_data['obj_bbox'][0] = torch.FloatTensor([0, 0, 1, 1])
        #meta_data['obj_class'][0] = self.vocab['object_name_to_idx']['__image__']
        meta_data['obj_class'][0] = caption
        meta_data['is_valid_obj'][0] = 1.0

        clip_text_ids = self.tokenize_caption(caption)
        meta_data['obj_class_text_ids'][0] = clip_text_ids

        return meta_data

    def load_image(self, image_path):
        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                image = image.convert('RGB')
        return image

    def __len__(self):
        return len(self.image_ids)

    def tokenize_caption(self, caption):
        captions = []
        #if random.random() < self.proportion_empty_prompts:
        if random.random() < 0.05:
            captions.append("")
        else:
            captions.append(caption)
        clip_inputs = self.tokenizers(
            captions, max_length=self.tokenizers.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return clip_inputs.input_ids

    def resize_img(self, image, obj_bbox, obj_class):
        # resize to self.resolution
        #pdb.set_trace()
        ori_width, ori_height = image.size
        #target_size = self.image_size
        #res_height, res_width = self.image_size
        res_min_size = self.min_image_size
        if ori_height < ori_width:
            resize_height = res_min_size
            aspect_r = ori_width / ori_height
            resize_width = int(resize_height * aspect_r)
            im_resized = image.resize((resize_width, resize_height))

            rescale = resize_height / ori_height
            re_obj_bbox = obj_bbox * rescale
        else:
            resize_width = res_min_size
            aspect_r = ori_height / ori_width
            resize_height = int(resize_width * aspect_r)
            im_resized = image.resize((resize_width, resize_height))

            rescale = resize_height / ori_height
            re_obj_bbox = obj_bbox * rescale

        return im_resized, re_obj_bbox, obj_class

    def draw_image(self, image, obj_bbox, obj_class, img_save):
        dw_img = PIL.Image.fromarray(np.uint8(image * 255))
        draw = PIL.ImageDraw.Draw(dw_img)
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        #draw.rectangle([100, 100, 300, 300], outline = (0, 255, 255), fill = (255, 0, 0), width = 10)
        for iix in range(len(obj_bbox)):
            rec = obj_bbox[iix]
            d_rec = [int(xx) for xx in rec]
            draw.rectangle(d_rec, outline = color, width = 3)

            text = obj_class[iix]
            font = ImageFont.truetype("/home/jovyan/boomcheng-data/tools/font/msyh.ttf", size=10)
            draw.text((d_rec[0], d_rec[1]), text, font = font, fill="red", align="left")
        dw_img.save(img_save)

    def draw_image_xywh(self, image, obj_bbox, obj_class, img_save):
        dw_img = PIL.Image.fromarray(np.uint8(image * 255))
        draw = PIL.ImageDraw.Draw(dw_img)
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        #draw.rectangle([100, 100, 300, 300], outline = (0, 255, 255), fill = (255, 0, 0), width = 10)
        for iix in range(len(obj_bbox)):
            rec = obj_bbox[iix]
            d_rec = [int(xx) for xx in rec]
            d_rec[2] += d_rec[0]
            d_rec[3] += d_rec[1]
            draw.rectangle(d_rec, outline = color, width = 3)

            text = obj_class[iix]
            font = ImageFont.truetype("/home/jovyan/boomcheng-data/tools/font/msyh.ttf", size=10)
            draw.text((d_rec[0], d_rec[1]), text, font = font, fill="red", align="left")
        dw_img.save(img_save)


    def __getitem__(self, index):
        """
        Get the pixels of an image, and a random synthetic scene graph for that
        image constructed on-the-fly from its COCO object annotations. We assume
        that the image will have height H, width W, C channels; there will be O
        object annotations, each of which will have both a bounding box and a
        segmentation mask of shape (M, M). There will be T triples in the scene
        graph.

        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system
        - masks: LongTensor of shape (O, M, M) giving segmentation masks for
          objects, where 0 is background and 0 is object.

        """
        f_idx, f_image_path, f_obj_nums = self.image_ids[index]

        image = self.load_image(f_image_path)
        W, H = image.size
        caption, image_w, image_h, list_bbox_info, url = self.image_id_to_objects[f_idx]

        if W != image_w or H != image_h:
            index = 0
            f_idx, f_image_path, f_obj_nums = self.image_ids[index]
            image = self.load_image(f_image_path)
            caption, image_w, image_h, list_bbox_info, url = self.image_id_to_objects[f_idx]
            

        f_img_nm = f_image_path.split("/")[-1]

        num_obj = len(list_bbox_info)
        obj_bbox = [obj[1] for obj in list_bbox_info]
        obj_bbox = np.array(obj_bbox)
        obj_class = [obj[0] for obj in list_bbox_info]
        is_valid_obj = [True for _ in range(num_obj)]


        #self.draw_image_xywh(np.array(image, dtype=np.float32) / 255.0, obj_bbox, obj_class, "./image_demo/%s-s2-flip.jpg" % f_img_nm)

        # filter invalid bbox
        # bbox : [x, y, w, h] -> [x1, y1, x2, y2]
        if True:
            W, H = image.size
            obj_bbox, is_valid_obj = self.filter_invalid_bbox(H=H, W=W, bbox=obj_bbox, is_valid_bbox=is_valid_obj)

        if True:
            image, obj_bbox, obj_class = self.resize_img(image, obj_bbox, obj_class)

        if self.return_origin_image:
            origin_image = np.array(image, dtype=np.float32) / 255.0
        image = np.array(image, dtype=np.float32) / 255.0


        #self.draw_image(image, obj_bbox, obj_class, "./image_demo/%s-s1-resize.jpg" % f_img_nm)

        H, W, _ = image.shape
        # get meta data
        meta_data = self.get_init_meta_data(f_idx, caption)
        meta_data['width'], meta_data['height'] = W, H
        meta_data['num_obj'] = num_obj

        # flip
        if self.left_right_flip:
            image, obj_bbox, obj_class = self.random_flip(image, obj_bbox, obj_class)


        # random crop image and its bbox
        if self.use_MinIoURandomCrop:
            r_obj_bbox = obj_bbox[is_valid_obj]
            r_obj_class = [obj_class[ii] for ii in range(len(is_valid_obj)) if is_valid_obj[ii]]

            try:
                image, updated_obj_bbox, updated_obj_class, tmp_is_valid_obj = self.MinIoUCenterCrop(image, r_obj_bbox, r_obj_class)
            except:
                print (f"=======================, index:{index}, f_idx:{f_idx}")
                return self.__getitem__(0)
                

            meta_data['new_height'] = image.shape[0]
            meta_data['new_width'] = image.shape[1]
            H, W, _ = image.shape

        obj_bbox, obj_class = updated_obj_bbox, updated_obj_class

        H, W, C = image.shape
        ############### condition_image #############
        list_cond_image = []
        cond_image = np.zeros_like(image, dtype=np.uint8)
        list_cond_image.append(cond_image)
        for iit in range(len(obj_bbox)):
            dot_bbox = obj_bbox[iit]
            dx1, dy1, dx2, dy2 = [int(xx) for xx in dot_bbox]
            cond_image = np.zeros_like(image, dtype=np.uint8)
            #cond_image[dy1:dy2, dx1:dx2] = 255
            cond_image[dy1:dy2, dx1:dx2] = 1
            list_cond_image.append(cond_image)

        obj_bbox = torch.FloatTensor(obj_bbox)

        obj_bbox[:, 0::2] = obj_bbox[:, 0::2] / W
        obj_bbox[:, 1::2] = obj_bbox[:, 1::2] / H

        num_selected = min(obj_bbox.shape[0], self.max_objects_per_image)
        selected_obj_idxs = random.sample(range(obj_bbox.shape[0]), num_selected)

        meta_data['obj_bbox'][1:1 + num_selected] = obj_bbox[selected_obj_idxs]
        list_text_select = [obj_class[iv] for iv in selected_obj_idxs]
        meta_data['obj_class'][1:1 + num_selected] = list_text_select 

        obj_cond_image = np.stack(list_cond_image, axis=0)
        meta_data['cond_image'] = np.zeros([self.layout_length, H, W, C])
        meta_data['cond_image'][0:len(list_cond_image)] = obj_cond_image

        meta_data['cond_image'][1:1 + num_selected] = obj_cond_image[1:][selected_obj_idxs]
        meta_data['cond_image'] = torch.from_numpy(meta_data['cond_image'].transpose(0,3,1,2))


        clip_text_ids = self.tokenize_caption(caption)
        meta_data['base_caption'] = caption
        meta_data['base_class_text_ids'] = clip_text_ids

        #meta_data['is_valid_obj'][1:1 + num_selected] = 1.0
        meta_data['num_selected'] =  1 + num_selected
        meta_data['url'] = url
        #meta_data['num_obj_select'] = num_selected

        # tokenizer
        for iit in range(len(list_text_select)):
            text = list_text_select[iit]
            clip_text_ids = self.tokenize_caption(text)
            meta_data['obj_class_text_ids'][1+iit] = clip_text_ids

        if self.return_origin_image:
            meta_data['origin_image'] = self.origin_transform(origin_image)

        return self.transform(image), meta_data


def grit_collate_fn_for_layout(batch):
    """
    Collate function to be used when wrapping GritSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (N, L) giving object categories
    - masks: FloatTensor of shape (N, L, H, W)
    - is_valid_obj: FloatTensor of shape (N, L)
    """

    all_meta_data = defaultdict(list)
    all_imgs = []

    for i, (img, meta_data) in enumerate(batch):
        all_imgs.append(img[None])
        for key, value in meta_data.items():
            all_meta_data[key].append(value)

    all_imgs = torch.cat(all_imgs)
    for key, value in all_meta_data.items():
        #if key in ['obj_bbox', 'obj_class', 'is_valid_obj'] or key.startswith('labels_from_layout_to_image_at_resolution'):
        if key in ['obj_bbox'] or key.startswith('labels_from_layout_to_image_at_resolution'):
            all_meta_data[key] = torch.stack(value)

    return all_imgs, all_meta_data


def build_grit_dsets(cfg, tokenizer, mode='train'):
    assert mode in ['train', 'val', 'test']
    params = cfg.data.parameters
    dataset = GritSceneGraphDataset(
        tokenizers=tokenizer,
        grit_json=params.grit_json,
        mode=mode,
        filter_mode=params.filter_mode,
        stuff_only=params.stuff_only,
        image_size=(params.image_size, params.image_size),
        mask_size=params.mask_size_for_layout_object,
        min_object_size=params.min_object_size,
        min_objects_per_image=params.min_objects_per_image,
        max_objects_per_image=params.max_objects_per_image,
        instance_whitelist=params.instance_whitelist,
        stuff_whitelist=params.stuff_whitelist,
        include_other=params.include_other,
        include_relationships=params.include_relationships,
        use_deprecated_stuff2017=params.use_deprecated_stuff2017,
        deprecated_coco_stuff_ids_txt=os.path.join(params.root_dir, params[mode].deprecated_stuff_ids_txt),
        image_dir=os.path.join(params.root_dir, params[mode].image_dir),
        instances_json=os.path.join(params.root_dir, params[mode].instances_json),
        stuff_json=os.path.join(params.root_dir, params[mode].stuff_json),
        max_num_samples=params[mode].max_num_samples,
        left_right_flip=params[mode].left_right_flip,
        use_MinIoURandomCrop=params[mode].use_MinIoURandomCrop,
        return_origin_image=params.return_origin_image,
        specific_image_ids=params[mode].specific_image_ids
    )

    num_objs = dataset.total_objects()
    num_imgs = len(dataset)
    print('%s dataset has %d images and %d objects' % (mode, num_imgs, num_objs))
    print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

    return dataset

