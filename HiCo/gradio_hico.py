import gradio as gr
import numpy as np
import cv2
import pandas as pd
import utils
import pdb
import PIL
from PIL import Image, ImageFont, ImageDraw
import json
import os
import torch
import copy
import datetime

SIZE_TO_CLICK_SIZE = {
    1024: 8,
    512: 5,
    256: 2
}
save_date_sec = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

save_dir = "./result-layout/"

from diffusers import  ControlNetModel, UniPCMultistepScheduler, DPMSolverMultistepScheduler, StableDiffusionHicoNetLayoutPipeline
base_model_path = ""
common = ""

controlnet_path = ""



HiCoNet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float32)

pipe = StableDiffusionHicoNetLayoutPipeline.from_pretrained(
    base_model_path, controlnet=[HiCoNet], torch_dtype=torch.float32
)
pipe.enable_attention_slicing()

#
# speed up diffusion process with faster scheduler and memory optimization
#pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# remove following line if xformers is not installed or when using Torch 2.0.
#pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
#pipe.enable_model_cpu_offload()

pipe.to("cuda")


#def get_demo(layout_to_image_generation_fn, cfg, model_fn, noise_schedule):
def get_demo():
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0),
              (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64)]

    def layout_to_image_generation(layout_bbox, layout_class, caption, negative_caption=None, classifier_free_scale=1.0, steps=50, seed=23, save_name="./demo_case"):
        base_bbox = copy.deepcopy(layout_bbox)
        base_class = copy.deepcopy(layout_class)
        image = np.zeros((512, 512, 3))

        base_bbox = np.array(base_bbox)
        #base_bbox[:,2:] += base_bbox[:,:2]      # x,y,w,h  -> x1,y1,x2,y2
        base_bbox = np.insert(base_bbox, obj=0, values=[0,0,512,512], axis=0)
        base_class.insert(0, caption)
        # gen condition image
        list_cond_image = []
        #cond_image = np.zeros_like(r_image, dtype=np.uint8)
        cond_image = np.zeros((512, 512))
        list_cond_image.append(cond_image)
        for iit in range(1, len(base_bbox)):
            dot_bbox = base_bbox[iit]
            dx1, dy1, dx2, dy2 = [int(xx) for xx in dot_bbox]
            #cond_image = np.zeros_like(r_image, dtype=np.uint8)
            cond_image = np.zeros((512, 512))
            cond_image[dy1:dy2, dx1:dx2] = 255
            #cond_image[dy1:dy2, dx1:dx2] = 1
            list_cond_image.append(cond_image)
        obj_cond_image = np.stack(list_cond_image, axis=0)

        layo_prompt = base_class
        layo_bbox = torch.FloatTensor(base_bbox)
        layo_cond = torch.FloatTensor(obj_cond_image)
        list_cond_image_pil = [PIL.Image.fromarray(dot_cond).convert('RGB') for dot_cond in list_cond_image]
        generator = torch.manual_seed(seed)
        caption = common + caption
        print (caption)
        image = pipe(
            caption, layo_prompt, guess_mode=False, generator=generator, negative_prompt=negative_caption,
            num_inference_steps=steps, image=list_cond_image_pil, guidance_scale=classifier_free_scale,
            width=512, height=512
        ).images[0]
        np_image = np.array(image)

        #return np_image
        save_date_sec = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path_gen_save_base = save_dir + save_name + "_" + save_date_sec +"_gen_base.png"
        path_gen_save_bbox = save_dir + save_name + "_" + save_date_sec +"_gen_bbox.png"

        image.save(path_gen_save_base)
        rect_image = draw_image(np_image/255, layout_bbox, layout_class, path_gen_save_bbox)
        return rect_image


    def clear_point(image, points):
        image = np.ones((512, 512, 3), dtype=np.uint8) * 100
        points['handle'] = []
        points['target'] = []
        return image, points

    def save_point(points, image, size):                                                                                                                                        
        src_point = points['handle']
        dst_point = points['target']

        #print ('[H, W]', src_point, dst_point)
        for i in range(len(src_point)):
            bbox_lf = src_point[i]
            bbox_rd = dst_point[i]
            print ("bbox %s, coord: " % i, bbox_lf, bbox_rd)

    def draw_image(image, obj_bbox, obj_class, img_save): 
        dw_img = PIL.Image.fromarray(np.uint8(image * 255))
        draw = PIL.ImageDraw.Draw(dw_img)
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        #draw.rectangle([100, 100, 300, 300], outline = (0, 255, 255), fill = (255, 0, 0), width = 10)
        for iix in range(len(obj_bbox)):
            rec = obj_bbox[iix]
            d_rec = [int(xx) for xx in rec]
            draw.rectangle(d_rec, outline = color, width = 3)

            text = obj_class[iix]
            font = ImageFont.truetype("./models_ckpt/font/msyh.ttf", size=10)
            draw.text((d_rec[0], d_rec[1]), text, font = font, fill="red", align="left")
        dw_img.save(img_save)

        return dw_img

    def save_layout_data_func(image_caption, custom_layout_dataframe, num_obj, points, size, save_name="demo_case"):
        num_obj = int(num_obj)
        out_data = []
        #out_data.append(custom_layout_dataframe['obj_caption'][0])  # caption
        out_data.append(image_caption)  # caption
        out_data.append(num_obj)
        out_data.append([size, size])

        cur_dir = os.getcwd()
        save_date_sec = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path_json_save = save_dir + save_name + "_" + save_date_sec + ".json"
        path_image_save = save_dir + save_name + "_" + save_date_sec + "_mask.png"
        out_data.append(path_image_save)
        print ("object numbers : %s" % num_obj)

        src_point = points['handle']
        dst_point = points['target']

        obj_bbox = []
        obj_class = []
        dot_info = []
        for i in range(num_obj):
            print (i, custom_layout_dataframe['obj_caption'][i], src_point[i], dst_point[i])
            dot_caption = custom_layout_dataframe['obj_caption'][i]
            bbox_lf = src_point[i]
            bbox_rd = dst_point[i]
            obj_bbox.append(bbox_lf + bbox_rd)
            obj_class.append(dot_caption)

            box_wh = list(map(lambda x: x[0]-x[1], zip(bbox_rd, bbox_lf)))
            dot_bbox = bbox_lf + box_wh
            dot_info.append([dot_caption, dot_bbox])    # [x, y w, h]
        out_data.append(dot_info)

        with open(path_json_save, "w", encoding="utf-8") as f:
            f.write(json.dumps(out_data, ensure_ascii=False, indent=4, separators=(',', ':')))


        image = np.zeros((512, 512, 3))
        # bbox: x1,y1, x2,y2
        draw_image(image, obj_bbox, obj_class, path_image_save)
        
        return image, obj_bbox, obj_class

    def add_points_to_image(image, points, size=5):
        image = utils.draw_handle_target_points(image, points['handle'], points['target'], size)
        return image

    def on_click(image, target_point, points, size, evt: gr.SelectData): 
        if target_point:
            points['target'].append([evt.index[0], evt.index[1]])
            image = add_points_to_image(image, points, size=SIZE_TO_CLICK_SIZE[size])
            return image, not target_point
        points['handle'].append([evt.index[0], evt.index[1]])
        image = add_points_to_image(image, points, size=SIZE_TO_CLICK_SIZE[size])
        print (points)
        return image, not target_point

    with gr.Blocks(css="#btn {background: gray; color: blue; width:50px;}") as demo:
    #with gr.Blocks(css="styles.css") as demo:
        gr.Markdown(
            """
        # LayoutDiffusion - 基础可控生成模型
        Get "layout image" and then "layout-to-image generation".  
        step1 : image-caption, object-nums, demo-name  
        step2 : input region-caption  
        step3 : input region-point, bbox  
        step4 : Button Order -> "save point", "save layout image", "Generate LayoutImage"  
        """
        )

        with gr.Row():
            num_obj = gr.Slider(value=3, step=1, minimum=1, maximum=10, label="object nums")
            classifier_free_scale = gr.Slider(value=1.0, minimum=0.5, maximum=10.0, step=0.5, label='Classifier free scale')
            steps = gr.Slider(value=50, minimum=25, maximum=200, label='Steps')

        with gr.Row():
            case_name_input = gr.Textbox(placeholder="input case name", value="demo_case", label="demo name")
            image_caption = gr.Textbox(placeholder="input image caption", value="An old man and his wife led a corgi for a walk on the beach in the setting sun", label="image caption")
            negative_caption = gr.Textbox(placeholder="negative caption", value="", label="negative prompt")
            seed = gr.Number(value=2333, precision=0, label='Seed', interactive=True)

        df_num_obj = 6
        df_fix = [[i, "region_prompt %s" % i] for i in range(1, df_num_obj+1)]
        with gr.Row():
            with gr.Column():
                custom_layout_dataframe = gr.Dataframe(
                    value=df_fix,
                    headers=["obj_id", "obj_caption"],
                    datatype=["number", "str"],
                    row_count=(df_num_obj, "fixed"),
                    col_count=(2, "fixed"),
                    interactive=True,
                )

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    layout_image = gr.Image(label='Layout Image', shape=(512, 512), value=np.ones((512, 512, 3), dtype=np.uint8) * 100).style(width=512, height=512)
                    generated_image = gr.Image(label='Generated Image', shape=(512, 512)).style(width=512, height=512)

                with gr.Row():
                    clear_btn_point = gr.Button('clear point', elem_id="btn")
                    save_btn_point = gr.Button('save point', elem_id="btn")
                    save_layout_data = gr.Button('save layout data', elem_id="btn")
                    generate_button = gr.Button(value='Generate LayoutImage', elem_id="btn")

        points = gr.State({'target': [], 'handle': []})
        size = gr.State(512)
        target_point = gr.State(False)
        image_mask = gr.State()
        obj_bbox = gr.State()
        obj_class = gr.State()
        
        save_btn_point.click(save_point, inputs=[points, layout_image, size])
        clear_btn_point.click(clear_point, inputs=[layout_image, points], outputs=[layout_image, points])

        layout_image.select(on_click, inputs=[layout_image, target_point, points, size], outputs=[layout_image, target_point])

        save_layout_data.click(save_layout_data_func, inputs=[image_caption, custom_layout_dataframe, num_obj, points, size, case_name_input],
                                                      outputs=[image_mask, obj_bbox, obj_class])


        generate_button.click(
            fn=layout_to_image_generation, inputs=[obj_bbox, obj_class, image_caption, negative_caption, classifier_free_scale, steps, seed, case_name_input], outputs=generated_image
        )

    return demo

if __name__ == "__main__":
    demo = get_demo()

    demo.launch(server_name='0.0.0.0', server_port=9500)


