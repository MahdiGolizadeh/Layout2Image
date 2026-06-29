

### <div align="center">👉 HiCo: Hierarchical Controllable Diffusion Model for Layout-to-image Generation<div> 
### <div align="center"> 💥 NeurIPS 2024！ <div> 
#### <div align="center"> Bo Cheng, Yuhang Ma, Liebucha Wu, Shanyuan Liu, Ao Ma, Xiaoyu Wu, Dawei Leng†, Yuhui Yin(✝Corresponding Author) <div> 

<div align="center">
  <a href="https://360cvgroup.github.io/HiCo_T2I/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2410.14324"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:HiCo&color=red&logo=arxiv"></a> &ensp;
  <a href=""><img src="https://img.shields.io/static/v1?label=App&message=ComfyUI&&color=green"></a> &ensp;
</div>

---
## 🔥 News 
- **[2024/10/21]** We initialized this github repository and released the code .
- **[2024/10/18]** We released the paper of [HiCo](https://arxiv.org/abs/2410.14324).

## 🕓 Schedules
- **[Temporary uncertainty]** We plan to release the 2nd generation HiCo(more lightweight).

## 💻 Quick Demos
Image demos can be found on the [HiCo](https://360cvgroup.github.io/HiCo_T2I/). Some of them are contributed by the community. You can customize your own personalized generation using the following reasoning code.

## 🔧 Quick Start
### 0. Experimental environment
We tested our inference code on a machine with a 24GB 3090 GPU and CUDA environment version 12.1.

### 1. Setup repository and environment
```
git clone https://github.com/360CVGroup/HiCo_T2I.git
cd HiCo
conda create -n HiCo python=3.10
conda activate HiCo
pip install -r requirements.txt
cd diffusers
pip install .
```
### 2. Prepare the models
```
git lfs install

# HiCo checkpoint
git clone https://huggingface.co/qihoo360/HiCo_T2I models/controlnet

# stable-diffusion-v1-5
git clone https://huggingface.co/krnl/realisticVisionV51_v51VAE models/realisticVisionV51_v51VAE
```
### 3. Customize your own creation
```
CUDA_VISIBLE_DEVICES=0   infer-avg.py
```
## 🔥 Train

The json structure for dataset is: (like GRIT)
```
dataset 

├──base_info 
│  ├──id
│  ├──width
│  ├──height
│  ├──f_path
├──caption  
├──obj_nums  
├──img_size  
│  ├──H
│  ├──W
├──path_img (f_path)
├──list_bbox_info
│  ├──subcaption
│  ├──coordinates(x1,y1,x2,y2)
│  │......
├──crop_location

```
Then you can train the code.
```
sh run.sh
```

## BibTeX
```
@misc{cheng2024hicohierarchicalcontrollablediffusion,
      title={HiCo: Hierarchical Controllable Diffusion Model for Layout-to-image Generation}, 
      author={Bo Cheng and Yuhang Ma and Liebucha Wu and Shanyuan Liu and Ao Ma and Xiaoyu Wu and Dawei Leng and Yuhui Yin},
      year={2024},
      eprint={2410.14324},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.14324}, 
}
```
## License
This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).

