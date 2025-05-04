


## Updates

04 May, 2025: Initial upload



## Installation
**Step 0.** Install PyTorch and Torchvision following [official instructions](https://pytorch.org/get-started/locally/), e.g.,

```shell
pip install torch torchvision
# FYI, we're using torch==1.9.1 and torchvision==0.10.1
# We used docker image pytorch:1.9.1-cuda11.1-cudnn8-devel
```

**Step 1.** Install [MMCV](https://github.com/open-mmlab/mmcv).
```shell
pip install mmcv-full
# FYI, we're using mmcv-full==1.4.0 
```

**Step 2.** Install [CLIP](https://github.com/openai/CLIP).
```shell
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

**Step 3.** Install DPL.
```shell
git clone https://github.com/shijueganzhi/DPL.git
cd DPL
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

**Step 4.** Maybe you need. (if error occurs)
```shell
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
sudo apt-get install libmagickwand-dev
pip install yapf==0.40.1
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

## Dataset Preparation & Pretrained Checkpoint
In our paper, we experiment with Cityscapes, CamVid, and WildDash2.

- Example directory hierarchy
  ```
  DPL
  |--- data
  |    |--- cityscapes
  |    |    |---leftImg8bit
  |    |    |---gtFine
  |    |--- camvid11
  |    |    |---img
  |    |    |---mask
  |    |--- wilddash2
  |    |    |---img
  |    |    |---mask
  |--- work_dirs
  |    |--- output_dirs (config_name)
  |    | ...
  | ...
  ```

**Dataset**
- [Cityscapes](https://www.cityscapes-dataset.com/)
- [Camvid](https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
- [WildDash2](https://www.wilddash.cc/)

**Pretrained Checkpoint**
- [Cityscapes](https://drive.google.com/file/d/1acN1JK__LKzGV5TynQExiUVTot5PxE_D/view?usp=sharing)
- [CamVid](https://drive.google.com/file/d/1naC6bAfEmvoSaigPP3odNgPlra-ElODj/view?usp=sharing)
- [WildDash2](https://drive.google.com/file/d/1pmBRPKH8gvaC_ZsDQCYxjU9XNAOGw_e1/view?usp=sharing)

## training DPL
DPL trains segmentation model with single or dual path.
You need to prepair fixed-masks (pseudo-masks) for single path training.

**Step 0.** Download and convert the CLIP models, e.g.,
```shell
python tools/maskclip_utils/convert_clip_weights.py --model ViT16
# Other options for model: RN50, RN101, RN50x4, RN50x16, RN50x64, ViT32, ViT16, ViT14
```

**Step 1.** Prepare the text embeddings of the target dataset, e.g.,
```shell
python tools/maskclip_utils/prompt_engineering.py --model ViT16 --class-set city_carb
# Other options for model: RN50, RN101, RN50x4, RN50x16, ViT32, ViT16
# Other options for class-set: camvid, wilddash2
# Default option is ViT16, city_carb
```

**Train.** Here, we give an example of multiple GPUs on a single machine. 
```shell
# Please see this file for the detail of execution.
# You can change detailed configuration by changing config files (e.g., CARB/configs/carb/cityscapes_carb_dual.py)
bash tools/train.sh 
```

## Inference DPL
```shell
# Please see this file for the detail of execution.
bash tools/test.sh
```

## Acknoledgement
This is highly borrowed from [MaskCLIP](https://github.com/chongzhou96/MaskCLIP), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). Thanks to Chong, zhou.


