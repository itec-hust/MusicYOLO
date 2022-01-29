# MusicYOLO
MusicYOLO framework uses the object detection model, YOLOX, to locate notes in the spectrogram. Its performance on the ISMIR2014 dataset, MIR-ST500 dataset and SSVD dataset show that MusicYOLO significantly improves onset/offset detection compared with previous approaches.

## Installation

Step1. Install pytorch.
```shell
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
```

Step1. Install YOLOX.
```shell
git clone git@github.com:xk-wang/MusicYOLO.git
cd MusicYOLO
pip3 install -U pip && pip3 install -r requirements.txt
pip3 install -v -e .  # or  python3 setup.py develop
```
Step2. Install apex.

```shell
# skip this step if you don't want to train model.
cd apex
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" .
```

Step3. Install pycocotools.

```shell
pip3 install cython;
cd cocoapi/PythonAPI && pip3 install -v .
```

## Inference

Download the pretrained musicyolo1 and musicyolo2 models described in our paper. Put these two models under the models folder. The models are stored in BaiduYun https://pan.baidu.com/s/1TbE36ydi-6EZXwxo5DwfLg?pwd=1234 code: 1234

### SSVD & ISMIR2014

Step1. Download SSVD-v2.0 from https://github.com/xk-wang/SSVD-v2.0

Step2. Onset/offset detection (use musicyolo2.pth)
```shell
python3 tools/predict.py -f exps/example/custom/yolox_singing.py -c models/musicyolo2.pth --audiodir $SSVD_TEST_SET_PATH --savedir $SAVE_PATH --ext .flac --device gpu
```

Step3. Evaluate
```shell
python3 tools/note_eval.py --label $SSVD_TEST_SET_PATH --result $SAVE_PATH --offset
```

Similar process for ISMIR2014 dataset.

### MIR-ST500

Since MIR-ST500 dataset is a mixture of vocals and accompaniments, we need to separate vocals and accompaniments with spleeter first. Besides, since the singing duration of each audio in MIR-ST500 dataset is too long, we will first cut each audio into short audios of about 35s for on/offset detection.

Step1. Audio source seperation
```shell
python3 tools/util/do_spleeter.py $MIR_ST500_DIR
```

Step2. Split audio
```shell
python3 tools/util/split_mst.py --mst_path $MST_TEST_VOCAL_PATH --dest_dir $SPLIT_PATH
```

Step3. Onset/offset detection (use musicyolo1.pth)
```shell
python3 tools/predict.py -f exps/example/custom/yolox_singing.py -c models/musicyolo1.pth --audiodir $SPLIT_PATH --savedir $SAVE_PATH --ext .wav --device gpu
```

Step4. Merge results

Because we split the MIR-ST500 test set audio earlier, the results are also splited. Here we merge the split results.
```shell
python3 tools/util/merge_res.py --audio_dir $SPLIT_PATH --origin_dir $SAVE_PATH --final_dir $MERGE_PATH
```

Step5. Evaluate
```shell
python3 tools/note_eval.py --label $MIR_ST500_TEST_LABEL_PATH --result $MERGE_PATH --offset
```

Step1. Download a pretrained model from the benchmark table.

Step2. Use either -n or -f to specify your detector's config. For example:

```shell
python3 tools/demo.py image -n yolox-s -c /path/to/your/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```
or
```shell
python3 tools/demo.py image -f exps/default/yolox_s.py -c /path/to/your/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```
Demo for video:
```shell
python3 tools/demo.py video -n yolox-s -c /path/to/your/yolox_s.pth --path /path/to/your/video --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```

## Train yourself

Download yolox-s weight from https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth . Put the model weight under models folder.

### Train on SSVD (get musicyolo2)

Step1. Get SSVD train set

Download SSVD-v2.0 from https://github.com/xk-wang/SSVD-v2.0. Put the images folder under the datasets folder.

Step2. Train:

```shell
python3 tools/train.py -f exps/example/custom/yolox_singing.py -d 1 -b 16 --fp16 -o -c models/yolox_s.pth
```

### Train on MIR-ST500 (get musicyolo1)

### Prepair note object detection dataset
Because there are a few audios for SSVD training set, we use Labelme software to annotate note object manually. There are a lot of data in MIR-ST500 training set, so we design a set of automatic annotation tools.

Step1. Audio source seperation
```shell
python3 tools/util/do_spleeter.py $MIR_ST500_TRAIN_DIR
```

Step2. Split audio
```shell
python3 tools/util/split_mst.py --mst_path $MIR_ST500_TRAIN_DIR --dest_dir $TRAIN_SPLIT_PATH
```

Step3. Automatic annotation

```shell
python3 tools/util/automatic_annotation.py --audiodir $TRAIN_SPLIT_PATH --imgdir $MST_NOTE_PATH
```

Step4. Automatic annotation

Divide the training set and validation set by yourself. We break up the images and divide them according to the ratio of 7:3 to get the training set and validation set. The images and annotations are put under $YOU_MIR_ST500_IMAGES folder.

Step4. Coco dataset format

The MIR-st500 note object detection dataset is organized in a format similar to the images folder in SSVD v2.0 dataset.

```shell
python3 tools/util/labelme2coco.py --annotationpath $YOU_MIR_ST500_IMAGES/train --jsonpath $IMAGE_DIR/train/_annotations.coco.json

python3 tools/util/labelme2coco.py --annotationpath $YOU_MIR_ST500_IMAGES/valid --jsonpath $IMAGE_DIR/valid/_annotations.coco.json
```

then put the MIR-ST500 note object detection dataset under the datasets folder like SSVD.

### Train

the similar process like train on SSVD dataset.

## Citation

```latex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}

@inproceedings{musicyolo2022,
  title={A SIGHT-SINGING ONSET/OFFSET DETECTION FRAMEWORK BASED ON OBJECT DETECTION INSTEAD OF SPECTRUM FRAMES.},
  author={X. Wang, W. Xu, W. Yang and W. Cheng},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={},
  year={2022},
}
```
