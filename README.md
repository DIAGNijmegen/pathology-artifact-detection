![artifact-segmentation](https://github.com/DIAGNijmegen/pathology-artifact-detection/blob/main/images/header.png?raw=true)

# Multi-class semantic segmentation of artifacts
Due to the presence of artifacts in digitized histopathology images, tissue regions that are important for diagnosis can be unclear or even completely unusable. As a result, artifacts may interfere with an accurate diagnosis by pathologists or lead to erroneous analyses by deep learning algorithms. By detecting commonly found artifacts, it is possible to improve the process of automated diagnosis. Images with too many artifacts can automatically be rejected by our quality control system, potentially saving many hours of manual inspection.

## Included artifacts
The system detects the following artifact types:
1. Tissue folds
2. Ink
3. Dust
4. Air bubbles 
5. Marker
6. Out-of-focus

## Demo
An interactive demo is accessible via Grand-Challenge:
* URL: [grand-challenge.org/artifact-segmentation](https://grand-challenge.org/algorithms/quality-assessment-of-whole-slide-images-through-a/)

*To be able to make use of the demo, please click on `Request Access`*.

## Dataset
The following slides were contained in the training data:

| Tissue        | Stain              | Whole slide scanner                                                                           | #Slides | #Annotations |
|---------------|--------------------|-----------------------------------------------------------------------------------------------|---------|--------------|
| Breast        | HE <br /> CD3  <br /> CD45RO <br /> CD8  | 3DHistech Pannoramic 250 Flash II <br /> Hamamatsu NanoZoomer-XR C12000-01 <br /> Philips Ultrafast Scanner |       47|          1456|
| Bonemarrow    | HE <br /> PAS             | 3DHistech P1000 scanners                                                                      |        8|            71|
| Colon         | CK20               | 3DHistech Pannoramic 250 Flash II                                                             |       25|           735|
| Kidney        | HE <br /> PAS             | Leica Aperio XT ScanScope <br /> 3DHistech P1000 scanners                                            |        9|           211|
| Lymphoma      | HE                 | 3DHistech P1000 scanners                                                                      |        9|            48|
| Pancreas      | CD3 <br /> CK81 <br /> CKPAN <br /> CD8 | 3DHistech Pannoramic 250 Flash II <br /> 3DHistech P1000 scanners                                    |       10|           451|
| Prostate      | HE                 | Leica Aperio AT2 <br /> Hamamatsu C9600-12 <br /> 3DHistech P1000 scanners                                  |       20|           166|
| Ovary         | HE <br /> P53 <br /> Ki67        | 3DHistech P1000 scanners                                                                      |        9|            71|
| Miscellaneous | HE <br /> Ki67            | 3DHistech P1000 scanners                                                                      |        5|            69|

## Inference
Prepare a configuration file (e.g., named `config.yml`):

```yaml
# paths to checkpoints and output folder
artifact_network: /.../path/to/checkpoints/artifact_network.pth
quality_network: /.../path/to/checkpoints/quality_score_network.ckpt
output_folder: /.../path/to/output/directory
# architecure settings
architecture: deeplabv3plus
encoder_name: efficientnet-b2
encoder_weights: imagenet
use_batchnorm: true
pooling: avg
dropout: 0.4
spacing: 4.0
# torch settings
device: cuda
workers: 4
# tissue settings
minimum_region: 5000
minimum_hole: 1000
remove_mask: true
# artifact settings
overlap: 256
tile_size: 1024
tissue_spacing: 4.0
region_threshold: 10.0
hole_threshold: 200.0
num_classes: 7
# slides to process
input_path: /.../path/to/input/folder/*.tif
# tissue-background segmentation mask
mask_path: /.../path/to/tissue-background-masks/folder/{image}_tb_mask.tif
```

Run `infer.py` using the Docker image:

```bash
#!/usr/bin/env bash
cd "/home/user/source/artifact-segmentation/"
git pull
python3.7 /home/user/source/artifact-segmentation/src/infer.py \
  --config /.../path/to/config/config.yml
```

## Training
Copy (and change paramaters of) the configuration file `training.yml`:

Run `train.py` using the Docker image:

```bash
#!/usr/bin/env bash
cd "/home/user/source/artifact-segmentation/"
git pull
python3.7 /home/user/source/artifact-segmentation/src/train.py \
  --config /.../path/to/config/training.yml
```

## Requirements
Successfully tested using the following settings:
- `--gpu-count="1"`
- `--require-gpu-mem="11G"`
- `--require-cpus="2"`
- `--require-mem="48G"`
