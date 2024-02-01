## The Segment Anything Model (SAM) to EM image segmentation tasks (EM-SAM)

### Installation and SAM checkpoints

* This project is based on **pytorch_connectomics** framework development, in the execution of the code before please make sure you have correctly installed this framework. The installation instructions please click [here](https://github.com/Biomed-ssSEM-Lab/pytorch_connectomics?tab=readme-ov-file#installation)

* The checkpoints for the SAM model you can download by clicking [here](https://github.com/facebookresearch/segment-anything#model-checkpoints), there are three versions and we are using '**vit_b**' this version. The '**vit_b**' should be downloaded in the ***sam_checkpoints*** folder. If you want to use another version, please download it in the same way.

### Configurations
Please adjust the parameters in the provided configuration file ***(EM-Sam-Seamantic-Seg-File.yaml)*** to accommodate your dataset directory and hardware configuration.
```yaml
SOLVER:
  ITERATION_SAVE: 10000  # You can change this as you need
  ITERATION_TOTAL: 200000  # You can change this as you need
  SAMPLES_PER_BATCH: 2  # You can change this as you need
DATASET:
  IMAGE_NAME: 'image.tif'
  LABEL_NAME: 'imagelabel.tif'
  INPUT_PATH: 'datasets/EM-SAM/train_or_inference'
  OUTPUT_PATH: 'Path/to/save/checkpoints/during/training'
INFERENCE:
  IMAGE_NAME: "seg_raw_image.tif"
  OUTPUT_NAME: 'seg_output_name'
  OUTPUT_PATH: "Path/to/save/results/of/segmentation"
  OUTPUT_ACT: ["softmax"]
```

### Command
After completing the configuration file changes, you can run the following commands to complete the training of the model as well as the inference.

**Training command:**
```bash
CUDA_VIISBLE_DEVICES=0,1,2,3,4,5,6 python main.py \
--config-base config/EM-Sam-Base.yaml \
--config-file config/EM-Sam-Seamantic-Seg-File.yaml
```
**Inference command:**
```bash
CUDA_VIISBLE_DEVICES=0,1,2,3,4,5,6 python main.py \
--config-base config/EM-Sam-Base.yaml \
--config-file config/EM-Sam-Seamantic-Seg-File.yaml \
--inference \
--checkpoint path/of/your/checkpoints/checkpoints/name
```

### Dataset
[**AxonCallosumEM Dataset**](https://drive.google.com/drive/folders/1uNmICvrdD9G1jNFzpgzXT0a3D7skl6-8?usp=drive_link):The dataset we released to train the model and inference at present.

[**Introduction video**](https://drive.google.com/drive/folders/1uNmICvrdD9G1jNFzpgzXT0a3D7skl6-8?usp=drive_link): A brief description of the AxonCallosumEM Dataset dataset.

