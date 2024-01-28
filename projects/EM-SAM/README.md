## The Segment Anything Model (SAM) to EM image segmentation tasks (EM-SAM)

### Installation and SAM checkpoints

* This project is based on **pytorch_connectomics** framework development, in the execution of the code before please make sure you have correctly installed this framework. The installation instructions please click [here](https://github.com/Biomed-ssSEM-Lab/pytorch_connectomics?tab=readme-ov-file#installation)

* The checkpoints for the SAM model you can download by clicking [here](https://github.com/facebookresearch/segment-anything#model-checkpoints), there are three versions and we are using '**vit_b**' this version. The '**vit_b**' should be downloaded in the ***sam_checkpoints*** folder. If you want to use another version, please download it in the same way.

### Configurations
You can change the following parameters in the configuration file we provide (*EM-Sam-Seamantic-Seg.yaml*) to apply to your data path and device.
```yaml
SYSTEM:
  NUM_GPUS: number #The GPU numbers in your device you need to use
  NUM_CPUS: number #The CPU numbers in your device you need to use
DATASET:
  IMAGE_NAME: 'The training images name'
  LABEL_NAME: 'The label images name corresponding to the training images'
  INPUT_PATH: 'The data inputpath you need to use in your device, inculding training images, label images and test images. We refer to put this datas in same folder.'
  OUTPUT_PATH: 'The path to save the checkpoints of the model during training.'
INFERENCE:
  IMAGE_NAME: 'The test images name'
  OUTPUT_PATH: 'The path to save the segmentation results of test images '
  OUTPUT_NAME: 'The name of the segmentation results'
```

### Command
After completing the configuration file changes, you can run the following commands to complete the training of the model as well as the inference.

**Training command:**
```bash
CUDA_VIISBLE_DEVICES=0,1,2,3,4,5,6 python main.py \
--config-file config/EM-Sam-Seamantic-Seg.yaml \
--config-base config/EM-Sam-Seamantic-Seg.yaml
```
**Inference command:**
```bash
CUDA_VIISBLE_DEVICES=0,1,2,3,4,5,6 python main.py \
--config-file config/EM-Sam-Seamantic-Seg.yaml \
--config-base config/EM-Sam-Seamantic-Seg.yaml \
--inference \
--checkpoint path of your checkpoints/checkpoints name
```

### Dataset
[**AxonCallosumEM Dataset**](https://drive.google.com/drive/folders/1uNmICvrdD9G1jNFzpgzXT0a3D7skl6-8?usp=drive_link):The dataset we released to train the model and inference at present.
[**Introduction video**](https://drive.google.com/drive/folders/1uNmICvrdD9G1jNFzpgzXT0a3D7skl6-8?usp=drive_link): A brief description of the AxonCallosumEM Dataset dataset.

