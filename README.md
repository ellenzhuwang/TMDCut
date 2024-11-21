<h1>Improving Open-vocabulary Segmentation using Diffused Cuts</h1>

## Abstract
Utilizing structure in the input data to derive embeddings has become a key ingredient that determines performance of machine learning models for Vision. For instance, using geometric information such as neighborhoods in Convolutional or more generally, Graph Neural Networks is known to be beneficial in many downstream tasks. However, these information are taken in a sample wise manner. That is, neighborhood information in one sample does not affect predictions, in the form of forward pass, in other samples. We propose a diffusion based layer to combine features from different samples for segmentation. Our layer can be used to incorporate information from multiple modalities for segmentation purposes within a single pipeline with no recurrences or recursions in them. Our pipeline uses diffused vision foundation models and CLIP to inform features across samples for novel concept segmentation as in training-free cases. We then incorporate language modality within our framework using CLIP embeddings for cuts guidance to enhance open-vocabulary semantic segmentation. Our empirical results show significant improvements on various benchmark datasets such as Pascal VOC and MS-COCO datasts. Our analysis on several downstream tasks suggest that diffusing information across samples while inference can provide significant performance improvements.

## Dependencies and Installation
```
# install torch and dependencies
pip install -r requirements.txt
```

## Datasets
We follow the [MMSeg data preparation document](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) to download and pre-process the datasets. 

## Open-vocabulary semantic segmentation evaluation
Please modify the settings in `configs/base_config.py` before running the evaluation.

Single-GPU:

```
python eval.py --config ./config/cfg_DATASET.py --workdir YOUR_WORK_DIR
```
## To Run either single cut or multiple cuts based on prediction 

```bash
python tmd_single_cut.py --dataset_path 'folder containing images' --save-feat-dir 'folder where the features are to be saved' --predict_n 'n_objects.json file for the specified dataset'
```
Or

```bash
python tmd_multiple_cuts_predicted.py --dataset_path 'folder containing images' --save-feat-dir 'folder where the features are to be saved' --predict_n 'n_objects.json file for the specified dataset'
```

## To calculate corloc scores
```bash
python corloc_calucaltion.py --dataset VOC12/VOC07/COCO20k --set trainval/val --output_dir 'path where the corloc scores are to be saved'  --box_path 'folder where the features are saved'
```
