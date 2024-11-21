_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_voc21.txt',
    prob_thd= 0.2     # 0.2  OURS  0.1 CLIP-B/16
)

# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = ''

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 336), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='/home/ellen/tokencut_tmd/diffused_cut/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages', seg_map_path='/home/ellen/tokencut_tmd/diffused_cut/datasets/VOC2012/VOCdevkit/VOC2012/SegmentationClass'),
        ann_file='/home/ellen/tokencut_tmd/diffused_cut/datasets/VOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))