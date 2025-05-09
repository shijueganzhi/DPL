_base_ = [
    '../_base_/models/dpl.py', '../_base_/datasets/cityscapes_w.py', 
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_60k_lr_0.005.py'
]

suppress_labels = list(range(0, 19))
model = dict(
    decode_head=dict(
        num_classes=19,
        text_categories=19,
        text_embeddings_path='pretrain/city_carb_ViT16_clip_text.pth',
        clip_unlabeled_cats=suppress_labels,
        warmup_iter=16000,
        patch_size=(512, 256),
        resize_rate=1,
        resize_offset=1,
        adaptive=True,
        get_train_mask=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_masked=True, loss_weight=1.0),
        sam_model_type='vit_h',
        sam_checkpoint='pretrained/sam_vit_h.pth',
        loss_sam_weight=0.3,
        sam_loss_warmup_iters=1000,
        sam_loss_update_freq=2000,
    ),
)

find_unused_parameters=True
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'],
         meta_keys = ['filename', 'ori_filename', 'ori_shape',
                      'img_shape', 'pad_shape', 'scale_factor', 'flip',
                      'flip_direction', 'img_norm_cfg', 'img_label']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'],
                 meta_keys=['filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 'img_label']),
        ])
]
data = dict(
    samples_per_gpu=2,
    train=dict(
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        img_labels='metadata/cityscapes/labels.npy',
        pipeline=train_pipeline
    ),
    val=dict(
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),
    test=dict(
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline)
) 