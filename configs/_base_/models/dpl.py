# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DPLHead',
        vit=True,
        in_channels=2048,
        channels=512,
        num_classes=20,
        dropout_ratio=0,
        norm_cfg=norm_cfg,
        # loss_decode=dict(
        #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        decode_module_cfg=dict(
            type='ASPPHeadV2',
            input_transform=None,
            dilations=(6, 12, 18, 24)
        ),
        text_categories=20,
        text_channels=512,
        clip_channels=768,
        text_embeddings_path='pretrain/voc_ViT16_clip_text.pth',
        clip_unlabeled_cats=list(range(0, 20)),
        clip_cfg=dict(
            type='VisionTransformer',
            img_size=(224, 224),
            patch_size=16,
            patch_bias=False,
            in_channels=3,
            embed_dims=768,
            num_layers=12,
            num_heads=12,
            mlp_ratio=4,
            out_indices=-1,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            with_cls_token=True,
            output_cls_token=False,
            norm_cfg=dict(type='LN', eps=1e-6),
            act_cfg=dict(type='GELU'),
            patch_norm=False,
            pre_norm = True,
            final_norm=True,
            return_qkv=True,
            interpolate_mode='bicubic',
            num_fcs=2,
            norm_eval=False
        ),
        clip_weights_path='pretrain/ViT16_clip_weights.pth',
        reset_counter=True
    ),
    feed_img_to_decode_head=True,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
) 