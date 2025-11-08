checkpoint_config = dict(interval=7330)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]

num_bins = 2000
num_classes = 80
num_embeddings = 128
num_embeddings_depth = 128
num_embeddings_others = 128  # other tasks token
num_vocal = num_bins+1 + num_classes + 2 + num_embeddings + \
    num_embeddings_depth + num_embeddings_others

model = dict(
    type='AiT',
    backbone=dict(
        type="SwinV2TransformerRPE2FC",
        pretrain_img_size=192,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[30, 30, 30, 15],
        use_shift=[True, True, True, True],
        pretrain_window_size=[12, 12, 12, 6],
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
        out_indices=(3,),
        init_cfg=dict(type='Pretrained',
                      checkpoint='swin_v2_base_densesimmim.pth'),
    ),
    transformer=dict(
        type='ARTransformer',
        in_chans=1024,
        d_model=256,
        drop_path=0.1,
        drop_out=0.1,
        nhead=8,
        dim_feedforward=1024,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_vocal=num_vocal,
        num_bins=num_bins,
        num_classes=num_classes,
        num_embeddings=num_embeddings,
        num_embeddings_depth=num_embeddings_depth,
        dec_length=2100,
        n_rows=20,
        n_cols=20,
        pos_enc='sine',
        pred_eos=False,
        soft_vae=False,
        soft_transformer=False,
        top_p=0.3
    ),
    task_heads=dict(
        det=dict(
            type='DetHead',
            task_id=0,
            loss_weight=1.,
            num_classes=num_classes,
            num_bins=num_bins,
            coord_norm='abs',  # abs or rel
            norm_val=640,
            sync_cls_avg_factor=True,
            seq_aug=True),
        insseg=dict(
            type='InsSegHead',
            task_id=1,
            loss_weight=1.,
            num_classes=num_classes,
            num_bins=num_bins,
            coord_norm='abs',  # abs or rel
            norm_val=640,
            sync_cls_avg_factor=True,
            vae_cfg=dict(
                type='VQVAE',
                token_length=16,
                mask_size=64,
                embedding_dim=512,
                hidden_dim=128,
                num_resnet_blocks=2,
                num_embeddings=num_embeddings,
                pretrained='vqvae_insseg.pt',
                freeze=True
            ),
            mask_weight=0.2,
            seq_aug=True),
    ))


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# large scale jitter pipeline from configs/common/lsj_100e_coco_instance.py
image_size = (640, 640)
file_client_args = dict(backend='disk')

runner = dict(type='IterBasedRunnerMultitask', max_iters=366500)
evaluation = dict(interval=73300)

# learning policy

lr_config = dict(
    policy='LinearAnnealing',
    by_epoch=False,
    min_lr_ratio=0.01,
    warmup='linear',
    warmup_by_epoch=False,
    warmup_iters=14660,
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.05,
    constructor='SwinLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=[2, 2, 18, 2], layer_decay_rate=0.85,
        no_decay_names=['relative_position_bias_table', 'rpe_mlp', 'logit_scale',
                        'det_embed', 'voc_embed', 'enc_embed', 'dec_embed', 'mask_embed'],
    ))
optimizer_config = dict(grad_clip={'max_norm': 50, 'norm_type': 2})

insseg_train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.1, 3.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute',
        crop_size=image_size,
        recompute_bbox=False,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=image_size),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
    dict(type='AddKey', kv={'task_type': 'insseg'})
]

insseg_test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=image_size),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

det_dataset_type = 'CocoDataset'

task = dict(
    insseg=dict(  # len=117266
        times=1,
        data=dict(
            train=dict(type=det_dataset_type, ann_file='data/coco/annotations/instances_train2017.json',
                       img_prefix='data/coco/train2017/', pipeline=insseg_train_pipeline, samples_total_gpu=16),
            val=dict(type=det_dataset_type, ann_file='data/coco/annotations/instances_val2017.json',
                     img_prefix='data/coco/val2017/', pipeline=insseg_test_pipeline, samples_per_gpu=8, workers_per_gpu=2),
        )
    ),
)


# enable fp16
fp16 = dict(loss_scale='dynamic')

load_from = 'ait_det_swinv2b_wodec.pth'
