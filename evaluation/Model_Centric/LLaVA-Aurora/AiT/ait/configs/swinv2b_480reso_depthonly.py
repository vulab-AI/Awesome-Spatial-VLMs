checkpoint_config = dict(interval=5050)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

num_bins = 2000
num_classes = 80
num_embeddings = 128
num_embeddings_depth = 128
num_vocal = num_bins+1 + num_classes + 2 + num_embeddings + num_embeddings_depth

model = dict(
    type='AiT',
    backbone=dict(
        type="SwinV2TransformerRPE2FC",
        pretrain_img_size=192,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[30, 30, 30, 15],
        use_shift=[True, True, False, False],
        pretrain_window_size=[12, 12, 12, 6],
        ape=False,
        drop_path_rate=0.1,
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
        n_rows=15,
        n_cols=15,
        pos_enc='sine',
        pred_eos=False,
        soft_vae=True,
        soft_transformer=True,
        top_p=0.
    ),
    task_heads=dict(
        depth=dict(
            type='DepthHead',
            task_id=2,
            loss_weight=1.,
            depth_token_offset=num_bins+1 + num_classes + 2 + num_embeddings,
            vae_cfg=dict(
                type='VQVAE',
                use_norm=False,
                token_length=15*15,
                mask_size=480,
                embedding_dim=512,
                hidden_dim=256,
                num_resnet_blocks=2,
                num_embeddings=num_embeddings_depth,
                tau=0.8,
                pretrained='vqvae_depth.pt',
                freeze=True
            ),
            decoder_loss_weight=1.0,
            soft_vae=True),
    ))


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

file_client_args = dict(backend='disk')

runner = dict(type='IterBasedRunnerMultitask', max_iters=25250)
evaluation = dict(interval=25250)

# learning policy

lr_config = dict(
    policy='Step',
    step=[18180],  # 18/25 epoch
    by_epoch=False,
    warmup_ratio=0.1,
    warmup='linear',
    warmup_by_epoch=False,
    warmup_iters=500,
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.05,
    constructor='SwinLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=[2, 2, 18, 2], layer_decay_rate=0.9,
        no_decay_names=['relative_position_bias_table', 'rpe_mlp', 'logit_scale',
                        'det_embed', 'voc_embed', 'enc_embed', 'dec_embed', 'mask_embed'],
    ))
optimizer_config = dict(grad_clip={'max_norm': 10, 'norm_type': 2})

task = dict(
    depth=dict(  # len=24231
        times=1,
        data=dict(
            train=dict(type='nyudepthv2', data_path='data', filenames_path='code/dataset/depth/filenames/',
                       is_train=True, crop_size=(480, 480), samples_total_gpu=24),
            val=dict(type='nyudepthv2', data_path='data', filenames_path='code/dataset/depth/filenames/',
                     is_train=False, crop_size=(480, 480), samples_per_gpu=2, workers_per_gpu=8),
        )
    ),
)

# enable fp16
fp16 = dict(loss_scale='dynamic')
