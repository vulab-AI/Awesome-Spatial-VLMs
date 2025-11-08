image_size = 320
model = dict(
    image_size=image_size,
    num_resnet_blocks=2,
    downsample_ratio=32,
    num_tokens=128,
    codebook_dim=512,
    hidden_dim=16,
    use_norm=False,
    channels=1,
    train_objective='regression',
    max_value=10.,
    residul_type='v1',
    loss_type='mse',
)

train_setting = dict(
    output_dir='outputs_ade',
    data=dict(
        is_train=True,
        data_path='ADE_depth/',
        crop_size=(image_size, image_size),
        mask=False
    ),
    opt_params=dict(
        epochs=20,
        batch_size=8,
        learning_rate=3e-4,
        lr_decay_rate=0.98,
        schedule_step=500,
        schedule_type='exp',
    )
)


