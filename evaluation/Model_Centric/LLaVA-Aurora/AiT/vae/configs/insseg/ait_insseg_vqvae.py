from torchvision import transforms as T

image_size = 64
model = dict(
    image_size=image_size,
    num_resnet_blocks=2,
    downsample_ratio=16,
    num_tokens=128,
    codebook_dim=512,
    hidden_dim=16,
    channels=1,
    use_norm=True,
    train_objective='regression',
    max_value=1.,
    residul_type='v1',
    loss_type='mse',
)

train_setting = dict(
    output_dir='outputs/',
    data=dict(
        image_folder='data/maskcoco/instances_train2017',
        pipeline=[
            dict(type='Resize', size=image_size,
                 interpolation=T.InterpolationMode.BILINEAR),
            dict(type='CenterCrop', size=image_size),
            dict(type='CustomToTensor'),
            dict(type='Uint8Remap'),
        ],
    ),
    opt_params=dict(
        epochs=20,
        batch_size=512,
        learning_rate=3e-4,
        warmup_ratio=1e-3,
        warmup_steps=500,
        weight_decay=0.0,
        schedule_type='cosine',
    )
)

test_setting = dict(
    coco_dir='data/coco',
    target_size=(image_size, image_size),
    iou_type=['segm', 'boundary'],
    max_samples=5000,
    seed=1234,
)
