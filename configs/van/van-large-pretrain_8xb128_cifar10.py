_base_ = [
    "../_base_/models/van/van_large.py",
    "../_base_/datasets/cifar10_bs16.py",
    "../_base_/schedules/imagenet_bs1024_adamw_swin.py",
    "../_base_/default_runtime.py",
]

# model setting
model = dict(
    init_cfg=dict(
        type="Pretrained",
        checkpoint="./checkpoints/van-large_8xb128_in1k_20220501-f212ba21.pth",
    ),
    head=dict(num_classes=10),
)

# dataset setting
data_preprocessor = dict(
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor["mean"][::-1]
bgr_std = data_preprocessor["std"][::-1]

train_pipeline = [
    # dict(type="LoadImageFromFile"),
    dict(
        type="RandomResizedCrop", scale=224, backend="pillow", interpolation="bicubic"
    ),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(
        type="RandAugment",
        policies="timm_increasing",
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[round(x) for x in bgr_mean], interpolation="bicubic"),
    ),
    dict(type="ColorJitter", brightness=0.4, contrast=0.4, saturation=0.4),
    dict(
        type="RandomErasing",
        erase_prob=0.25,
        mode="rand",
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std,
    ),
    dict(type="PackInputs"),
]

test_pipeline = [
    # dict(type="LoadImageFromFile"),
    dict(
        type="ResizeEdge",
        scale=248,
        edge="short",
        backend="pillow",
        interpolation="bicubic",
    ),
    dict(type="CenterCrop", crop_size=224),
    dict(type="PackInputs"),
]

train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline),
    # batch_size=128,
)
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

val_evaluator = dict(type="Accuracy", topk=(1, 5))

# schedules setting
optim_wrapper = dict(optimizer=dict(lr=5e-4 * 128 / 512))
auto_scale_lr = dict(base_batch_size=128)

# visual
custom_imports = dict(
    imports=["swanlab.integration.mmengine"], allow_failed_imports=False
)
vis_backends = [
    dict(type="LocalVisBackend"),
    dict(
        type="SwanlabVisBackend",
        save_dir="./swanlog",  # swanlab save path
        init_kwargs={  # swanlab.init args
            "project": "MMPretrain",  # project name on swanlab
            "experiment_name": "van-l-pretrain_bs128_cifar",  # experiment name on swanlab
            "description": "van-large exp use mmpretrain",  # experiment description (can be null)
            "workspace": "SwanLab",  # Your Organization on swanlab
            # "cloud": False,                       # Upload to cloud
        },
    ),
    dict(
        type="WandbVisBackend",
        init_kwargs={"project": "MMPretrain", "name": "van-l-pretrain_8xb16_cifar"},
    ),
]

visualizer = dict(vis_backends=vis_backends)
