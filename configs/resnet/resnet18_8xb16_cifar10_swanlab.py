_base_ = [
    "../_base_/models/resnet18_cifar.py",
    "../_base_/datasets/cifar10_bs16.py",
    "../_base_/schedules/cifar10_bs128.py",
    "../_base_/default_runtime.py",
]

# custom_imports = dict(
#     imports=["swanlab.integration.mmengine"], allow_failed_imports=False
# )
# vis_backends = [
#     dict(type="LocalVisBackend"),
#     dict(
#         type="SwanlabVisBackend",
#         save_dir="./swanlog",  # swanlab save path
#         init_kwargs={  # swanlab.init args
#             "project": "MMPretrain",  # project name on swanlab
#             "experiment_name": "resnet18_8xb16_cifar",  # experiment name on swanlab
#             "description": "resnet18 exp use mmpretrain",  # experiment description (can be null)
#             "workspace": "SwanLab",  # Your Organization on swanlab
#             # "cloud": False,                       # Upload to cloud
#         },
#     ),
#     dict(
#         type="WandbVisBackend",
#         init_kwargs={"project": "MMPretrain", "name": "resnet18_8xb16_cifar"},
#     ),
# ]

# visualizer = dict(vis_backends=vis_backends)
