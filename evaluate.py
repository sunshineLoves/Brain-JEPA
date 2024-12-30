import os
import yaml
import torch
from downstream_tasks.models_vit import VisionTransformer
from downstream_eval import Config, load_args_from_yaml
from src.datasets.adni_dx_datasets import make_adni_dx

output_dir = '/mnt/2T-LabOwned/yuquan/ai4science/Brain-JEPA/output_dir/adni_dx/fine_tune_classification/jepa-ep300_2024-12-12_17-07-01/ft_output'
config_file = os.path.join(output_dir, 'config.yaml')

yaml_args = load_args_from_yaml(config_file)
args = Config(yaml_args)

model = VisionTransformer(
        args,
        model_name=args.model_name,
        attn_mode=args.attn_mode,
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
        device='cuda',
        add_w=args.add_w
    )

checkpoint = torch.load(os.path.join(output_dir, 'checkpoint-49.pth'))
model.load_state_dict(checkpoint['model'])

data_loader_train, data_loader_val, data_loader_test, train_dataset, valid_dataset, test_dataset = make_adni_dx(
    batch_size=args.batch_size,
    pin_mem=args.pin_mem,
    num_workers=args.num_workers,
    drop_last=False,
    # processed_dir=f'path/to/dataset',
    use_normalization=args.use_normalization,
    downsample=args.downsample
)

# model.eval()
for i, (inputs, targets) in enumerate(data_loader_test):
    outputs = model(inputs)
    print(outputs)
    break
