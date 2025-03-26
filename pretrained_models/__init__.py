import torch

model_path = r'D:\tracking\OSTrack-main\pretrained_models\mae_pretrain_vit_base.pth'
pretrained_dict = torch.load(model_path, map_location='cpu')

model_state_dict = pretrained_dict['model']
print(model_state_dict.keys())

for key, value in model_state_dict.items():
    print(key, value.shape)


