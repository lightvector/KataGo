
# cd /g/Projects/KataGo/python

# python noise.py --base-dir "G:\Projects\KataGo\Training\BaseDir" --training-name "kata1-b28c512nbt" --model-kind "b28c512nbt" --noise-scale 5.0

# ./selfplay/export_model_for_selfplay.sh "noisy-5.0" "/g/Projects/KataGo/Training/BaseDir" "1"



import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
from model_pytorch import Model
import modelconfigs
import os
import logging
import load_model
import time
import shutil

def save(savepath, model, swa_model, optimizer, train_state, metrics, model_config):
    state_dict = {}
    state_dict["model"] = model.state_dict()
    state_dict["config"] = model_config
    if swa_model is not None:
        state_dict["swa_model"] = swa_model.state_dict()
    state_dict["optimizer"] = optimizer.state_dict()
    state_dict["train_state"] = train_state
    state_dict["metrics"] = metrics
    torch.save(state_dict, savepath + "/model.ckpt")


def main():
    # 参数解析
    parser = argparse.ArgumentParser(description="Add noise to a model checkpoint and export to torchmodels_toexport, copying train.py logic.")
    parser.add_argument('--base-dir', required=True, help='Base directory for training')
    parser.add_argument('--training-name', required=True, help='Name of the training run')
    parser.add_argument('--noise-scale', type=float, required=True, help='Scale of the noise to add to the weights')
    parser.add_argument('--model-kind', required=True, help='Model kind to use')
    parser.add_argument('--pos-len', type=int, default=19, help='Board size (e.g., 19 for 19x19)')
    parser.add_argument('--export-prefix', default='noisy', help='Prefix for exported model names')
    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(level=logging.INFO)

    # 检查点文件路径、保存路径及待导出路径
    train_path = os.path.join(args.base_dir, "train", args.training_name)
    checkpoint_path = os.path.join(train_path, "checkpoint.ckpt")
    export_dir = os.path.join(train_path, "noise")
    for_export_dir = os.path.join(args.base_dir, "torchmodels_toexport")
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    # 加载 checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 获取模型配置
    if 'config' in state_dict:
        model_config = state_dict['config']
        logging.info("Using config from checkpoint")
    else:
        model_config = modelconfigs.config_of_name[args.model_kind]
        logging.info("Using default config for model kind: %s", args.model_kind)
    
    # 创建模型
    raw_model = Model(model_config, args.pos_len)
    raw_model.initialize()
    
    # 加载模型权重
    model_state_dict = load_model.load_model_state_dict(state_dict)
    raw_model.load_state_dict(model_state_dict)  #, strict=False 使用 strict=False 以避免不必要的错误

    # # 添加噪声
    # with torch.no_grad():
    #     for param in raw_model.parameters():
    #         noise = torch.randn_like(param.data) * args.noise_scale
    #         param.data += noise
    #         logging.info(f"Added noise to parameter {param.shape} with scale {args.noise_scale}")

    # 添加噪声，跳过规范化层参数
    with torch.no_grad():
        for name, param in raw_model.named_parameters():
            if 'conv_spatial' in name or 'linear_global' in name:
                continue  # 跳过空间卷积层/全局线性层/中间层 or 'intermediate' in name
            # if any(s in name for s in ['value_head.linear_valuehead', 'value_head.conv_ownership', 'value_head.conv_scoring']):
            #     continue  # 跳过价值头的敏感输出权重

            # if 'weight' in name:
            noise_scale = args.noise_scale
            if 'blocks' in name:
                noise_scale *= 0.1  # 对 blocks 的噪声尺度降低1个数量级
            noise = torch.randn_like(param.data) * noise_scale
            param.data += noise
            logging.info(f"Added noise to parameter {name} with shape {param.shape} and scale {noise_scale}")


    # 模拟 train.py 的模型设置
    model = nn.DataParallel(raw_model)  # 复制 train.py 的 DataParallel 包装

    # 创建优化器，复制 train.py 的设置
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,  # train.py 使用配置文件中的 lr，这里使用默认值
        betas=(0.9, 0.999),  # train.py 默认值
        weight_decay=1e-4  # train.py 常用值
    )

    # 尝试加载输入 checkpoint 的优化器状态
    if 'optimizer' in state_dict:
        try:
            optimizer.load_state_dict(state_dict['optimizer'])
            logging.info("Loaded optimizer state from checkpoint")
        except Exception as e:
            logging.warning(f"Could not load optimizer state: {e}, using default")
    else:
        logging.warning("No optimizer state in checkpoint, using default")

    # 创建 SWA 模型，复制 train.py
    if 'swa_model' in state_dict:
        swa_model = AveragedModel(raw_model)  # 使用 raw_model，避免额外的 module 包装
        try:
            # 调整 checkpoint 中的 swa_model 键以匹配
            swa_state_dict = state_dict['swa_model']
            new_swa_state_dict = {}
            for k, v in swa_state_dict.items():
                if k.startswith("module.") or k.startswith("n_averaged"):
                    new_swa_state_dict[k] = v  # 保持 module. 前缀
                else:
                    new_swa_state_dict["module." + k] = v  # 添加 module. 前缀
            swa_model.load_state_dict(new_swa_state_dict)  #, strict=False 使用 strict=False 以避免不必要的错误
            logging.info("Loaded swa_model state from checkpoint")
        except Exception as e:
            logging.warning(f"Could not load swa_model state: {e}, using noisy model weights")
            swa_model = AveragedModel(raw_model)  # 回退到当前模型
    else:
        swa_model = None
        logging.info("No swa_model in checkpoint, skipping SWA")

    # # 创建 SWA 模型，复制 train.py
    # if 'swa_model' in state_dict:
    #     swa_model = AveragedModel(raw_model)  # 使用 raw_model
    #     try:
    #         swa_state_dict = state_dict['swa_model']
    #         new_swa_state_dict = {}
    #         for k, v in swa_state_dict.items():
    #             # 修正键以匹配 nn.DataParallel 包装
    #             if k.startswith("module."):
    #                 new_swa_state_dict[k.replace("module.", "", 1)] = v
    #             else:
    #                 new_swa_state_dict[k] = v
    #         # 单独处理 n_averaged
    #         if "module.n_averaged" in swa_state_dict:
    #             new_swa_state_dict["n_averaged"] = swa_state_dict["module.n_averaged"]
    #         swa_model.load_state_dict(new_swa_state_dict)
    #         logging.info("Loaded swa_model state from checkpoint")
    #     except Exception as e:
    #         logging.warning(f"Could not load swa_model state: {e}, using original model weights without noise")
    #         swa_model = AveragedModel(raw_model)
    # else:
    #     swa_model = None
    #     logging.info("No swa_model in checkpoint, skipping SWA")


    # 提取 train_state 和 metrics
    train_state = state_dict.get('train_state', {
        'global_step_samples': 0,
        'total_num_data_rows': 0,
        'global_step': 0
    })
    metrics = state_dict.get('metrics', {})

    if not metrics:
        logging.warning("Metrics not found in checkpoint, using empty metrics")
    else:
        logging.info("Loaded metrics from checkpoint")

    # 模拟 train.py 的命名方式
    args.export_prefix += f"-{args.noise_scale}"

    step_samples = train_state.get('global_step_samples', 0)
    data_rows = train_state.get('total_num_data_rows', 0)
    model_name = f"{args.export_prefix}-s{step_samples}-d{data_rows}"

    save_path = os.path.join(export_dir, model_name)
    save_path_tmp = os.path.join(export_dir, f"{model_name}.tmp")

    export_model_dir = os.path.join(for_export_dir, model_name)
    export_model_path = os.path.join(export_model_dir, "model.ckpt")

    # 创建临时目录并保存模型
    if os.path.exists(save_path):
        logging.info(f"Model already exists at {save_path}, skipping save")
    else:
        os.makedirs(save_path_tmp)
        save(save_path_tmp, model, swa_model, optimizer, train_state, metrics, model_config)
        time.sleep(5)  # 短暂等待以确保文件写入完成
        os.rename(save_path_tmp, save_path)

    # 复制到 for_export_dir
    if os.path.exists(export_model_path):
        logging.info(f"Model already exists at {export_model_path}, skipping copy")
    else:
        os.makedirs(export_model_dir, exist_ok=True)
        shutil.copy(os.path.join(save_path, "model.ckpt"), export_model_path)
        logging.info(f"Copied model to {export_model_path}")

    logging.info(f"Added noise with scale {args.noise_scale} to the model")
    logging.info(f"Saved to {save_path}/model.ckpt, and copied to {export_model_path}")
    logging.info(f"Model name: {model_name}")

if __name__ == '__main__':
    main()