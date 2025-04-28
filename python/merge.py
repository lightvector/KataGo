
# cd /g/Projects/KataGo-Noise/python

# python merge.py --base-dir "G:\Projects\KataGo-Noise\Training\BaseDir" --training-name "kata1-b28c512nbt" --model-kind "b28c512nbt"


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

def get_raw_model(model_config, pos_len):
    # 创建模型实例
    raw_model = Model(model_config, pos_len)
    raw_model.initialize()
    return raw_model

def get_swa_model(raw_model, state_dict):
    # 创建 SWA 模型实例
    if 'swa_model' in state_dict:
        swa_model = AveragedModel(raw_model)  # 使用原始模型，避免额外的 module 包装

        # 调整 checkpoint 中的 swa_model 键以匹配
        swa_state_dict = state_dict['swa_model']
        new_swa_state_dict = {}
        for k, v in swa_state_dict.items():
            if k.startswith("module.") or k.startswith("n_averaged"):
                new_swa_state_dict[k] = v  # 保持 module. 前缀
            else:
                new_swa_state_dict["module." + k] = v  # 添加 module. 前缀
        swa_model.load_state_dict(new_swa_state_dict)
    else:
        swa_model = None
    return swa_model

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description="Merge and export models")
    parser.add_argument('--base-dir', required=True, help='Base directory for training')
    parser.add_argument('--training-name', required=True, help='Name of the training run')
    parser.add_argument('--model-kind', required=True, help='Model kind to use')
    parser.add_argument('--pos-len', type=int, default=19, help='Board size (e.g., 19 for 19x19)')
    parser.add_argument('--export-prefix', default='merged', help='Prefix for exported model names')
    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H-%M-%S%z:'  # 指定日期时间格式和时区
    )

    # 检查点文件路径、保存路径及待导出路径
    train_path = os.path.join(args.base_dir, "train", args.training_name)
    source_path = os.path.join(train_path, "sources")
    source_files = [f for f in os.listdir(source_path) if f.endswith('.ckpt')]
    source_files.sort(key=lambda f: os.path.getmtime(os.path.join(source_path, f)))
    source_raws = {}
    checkpoint_path = os.path.join(train_path, "checkpoint.ckpt") # '_' ************************************************************************

    export_dir = os.path.join(train_path, "merged")
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
    
    # # 创建模型
    # raw_model = Model(model_config, args.pos_len)
    # raw_model.initialize()
    raw_model = get_raw_model(model_config, args.pos_len)
    
    # 加载模型权重
    model_state_dict = load_model.load_model_state_dict(state_dict)
    raw_model.load_state_dict(model_state_dict)


    # 模拟 train.py 的模型设置
    model = nn.DataParallel(raw_model)

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

    swa_model = get_swa_model(raw_model, state_dict)

    # 遍历 source_path 中的所有模型文件并更新 SWA 模型
    logging.info(f"Updating SWA model with models from {source_path}")
    logging.info(f"Found {len(source_files)} source checkpoints.")

    for source_file in source_files:
        source_ckpt_path = os.path.join(source_path, source_file)
        logging.info(f"Processing source checkpoint: {source_ckpt_path}")

        source_state_dict = torch.load(source_ckpt_path, map_location='cpu', weights_only=False)
        source_model_state_dict = load_model.load_model_state_dict(source_state_dict)

        # 创建一个临时模型以加载源模型的权重
        temp_raw_model = get_raw_model(model_config, args.pos_len)
        temp_raw_model.load_state_dict(source_model_state_dict)

        temp_swa_model = get_swa_model(temp_raw_model, source_state_dict)
        for i in range(5):
            temp_swa_model.update_parameters(temp_raw_model)

        # 将 temp_SWA 权重复制到 temp_raw 中
        temp_raw_model.load_state_dict(temp_swa_model.module.state_dict())

        source_raws[source_file] = temp_raw_model

        del temp_raw_model
        del temp_swa_model
        del source_model_state_dict
        del source_state_dict
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    iterations = 50
    for i in range(iterations):
        logging.info(f"Starting iteration {i+1}/{iterations}")
        logging.info("Update the SWA model with current model weights")
        swa_model.update_parameters(raw_model)

        for source_file in source_raws:
            # 用临时模型的权重更新 SWA 模型
            swa_model.update_parameters(source_raws[source_file])
            logging.info(f"Updated SWA model with {source_file}")
        logging.info(f"Iteration {i+1}/{iterations} completed.")
    
    raw_model.load_state_dict(swa_model.module.state_dict())
    logging.info("Merging completed.")


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
    args.export_prefix += f"-{len(source_files)}sources"

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

    logging.info(f"Saved to {save_path}/model.ckpt, copying ...")
    # 复制到 for_export_dir
    if os.path.exists(export_model_path):
        logging.info(f"Model already exists at {export_model_path}, skipping copy")
    else:
        os.makedirs(export_model_dir, exist_ok=True)
        shutil.copy(os.path.join(save_path, "model.ckpt"), export_model_path)
        logging.info(f"Copied model to {export_model_path}")

    logging.info(f"Model name: {model_name}")

if __name__ == '__main__':
    main()