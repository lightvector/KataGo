import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
from model_pytorch import Model
import numpy as np
import modelconfigs
import os
import logging
import load_model
import time
import shutil
import json
import traceback
import re
import torch

NUM_SHORTTERM_CHECKPOINTS_TO_KEEP = 4

def save(savepath, traindir, model, swa_model, optimizer, train_state, metrics, model_config):
    state_dict = {}
    state_dict["model"] = model.state_dict()
    state_dict["config"] = model_config
    if swa_model is not None:
        state_dict["swa_model"] = swa_model.state_dict()
    
    # Skip optimizer state to avoid parameter group mismatches
    # state_dict["optimizer"] = optimizer.state_dict()
    
    state_dict["train_state"] = train_state
    
    # Remove metrics to reduce file size and avoid compatibility issues
    # state_dict["metrics"] = metrics
    
    # Clean up train_state similar to clean_checkpoint.py
    if "old_train_data_dirs" in train_state:
        train_state_copy = train_state.copy()
        train_state_copy.pop("old_train_data_dirs", None)
        train_state_copy.pop("data_files_used", None)
        state_dict["train_state"] = train_state_copy
    
    torch.save(state_dict, savepath + "/model.ckpt")

    logging.info("Saving checkpoint: " + get_checkpoint_path(traindir))
    for i in reversed(range(NUM_SHORTTERM_CHECKPOINTS_TO_KEEP-1)):
        if os.path.exists(get_checkpoint_prev_path(i, traindir)):
            os.replace(get_checkpoint_prev_path(i, traindir), get_checkpoint_prev_path(i+1, traindir))
    if os.path.exists(get_checkpoint_path(traindir)):
        shutil.copy(get_checkpoint_path(traindir), get_checkpoint_prev_path(0, traindir))
    torch.save(state_dict, get_checkpoint_path(traindir) + ".tmp")
    os.replace(get_checkpoint_path(traindir) + ".tmp", get_checkpoint_path(traindir))

def get_checkpoint_path(traindir = None):
    return os.path.join(traindir,"checkpoint.ckpt")

def get_checkpoint_prev_path(i, traindir = None):
    return os.path.join(traindir,f"checkpoint_prev{i}.ckpt")

def get_raw_model(model_config, pos_len):
    """创建模型实例"""
    raw_model = Model(model_config, pos_len)
    raw_model.initialize()
    return raw_model

def get_swa_model(raw_model, state_dict, swa_scale=None):
    """创建 SWA 模型实例"""
    if swa_scale is None:
        swa_scale = 4
    if 'swa_model' in state_dict:
        new_factor = 1.0 / swa_scale
        ema_avg = lambda avg_param, cur_param, num_averaged: avg_param + new_factor * (cur_param - avg_param)
        swa_model = AveragedModel(raw_model, avg_fn=ema_avg)

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

# # 全局参数缓存
delta_cache = {}

def generate_skew_normal(shape, skewness, loc, scale, device, dtype, group_name):
    """使用 PyTorch 生成 Skew-Normal 分布样本。
    
    参数:
        shape: 输出张量形状
        skewness: 偏度参数
        loc: 均值
        scale: 标准差
        device: 目标设备（如 'cuda' 或 'cpu'）
        dtype: 数据类型（如 torch.float32）
        group_name: 参数组名称
    
    返回:
        Skew-Normal 分布的样本张量
    """
    # 计算 delta
    cache_key = (skewness, device, dtype)
    if cache_key not in delta_cache:
        alpha = torch.tensor(skewness, device=device, dtype=dtype)
        delta_cache[cache_key] = alpha / torch.sqrt(1 + alpha**2)

        logging.info(f"Calculated delta for group {group_name}: {delta_cache[cache_key].item()}")

    delta = delta_cache[cache_key]
    
    # 生成两个独立的标准正态样本
    U1 = torch.randn(shape, device=device, dtype=dtype)
    U2 = torch.randn(shape, device=device, dtype=dtype)
    
    # 计算 Skew-Normal 样本
    Z = delta * torch.abs(U1) + torch.sqrt(1 - delta**2) * U2
    
    # 缩放和平移
    X = loc + scale * Z
    
    return X

def generate_noise(param, group_name, noise_scale):
    """生成模拟权重更新经验分布的噪声。
    
    参数:
        param: 需要生成噪声的参数
        group_name: 参数组名称
        noise_scale: 噪声强度的缩放因子
    
    返回:
        具有适当统计特性的噪声张量
    """
    # global param_cache
    # cache_key = (group_name)
    
    # 从ratio_data获取详细的统计信息
    mu_abs = ratio_data['abs_means'].get(group_name, 1e-6)
    var_abs = ratio_data['abs_vars'].get(group_name, 1e-11)
    sigma_abs = math.sqrt(var_abs)
    skewness = ratio_data['skewness'].get(group_name, 3.0)
    mu_rel = ratio_data['update_means'].get(group_name, 0.0)
    var_rel = ratio_data['update_vars'].get(group_name, 0.01)
    sigma_rel = math.sqrt(var_rel)
    
    # 应用噪声缩放
    noise_scale = float(noise_scale)
    adjusted_mu = mu_abs * noise_scale
    adjusted_sigma = sigma_abs * noise_scale
    bound = adjusted_sigma * 10

    # 使用 PyTorch 生成 Skew-Normal 噪声
    X = generate_skew_normal(
        shape=param.data.shape,
        skewness=skewness,
        loc=adjusted_mu,
        scale=adjusted_sigma,
        device=param.device,
        dtype=param.dtype,
        group_name=group_name
    )

    # 裁剪噪声范围
    X = torch.clamp(X, adjusted_mu - bound, adjusted_mu + bound)
    
    # 生成方向噪声
    direction = torch.randn_like(param.data) * sigma_rel + mu_rel
    
    # 组合幅度和方向
    noise_sign = torch.sign(direction)
    noise = X * param.data.abs() * noise_sign
    
    # raw_mean = X.mean().item()
    # raw_std = X.std().item()
    # noise_mean = noise.mean().item()
    # noise_std = noise.std().item()

    # # # if abs(raw_mean) > 1e-5 and abs(raw_std) > 1e-4:
    # logging.info(f"Noise generated for {group_name}: \nraw_mean={raw_mean}, \nraw_std={raw_std}, \nnoise_mean={noise_mean}, \nnoise_std={noise_std}")

    return noise

def parse_statistics_block(block):
    """解析统计信息块，提取模块数据"""
    stats_data = {
        "abs_means": {},
        "abs_vars": {},
        "skewness": {},
        "kurtosis": {},
        "update_means": {},
        "update_vars": {},
        "data_lengths": {}
    }
    
    # 首先清理数据块，移除可能干扰解析的内容
    clean_block = re.sub(r'\n(?:pacc1:|p0loss|vloss|tdvloss|oloss|sloss|fploss|gnorm|loss)\s+.*?\n', '\n', block)
    
    # 解析模块特定的 abs_update_ratio 数据
    abs_pattern = r'(conv_spatial|linear_global|norm_beta|norm_gamma|blocks|policy_head|value_head|intermediate_policy|intermediate_value)\s+abs_update_ratio:.*?mean=([\d\.eE\-\+]+).*?var=([\d\.eE\-\+]+)(?:.*?skewness=([\d\.eE\-\+]+))?(?:.*?kurtosis=([\d\.eE\-\+]+))?.*?data length:\s*(\d+)'
    for match in re.finditer(abs_pattern, clean_block, re.DOTALL):
        groups = match.groups()
        module = groups[0]
        mean = groups[1]
        var = groups[2]
        skewness = groups[3] if len(groups) > 3 and groups[3] is not None else None
        kurtosis = groups[4] if len(groups) > 4 and groups[4] is not None else None
        length = groups[5]
        
        if mean and var and length:  # 确保必需字段存在
            stats_data["abs_means"][module] = float(mean)
            stats_data["abs_vars"][module] = float(var)
            stats_data["data_lengths"][f"abs_{module}"] = int(length)
            if skewness:
                stats_data["skewness"][module] = float(skewness)
            if kurtosis:
                stats_data["kurtosis"][module] = float(kurtosis)
    
    # 解析 update_ratio 数据
    update_pattern = r'(conv_spatial|linear_global|norm_beta|norm_gamma|blocks|policy_head|value_head|intermediate_policy|intermediate_value)\s+update_ratio:.*?mean=([\d\.eE\-\+]+).*?var=([\d\.eE\-\+]+).*?data length:\s*(\d+)'
    for match in re.finditer(update_pattern, clean_block, re.DOTALL):
        groups = match.groups()
        module = groups[0]
        mean = groups[1]
        var = groups[2]
        length = groups[3]
        
        if mean and var and length:  # 确保必需字段存在
            stats_data["update_means"][module] = float(mean)
            stats_data["update_vars"][module] = float(var)
            stats_data["data_lengths"][f"update_{module}"] = int(length)
    
    logging.info(f"Parsed stats block: {len(stats_data['abs_means'])} abs modules, {len(stats_data['update_means'])} update modules")
    if len(stats_data['abs_means']) == 0:
        logging.debug(f"Failed to parse abs_update_ratio in block: {clean_block[:500].replace('\n', '\\n')}...")
    if len(stats_data['update_means']) == 0:
        logging.debug(f"Failed to parse update_ratio in block: {clean_block[:500].replace('\n', '\\n')}...")
    return stats_data

def extract_stats_from_stdout(file_path, num_lines_to_check=2000):
    """从 stdout.txt 中提取最新的统计信息"""
    logging.info(f"Attempting to extract stats from {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取文件末尾的内容
            lines = f.readlines()[-num_lines_to_check:]  # 直接读取最后 2000 行
            content = ''.join(lines)
            
            logging.debug(f"Read {len(lines)} lines from {file_path}, total {len(content)} characters")
            
            # 预处理 - 移除干扰项，类似trend_update.py的方法
            cleaned_content = re.sub(r'\n(?:pacc1:|p0loss|vloss|tdvloss|oloss|sloss|fploss|skloss|smloss|'
                           r'norm_normal|gnorm_batch|exgnorm|loss|time_since_last_print|Accumulating SWA).*?\n', 
                           '\n\n\n', content)
            
            # 按多个连续换行分割，找到包含完整数据的块
            blocks = re.split(r'\n{3,}', cleaned_content)
            valid_blocks = []
            
            for block in blocks:
                if 'abs_update_ratio:' in block and 'update_ratio:' in block:
                    valid_blocks.append(block)
            
            if not valid_blocks:
                logging.warning("No valid data blocks found")
                return {
                    "abs_means": {}, "abs_vars": {}, "skewness": {}, "kurtosis": {},
                    "update_means": {}, "update_vars": {}, "data_lengths": {}
                }
            
            # 使用最后一个有效块
            stats_block = valid_blocks[-1]
            logging.debug(f"Using last valid block: {stats_block[:500].replace('\n', '\\n')}...")
            
            stats_data = parse_statistics_block(stats_block)
            logging.info(f"Extracted stats: {len(stats_data['abs_means'])} abs modules, {len(stats_data['update_means'])} update modules")
            return stats_data
    
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        traceback.print_exc()
    
    logging.warning("Failed to extract stats, returning empty data")
    return {
        "abs_means": {}, "abs_vars": {}, "skewness": {}, "kurtosis": {},
        "update_means": {}, "update_vars": {}, "data_lengths": {}
    }

def read_old_ratios_data(file_path):
    """读取旧的 ratios.txt 数据"""
    logging.info(f"Reading old ratios from {file_path}")
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                old_data = json.load(f)
                logging.info(f"Loaded old ratios with {len(old_data.get('abs_means', {}))} abs modules")
                return old_data
        else:
            logging.info(f"{file_path} does not exist, returning empty data")
    except Exception as e:
        logging.error(f"Failed to read {file_path}: {e}")
    
    return {
        "abs_means": {}, "abs_vars": {}, "skewness": {}, "kurtosis": {},
        "update_means": {}, "update_vars": {}, "data_lengths": {}
    }

def combine_ratio_data(old_data, new_data):
    """合并旧数据和新数据"""
    logging.info("Combining old and new ratio data")
    combined_data = {
        'abs_means': {},
        'abs_vars': {},
        'skewness': {},
        'kurtosis': {},
        'update_means': {},
        'update_vars': {},
        'data_lengths': old_data.get('data_lengths', {}).copy()
    }
    
    # 处理 abs 数据
    for key in set(new_data['abs_means'].keys()) | set(old_data.get('abs_means', {}).keys()):
        old_length = old_data.get('data_lengths', {}).get(f"abs_{key}", 0)
        new_length = new_data['data_lengths'].get(f"abs_{key}", 0)
        
        if key in new_data['abs_means'] and key in old_data.get('abs_means', {}) and old_length > 0 and new_length > 0:
            logging.info(f"Combining data for {key}: old length {old_length}, new length {new_length}")
            # 合并旧数据和新数据
            total_length = old_length + new_length
            weight_old = old_length / total_length
            weight_new = new_length / total_length
            
            combined_data['abs_means'][key] = (
                old_data['abs_means'][key] * weight_old + 
                new_data['abs_means'][key] * weight_new
                )
            combined_data['abs_vars'][key] = (
                old_data['abs_vars'][key] * weight_old + 
                new_data['abs_vars'][key] * weight_new + 
                weight_old * weight_new * (
                    old_data['abs_means'][key] - 
                    new_data['abs_means'][key]
                    ) ** 2
                )
            
            if combined_data['abs_means'][key] == old_data['abs_means'][key] and combined_data['abs_vars'][key] == old_data['abs_vars'][key]:
                logging.info(f"Abs data for {key} unchanged, keeping old data")
                combined_data['data_lengths'][f"abs_{key}"] = old_length
                combined_data['skewness'][key] = old_data['skewness'][key]
                combined_data['kurtosis'][key] = old_data['kurtosis'][key]
                continue

            combined_data['data_lengths'][f"abs_{key}"] = total_length
            
            # 处理偏度和峰度
            old_third_moment = old_data['skewness'][key] * (old_data['abs_vars'][key] ** 1.5)
            new_third_moment = new_data['skewness'][key] * (new_data['abs_vars'][key] ** 1.5)
            delta = new_data['abs_means'][key] - old_data['abs_means'][key]
            combined_third_moment = (
                weight_old * old_third_moment + 
                weight_new * new_third_moment +
                3 * weight_old * weight_new * delta * (
                    weight_new * old_data['abs_vars'][key] - 
                    weight_old * new_data['abs_vars'][key]
                ) +
                weight_old * weight_new * (weight_old - weight_new) * (delta ** 3)
            )
            combined_data['skewness'][key] = combined_third_moment / (combined_data['abs_vars'][key] ** 1.5)
        
            old_fourth_moment = old_data['kurtosis'][key] * (old_data['abs_vars'][key] ** 2)
            new_fourth_moment = new_data['kurtosis'][key] * (new_data['abs_vars'][key] ** 2)
            delta = new_data['abs_means'][key] - old_data['abs_means'][key]
            combined_fourth_moment = (
                weight_old * old_fourth_moment + 
                weight_new * new_fourth_moment +
                6 * weight_old * weight_new * (
                    weight_new * old_data['abs_vars'][key] - 
                    weight_old * new_data['abs_vars'][key]
                ) * (delta ** 2) +
                4 * weight_old * weight_new * (weight_new - weight_old) * (delta ** 3) +
                weight_old * weight_new * (weight_old ** 2 + weight_new ** 2) * (delta ** 4)
            )
            combined_data['kurtosis'][key] = combined_fourth_moment / (combined_data['abs_vars'][key] ** 2)
        elif key in new_data['abs_means']:
            logging.info(f"New data for {key}: length {new_length}")
            # 仅有新数据
            combined_data['abs_means'][key] = new_data['abs_means'][key]
            combined_data['abs_vars'][key] = new_data['abs_vars'][key]
            combined_data['data_lengths'][f"abs_{key}"] = new_length
            combined_data['skewness'][key] = new_data['skewness'][key]
            combined_data['kurtosis'][key] = new_data['kurtosis'][key]
        elif key in old_data.get('abs_means', {}):
            logging.info(f"Old data for {key}: length {old_length}")
            # 仅有旧数据 - 保留旧数据
            combined_data['abs_means'][key] = old_data['abs_means'][key]
            combined_data['abs_vars'][key] = old_data['abs_vars'][key]
            combined_data['data_lengths'][f"abs_{key}"] = old_length
            combined_data['skewness'][key] = old_data['skewness'][key]
            combined_data['kurtosis'][key] = old_data['kurtosis'][key]
    
    # 处理 update 数据
    for key in set(new_data['update_means'].keys()) | set(old_data.get('update_means', {}).keys()):
        old_length = old_data.get('data_lengths', {}).get(f"update_{key}", 0)
        new_length = new_data['data_lengths'].get(f"update_{key}", 0)
        
        if key in new_data['update_means'] and key in old_data.get('update_means', {}) and old_length > 0 and new_length > 0:
            logging.info(f"Combining data for {key}: old length {old_length}, new length {new_length}")
            total_length = old_length + new_length
            weight_old = old_length / total_length
            weight_new = new_length / total_length
            
            combined_data['update_means'][key] = (
                old_data['update_means'][key] * weight_old + 
                new_data['update_means'][key] * weight_new
                )
            combined_data['update_vars'][key] = (
                old_data['update_vars'][key] * weight_old + 
                new_data['update_vars'][key] * weight_new + 
                weight_old * weight_new * (
                    old_data['update_means'][key] - 
                    new_data['update_means'][key]
                    ) ** 2)
            
            if combined_data['update_means'][key] == old_data['update_means'][key] and combined_data['update_vars'][key] == old_data['update_vars'][key]:
                logging.info(f"Relative data for {key} unchanged, keeping old data")
                combined_data['data_lengths'][f"update_{key}"] = old_length
                continue

            combined_data['data_lengths'][f"update_{key}"] = total_length
        elif key in new_data['update_means']:
            logging.info(f"New data for {key}: length {new_length}")
            combined_data['update_means'][key] = new_data['update_means'][key]
            combined_data['update_vars'][key] = new_data['update_vars'][key]
            combined_data['data_lengths'][f"update_{key}"] = new_length
        elif key in old_data.get('update_means', {}):
            logging.info(f"Old data for {key}: length {old_length}")
            # 仅有旧数据 - 保留旧数据
            combined_data['update_means'][key] = old_data['update_means'][key]
            combined_data['update_vars'][key] = old_data['update_vars'][key]
            combined_data['data_lengths'][f"update_{key}"] = old_length
    
    logging.info(f"Combined data: {len(combined_data['abs_means'])} abs modules, {len(combined_data['update_means'])} update modules")
    return combined_data

def sort_ratio_data(data):
    """按模块顺序排序统计数据"""
    module_order = ['conv_spatial', 'linear_global', 'norm_beta', 'norm_gamma', 'blocks', 
                    'policy_head', 'value_head', 'intermediate_policy', 'intermediate_value']
    sorted_data = {
        'abs_means': {}, 'abs_vars': {}, 'skewness': {}, 'kurtosis': {},
        'update_means': {}, 'update_vars': {}, 'data_lengths': {}
    }
    
    for module in module_order:
        if module in data['abs_means']:
            sorted_data['abs_means'][module] = data['abs_means'][module]
            sorted_data['abs_vars'][module] = data['abs_vars'][module]
            sorted_data['data_lengths'][f"abs_{module}"] = data['data_lengths'].get(f"abs_{module}", 0)
            if module in data['skewness']:
                sorted_data['skewness'][module] = data['skewness'][module]
            if module in data['kurtosis']:
                sorted_data['kurtosis'][module] = data['kurtosis'][module]
        
        if module in data['update_means']:
            sorted_data['update_means'][module] = data['update_means'][module]
            sorted_data['update_vars'][module] = data['update_vars'][module]
            sorted_data['data_lengths'][f"update_{module}"] = data['data_lengths'].get(f"update_{module}", 0)
    
    logging.info(f"Sorted data: {len(sorted_data['abs_means'])} abs modules, {len(sorted_data['update_means'])} update modules")
    return sorted_data

def save_ratio_data(data, file_path):
    """保存统计数据到文件"""
    logging.info(f"Saving data to {file_path}")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Successfully saved data to {file_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save data to {file_path}: {e}")
        return False

def read_latest_update_ratios(train_dir, num_lines_to_check=2000):
    """从 stdout.txt 文件中高效读取最新的 update ratio 数据，并与 ratios.txt 中的数据进行加权平均"""
    stdout_file_path = os.path.join(train_dir, "stdout.txt")
    ratios_file_path = os.path.join(train_dir, "ratios.txt")
    
    logging.info(f"Starting process with train_dir: {train_dir}")
    new_data = extract_stats_from_stdout(stdout_file_path, num_lines_to_check)
    if not any(new_data.values()):
        logging.warning("No new stats extracted from stdout.txt")
    
    old_data = read_old_ratios_data(ratios_file_path)
    combined_data = combine_ratio_data(old_data, new_data)
    sorted_data = sort_ratio_data(combined_data)
    save_ratio_data(sorted_data, ratios_file_path)
    
    logging.info("Process completed")
    return sorted_data

# 定义全局变量以存储参数组的 abs_update_ratio 和 update_ratio 的统计数据
ratio_data = {}

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description="Add noise to a model checkpoint and export to torchmodels_toexport, copying train.py logic.")
    parser.add_argument('--base-dir', required=True, help='Base directory for training')
    parser.add_argument('--training-name', required=True, help='Name of the training run')
    parser.add_argument('--noise-scale', type=float, required=True, help='Scale of the noise to add to the weights')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations to add noise')
    parser.add_argument('--model-kind', required=True, help='Model kind to use')
    parser.add_argument('--pos-len', type=int, default=19, help='Board size (e.g., 19 for 19x19)')
    parser.add_argument('--export-prefix', default='noisy', help='Prefix for exported model names')
    args = parser.parse_args()

    # 设置日志
    handlers = [logging.StreamHandler()]
    log_file = os.path.join(args.base_dir, "logs/outnoise.txt")
    handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.DEBUG,
        handlers=handlers,
        datefmt='%Y-%m-%d %H-%M-%S%z:'
    )
    # 检查点文件路径、保存路径及待导出路径
    train_path = os.path.join(args.base_dir, "train", args.training_name)
    checkpoint_path = os.path.join(train_path, "checkpoint.ckpt") # '_' ************************************************************************
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
    raw_model = get_raw_model(model_config, args.pos_len)
    
    # 加载模型权重
    model_state_dict = load_model.load_model_state_dict(state_dict)
    raw_model.load_state_dict(model_state_dict)

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

    swa_model = get_swa_model(raw_model, state_dict)
    swa_backup = None
    # swa_backup_flag = False
    logging.info(f"SWA model loaded: {swa_model is not None}")

    # 读取最新的 update ratio 数据
    global ratio_data
    ratio_data = read_latest_update_ratios(train_path)

    # 打印 ratio_data 的详细信息
    logging.info(f"Loaded ratio_data with {len(ratio_data)} sections")
    for section, data in ratio_data.items():
        if isinstance(data, dict):
            logging.info(f"  Section '{section}' has {len(data)} parameter groups")
        else:
            logging.info(f"  Section '{section}': {data}")

    logging.info("Detailed ratio_data:")
    for section, data in ratio_data.items():
        if isinstance(data, dict):
            logging.info(f"Section: {section}")
            for group_name, value in data.items():
                logging.info(f"  {group_name}: {value}")

    with torch.no_grad():
        iterations = args.iterations  # 迭代次数
        noise_scale = args.noise_scale  # 噪声缩放因子

        update_interval = 500 if iterations > 1000 else iterations // 2

        for j in range(1, iterations + 1):
            if j == 1:
                raw_backup = raw_model
                raw_model.load_state_dict(swa_model.module.state_dict())
                swa_model.update_parameters(raw_backup)
                swa_backup = raw_model
                logging.info(f"Backup SWA as raw & Accumulating SWA")

            # 噪声生成循环
            for name, param in raw_model.named_parameters():
                if 'norm_trunkfinal' in name or 'norm_intermediate_trunkfinal' in name:
                    continue

                # 确定参数组
                group_name = None
                if 'conv_spatial' in name:
                    group_name = 'conv_spatial'
                elif 'linear_global' in name:
                    group_name = 'linear_global'
                elif 'norm.' in name:
                    if 'beta' in name:
                        group_name = 'norm_beta'
                    elif 'gamma' in name:
                        group_name = 'norm_gamma'
                elif 'blocks' in name and 'weight' in name:
                    group_name = 'blocks'
                elif 'policy_head' in name:
                    if 'intermediate' in name:
                        group_name = 'intermediate_policy'
                    else:
                        group_name = 'policy_head'
                elif 'value_head' in name:
                    if 'intermediate' in name:
                        group_name = 'intermediate_value'
                    else:
                        group_name = 'value_head'

                if group_name and group_name in ratio_data['abs_means']:
                    noise = generate_noise(param, group_name, noise_scale)
                    param.data.add_(noise)

            if j % update_interval == 0:
                # swa_backup_flag = False
                swa_model.update_parameters(raw_model)
                swa_model.update_parameters(swa_backup)
                raw_model.load_state_dict(swa_model.module.state_dict())
                logging.info(f"Accumulating/Restoring SWA & Replace raw")
            
            # if (j % (update_interval * 2) == 0 or j == iterations) and not swa_backup_flag:
            #     swa_backup_flag = True
            #     swa_model.update_parameters(swa_backup)
            #     raw_model.load_state_dict(swa_model.module.state_dict())
            #     logging.info(f"Restoring SWA & Replace raw")

            if j % 100 == 0:
                logging.info(f"Iteration {j} completed")

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
    args.export_prefix += f"-{args.noise_scale}-{iterations}iters"

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
        save(save_path_tmp, train_path, model, swa_model, optimizer, train_state, metrics, model_config)
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

    logging.info(f"Added noise with scale {args.noise_scale} to the model")
    logging.info(f"Model name: {model_name}")

if __name__ == '__main__':
    main()