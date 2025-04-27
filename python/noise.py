
# cd /g/Projects/KataGo-Noise/python

# python noise.py --base-dir "G:\Projects\KataGo-Noise\Training\BaseDir" --training-name "kata1-b28c512nbt" --model-kind "b28c512nbt" --noise-scale 5.0

# ./selfplay/export_model_for_selfplay.sh "noisy-5.0" "/g/Projects/KataGo-Noise/Training/BaseDir" "1"



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
import json
import traceback

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

def generate_noise(param, group_name, noise_scale, iterations = 1000, i = 0):
    # 1. 获取两种比率的均值和方差
    mu_abs = abs_update_ratio_means[group_name]
    sigma_abs = torch.sqrt(torch.tensor(abs_update_ratio_vars[group_name], device=param.device))
    
    mu_rel = update_ratio_means[group_name]
    mu_rel_t = torch.tensor(mu_rel, device=param.device)
    sigma_rel = torch.sqrt(torch.tensor(update_ratio_vars[group_name], device=param.device))
    
    # 2. 生成幅度噪声 - 使用真正的截断正态分布
    # 实现截断正态分布采样 (μ, σ, 下限=0, 上限=None)
    # 为提高效率，使用拒绝采样法
    # 生成标准正态分布样本
    z = torch.randn_like(param.data)
    # 调整 mu 和 sigma 避免生成过多负值
    adjusted_mu = mu_abs + sigma_abs * noise_scale 
    adjusted_sigma = sigma_abs * noise_scale
    
    # 计算截断边界对应的标准化值
    alpha = -adjusted_mu / (adjusted_sigma + 1e-10)  # 截断下限(0)对应的标准化值
    
    # 使用拒绝采样确保非负
    # 对于靠近边界的区域，不断重采样直到满足条件
    rejection_mask = (z < alpha)
    while rejection_mask.any():
        # 只对拒绝的位置重新采样
        z_new = torch.randn(rejection_mask.sum(), device=param.device)
        z.masked_scatter_(rejection_mask, z_new)
        # 更新拒绝掩码
        rejection_mask = (z < alpha)
    
    # 将标准正态转换为目标分布
    magnitude = adjusted_mu + adjusted_sigma * z
    
    # 3. 生成方向噪声 - 自适应稳定性阈值
    # 基于层类型动态调整稳定性阈值
    base_stability_threshold = 0.5
    if 'norm' in group_name:
        stability_threshold = base_stability_threshold * 0.8  # 规范化层更保守
    elif 'value_head' in group_name:
        stability_threshold = base_stability_threshold * 1.2  # 价值头更激进
    else:
        stability_threshold = base_stability_threshold
    
    direction_stability = mu_rel_t / (sigma_rel + 1e-10)
    
    if direction_stability > stability_threshold:
        # 方向相对稳定 - 完全从分布采样
        direction = torch.randn_like(param.data) * sigma_rel + mu_rel
    else:
        # 方向不稳定 - 混合使用符号信息和受限噪声
        # 计算稳定性比例，越接近阈值，越多地使用分布信息
        stability_ratio = direction_stability / stability_threshold
        # 受限噪声比例随稳定性增加
        noise_scale = 0.1 + 0.3 * stability_ratio
        sign = 1 if mu_rel >= 0 else -1
        
        # 混合均值和符号导向的噪声
        random_part = torch.randn_like(param.data) * sigma_rel * noise_scale
        direction = sign * (random_part + mu_rel_t)
    
    # 4. 合成噪声 - 将幅度与方向结合
    noise_sign = torch.sign(direction)
    noise = magnitude * param.data.abs() * noise_sign
    
    # 5. 应用动态衰减因子 - 随迭代递减
    # 使用指数衰减系数，在高迭代次数时减少噪声影响
    iteration_fraction = min(i / iterations, 1.0)  # 当前迭代在总迭代中的比例
    base_decay = 1.0  # 初始因子
    min_decay = 0.7   # 最低因子
    
    # 非线性递减，前期保持较高，后期迅速降低
    decay_exponent = 2.0
    decay_factor = base_decay - (base_decay - min_decay) * (iteration_fraction ** decay_exponent)
    
    # 为不同层类型应用不同的衰减策略
    if 'value_head' in group_name or 'policy_head' in group_name:
        # 输出层应用更强的衰减，保持更多原始信息
        decay_factor *= 0.9
    
    # 6. 返回最终噪声
    return noise * decay_factor

# 定义各参数组的 abs_update_ratio 和 update_ratio 的均值和方差
# mean = sum(v if isinstance(v, float) else v[0] for v in values) / len(values)
# var = sum((v - mean) ** 2 if isinstance(v, float) else (v[0] - mean) ** 2 for v in values) / len(values)

def read_latest_update_ratios(train_dir, num_lines_to_check=1200):
    """从 stdout.txt 文件中高效读取最新的 update ratio 数据，并与 ratios.txt 中的数据进行加权平均"""
    
    stdout_file_path = os.path.join(train_dir, "stdout.txt")
    ratios_file_path = os.path.join(train_dir, "ratios.txt")
    
    # 从 stdout.txt 读取数据
    abs_means = {}
    abs_vars = {}
    update_means = {}
    update_vars = {}
    data_lengths = {}  # 存储每个参数组的数据长度
    
    current_section = None
    current_key = None
    
    try:
        with open(stdout_file_path, 'r', encoding='utf-8') as f:
            # 移动到文件末尾
            f.seek(0, 2)
            position = f.tell()
            lines = []
            
            # 向前读取指定行数
            while position > 0 and len(lines) < num_lines_to_check:
                position -= 1
                f.seek(position)
                char = f.read(1)
                if char == '\n':
                    lines.append(f.readline().strip())
            
            # 反转行列表，按文件顺序解析
            lines.reverse()
            
            for line in lines:
                # 识别当前部分
                if line == "abs_update_ratio:":
                    current_section = "abs"
                    continue
                elif line == "update_ratio:":
                    current_section = "update"
                    continue
                elif line in ["weight:", "gradient:", "pacc1:"]:
                    current_section = None
                    continue
                
                if not current_section:
                    continue
                
                # 获取参数组名称
                if current_section == "abs" and " abs_update_ratio:" in line:
                    current_key = line.split(" abs_update_ratio:")[0]
                elif current_section == "update" and " update_ratio:" in line:
                    current_key = line.split(" update_ratio:")[0]
                
                # 提取数值
                if line.startswith("mean=") and current_key:
                    value = float(line.split("=")[1].strip().rstrip(","))
                    if current_section == "abs":
                        abs_means[current_key] = value
                    else:
                        update_means[current_key] = value
                
                if line.startswith("var=") and current_key:
                    value = float(line.split("=")[1].strip().rstrip(","))
                    if current_section == "abs":
                        abs_vars[current_key] = value
                    else:
                        update_vars[current_key] = value
                
                # 提取数据长度
                if line.startswith("data length:") and current_key:
                    length = int(line.split(":")[1].strip())
                    key_with_section = f"{current_section}_{current_key}"
                    data_lengths[key_with_section] = length
        
        # 尝试读取 ratios.txt 文件中的数据
        old_data = {}
        try:
            if os.path.exists(ratios_file_path):
                with open(ratios_file_path, 'r', encoding='utf-8') as f:
                    old_data = json.load(f)
        except Exception as e:
            print(f"Failed to read {ratios_file_path}: {e}")
            old_data = {}
        
        # 提取旧数据
        old_abs_means = old_data.get('abs_means', {})
        old_abs_vars = old_data.get('abs_vars', {})
        old_update_means = old_data.get('update_means', {})
        old_update_vars = old_data.get('update_vars', {})
        old_data_lengths = old_data.get('data_lengths', {})
        
        # 合并数据
        combined_abs_means = {}
        combined_abs_vars = {}
        combined_update_means = {}
        combined_update_vars = {}
        combined_data_lengths = old_data_lengths.copy()
        
        # 处理 abs 数据
        for key in set(abs_means.keys()) | set(old_abs_means.keys()):
            old_length = old_data_lengths.get(f"abs_{key}", 0)
            new_length = data_lengths.get(f"abs_{key}", 0)
            
            if key in abs_means and key in old_abs_means and old_length > 0 and new_length > 0:
                # 两个来源都有数据，做加权平均
                total_length = old_length + new_length
                weight_old = old_length / total_length
                weight_new = new_length / total_length

                # 调整权重
                weight_new += weight_old / 2
                weight_old -= weight_old / 2
                
                combined_abs_means[key] = old_abs_means[key] * weight_old + abs_means[key] * weight_new
                combined_abs_vars[key] = old_abs_vars[key] * weight_old + abs_vars[key] * weight_new
                combined_data_lengths[f"abs_{key}"] = total_length
            elif key in abs_means:
                # 只有新数据
                combined_abs_means[key] = abs_means[key]
                combined_abs_vars[key] = abs_vars[key]
                combined_data_lengths[f"abs_{key}"] = new_length
            else:
                # 只有旧数据
                combined_abs_means[key] = old_abs_means[key]
                combined_abs_vars[key] = old_abs_vars[key]
                # 数据长度保持不变

            if combined_abs_means[key] == old_abs_means[key] and combined_abs_vars[key] == old_abs_vars[key]:
                combined_data_lengths[f"abs_{key}"] = old_length
        
        # 处理 update 数据
        for key in set(update_means.keys()) | set(old_update_means.keys()):
            old_length = old_data_lengths.get(f"update_{key}", 0)
            new_length = data_lengths.get(f"update_{key}", 0)
            
            if key in update_means and key in old_update_means and old_length > 0 and new_length > 0:
                # 两个来源都有数据，做加权平均
                total_length = old_length + new_length
                weight_old = old_length / total_length
                weight_new = new_length / total_length

                # 调整权重
                weight_new += weight_old / 2
                weight_old -= weight_old / 2
                
                combined_update_means[key] = old_update_means[key] * weight_old + update_means[key] * weight_new
                combined_update_vars[key] = old_update_vars[key] * weight_old + update_vars[key] * weight_new
                combined_data_lengths[f"update_{key}"] = total_length
            elif key in update_means:
                # 只有新数据
                combined_update_means[key] = update_means[key]
                combined_update_vars[key] = update_vars[key]
                combined_data_lengths[f"update_{key}"] = new_length
            else:
                # 只有旧数据
                combined_update_means[key] = old_update_means[key]
                combined_update_vars[key] = old_update_vars[key]
                # 数据长度保持不变

            if combined_update_means[key] == old_update_means[key] and combined_update_vars[key] == old_update_vars[key]:
                combined_data_lengths[f"update_{key}"] = old_length
        
        # 保存合并后的数据到 ratios.txt
        combined_data = {
            'abs_means': combined_abs_means,
            'abs_vars': combined_abs_vars,
            'update_means': combined_update_means,
            'update_vars': combined_update_vars,
            'data_lengths': combined_data_lengths
        }
        
        try:
            with open(ratios_file_path, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=2)
            print(f"Successfully wrote combined data to {ratios_file_path}")
        except Exception as e:
            print(f"Failed to write combined data to {ratios_file_path}: {e}")
        
        # 返回合并后的数据
        return combined_abs_means, combined_abs_vars, combined_update_means, combined_update_vars
    
    except Exception as e:
        print(f"Error reading stdout.txt: {e}")
        traceback.print_exc()
        return {}, {}, {}, {}

abs_update_ratio_means = {
    'conv_spatial': 0.00016860119930313147,
    'linear_global': 0.00018781224798213123,
    'norm_beta': 1.9531230858301595e-05,
    'norm_gamma': 1.4373712142061939e-05,
    'blocks': 0.00011361872836163889,
    'policy_head': 8.902986346480544e-06,
    'value_head': 6.042755581183511e-05,
    'intermediate_policy': 1.2608511123735288e-05,
    'intermediate_value': 3.6676121624521245e-05,
}

abs_update_ratio_vars = {
    'conv_spatial': 2.4825361606404196e-09,
    'linear_global': 3.090347689371258e-09,
    'norm_beta': 7.176491128926009e-10,
    'norm_gamma': 3.126933457492505e-10,
    'blocks': 1.121679580364991e-09,
    'policy_head': 1.420171322525905e-10,
    'value_head': 3.008553955426278e-08,
    'intermediate_policy': 1.5907402191500531e-10,
    'intermediate_value': 5.646077578571462e-09,
}

update_ratio_means = {
    'conv_spatial': 4.422751798966035e-07,
    'linear_global': -1.193874497684425e-05,
    'norm_beta': -8.453670681838206e-09,
    'norm_gamma': 4.755875243357739e-09,
    'blocks': -3.4202291084586074e-06,
    'policy_head': -8.506532525786191e-10,
    'value_head': -0.0004893960702194334,
    'intermediate_policy': -1.8701952071983665e-07,
    'intermediate_value': 4.975334504663584e-05,
}

update_ratio_vars = {
    'conv_spatial': 2.185847385536089e-09,
    'linear_global': 2.8528927561190385e-07,
    'norm_beta': 5.201187664432502e-10,
    'norm_gamma': 4.903276613859002e-11,
    'blocks': 0.0030210504626944404,
    'policy_head': 1.077941427073591e-10,
    'value_head': 0.18421801176135222,
    'intermediate_policy': 7.340983634223755e-10,
    'intermediate_value': 2.107397687042172e-05,
}


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
        level=logging.INFO,
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
    logging.info(f"SWA model loaded: {swa_model is not None}")

    new_abs_means, new_abs_vars, new_update_means, new_update_vars = read_latest_update_ratios(train_path)
    if new_abs_means: abs_update_ratio_means.update(new_abs_means)
    if new_abs_vars: abs_update_ratio_vars.update(new_abs_vars)
    if new_update_means: update_ratio_means.update(new_update_means)
    if new_update_vars: update_ratio_vars.update(new_update_vars)
    logging.info(f"Loaded update ratios from file: {train_path}/stdout.txt")
    for key, value in new_abs_means.items():
        logging.info(f"abs_update_ratio_means[{key}] = {value}")
    for key, value in new_abs_vars.items():
        logging.info(f"abs_update_ratio_vars[{key}] = {value}")
    for key, value in new_update_means.items():
        logging.info(f"update_ratio_means[{key}] = {value}")
    for key, value in new_update_vars.items():
        logging.info(f"update_ratio_vars[{key}] = {value}")

    # 添加噪声，跳过某些层
    with torch.no_grad():
        # ~ 0.45s / round
        # 1000: 7.5mins
        # 10000: 1.25h
        # 50000: 6.25h
        # 100000: 12.5h
        iterations = args.iterations  # 迭代次数
        for i in range(iterations):
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

                # 改进的噪声生成方案
                if group_name and group_name in abs_update_ratio_means:
                    noise = generate_noise(param, group_name, args.noise_scale, iterations, i)
                    param.data.add_(noise)
                    
                # param.data.zero_()
                # logging.info(f"SWA Parameter {name} - mean: {abs_data_mean}, max: {abs_data.max().item()}, min: {abs_data.min().item()}")
                # logging.info(f"Added noise to parameter {name} with shape {param.shape}. --- noise scale: {noise_std} --- mean: {abs_data_mean}")
            
            logging.info(f"Accumulating SWA")
            swa_model.update_parameters(raw_model)
            if (i+1) % 10 == 0:
                logging.info(f"SWA iteration {i+1} completed")
            if (i+1) % 500 == 0:
                # Update raw_model with the current SWA averaged parameters
                raw_model.load_state_dict(swa_model.module.state_dict())
                logging.info(f"Updated raw_model with SWA parameters at iteration {i+1}")
                # for name, param in raw_model.named_parameters():
                #     logging.info(f"\nRaw parameter {name} - mean: {param.data.mean().item()}, max: {param.data.max().item()}, min: {param.data.min().item()} \nSWA parameter {name} - mean: {swa_model.module.state_dict()[name].mean().item()}, max: {swa_model.module.state_dict()[name].max().item()}, min: {swa_model.module.state_dict()[name].min().item()}")


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

    logging.info(f"Added noise with scale {args.noise_scale} to the model")
    logging.info(f"Model name: {model_name}")

if __name__ == '__main__':
    main()