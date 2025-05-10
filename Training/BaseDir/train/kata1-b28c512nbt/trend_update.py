#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
from collections import defaultdict
import logging

def parse_data_block(block):
    """解析单个数据块"""
    
    sections = {}
    sections['abs_update_ratio'] = {}
    sections['update_ratio'] = {}
    
    # 找到所有 abs_update_ratio 部分，添加可选的偏度和峰度捕获
    abs_pattern = r'(conv_spatial|linear_global|norm_beta|norm_gamma|blocks|policy_head|value_head|intermediate_policy|intermediate_value) abs_update_ratio:.*?mean=([\d\.eE\-\+]+).*?var=([\d\.eE\-\+]+)(?:.*?skewness=([\d\.eE\-\+]+))?(?:.*?kurtosis=([\d\.eE\-\+]+))?.*?data length: (\d+)'
    for match in re.finditer(abs_pattern, block, re.DOTALL):
        groups = match.groups()
        module = groups[0]
        stats = {
            'mean': float(groups[1]),
            'var': float(groups[2]),
            'data_length': int(groups[-1])
        }
        
        # 添加偏度和峰度（如果存在）
        if len(groups) > 3 and groups[3] is not None:
            stats['skewness'] = float(groups[3])
        if len(groups) > 4 and groups[4] is not None:
            stats['kurtosis'] = float(groups[4])
            
        sections['abs_update_ratio'][module] = stats
    
    # 找到所有 update_ratio 部分
    update_pattern = r'(conv_spatial|linear_global|norm_beta|norm_gamma|blocks|policy_head|value_head|intermediate_policy|intermediate_value) update_ratio:.*?mean=([\d\.eE\-\+]+).*?var=([\d\.eE\-\+]+).*?data length: (\d+)'
    for match in re.finditer(update_pattern, block, re.DOTALL):
        module, mean, var, length = match.groups()
        sections['update_ratio'][module] = {
            'mean': float(mean),
            'var': float(var),
            'data_length': int(length)
        }
    
    return sections

def parse_ratios_file(file_path):
    """解析 ratioData 中的文件，提取所有数据块"""
    
    logging.info(f"开始解析文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    logging.info(f"文件内容长度: {len(content)} 字节")
    
    # 首先，检查文件是否包含必要的数据块标记
    if 'abs_update_ratio:' not in content or 'update_ratio:' not in content:
        logging.warning(f"文件 {file_path} 不包含必要的数据块标记")
        return []
    
    # 步骤1: 预处理 - 移除无关的监控指标行
    # 移除以pacc1:、p0loss、vloss等开头的行，这些是监控数据而非更新比率数据
    cleaned_content = re.sub(r'\n(?:pacc1:|p0loss|vloss|tdvloss|oloss|sloss|fploss|skloss|smloss|sbcdfloss|'
                           r'sbpdfloss|sdregloss|leadloss|vtimeloss|evstloss|esstloss|qwlloss|qscloss|loss|'
                           r'ptentr|ptsoftentr|sekiweightscale|norm_normal|gnorm_batch|exgnorm|pslr_batch|wdnormal|'
                           r'gnorm_cap|window_start|window_end|time_since_last_print|Accumulating SWA).*?\n', '\n\n\n', content)
    
    # 步骤2: 按照多个连续换行符分割清理后的内容
    raw_blocks = re.split(r'\n{3,}', cleaned_content)
    logging.info(f"预处理后按空行分割得到 {len(raw_blocks)} 个原始块")
    
    # 识别包含完整数据结构的块
    complete_blocks = []
    for i, block in enumerate(raw_blocks):
        # 清理块，移除前导和尾随空白
        block = block.strip()
        if not block:
            continue
            
        if 'abs_update_ratio:' in block and 'update_ratio:' in block:
            complete_blocks.append(block)
            logging.debug(f"找到完整数据块 #{i}")
        elif 'abs_update_ratio:' in block and i+1 < len(raw_blocks) and 'update_ratio:' in raw_blocks[i+1]:
            # 处理跨块的情况
            combined_block = block + "\n\n" + raw_blocks[i+1]
            complete_blocks.append(combined_block)
            logging.debug(f"合并块 #{i} 和 #{i+1} 为完整数据块")
    
    logging.info(f"找到 {len(complete_blocks)} 个包含完整数据结构的块")
    
    # 解析每个完整块
    data_blocks = []
    for i, block in enumerate(complete_blocks):
        logging.info(f"处理完整数据块 {i+1}/{len(complete_blocks)}")
        
        # 清理数据块：再次确保移除任何可能遗漏的pacc1行
        clean_block = re.sub(r'\npacc1:.*?\n', '\n', block)
        
        # 再次确认清理后的块包含必要的数据
        if 'abs_update_ratio:' in clean_block and 'update_ratio:' in clean_block:
            parsed_block = parse_data_block(clean_block)
            
            # 如果解析成功（至少有一个模块被找到）
            if (len(parsed_block['abs_update_ratio']) > 0 or 
                len(parsed_block['update_ratio']) > 0):
                data_blocks.append(parsed_block)
                logging.info(f"  成功解析数据块 {i+1}，找到 {len(parsed_block['abs_update_ratio'])} 个abs模块和 {len(parsed_block['update_ratio'])} 个update模块")
            else:
                logging.warning(f"  数据块 {i+1} 解析后不包含任何模块信息")
                logging.debug(f"  块内容片段: {clean_block[:200].replace('\n', '\\n')}...")
        else:
            logging.warning(f"  清理后的数据块 {i+1} 不包含必要的标记")
    
    logging.info(f"共解析出 {len(data_blocks)} 个有效数据块")
    return data_blocks

def get_latest_data_from_trends(trend_file_path):
    """从 ratios_trend.txt 获取最新的数据点"""
    
    try:
        with open(trend_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按空行分割数据块
        blocks = content.split('\n\n\n\n\n')
        
        # 获取最后一个有效数据块
        for block in reversed(blocks):
            if 'abs_update_ratio:' in block and 'update_ratio:' in block:
                return parse_data_block(block)
    except FileNotFoundError:
        logging.info(f"警告: {trend_file_path} 不存在，将创建新文件")
    
    # 如果没有找到有效数据或文件不存在，返回空结构
    return {'abs_update_ratio': {}, 'update_ratio': {}}

def compute_weighted_average(base_data, new_data):
    """计算基础数据和单个新数据的加权平均值"""
    
    result = {
        'abs_update_ratio': defaultdict(lambda: {'mean': 0.0, 'var': 0.0, 'skewness': 0.0, 'kurtosis': 0.0, 'data_length': 0}),
        'update_ratio': defaultdict(lambda: {'mean': 0.0, 'var': 0.0, 'data_length': 0})
    }
    
    # 初始化结果，从基础数据开始
    for ratio_type in ['abs_update_ratio', 'update_ratio']:
        for module, stats in base_data[ratio_type].items():
            result[ratio_type][module] = stats.copy()
    
    # 处理新数据
    for ratio_type in ['abs_update_ratio', 'update_ratio']:
        for module, new_stats in new_data[ratio_type].items():
            if module not in result[ratio_type]:
                # 如果是新模块，直接添加
                result[ratio_type][module] = new_stats.copy()
            else:
                # 已存在的模块，执行加权平均
                old_stats = result[ratio_type][module]
                old_length = old_stats['data_length']
                new_length = new_stats['data_length']
                total_length = old_length + new_length
                
                # 计算加权平均
                if total_length > 0:
                    weight_old = old_length / total_length
                    weight_new = new_length / total_length
                    
                    # 均值的加权平均
                    result[ratio_type][module]['mean'] = (
                        old_stats['mean'] * weight_old + 
                        new_stats['mean'] * weight_new
                    )
                    
                    # 方差的加权平均
                    result[ratio_type][module]['var'] = (
                        old_stats['var'] * weight_old + 
                        new_stats['var'] * weight_new + 
                        weight_old * weight_new * (old_stats['mean'] - new_stats['mean']) ** 2
                    )
                    
                    result[ratio_type][module]['data_length'] = total_length
                    
                    # 处理偏度 (skewness)
                    if 'skewness' in old_stats and 'skewness' in new_stats:
                        # 将偏度转换为三阶中心矩
                        old_third_moment = old_stats['skewness'] * (old_stats['var'] ** 1.5)
                        new_third_moment = new_stats['skewness'] * (new_stats['var'] ** 1.5)
                        
                        # 计算组合的三阶中心矩（需要考虑均值偏移）
                        delta = new_stats['mean'] - old_stats['mean']
                        combined_third_moment = (
                            weight_old * old_third_moment + 
                            weight_new * new_third_moment +
                            3 * weight_old * weight_new * delta * (
                                weight_new * old_stats['var'] - 
                                weight_old * new_stats['var']
                            ) +
                            weight_old * weight_new * (weight_old - weight_new) * (delta ** 3)
                        )
                        
                        # 使用组合方差将结果转换回偏度
                        result[ratio_type][module]['skewness'] = combined_third_moment / (result[ratio_type][module]['var'] ** 1.5)
                    elif 'skewness' in new_stats:
                        result[ratio_type][module]['skewness'] = new_stats['skewness']
                    # 如果只有旧数据有偏度，保留原值
                    
                    # 处理峰度 (kurtosis)
                    if 'kurtosis' in old_stats and 'kurtosis' in new_stats:
                        # 将峰度转换为四阶中心矩
                        old_fourth_moment = old_stats['kurtosis'] * (old_stats['var'] ** 2)
                        new_fourth_moment = new_stats['kurtosis'] * (new_stats['var'] ** 2)
                        
                        # 计算组合的四阶中心矩（需要考虑均值偏移）
                        delta = new_stats['mean'] - old_stats['mean']
                        combined_fourth_moment = (
                            weight_old * old_fourth_moment + 
                            weight_new * new_fourth_moment +
                            6 * weight_old * weight_new * (
                                weight_new * old_stats['var'] - 
                                weight_old * new_stats['var']
                            ) * (delta ** 2) +
                            4 * weight_old * weight_new * (
                                weight_new - weight_old
                            ) * (delta ** 3) +
                            weight_old * weight_new * (
                                weight_old ** 2 + weight_new ** 2
                            ) * (delta ** 4)
                        )
                        
                        # 使用组合方差将结果转换回峰度
                        result[ratio_type][module]['kurtosis'] = combined_fourth_moment / (result[ratio_type][module]['var'] ** 2)
                    elif 'kurtosis' in new_stats:
                        result[ratio_type][module]['kurtosis'] = new_stats['kurtosis']
                    # 如果只有旧数据有峰度，保留原值
    
    return result

def format_data_for_trend_file(data):
    """将数据格式化为 ratios_trend.txt 所需的格式"""
    
    output = []
    
    # 定义模块的自定义排序顺序
    module_order = [
        'conv_spatial', 
        'linear_global', 
        'norm_beta', 
        'norm_gamma', 
        'blocks', 
        'policy_head', 
        'value_head', 
        'intermediate_policy', 
        'intermediate_value'
    ]
    
    # 添加 abs_update_ratio 部分
    output.append("abs_update_ratio:")
    
    # 获取所有模块
    all_modules = set(data['abs_update_ratio'].keys())
    # 按自定义顺序排序模块
    # 首先处理已知的模块（按预定义顺序）
    for module in module_order:
        if module in data['abs_update_ratio']:
            stats = data['abs_update_ratio'][module]
            output.append(f"{module} abs_update_ratio: ")
            output.append(f"mean={stats['mean']}, ")
            output.append(f"var={stats['var']}, ")
            output.append(f"skewness={stats['skewness']}, ")
            output.append(f"kurtosis={stats['kurtosis']}, ")
            output.append(f"data length: {stats['data_length']}")
            all_modules.remove(module)
    
    # # 然后处理任何其他未知模块（按字母顺序）
    # for module in sorted(all_modules):
    #     stats = data['abs_update_ratio'][module]
    #     output.append(f"{module} abs_update_ratio: ")
    #     output.append(f"mean={stats['mean']}, ")
    #     output.append(f"var={stats['var']}, ")
    #     output.append(f"skewness={stats['skewness']}, ")
    #     output.append(f"kurtosis={stats['kurtosis']}, ")
    #     output.append(f"data length: {stats['data_length']}")
    
    # 添加 update_ratio 部分
    output.append("update_ratio:")
    
    # 获取所有模块
    all_modules = set(data['update_ratio'].keys())
    
    # 按自定义顺序排序模块
    # 首先处理已知的模块（按预定义顺序）
    for module in module_order:
        if module in data['update_ratio']:
            stats = data['update_ratio'][module]
            output.append(f"{module} update_ratio: ")
            output.append(f"mean={stats['mean']}, ")
            output.append(f"var={stats['var']}, ")
            output.append(f"data length: {stats['data_length']}")
            all_modules.remove(module)
    
    # # 然后处理任何其他未知模块（按字母顺序）
    # for module in sorted(all_modules):
    #     stats = data['update_ratio'][module]
    #     output.append(f"{module} update_ratio: ")
    #     output.append(f"mean={stats['mean']}, ")
    #     output.append(f"var={stats['var']}, ")
    #     output.append(f"data length: {stats['data_length']}")
    
    # 连接所有行并添加足够的空行分隔
    return "\n".join(output) + "\n\n\n\n\n"

def main():
    # 设置日志
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H-%M-%S%z:'  # 指定日期时间格式和时区
    )

    # 文件和目录路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    trend_file_path = os.path.join(base_dir, "ratios_trend.txt")
    ratio_data_dir = os.path.join(base_dir, "ratioData")
    
    # 确保 ratioData 目录存在
    if not os.path.exists(ratio_data_dir):
        os.makedirs(ratio_data_dir)
        logging.info(f"创建目录 {ratio_data_dir}")
    
    # 获取并排序 ratioData 目录中的所有 .txt 文件，按修改时间排序
    ratio_files = [
        os.path.join(ratio_data_dir, f) 
        for f in os.listdir(ratio_data_dir) 
        if f.endswith('.txt')
    ]
    ratio_files.sort(key=lambda f: os.path.getmtime(f))
    
    if not ratio_files:
        logging.info("没有找到新的比率数据文件，不需要更新")
        return
    
    logging.info(f"找到 {len(ratio_files)} 个新的比率数据文件，按时间顺序处理")
    
    # 处理已完成的文件列表
    processed_files = []
    total_blocks_processed = 0
    
    # 依次处理每个文件
    for file_path in ratio_files:
        file_name = os.path.basename(file_path)
        logging.info(f"处理文件: {file_name}")
        
        # 解析当前文件中的所有数据块
        data_blocks = parse_ratios_file(file_path)

        if not data_blocks:
            logging.warning(f"  文件 {file_name} 中没有找到有效的数据块")
            continue
        
        logging.info(f"  发现 {len(data_blocks)} 个数据块，逐个处理")
        
        # 依次处理每个数据块
        current_data = get_latest_data_from_trends(trend_file_path)
        for i, block_data in enumerate(data_blocks):
            logging.info(f"  处理第 {i+1}/{len(data_blocks)} 个数据块")

            # 检查 block_data 是否为空或无效
            if not block_data or not block_data.get('abs_update_ratio') or not block_data.get('update_ratio'):
                logging.warning(f"    跳过无效或空的数据块 {i+1}")
                continue
            
            # 计算加权平均
            new_data = compute_weighted_average(current_data, block_data)
            
            # 格式化新数据
            formatted_data = format_data_for_trend_file(new_data)
            
            # 将新数据附加到趋势文件
            with open(trend_file_path, 'a', encoding='utf-8') as f:
                f.write(formatted_data)
            
            total_blocks_processed += 1
        
        # 添加到已处理列表
        processed_files.append(file_path)
    
    # 处理完成后移动已处理的文件
    processed_dir = os.path.join(ratio_data_dir, "processed")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    for file_path in processed_files:
        filename = os.path.basename(file_path)
        os.rename(file_path, os.path.join(processed_dir, filename))
    
    logging.info(f"已处理 {len(processed_files)} 个文件，共 {total_blocks_processed} 个数据块")
    logging.info(f"已将处理过的文件移至 {processed_dir}")

if __name__ == "__main__":
    main()