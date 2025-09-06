#!/usr/bin/env python3
"""
详细比较TODE和TODE-TIR模型的参数差异
"""

import torch
import torch.nn as nn
from models.Tode import Tode
from models.TodeTIR import TodeTIR
from models.swin_transformer import SwinTransformer

def count_parameters(model):
    """
    统计模型参数总量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_number(num):
    """
    格式化数字显示
    """
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

def get_module_params(module):
    """
    获取模块的参数统计
    """
    params = {}
    for name, child in module.named_modules():
        if len(list(child.children())) == 0:  # 叶子节点
            total = sum(p.numel() for p in child.parameters())
            trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
            if total > 0:
                params[name] = {'total': total, 'trainable': trainable}
    return params

def compare_models():
    """
    详细比较TODE和TODE-TIR模型的参数差异
    """
    print("=" * 80)
    print("TODE vs TODE-TIR 模型参数详细比较")
    print("=" * 80)
    
    # 创建模型
    print("\n正在创建模型...")
    tode_model = Tode.build(lambda_val=1, res=True)
    todetir_model = TodeTIR.build(lambda_val=1, res=True)
    
    # 获取各模块参数
    tode_params = get_module_params(tode_model)
    todetir_params = get_module_params(todetir_model)
    
    # 统计总体参数
    tode_total, tode_trainable = count_parameters(tode_model)
    todetir_total, todetir_trainable = count_parameters(todetir_model)
    
    print(f"\n总体参数统计:")
    print(f"TODE模型: {format_number(tode_total)} 参数 ({tode_total:,})")
    print(f"TODE-TIR模型: {format_number(todetir_total)} 参数 ({todetir_total:,})")
    print(f"参数增加量: {format_number(todetir_total - tode_total)} 参数 ({todetir_total - tode_total:,})")
    print(f"增加比例: {((todetir_total - tode_total) / tode_total * 100):.2f}%")
    
    # 详细比较各模块
    print(f"\n{'='*80}")
    print("各模块参数详细比较")
    print(f"{'='*80}")
    
    # 获取所有模块名称
    all_modules = set(tode_params.keys()) | set(todetir_params.keys())
    
    # 按模块类型分组
    encoder_modules = [name for name in all_modules if 'encoder' in name]
    decoder_modules = [name for name in all_modules if 'decoder' in name or 'up' in name or 'se' in name]
    final_modules = [name for name in all_modules if 'final' in name]
    other_modules = [name for name in all_modules if name not in encoder_modules + decoder_modules + final_modules]
    
    # 比较编码器模块
    print(f"\n1. 编码器模块 (Swin Transformer) 参数比较:")
    print(f"{'-'*60}")
    encoder_diff = 0
    for name in sorted(encoder_modules):
        tode_count = tode_params.get(name, {}).get('total', 0)
        todetir_count = todetir_params.get(name, {}).get('total', 0)
        diff = todetir_count - tode_count
        encoder_diff += diff
        
        if diff != 0:
            print(f"  {name}:")
            print(f"    TODE: {format_number(tode_count)} ({tode_count:,})")
            print(f"    TODE-TIR: {format_number(todetir_count)} ({todetir_count:,})")
            print(f"    差异: {format_number(diff)} ({diff:,})")
            print(f"    增加比例: {(diff/tode_count*100):.2f}%" if tode_count > 0 else "    新增模块")
            print()
    
    print(f"编码器总差异: {format_number(encoder_diff)} ({encoder_diff:,})")
    
    # 比较解码器模块
    print(f"\n2. 解码器模块参数比较:")
    print(f"{'-'*60}")
    decoder_diff = 0
    for name in sorted(decoder_modules):
        tode_count = tode_params.get(name, {}).get('total', 0)
        todetir_count = todetir_params.get(name, {}).get('total', 0)
        diff = todetir_count - tode_count
        decoder_diff += diff
        
        if diff != 0:
            print(f"  {name}:")
            print(f"    TODE: {format_number(tode_count)} ({tode_count:,})")
            print(f"    TODE-TIR: {format_number(todetir_count)} ({todetir_count:,})")
            print(f"    差异: {format_number(diff)} ({diff:,})")
            print()
        elif tode_count > 0:  # 相同参数量的模块
            print(f"  {name}: 相同参数 {format_number(tode_count)} ({tode_count:,})")
    
    print(f"解码器总差异: {format_number(decoder_diff)} ({decoder_diff:,})")
    
    # 比较最终输出模块
    print(f"\n3. 最终输出模块参数比较:")
    print(f"{'-'*60}")
    final_diff = 0
    for name in sorted(final_modules):
        tode_count = tode_params.get(name, {}).get('total', 0)
        todetir_count = todetir_params.get(name, {}).get('total', 0)
        diff = todetir_count - tode_count
        final_diff += diff
        
        if diff != 0:
            print(f"  {name}:")
            print(f"    TODE: {format_number(tode_count)} ({tode_count:,})")
            print(f"    TODE-TIR: {format_number(todetir_count)} ({todetir_count:,})")
            print(f"    差异: {format_number(diff)} ({diff:,})")
            print()
        elif tode_count > 0:
            print(f"  {name}: 相同参数 {format_number(tode_count)} ({tode_count:,})")
    
    print(f"最终输出模块总差异: {format_number(final_diff)} ({final_diff:,})")
    
    # 其他模块
    if other_modules:
        print(f"\n4. 其他模块参数比较:")
        print(f"{'-'*60}")
        other_diff = 0
        for name in sorted(other_modules):
            tode_count = tode_params.get(name, {}).get('total', 0)
            todetir_count = todetir_params.get(name, {}).get('total', 0)
            diff = todetir_count - tode_count
            other_diff += diff
            
            if diff != 0:
                print(f"  {name}:")
                print(f"    TODE: {format_number(tode_count)} ({tode_count:,})")
                print(f"    TODE-TIR: {format_number(todetir_count)} ({todetir_count:,})")
                print(f"    差异: {format_number(diff)} ({diff:,})")
                print()
            elif tode_count > 0:
                print(f"  {name}: 相同参数 {format_number(tode_count)} ({tode_count:,})")
        
        print(f"其他模块总差异: {format_number(other_diff)} ({other_diff:,})")
    
    # 总结
    print(f"\n{'='*80}")
    print("参数差异总结")
    print(f"{'='*80}")
    print(f"编码器差异: {format_number(encoder_diff)} ({encoder_diff:,})")
    print(f"解码器差异: {format_number(decoder_diff)} ({decoder_diff:,})")
    print(f"最终输出模块差异: {format_number(final_diff)} ({final_diff:,})")
    if other_modules:
        print(f"其他模块差异: {format_number(other_diff)} ({other_diff:,})")
    
    total_diff = todetir_total - tode_total
    print(f"\n总差异: {format_number(total_diff)} ({total_diff:,})")
    print(f"差异验证: {total_diff == (encoder_diff + decoder_diff + final_diff + (other_diff if other_modules else 0))}")
    
    # 分析主要差异来源
    print(f"\n主要差异来源分析:")
    if encoder_diff > 0:
        print(f"- 编码器 (Swin Transformer): 输入通道从4增加到5，增加了 {format_number(encoder_diff)} 参数")
    if decoder_diff > 0:
        print(f"- 解码器: 增加了 {format_number(decoder_diff)} 参数")
    if final_diff > 0:
        print(f"- 最终输出层: 增加了 {format_number(final_diff)} 参数")
    
    # 清理内存
    del tode_model, todetir_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    compare_models() 