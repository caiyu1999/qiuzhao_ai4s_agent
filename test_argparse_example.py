#!/usr/bin/env python3
"""
使用argparse模块处理命令行参数的示例
"""

import argparse
import sys
import os

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="OpenEvolve Graph 程序参数配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python test_argparse_example.py --config config.yaml --iterations 100
  python test_argparse_example.py --resume --checkpoint ./checkpoint_10
  python test_argparse_example.py --debug --log-level DEBUG
        """
    )
    
    # 基本参数
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)"
    )
    
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=1000,
        help="最大迭代次数 (默认: 1000)"
    )
    
    # 布尔参数
    parser.add_argument(
        "--resume", 
        action="store_true",
        help="是否从检查点恢复"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="启用调试模式"
    )
    
    # 选择参数
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)"
    )
    
    # 检查点相关
    parser.add_argument(
        "--checkpoint", 
        type=str,
        help="检查点路径"
    )
    
    parser.add_argument(
        "--resume-iteration", 
        type=int,
        default=0,
        help="恢复的迭代次数 (默认: 0)"
    )
    
    # 岛屿配置
    parser.add_argument(
        "--num-islands", 
        type=int,
        default=4,
        help="岛屿数量 (默认: 4)"
    )
    
    # 文件路径
    parser.add_argument(
        "--init-program", 
        type=str,
        help="初始程序文件路径"
    )
    
    parser.add_argument(
        "--evaluator-file", 
        type=str,
        help="评估器文件路径"
    )
    
    # 列表参数
    parser.add_argument(
        "--meeting-intervals", 
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
        help="会议间隔列表 (默认: 2 4 8 16)"
    )
    
    return parser

def validate_args(args):
    """验证参数的有效性"""
    errors = []
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        errors.append(f"配置文件不存在: {args.config}")
    
    # 检查初始程序文件
    if args.init_program and not os.path.exists(args.init_program):
        errors.append(f"初始程序文件不存在: {args.init_program}")
    
    # 检查评估器文件
    if args.evaluator_file and not os.path.exists(args.evaluator_file):
        errors.append(f"评估器文件不存在: {args.evaluator_file}")
    
    # 检查检查点
    if args.resume and not args.checkpoint:
        errors.append("启用恢复模式时必须指定检查点路径")
    
    if args.checkpoint and not os.path.exists(args.checkpoint):
        errors.append(f"检查点路径不存在: {args.checkpoint}")
    
    # 检查数值范围
    if args.iterations <= 0:
        errors.append("迭代次数必须大于0")
    
    if args.num_islands <= 0:
        errors.append("岛屿数量必须大于0")
    
    if args.resume_iteration < 0:
        errors.append("恢复迭代次数不能为负数")
    
    if errors:
        print("参数验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 验证参数
    if not validate_args(args):
        sys.exit(1)
    
    # 打印参数信息
    print("=== 程序参数配置 ===")
    print(f"配置文件: {args.config}")
    print(f"最大迭代次数: {args.iterations}")
    print(f"岛屿数量: {args.num_islands}")
    print(f"日志级别: {args.log_level}")
    print(f"调试模式: {args.debug}")
    print(f"恢复模式: {args.resume}")
    
    if args.resume:
        print(f"检查点路径: {args.checkpoint}")
        print(f"恢复迭代次数: {args.resume_iteration}")
    
    if args.init_program:
        print(f"初始程序: {args.init_program}")
    
    if args.evaluator_file:
        print(f"评估器文件: {args.evaluator_file}")
    
    print(f"会议间隔: {args.meeting_intervals}")
    print()
    
    # 这里可以调用您的实际程序逻辑
    print("参数验证通过，开始执行程序...")
    
    # 示例：创建配置字典
    config_dict = {
        "config_file": args.config,
        "max_iterations": args.iterations,
        "num_islands": args.num_islands,
        "log_level": args.log_level,
        "debug": args.debug,
        "resume": args.resume,
        "checkpoint": args.checkpoint,
        "resume_iteration": args.resume_iteration,
        "init_program": args.init_program,
        "evaluator_file": args.evaluator_file,
        "meeting_intervals": args.meeting_intervals
    }
    
    print("配置字典:")
    for key, value in config_dict.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 