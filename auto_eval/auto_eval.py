      
#!/usr/bin/env python3
import os
import time
import argparse
import subprocess
import json
import requests
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Automated model evaluation script')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory containing model checkpoints')
    parser.add_argument('--checkpoint_dirs', type=str, nargs='+',
                        help='Specific checkpoint directory names to evaluate under base_dir')
    parser.add_argument('--result_base_dir', type=str,
                        default='/workspace/lurui/glm-simple-evals-1007/glm-simple-evals/RL_auto_results',
                        help='Base directory for evaluation results')
    parser.add_argument('--eval_script', type=str,
                        default='/workspace/lurui/glm-simple-evals-1007/glm-simple-evals/test_math_mcts.sh',
                        help='Path to evaluation script')
    parser.add_argument('--base_script', type=str,
                        default='up_model.sh',
                        help='Base script for model loading')
    parser.add_argument('--api_url', type=str,
                        default='http://127.0.0.1:8007/v1/models',
                        help='API endpoint for model status check')
    return parser.parse_args()


def get_checkpoint_paths(base_dir, checkpoint_dirs=None):
    """获取指定目录下的所有checkpoint路径"""
    base_path = Path(base_dir)

    if not checkpoint_dirs:
        checkpoint_dirs = [d.name for d in base_path.iterdir() if d.is_dir()]

    print(f"Checking directories: {', '.join(checkpoint_dirs)}")

    all_checkpoints = []
    for dir_name in checkpoint_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists():
            checkpoints = list(dir_path.glob('**/*_actor_global_step*'))
            if checkpoints:
                print(f"Found {len(checkpoints)} checkpoints in {dir_name}")
                all_checkpoints.extend([str(p) for p in checkpoints])

    return all_checkpoints


def get_model_name(path):
    """从路径中提取模型名称"""
    parts = Path(path).parts
    return f"{parts[-2]}/{parts[-1]}"


def check_model_status(api_url):
    """检查模型服务状态"""
    try:
        response = requests.get(api_url)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def wait_for_model_ready(api_url, initial_wait=40, check_interval=30, max_retries=10):
    """等待模型服务就绪"""
    print(f"\nWaiting {initial_wait}s for model loading...")
    time.sleep(initial_wait)

    retries = 0
    while retries < max_retries:
        if check_model_status(api_url):
            print("Model service is ready")
            return True
        print(
            f"Model service not ready, waiting {check_interval}s... ({retries+1}/{max_retries})")
        time.sleep(check_interval)
        retries += 1

    print("Model service failed to become ready")
    return False


def check_result_exists(result_dir, model_name):
    """检查是否已有评测结果"""
    # result_path = Path(result_dir) / f"livecodebench_{model_name}" / "results.json"
    result_path1 = Path(result_dir) / f"{model_name}/simple_evals/omni-math.json"
    result_path2 = Path(result_dir) / f"{model_name}/simple_evals/math500.json"
    # print(f"Checking for existing result: {result_path}")
    # exit(1)
    return result_path1.exists() and result_path2.exists()


def run_script_non_blocking(cmd):
    """以非阻塞方式运行脚本"""
    try:
        # full_cmd = f"nohup {cmd} > /dev/null 2>&1 &"
        # full_cmd = f"nohup {cmd}"
        full_cmd = f"{cmd}"
        print(f"Running command: {full_cmd}")
        subprocess.run(full_cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def run_command(cmd, non_blocking=False, show_output=False):
    # print(cmd)
    # exit(1)
    # """运行shell命令并返回结果"""
    if non_blocking:
        return run_script_non_blocking(cmd)

    try:
        if show_output:
            # 直接显示命令输出
            print(f"Running command: {cmd}")
            process = subprocess.run(cmd, shell=True, check=True)
        else:
            print(f"Running command: {cmd}")
            # 隐藏命令输出
            process = subprocess.run(cmd, shell=True, check=True,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        print(f"Command failed: {cmd}")
        return False
    # print(f"Running command: {cmd}")
    # process = subprocess.run(cmd, shell=True, check=True)
    # return True


def cleanup_previous_services():
    """清理之前可能运行的服务"""
    try:
        cmd = "pkill -f -9 'import spawn_main'"
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        time.sleep(10)  # 等待进程完全终止
        cmd = "pkill -f 'vllm.entrypoints.openai.api_server'"
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        time.sleep(10)  # 等待进程完全终止
    except:
        pass


def main():
    args = parse_args()

    while True:
        checkpoint_paths = get_checkpoint_paths(
            args.base_dir, args.checkpoint_dirs
        )
        # if not checkpoint_paths:
        #     print("No checkpoints found.")
        #     break

        for checkpoint_path in sorted(checkpoint_paths):
            model_name = get_model_name(checkpoint_path)

            # 检查是否已经评测过
            if check_result_exists(args.result_base_dir, model_name):
                print(f"\nSkipping {model_name} (already evaluated)")
                continue

            print(f"\n{'='*50}")
            print(f"Processing: {model_name}")
            print(f"{'='*50}")

            # 清理之前的服务
            cleanup_previous_services()

            # 运行模型加载脚本（非阻塞方式）
            print("Starting model service...")
            load_cmd = f"bash {args.base_script} {checkpoint_path} {model_name}"
            if not run_command(load_cmd, non_blocking=True, show_output=True):
                print("Failed to start model service")
                continue

            # 等待模型服务就绪
            if not wait_for_model_ready(args.api_url):
                continue

            # 运行评测脚本，显示输出
            print("\nRunning evaluation...")
            eval_cmd = f"bash {args.eval_script}"
            if not run_command(eval_cmd, show_output=True):
                print("Evaluation failed")
                continue

            print(f"\nEvaluation completed for {model_name}")

        print("\nAll checkpoints processed. Checking for new checkpoints in 10s...")
        time.sleep(10)


if __name__ == "__main__":
    main()

    