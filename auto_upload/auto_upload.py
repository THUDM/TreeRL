      
import os
import time
import json
import argparse
import wandb


def upload_results_to_wandb(
    base_path,
    project_name,
    run_name,
):
    
    data_eval_res = {}

    wandb.init(
        project=project_name,
        name=run_name,  # 主运行名
        group=run_name,  # 分组到主运行名
        resume="allow",  # 自动覆盖现有运行
        id=run_name
    )
    
    # 跟踪已经上传的 step
    uploaded_steps = set()
    
    print(os.listdir(base_path))
    
    base_path_ls = [item for item in list(os.listdir(base_path)) if item.startswith("_actor")]

    for step_dir in sorted(base_path_ls, key=lambda x: int(x.split('step')[-1])):
        try:
            step = int(step_dir.split('step')[-1])
        except ValueError:
            continue

        if step in uploaded_steps:
            continue

        step_path = os.path.join(base_path, step_dir, "simple_evals")
        if os.path.isdir(step_path):
            for json_file in os.listdir(step_path):
                # if json_file.endswith('average.json'):
                if json_file.endswith('json'):
                    dataset_name = os.path.splitext(json_file)[0]
                    results_path = os.path.join(step_path, json_file)
                    if os.path.exists(results_path):
                        with open(results_path, 'r') as f:
                            results = json.load(f)
                        
                        if dataset_name not in data_eval_res:
                            data_eval_res[dataset_name] = {}

                        pass_metrics = {
                            f"{dataset_name}/{key}": value for key, value in results.items() if value is not None
                        }

                        pass_metrics['trainer/global_step'] = step
                        wandb.log(
                            {**pass_metrics, "section": dataset_name}, commit=True
                        )
                        uploaded_steps.add(step)
                        print(f"上传 step {step} 的结果: {pass_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="上传评测结果到 wandb")
    parser.add_argument("--base_path", type=str,
                        required=True, help="包含 step 结果的基路径")
    parser.add_argument("--project_name", type=str,
                        required=True, help="wandb 项目名称")
    parser.add_argument("--run_name", type=str,
                        required=True, help="wandb 运行名称")

    args = parser.parse_args()

    upload_results_to_wandb(
        base_path=args.base_path,
        project_name=args.project_name,
        run_name=args.run_name,
    )

    