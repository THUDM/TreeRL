import random
from vllm import LLM
from transformers import AutoTokenizer
try:
    from openrlhf.trainer.ppo_utils.entropy_chain_local_manager import EntropyGuidedChainLocalManager
except:
    from entropy_chain_local_manager import EntropyGuidedChainLocalManager

import math
import os
import json
from multiprocessing import Process
from filelock import FileLock

from IPython import embed
from tqdm import tqdm

# tokenizer = AutoTokenizer.from_pretrained(
#     "/workspace/reason_data/checkpoint/glm-o1-2w-sft",
#     trust_remote_code=True
# )


def select_paths_with_ratio(paths, num_traces=32):
    # Shuffle the paths to ensure random selection order
    random.shuffle(paths)

    selected_paths = []
    remaining_paths = []

    # Traverse the shuffled paths and select the first pass_ratio == 1 path
    for path in paths:
        if path[-1]["pass_ratio"] == 1 and len(selected_paths) == 0:
            selected_paths.append(path)
        else:
            remaining_paths.append(path)

    # Calculate how many additional paths we need
    remaining_num_traces = num_traces - len(selected_paths)

    # Randomly select remaining_num_traces paths from the remaining_paths if possible
    if remaining_num_traces > 0:
        selected_paths.extend(random.sample(remaining_paths, min(
            remaining_num_traces, len(remaining_paths))))

    # Shuffle the selected paths to ensure they are returned in random order
    random.shuffle(selected_paths)
    assert len(
        selected_paths) == num_traces, f"len(selected_paths) = {len(selected_paths)} != num_traces = {num_traces}"

    return selected_paths


def normalize_selected_terminals(paths):
    leaf_orm_value = [path[-1]["value"] for path in paths]
    _sum = sum(leaf_orm_value)
    num = len(leaf_orm_value) - 1
    if num == 0:
        return paths
    else:
        mean = [(_sum - leaf_orm_value[i]) /
                num for i in range(len(leaf_orm_value))]
        orm_normalized = [leaf_orm_value[i] - mean[i]
                          for i in range(len(leaf_orm_value))]
        for i in range(len(orm_normalized)):
            paths[i][-1]["value"] = orm_normalized[i]
        return paths


def parallel_entropy_guided_tree(
    item,
    llm,
    # tokenizer,
    args=None,
    tokenize_fn=None,
    decode_fn=None,
    system_prompt=None,
):
    manager = EntropyGuidedChainLocalManager(
        args=args,
        llm=llm,
        encode_fn=tokenize_fn,
        decode_fn=decode_fn,
        evaluator_urls=args['evaluator_urls'],
        extractor_urls=args['extractor_urls'],
        eos_tokens_set=args['eos_tokens'],
    )

    result = manager.process_single_item(item, args)
    paths = result["paths"]
    return paths


def process_single_data_for_each_gpu(data_batch, gpu_id, tokenizer_path, evaluator_urls, extractor_urls, eos_tokens, output_file, tokenize_fn):
    '''
    仅用作评测本地 vllm 推理性能，不进入 RL 训练
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(
        model=tokenizer_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        seed=3407
    )

    for data in tqdm(data_batch, desc=f"GPU {gpu_id} progress"):
        item = {
            "problem": data["Question"],
            "golden_answer": data["Answer"],
            # "difficulty": data["difficulty"]
        }
        args = {
            "temperature": 1.2,
            "top_p": 0.9,
            "m": 8,
            "n": 4,
            "l": 2,
            "t": 1,
            "evaluator_urls": evaluator_urls,
            "extractor_urls": extractor_urls,
            "eos_tokens": eos_tokens,
            "use_pure_binary": False,
            "entropy_rm_urls": ["http://172.18.73.102:8000/v1"],
            "num_traces": 32,
            "use_pure_RM" : True,
            "use_orm_reward" : False,
            "use_chain_reward" : False,
            "step_level_norm" : False,
            "use_state_value_reward" : False,
            "use_value_only" : True,
            "balance_ratio": 0.2,
            "average_one_generation" : False,
            "advantage_mix_allancestor" : False,
            "use_weighted_value": False,
            "use_all_terminals": False,
            "a": 0.5,
            "b": 0.5,
        }

        manager = EntropyGuidedChainLocalManager(
            args=args,
            llm=llm,
            encode_fn=tokenize_fn,
            decode_fn=decode_fn,
            evaluator_urls=args['evaluator_urls'],
            extractor_urls=args['extractor_urls'],
            eos_tokens_set=args['eos_tokens'],
        )

        result = manager.process_single_item(item, args)

        # if output_file:
        #     lock = FileLock(f"{output_file}.lock")
        #     with lock:
        #         with open(output_file, "a", encoding="utf-8") as f:
        #             json.dump(result, f, ensure_ascii=False)
        #             f.write("\n")
        if output_file:
            with open(output_file, "a", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")


if __name__ == '__main__':
    # RL 调用参数
    # MODEL_PATH = "/data/o1-cloud/checkpoints/sft/glm_9b_1102"

    # MODEL_PATH = "/data/share/checkpoint/glm-o1-2w-sft"
    MODEL_PATH = "/data/o1-cloud/checkpoints/rl/qwen-14b-o1/epoch_3"
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    # def tokenize_fn(texts, max_length=2048, device="cpu", system_prompt=None):
    #     sample_input_ids = tokenizer.encode(
    #         "[gMASK]<sop><|user|>\n" + texts[0][0],
    #         add_special_tokens=False
    #     )
    #     sample_input_ids = sample_input_ids[-max_length:] + \
    #         tokenizer.encode("<|assistant|>\n", add_special_tokens=False)
    #     return sample_input_ids

    import torch

    def tokenize_fn(texts, max_length=2048, device="cpu", system_prompt=None):
        sample_input_ids = tokenizer.encode(
            "[gMASK]<sop><|user|>\n" + texts[0][0], add_special_tokens=False)
        sample_input_ids = sample_input_ids[-max_length:] + \
            tokenizer.encode("<|assistant|>\n", add_special_tokens=False)
        output = {"input_ids": [torch.tensor(sample_input_ids)]}
        return output

    def decode_fn(ids):
        return tokenizer.decode(ids, skip_special_tokens=False)

    # item = {
    #     "problem": "The graph of $$x^4=x^2 y^2$$ is a union of $$n$$ different lines. What is the value of $$n$$ ?",
    #     "golden_answer": "3"
    # }
    # llm = LLM(
    #     model=MODEL_PATH,
    #     tensor_parallel_size=1,
    #     trust_remote_code=True,
    #     seed=3407
    # )
    # args = {
    #     "temperature": 1.2,
    #     "top_p": 0.9,
    #     "m": 8,
    #     "n": 4,
    #     "l": 2,
    #     "evaluator_urls": ["http://172.18.75.109:8000/v1"],
    #     "extractor_urls": ["http://172.18.75.109:8000/v1"],
    #     # "eos_tokens": ["<|user|>", "<|endoftext|>", "<|observation|>"],
    #     "eos_tokens": [151329, 151336, 151338],
    #     "num_traces": 32,
    #     "entropy_rm_urls": ["http://172.18.73.102:8000/v1"],
    #     "use_pure_binary" : True,
    #     "use_pure_RM" : False,
    #     "use_orm_reward" : False,
    #     "use_chain_reward" : False,
    #     "step_level_norm" : True,
    #     "use_state_value_reward" :False,
    # }
    # paths = parallel_entropy_guided_tree(item, llm, args, tokenize_fn, decode_fn)
    # with open("/workspace/lurui/openrlhf-mcts/data/entropy_paths.jsonl","w",encoding="utf-8") as f:
    #     json.dump({"path":paths},f,ensure_ascii=False)

    # 以下是用于本地评测 omnimath-500 passrate 的代码
    eval_path = "/workspace/lurui/rm_simple_evals/data/math/MATH500.jsonl"
    output_file = "./res/output_8_4_2_2.jsonl"
    # tokenizer_path = "/data/share/checkpoint/glm-o1-2w-sft"
    tokenizer_path = MODEL_PATH
    evaluator_urls = ["http://172.18.74.194:8000/v1"]
    extractor_urls = ["http://172.18.75.153:8000/v1"]
    # eos_tokens = [151329, 151336, 151338]
    eos_tokens = ["<|user|>", "<|endoftext|>", "<|observation|>"]

    # Read input data
    with open(eval_path, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]

    # Number of GPUs
    num_gpus = 8

    # Split data across GPUs
    data_batches = [datas[i::num_gpus] for i in range(num_gpus)]

    # Create a process for each GPU
    processes = []
    for gpu_id, data_batch in enumerate(data_batches):
        p = Process(target=process_single_data_for_each_gpu, args=(
            data_batch, gpu_id, tokenizer_path, evaluator_urls, extractor_urls, eos_tokens, output_file, tokenize_fn))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()
