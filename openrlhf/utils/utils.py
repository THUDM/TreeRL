from functools import partial
from multiprocessing.pool import Pool
from multiprocessing.pool import ThreadPool
import os
from pathlib import Path
import random
import re
import time
from openai import OpenAI

from datasets import Dataset, interleave_datasets, load_dataset
import ray
import requests
import torch
import torch.distributed
from tqdm import tqdm
from transformers import AutoTokenizer
    
from transformers.trainer import get_scheduler
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

# from openai import OpenAI
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import os


from openrlhf.utils import DeepspeedStrategy


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_sp_tokens(args):
    sp_tokens = dict()
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        sp_token = getattr(args, key, None)
        if sp_token is not None:
            sp_tokens[key] = sp_token
    return sp_tokens


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    # sp_tokens = get_sp_tokens(strategy.args)
    # tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, **sp_tokens)

    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True)

    # tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    # if "chatglm" in pretrain:
        # tokenizer.eos_token_id = tokenizer.get_command("<|user|>")  
        
    if tokenizer.pad_token is None and "glm" not in pretrain:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_strategy(args):
    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
    data_cache_dir=None
):
    if probabilities is not None:
        if type(datasets) != list:
            datasets = datasets.split(",")
        probabilities = list(map(float, probabilities.split(",")))

        assert len(probabilities) == len(datasets)
        probabilities = [p / sum(probabilities) for p in probabilities]
    else:
        # datasets = datasets.strip()
        assert isinstance(datasets, list)
        _datasets = []
        _probabilities = []
        for dataset in datasets:
            dt, prob = dataset.strip().split(",")
            _datasets.append(dt)
            _probabilities.append(float(prob))
        datasets = _datasets
        probabilities = _probabilities
        probabilities = [p / sum(probabilities) for p in probabilities]

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        dataset_subfold_list = dataset.split("@")
        strategy.print(f"dataset: {dataset}")
        # local dir with python script or common local file
        if os.path.isdir(os.path.join(os.getcwd(), dataset)) or dataset.endswith(
            (".json", ".jsonl", ".csv", ".parquet", ".txt")
        ):
            if dataset.endswith((".json", ".jsonl", ".csv", ".parquet", ".txt")):
                files = dataset
                data_type = os.path.splitext(files)[1][1:]
            else:
                path = Path(dataset)
                script = [str(file.resolve()) for file in Path(path).rglob("*.py")]
                extensions = ("*.json", "*.jsonl", "*.csv", "*.parquet", "*.txt")
                files = [str(file) for ext in extensions for file in Path(path).rglob(ext)]
                strategy.print(f"script: {script}")
                strategy.print(f"files: {files}")
                # For dir, follow python script or first file type
                data_type = script[0] if len(script) == 1 else os.path.splitext(files[0])[1][1:]
            # reformat data type
            if data_type in ["json", "jsonl"]:
                data_type = "json"
            # elif data_type == "txt":
                # data_type = "text"
            # elif data_type.endswith(".py"):
                # load local dir with python script
                # files = None
            # if data_type.endswith(".py"):
                # strategy.print(f"load {dataset} with script {data_type}")
            # else:
            strategy.print(f"load {files} from {dataset}")
            data = load_dataset(data_type, data_files=files)
        elif len(dataset_subfold_list) == 2:
            dataset = dataset_subfold_list[0]
            subfold = dataset_subfold_list[1]
            data = load_dataset(dataset, data_dir=subfold.strip())
        elif len(dataset_subfold_list) == 1:
            print("load dataset in subfolder")
            dataset = dataset_subfold_list[0]
            if data_cache_dir is not None:
                data = load_dataset(dataset, cache_dir=data_cache_dir)
            else:
                data = load_dataset(dataset, cache_dir="/workspace/zhenyu/data/hfdata")
        else:
            raise Exception(f"Dataset Name {dataset}: Format error")

        if "train" in data:
            train_data_list.append(data["train"].select(range(min(max_count, len(data["train"])))))
        else:
            # if isinstance(data, list):
            #     random.shuffle(data)
            #     data = data[:min(max_count, len(data))]
            # else:
            train_data_list.append(data.select(range(min(max_count, len(data)))))  # train will contains eval? TODO

        if return_eval:
            if "test" in data:
                eval_data = data["test"].select(range(min(int(max_count * 0.1), len(data["test"]))))
            elif "validation" in data:
                eval_data = data["validation"].select(range(min(int(max_count * 0.1), len(data["validation"]))))
            elif "train" in data:
                eval_data = data["train"].select(range(min(int(max_count * 0.1), int(len(data["train"]) * 0.01))))
            else:
                eval_data = data.select(range(min(int(max_count * 0.1), int(len(data) * 0.001))))
            eval_data_list.append(eval_data)

    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset


import json

def convert_hh_to_json(dataset_name, cache_dir="hfdata"):
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    train = [x for x in dataset["train"]]
    output_dir = f"{cache_dir}_json/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    open(f"{output_dir}/train.jsonl", "w").writelines([json.dumps(x, ensure_ascii=False) + "\n" for x in train])
    if "test" in dataset:
        test = [x for x in dataset["test"]]
        open(f"{output_dir}/test.jsonl", "w").writelines([json.dumps(x, ensure_ascii=False) + "\n" for x in test])
    

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr: float = 0.0
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))



def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1, min_lr: float = 0.0,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr=min_lr
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# and considering the give prolem to solve

# Problem: {question}

EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

    Expression 1: C
    Expression 2: C: (2,2)

Yes
(for multiple-choice question, matching the option or the value is both correct)

    Expression 1: C: (2,2)
    Expression 2: C

Yes
(for multiple-choice question, matching the option or the value is both correct)
    Expression 1: 3
    Expression 2: a = 3

Yes

    Expression 1: B
    Expression 2: $\boxed{B}$

Yes

    Expression 1: 500/3π
    Expression 2: \\frac{500}{3}\\pi
    
Yes
(\\frac{a}{b} denotes fractions a/b, and \\pi equals to π)

    Expression 1: 64
    Expression 2: \\text{[number]}

No

    Expression 1: 64
    Expression 2: \\text{[correct answer]}

No

    Expression 1: x = 2
    Expression 2: 2

Yes
---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s

""".strip()



EXTRACTION_TEMPLATE = """
Look at the following math problem and extract the final answer, such final results or option. If you cannot find an answer, return `No Answer`
## Question: 
{question}

## Answer: 
{answer}

Put the answer in the format of the following example: 

<ANSWER>: <your answer>

Example:
<ANSWER>: A
<ANSWER>: A: 130
<ANSWER>: a = 3
<ANSWER>: 100
<ANSWER>: No Answer

If the question is a multiple-choice question, extract the option and value that matches the answer. Show the answer directly.

Your response:
"""


def check_equality(urls, expr1: str, expr2: str, question: str):
    if expr1 == expr2:
        return True
    
    # prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2, "question": question}
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}
    response = query_chatglm_platform(prompt=prompt, urls=urls)
    if isinstance(response, str):
        return response.lower().strip() == "yes"
    else:
        return False


def check_result(urls, item):
    question, response, label = item
    original_response = response

    if "<answer>" in response and "</answer>" in response:
        resp_text = re.findall(r"<answer>([\s\S]+?)</answer>", response)
        if len(resp_text) > 0:
            resp_text = resp_text[-1].strip()
        else:
            resp_text = "\n".join(response.strip().split("\n")[-3:])
            resp_text = resp_text.strip()
    else:
        response = response.strip().split("\n")
        resp_text = [x for x in response if x.strip()]
        resp_text = "\n".join(resp_text[-3:])    

    # response = response.strip().split("\n")
    # resp_text = [x for x in response if x.strip()]
    # resp_text = "\n".join(resp_text[-3:])

    # resp_text = resp_text[-1]
    answer = None
    if "\\box" in resp_text or "\\boxed" in resp_text:
        # extract value in \box
        answer = re.findall(r'\\box\{([^{}]*(?:\{[^{}]*\})*[^{}]*)\}', resp_text)
        if len(answer) == 0:
            answer = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\})*[^{}]*)\}', resp_text)

    # if answer is None and "<answer>" in original_response:
    #     answer = re.findall(r"<answer>([\s\S]+?)</answer>", original_response)

    if answer: 
        answer = answer[0].strip()
    else:
        answer_template = EXTRACTION_TEMPLATE.format(question=question, answer=resp_text)
        extracted_answer = query_chatglm_platform(answer_template, urls=urls)
        # answer = extracted_answer[10:]
        # answer = answer.replace("<ANSWER>: ", "").strip()
        if extracted_answer is None:
            answer = ""
        else:
            # answer = extracted_answer[10:]
            answer = extracted_answer.replace("<ANSWER>: ", "").strip()
    check = check_equality(urls, answer, label, question)

    return answer, 1 if check else 0


def query_chatglm_platform(
    prompt, 
    history=[], 
    do_sample=True, 
    max_tokens=256, 
    urls=None
):
    # url = "http://xxx/v1/chat/completions"

    if isinstance(urls, str):
        urls = [urls]
        
    messages = []
    for turn in history:
        messages.append({
            "role": "user",
            "content": turn["prompt"],
        })
        messages.append({
            "role": "assistant",
            "content": turn["response"],
        })
    messages.append({
        "role": "user",
        "content": prompt,
    })

    temperature = 0.4
    top_p = 0.1
    payload = {
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        # "model": self.model_version,
        "max_tokens": max_tokens,
        "do_sample": do_sample,
        "stream": False,
        "seed": random.randint(1, 10000000),
    }

    # response = requests.post(self.url, data=payload, headers=self.headers, verify=False)
    answer = None
    
    # if ":8000" in urls[0]:
    #     client = OpenAI(
    #         base_url=random.choice(urls), 
    #         api_key="test",
    #     )
    # else:
    #     client = None
        
    for _ in range(4):
        url = random.choice(urls)
        if ":8080" in url:
            answer = query_chatglm_tgi(prompt, history, do_sample=True, url=url, temperature=temperature, top_p=top_p, max_tokens=max_tokens, max_retry=1)
        elif ":8000" in url:
            answer = query_vllm_platform(prompt, history, do_sample=True, url=url, max_tokens=max_tokens, max_retry=1)
        else:
            try:
                response = requests.post(url, json=payload, verify=False)
            except Exception as e:
                print(e)
                continue
            
            if response.status_code == 200:
                answer = json.loads(response.text)
                # print(answer)
                # if answer["choices"][0]["finish_reason"] != "eos_token":
                    # answer = None
                # else:
                answer = answer["choices"][0]["message"]["content"]
            else:
                print(f"******* try to send request to {url}, but got {response.text}")
                answer = None
                
        if answer is not None:
            break

    return answer


# def query_vllm_platform(client, prompt, history=[], do_sample=True, max_tokens=4096, max_retry=2):
    # # if url is None:
    #     # url = "http://172.18.64.19:9090/v1/chat/completions"
    # messages = []
    # for turn in history:
    #     messages.append({
    #         "role": "user",
    #         "content": turn["prompt"],
    #     })
    #     messages.append({
    #         "role": "assistant",
    #         "content": turn["response"],
    #     })
    # messages.append({
    #     "role": "user",
    #     "content": prompt,
    # })

    # for _ in range(max_retry):
    #     try:
    #         completion = client.chat.completions.create(
    #             model="Meta-Llama-3.1-70B-Instruct",
    #             messages=messages,
    #             max_tokens=max_tokens,
    #         )
    #         if completion.choices:
    #             answer = completion.choices[0].message.content
    #             break
    #         else:
    #             continue
    #     except Exception as e:
    #         print(f"error in vllm requests: {e}")
    #         # exit(0)
    #         time.sleep(1)
    #         continue
    # else:
    #     answer = None

    # return answer


def query_vllm_platform(prompt, history=[], do_sample=True, url=None, max_tokens=4096, max_retry=2):
    if url is None:
        url = "http://172.20.65.211:8000/v1"
    url = url + "/chat/completions"
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    # 这里 model 名称随便填都行
    request_data = {
        "model": "/workspace/yujiang/models/Meta-Llama-3.1-70B-Instruct",
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    for _ in range(max_retry):
        try:
            response = requests.post(
                url,
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                resp_json = response.json()
                content = resp_json['choices'][0]['message']['content'].strip()
                # print(f"vllm response: {content}")
                return content
            else:
                print(
                    f"Failed to fetch response: {response.status_code}, {response.text}"
                )
        except Exception as e:
            print(f"error in vllm, exception: {e}, url={url}")
            
    return None
query_vllm_platform("Are you ready?", [], do_sample=True, url="http://172.18.65.17:8000/v1", max_tokens=512, max_retry=1)


def query_chatglm_tgi(prompt, history=[], do_sample=False, max_tokens=256, max_retry=3, url=None, temperature=0.4, top_p=0.1):
    messages = ""
    for turn in history:
        ques, ans = turn["prompt"], turn["response"]
        messages += f"<|user|>\n{ques}<|assistant|>\n{ans}"

    messages += f"<|user|>\n{prompt}<|assistant|>\n"
    inputs = {
        "inputs": messages,
        "stream": False,
        "parameters": {
            "best_of": 1,
            "decoder_input_details": False,
            "details": False,
            "do_sample": do_sample,
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            "seed": None,
            # "temperature": temperature,
            # "top_p": top_p,
            "stop": ["<|endoftext|>", "<|user|>", "<|observation|>"]
        }
    }
   
    for _ in range(max_retry):
        try:
            output = requests.post(url, json=inputs)
            if output.status_code == 200:
                output = json.loads(output.text)
                # results.append(output[0]["generated_text"])
                result = output["generated_text"]
                break
            else:
                print(output.text)   
        except:
            print("error in tgi requests")
            continue
    else:
        result = None

    return result


def map_with_progress(f: callable, xs: list[Any], num_threads: int = 50):
    """
    Apply f to each element of xs, using a ThreadPool, and show progress.
    """
    # if os.getenv("debug"):
        # return list(map(f, xs, total=len(xs)))
    # else:
    if len(xs) == 0:
        return []
    
    # start = time.time()
    with ThreadPool(min(num_threads, len(xs))) as pool:
    # with Pool(min(num_threads, len(xs))) as pool:
        return list(pool.imap(f, xs))


# class GLM4Sampler(object):
#     """
#     Sample from TGI's completion API
#     """

#     def __init__(
#         self,
#         url: str = "https://api.chatglm.dev:8443/v1",
#         model: str = "glm-4-public",
#         api_key: str = 'None',
#         system_message: Optional[str] = None,
#         temperature: float = 0.1,
#         max_tokens: int = 1024,
#     ):
#         self.system_message = system_message
#         self.temperature = temperature
#         self.max_tokens = max_tokens
#         self.url = url
#         # self.url = "http://172.18.64.38:9090/v1"

#         self.model = model
#         os.environ["OPENAI_API_KEY"] = api_key if api_key else "test"
        
#         self.client = OpenAI(base_url=self.url, timeout=360)

#     def get_resp(self, message_list):
#         for i in range(5):
#             try:
#                 stream = self.client.chat.completions.create(
#                     messages=message_list,
#                     model=self.model,
#                     temperature=self.temperature,
#                     top_p=0.1,
#                     stream=True,
#                     max_tokens=self.max_tokens
#                 )
#                 output = ''
#                 for part in stream:
#                     output += part.choices[0].delta.content
#                 return output
#             except Exception as e:
#                 print(e)
#                 continue
#         print("failed")
#         return ''

#     def __call__(self, message_list) -> str:
#         if self.system_message:
#             message_list = [{"role": "system", "content": self.system_message}] + message_list
#         return self.get_resp(message_list)




def split_text_by_tags(text):
    """
    Splits the input text into segments, where each segment starts with
    <understand>, <action>, or <reflection> and includes everything up to
    the next tag or the end of the text, keeping the tags and content together.

    Args:
        text (str): The input text to split.

    Returns:
        List[str]: A list of segments, each including the tags and the enclosed content.
    """
    # Define the pattern to match the tags and their content
    pattern = r'(<(understand|action|reflection)>.*?(?:</\2>)?)(?=(<(understand|action|reflection)>|$))'

    # Use re.findall to find all matches, including the tags and their content
    matches = re.findall(pattern, text, re.DOTALL)

    # Extract the full match from each tuple in matches
    segments = [match[0] for match in matches]

    return segments

import random
from openai import OpenAI
def apply_chat_template_qwen(system_prompt, user, assistant):
    return f"<|im_start|>system\n{system_prompt}.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>\n"


def get_qwen_remote_reward_model_value(urls, question, response):
    # global count
    # count +=1
    # print(count)
    url = random.choice(urls)
    # print(url)

    client = OpenAI(
        api_key="EMPTY",
        base_url=url,
    )

    system_prompt = "Please reason step by step."
    # if len(question) + len(response) > 4096:
    #     response = response[:4096 - len(question)]

    conversation_str = apply_chat_template_qwen(system_prompt, question, response)
    # print(conversation_str)

    for _ in range(3):
        try:
        # if True:
            responses = client.embeddings.create(
                input=[conversation_str],
                model="Qwen72BRM",
            )

            for data in responses.data:
                # print("qwen rm data", float(data.embedding[-1]))
                return float(data.embedding[-1])
        except Exception as e:
            print(e)
            print("-- error in rm requests", url)
            continue
    return 0