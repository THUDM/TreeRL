import argparse
import time
import json
import random
import requests
from tqdm import tqdm
from functools import partial
from multiprocess import Pool, Queue, Process
import warnings
import openai
import re

random.seed(666)
warnings.filterwarnings("ignore")

NUM_PROCESS=150

QUEUE_SIZE=1000000

TEMPERATURE = 0.95
TOPP = 0.9
PROMPT_TEMPLATE = None


def query_chatglm(prompt, history=[]):
    def _build_prompt(prompt, history=[]):
        previous = ""
        for idx, r in enumerate(history):
            previous += f'##第 {idx + 1} 轮##\n\n问：{r["prompt"]}\n\n答：{r["response"]}\n\n'
        prompt = f"{previous}##第 {len(history) + 1} 轮##\n\n问：{prompt}\n\n答："
        return prompt

    url, header = "https://120.220.95.167:8443/v1/completions", "api-offline-32b.glm.ai"
    headers = {"Host": header}

    payload = json.dumps({
        "prompt": _build_prompt(prompt, history),
        "temperature": 0.95,
        "top_p": 0.7,
        "model": "chatglm2",
        "max_tokens": 4096,
        "do_sample": True,
        "stream": False,
        "seed": random.randint(1, 5000),
    })

    response = requests.post(url, data=payload, headers=headers, verify=False)
    if response.status_code == 200:
        answer = json.loads(response.text)
        answer = answer["choices"][0]["text"]
    else:
        answer = None

    return answer


def query_chatglm_platform(prompt, history=[], do_sample=True, max_tokens=4096):
    url = "http://172.18.64.19:9090/v1/chat/completions"

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

    payload = {
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOPP,
        # "model": self.model_version,
        "max_tokens": max_tokens,
        "do_sample": do_sample,
        "stream": False,
        "seed": random.randint(1, 10000000),
    }

    # response = requests.post(self.url, data=payload, headers=self.headers, verify=False)
    response = requests.post(url, json=payload, verify=False)
    
    if response.status_code == 200:
        answer = json.loads(response.text)
        # print(answer)
        # if answer["choices"][0]["finish_reason"] != "eos_token":
            # answer = None
        # else:
        answer = answer["choices"][0]["message"]["content"]
    else:
        print(response.text)
        answer = None

    return answer


def query_chatglm_tgi(prompt, history=[], do_sample=True, max_tokens=2048, max_retry=3):
    url = "http://172.18.64.8:8080/generate"
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
            "decoder_input_details": True,
            "details": False,
            "do_sample": do_sample,
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            "seed": None,
            "temperature": 0.9,
            "top_p": 0.9,
            "stop": ["<|endoftext|>", "<|user|>", "<|observation|>"]
        }
    }
   
    for _ in range(max_retry):
        output = requests.post(url, json=inputs)
        if output.status_code == 200:
            output = json.loads(output.text)
            # results.append(output[0]["generated_text"])
            result = output["generated_text"]
            break
        else:
            print(output.text)   
    else:
        result = None

    return result


def worker_build_training_pair(task_queue, done_queue, worker_func, is_glm=False):
    max_retry = 3
    
    for line in iter(task_queue.get, "STOP"):
        item = json.loads(line)
        response = None
        for _ in range(max_retry):
            try:
                response = worker_func(item)
            except Exception as e:
                print("error:", e)
                # exit()
                continue

            if response is not None:
                break
            # if not is_glm and response is not None:
            #     item["_gpt4_resp"] = response
            #     break
            # if is_glm  and response is not None:
            #     item["_glm_resp"] = gpt4_response
            #     break
            
            time.sleep(3)
        else:
            continue

        done_queue.put(item)

    done_queue.put("COMPLETE")


def build_training_file(input_file, output_file, worker_func, is_glm=False, num_process=None):
    if num_process is None:
        num_processes = NUM_PROCESS
    else:
        num_processes = num_process
        
    task_queue, done_queue = Queue(maxsize=QUEUE_SIZE), Queue(maxsize=QUEUE_SIZE)

    def read_data_into_queue():
        cnt = 0
        
        with open(input_file, "r") as r:
            print("read files")
            for line in r:
                task_queue.put(line)
                cnt += 1
                # cnt -= 1
                # if cnt <= 0:
                    # break
            print("read files done: ", cnt)

        for _ in range(num_processes):
            task_queue.put('STOP')

    processes = []
    for _ in range(num_processes):
        process = Process(target=partial(worker_build_training_pair, is_glm=is_glm),
                    args=(task_queue, done_queue, worker_func))
        process.start()
        processes.append(process)

    process = Process(target=read_data_into_queue)
    process.start()

    progress_bar = tqdm()
    print("----- GOGOGOGOGOGOGO !!!!!")
    with open(output_file, 'w') as w:
        num_finished = 0
        num_save = 0
        while num_finished < num_processes:
            item = done_queue.get()
            if item == 'COMPLETE':
                num_finished += 1
            else:
                w.write(json.dumps(item, ensure_ascii=False) + '\n')
                w.flush()
                num_save += 1
                print(f'save {num_save} examples to {output_file}', end='\r')
                progress_bar.update(1)

    progress_bar.close()
    

def standard_prompt_response(
    x, 
    response_key="response", 
    skip_response=False, 
    skip_generated=False, 
    backbone="gpt-3.5-turbo", 
    prompt_key="prompt", 
    num_generation=1,
    prompt_type="bench"
):

    # assert backbone == "chatglm_platform", "failed"
    if skip_response and response_key in x:
        print("skip")
        return x[response_key]
    if prompt_type == "bench":
        x[prompt_key] = x["contexts"][0]
    
    # if skip_generated and response_key in x:
        # return x["gpt4_turbo_response"]
        
    # if "messages" in x:    
    #     raise NotImplementedError 
    #     result = query_gpt4_with_standard_format(x)
    #     x["messages"].append(
    #         {"role": "assistant", "content": result}
    #     )
    #     if "gpt4_response" in x:
    #         x.pop("gpt4_response")
    #     x["gpt4_turbo_response"] = result
    #     # question = x['messages'][-2]["content"]
    #     x["sythetic_prompt"] = extract(result)
    # else:

    if "history" in x:
        history = x["history"]
    else:
        history = []
        
    prompt = x[prompt_key]

    # print(prompt)
    responses = []
    for i in range(num_generation):
        max_try = 1
        for _ in range(max_try):
            if backbone == "chatglm_platform":
                result = query_chatglm_platform(prompt, history)
            elif backbone == "tgi":
                result = query_chatglm_tgi(prompt, history, do_sample=True)
            elif backbone == "chatglm_ipo":
                result = query_chatglm(prompt, history)
            else:
                raise NotImplementedError

            if result is None:
                continue
        if result is not None:
            responses.append((f"reply_{i}", result))

        sleep_time = random.randint(1, 3)
        time.sleep(sleep_time)

    # print(responses)

    result = responses
    if len(result) > 0:        
        rnm = random.randint(0, 20)
        if rnm == 0:
            print(f"#### Question: {prompt} ------ \n Response: ", result[0])        # print("#### Original response: ", result)
            print()

    if num_generation == 1:
        result = result[0][1]

    x[response_key] = result
    return result


def prepare_template(prompt_filepath):
    print(f"Load prompt template from {prompt_filepath}...")
    global PROMPT_TEMPLATE
    PROMPT_TEMPLATE = open(prompt_filepath).read().strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--mode", type=str, default="response")
    parser.add_argument("--backbone", type=str, default="tgi")
    parser.add_argument("--prompt_key", type=str, default=None)
    # "gpt-4-1106-preview"
    parser.add_argument("--skip_response", action="store_true", default=False)
    parser.add_argument("--skip_generated", action="store_true", default=False)
    parser.add_argument("--prompt_template", type=str, default=None)
    parser.add_argument("--reference_key", type=str, default="answer")
    parser.add_argument("--response_key", type=str, default="response")
    parser.add_argument("--num_generation", type=int, default=16)
    parser.add_argument("--num_process", type=int, default=10)
    parser.add_argument("--prompt_type", type=str, default="train")
    parser.add_argument("--model_name", type=str, default="code_v3_0303")
    args = parser.parse_args()
    

    build_training_file(
        input_file=args.input_file,
        output_file=args.input_file.replace(".jsonl", f"_{args.model_name}.jsonl"),
        worker_func=partial(standard_prompt_response,
                            skip_response=args.skip_response, skip_generated=args.skip_generated, 
                            backbone=args.backbone, 
                            prompt_key=args.prompt_key, 
                            num_generation=args.num_generation,
                            prompt_type=args.prompt_type,
                            response_key=args.response_key),
        is_glm=False,
    )

