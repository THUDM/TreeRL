import random
import requests
import json
import re
import ray
import time
from openai import OpenAI
from vllm import SamplingParams
import torch
RETRY_COUNT = 10
MAX_CONTENT_FILTER_RETRY = 0

EXTRACTION_TEMPLATE = """
Look at the following math problem and extract the final answer, such as final results or option. 
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
<ANSWER>: 2: 1
If the question is a multiple-choice question, extract both the option and value that matches the answer.
Make sure your answer format matches the format required by the question.
"""


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
    Expression 2: C: 100

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
---
YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()


UNDERSTANDING_PROMPT = """
A math problem will be provided below. Your task is to offer a concise summary of the problem, including the known conditions and what is being asked to help understand the problem. Do not attempt to solve the problem or provide an answer, though a brief analysis is acceptable.
【Math problem】
{question}
"""
# Write as briefly as you can in no more than 3 sentences.


# QWEN_SYSTEM_PROMPT="""<|im_start|>system
# You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Solve the problem step by step and put your final answer in the format of `<answer>Your answer`. The following is an exmaple:
# ## Step1:
# ...
# ## Step N:
# <answer> The answer is ..
# <|im_end|>
# <|im_start|>user
# {prompt}
# <|im_end|>
# <|im_start|>assistant

# """
QWEN_SYSTEM_PROMPT = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Solve the following math problem step by step. You should divide your solution into several steps. 
For each step, you should provide a brief explanation of the purpose of this step and then the detailed mathematical reasoning to actually solving this step, they should be connected with a "\n". Do not mention the prahses "purpose of this step" or "detailed mathematical reasoning" directly in your response.
Present the steps in the format as follows:
```
A brief understanding of the problem.
### Step 1: 
### Step 2: 
...
### Step n: 
### Final Answer: The final answer
```
Here is one example:
{examples}
Please strictly adhere to the output format requirements and do not return any additional information.
<|im_end|>
<|im_start|>user
## New Math Problem
{prompt}
<|im_end|>
<|im_start|>assistant

"""

QWEN_SYSTEM_PROMPT = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Solve the following math problem step by step. You should divide your solution into several steps. 
For each step, you should provide a brief explanation of the purpose of this step and then the detailed mathematical reasoning to actually solving this step, they should be connected with a "
". Do not mention the prahses "purpose of this step" or "detailed mathematical reasoning" directly in your response. 
Present the steps in the format as follows:
```
A brief understanding of the problem.
### Step 1: 
### Step 2: 
...
### Step n: 
### Final Answer: The final answer
```<|im_end|>
<|im_start|>user
<|Math problem|>
{example_question}<|im_end|>
<|im_start|>assistant
{example_answer}<|im_end|>
<|im_start|>user
<|Math problem|>
{prompt}<|im_end|>
<|im_start|>assistant

"""

# QWEN_QA_PROMPT = """<|im_start|>system
# You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
# ```<|im_end|>
# <|im_start|>user
# {prompt}<|im_end|>
# <|im_start|>assistant
# {response}
# """
QWEN_QA_PROMPT = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
{response}"""

# Qwen_FEWSHOT_QAPROMPT = """<|im_start|>system
# You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
# <|im_start|>user
# {prompt}<|im_end|>
# <|im_start|>assistant
# {response}"""


GLM_QA_PROMPT = """[gMASK]<sop><|user|>
{prompt}<|assistant|>
{response}"""

def query_sglang_chat(
    prompt,
    urls,
    use_logits=False,
    skip_special_tokens=True,
    n=1,
    max_tokens=512,
    temperature=0.0,
    top_p=0.5
):
    messages = [{"role": "user", "content": prompt}]
    extra_body = {
        "skip_special_tokens": skip_special_tokens
    }

    for try_counter in range(RETRY_COUNT):
        try:
            request_data = {
                "model": "default",
                "temperature": temperature,
                "top_p": top_p,
                "messages": messages,
                "n": n,
                "max_tokens": max_tokens,
                "seed": random.randint(0, 100000),
                **extra_body
            }

            api_base = random.choice(urls)
            url = api_base + "/chat/completions"
            response = requests.post(
                url,
                json=request_data,
                headers={"content-type": "application/json"},
                timeout=1800
            )

            if response.status_code == 200:
                resp_json = response.json()
                choices = resp_json['choices']
                content_list = [choice['message']['content'].strip()
                                for choice in choices]

                return content_list[0]
            else:
                print(
                    f"Failed to fetch response, chat: {response.status_code}, {response.text},{api_base}")
        except Exception as e:
            sleep_time = 2 * try_counter + 1
            if sleep_time > 30:
                exit(1)
            print(f"Error: {str(e)}, sleeping for {sleep_time} seconds")
            time.sleep(sleep_time)

    return None

def extract_answer(
    question,
    response,
    qwen_urls,
):
    # resp_url = "http://172.19.192.137:9090/v1/chat/completions"
    response = response.strip().split('\n')
    # response = [x for x in response if x]
    resp_text = [x for x in response if x.strip()]
    resp_text = "\n".join(resp_text[-3:])

    answer = None
    if "\\box" in resp_text or "\\boxed" in resp_text:
        # extract value in \box
        answer = re.findall(
            r'\\box\{([^{}]*(?:\{[^{}]*\})*[^{}]*)\}', resp_text)
        if len(answer) == 0:
            answer = re.findall(
                r'\\boxed\{([^{}]*(?:\{[^{}]*\})*[^{}]*)\}', resp_text)
    elif "### Final Answer" in resp_text:
        answer = re.findall(r'<answer>([^<>]*)', resp_text)

    if answer:
        answer = answer[0].strip()
    else:
        # if "<answer>" in resp_text:
        # resp_text = resp_text.replace("<answer>", "").strip()
        # answer =
        answer_template = EXTRACTION_TEMPLATE.format(
            question=question, answer=resp_text)
        # if "sglang" in backbone:
        #     extracted_answer = query_sglang_chat(prompt = answer_template, urls=urls)
        # else:
        #     url = random.choice(urls)
        #     extracted_answer = query_chatglm_platform(
        #         answer_template, url=url
        #         )
        extracted_answer = query_sglang_chat(
            prompt=answer_template, urls=qwen_urls, temperature=0.0, top_p=0.5)
        if extracted_answer is None:
            answer = ""
        else:
            answer = extracted_answer.replace("<ANSWER>: ", "").strip()

    return answer


def check_equality(expr1: str, expr2: str, urls):
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}
    response = query_sglang_chat(prompt, urls)
    if isinstance(response, str):
        return "yes" in response.lower().strip()
    else:
        return 0


def check_result(
    question,
    response,
    label,
    # qwen_urls,
    urls
):
    answer = extract_answer(question, response, urls)
    if answer is None:
        return None, 0
    check = check_equality(answer, label, urls=urls)
    print("===",check,"===",answer,label)
    return answer, 1 if check else 0


# def query_local_vllm_completions_ids(
#     prompt_token_ids,
#     llm,
#     n=1,
#     skip_special_tokens=True,
#     max_tokens=2048,
#     stops=None,
#     temperature=0.9,
#     top_p=0.9,
#     min_tokens = 0,
#     model = "glm"
# ):  

#     sampling_params = SamplingParams(
#         temperature=temperature,
#         top_p=top_p,
#         max_tokens=max_tokens,
#         min_tokens = min_tokens,
#         skip_special_tokens = skip_special_tokens,
#         stop = stops,
#         # seed = random.randint(0, 100000),
#         n = n,
#     )
    
#     content_list = []
#     finish_reason_list = []
#     stop_token_list = []
#     token_num_list = []

#     for try_counter in range(RETRY_COUNT):
#         try:
#             outputs = ray.get(llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))
#             content_token_list = [[outs.token_ids for outs in output.outputs] for output in outputs]
#             content_str_list = [[outs.text for outs in output.outputs] for output in outputs]
#             finish_reason_list = [[outs.finish_reason for outs in output.outputs] for output in outputs]
#             stop_token_list = [[outs.stop_reason for outs in output.outputs] for output in outputs]
#             token_num_list = [[len(outs.token_ids) for outs in output.outputs] for output in outputs]
#             # print("===stop_token===\n", stop_token_list)
#             # pdb.set_trace()

#             return content_token_list, content_str_list, finish_reason_list, stop_token_list ,token_num_list

#         except Exception as e:
#             sleep_time = 2 * try_counter + 1
#             if sleep_time > 30:
#                 exit(1)
#             print(f"Error: {str(e)}, sleeping for {sleep_time} seconds")
#             time.sleep(sleep_time)
    
#     return None, None, None, None, None

def query_local_vllm_completions_ids(
    prompt_token_ids,
    llm,
    n=1,
    skip_special_tokens=True,
    max_tokens=2048,
    stops=None,
    temperature=0.9,
    top_p=0.9,
    min_tokens = 0,
    model = "glm"
):  

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        min_tokens = min_tokens,
        skip_special_tokens = skip_special_tokens,
        stop_token_ids = stops,
        # seed = random.randint(0, 100000),
        n = n,
    )
    
    content_list = []
    finish_reason_list = []
    stop_token_list = []
    token_num_list = []

    for try_counter in range(RETRY_COUNT):
        try:
            outputs = ray.get(llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))
            # outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
            # content_token_list = torch.tensor([[outs.token_ids for outs in output.outputs] for output in outputs])
            content_token_list = [[list(outs.token_ids) for outs in output.outputs] for output in outputs]
            content_str_list = [[outs.text for outs in output.outputs] for output in outputs]
            finish_reason_list = [[outs.finish_reason for outs in output.outputs] for output in outputs]
            stop_token_list = [[outs.stop_reason for outs in output.outputs] for output in outputs]
            token_num_list = [[len(outs.token_ids) for outs in output.outputs] for output in outputs]
            # print("===stop_token===\n", stop_token_list)
            # pdb.set_trace()
            # print(content_token_list)

            return content_token_list, content_str_list, finish_reason_list, stop_token_list, token_num_list

        except Exception as e:
            sleep_time = 2 * try_counter + 1
            if sleep_time > 30:
                exit(1)
            with open("/workspace/lurui/openrlhf-glm/logs/outputs/error.log", "a") as f:
                f.write(f"Error: {str(e)}, sleeping for {sleep_time} seconds\n")
            time.sleep(sleep_time)
    
    return None, None, None, None, None


def test_glm_model(urls):
    # def query_chatglm_tgi(prompt, history=[], do_sample=False, max_tokens=256, max_retry=3, url=None, temperature=0.4, top_p=0.1):
    messages = ""
    # for turn in history:
    #     ques, ans = turn["prompt"], turn["response"]
    #     messages += f"<|user|>\n{ques}<|assistant|>\n{ans}"

    messages += f"<|user|>\nhi 你好<|assistant|>\n"
    inputs = {
        "inputs": messages,
        "stream": False,
        "parameters": {
            "best_of": 1,
            "decoder_input_details": False,
            "details": False,
            "do_sample": True,
            "max_new_tokens": 128,
            "return_full_text": False,
            "seed": None,
            # "temperature": temperature,
            # "top_p": top_p,
            "stop": ["<|endoftext|>", "<|user|>", "<|observation|>"]
        }
    }

    for api_base in urls:
        try:
            output = requests.post(api_base, json=inputs)
            if output.status_code == 200:
                output = json.loads(output.text)
                # results.append(output[0]["generated_text"])
                result = output["generated_text"]
                break
            else:
                print(output.text)
        except Exception as e:
            print(f"error in tgi requests: {api_base}")
            exit(0)
            continue


def test_sglang_model(urls):
    import requests

    prompt = "Are you ready?"
    for api_base in list(set(urls)):
        url = api_base + "/chat/completions"
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        request_data = {
            "model": "default",
            "messages": messages,
            "temperature": 0.0,
            "stream": False,
        }

        try:
            response = requests.post(url, json=request_data, headers={
                "content-type": "application/json"
            }, timeout=10)
            print(response.text)
            print(response.json()['choices'][0]['message']['content'])
        except requests.exceptions.Timeout:
            print(response.text)
            print("Request timed out")
            exit(1)
# print(query_model_sglang("who are you?", url="http://172.19.192.138:8000/v1"))
import math
def generate_logits(urls, user_query, assistant_response):
    def send_request(user_query, assistant_response):
        payload = {
            "inputs": f"<|user|>\n{user_query}\n<|assistant|>\n{assistant_response}\n<|user|>",
            "parameters": {
                "best_of": 1,
                "decoder_input_details": False,
                "details": True,
                "do_sample": False,
                "max_new_tokens": 1,
                "return_full_text": False,
                "stop": ["<|user|>", "</s>", "<|assistant|>", "<eos>"],
                "seed": None,
                "top_n_tokens": 32
            }
        }
        headers = {
            'Content-Type': 'application/json'
        }
        # print(f"<|user|>\nHuman: {user_query}\nAssistant: <|assistant|>\n{assistant_response}\n<|user|>")
        RETRY_COUNT = 5
        response = None
        for try_counter in range(RETRY_COUNT):
            try:
                api_base = random.choice(urls)
                url = api_base
                response = requests.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    # print(response.json())
                    return response.json()
                else:
                    print(f"Failed to fetch response: {response.status_code}, {response.text}, {url}")
            except requests.exceptions.RequestException as e:
                sleep_time = min(2 ** try_counter, 30)
                print(f"Error: {str(e)}, sleeping for {sleep_time} seconds")
                time.sleep(sleep_time)
        return None

    def parse_response(response):
        if response:
            try:
                token_id = response["details"]["tokens"][0]["id"]
                if token_id == 0:
                    logprob = response["details"]["tokens"][0]["logprob"]
                else:
                    logprob = response["details"]["top_tokens"][0][-1]["logprob"]
                prob = math.exp(logprob)
                logits = math.log(31.0 * prob / (1.0 - prob + 1e-8))
                return logits
            except Exception as e:
                print(f"Error parsing response: {e}")
                return None

    response = send_request(user_query, assistant_response)
    return parse_response(response)
# print(generate_logits(["http://172.18.71.87:8000/v1"], "What is the value of $x$ in the equation $2x + 3 = 7$?", "The value of $x$ in the equation $2x + 3 = 7$ is $x = 2$."))

# def apply_chat_template_qwen(system_prompt, user, assistant):
#     return f"<|im_start|>system\n{system_prompt}.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>\n"


# def get_qwen_remote_reward_model_value(urls, question, response):
#     url = random.choice(urls)
#     # print(url)

#     client = OpenAI(
#         api_key="EMPTY",
#         base_url=url,
#     )

#     system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
#     if len(question) + len(response) > 8192:
#         response = response[:8192 - len(question)]

#     conversation_str = apply_chat_template_qwen(system_prompt, question, response)
#     # print(conversation_str)

#     for _ in range(3):
#         try:
#         # if True:
#             responses = client.embeddings.create(
#                 input=[conversation_str],
#                 model="Qwen72BRM",
#             )

#             for data in responses.data:
#                 a = 5
#                 b = 2
#                 result = 1/(1+a**(-float(data.embedding[-1])/b))
#                 return result
#         except Exception as e:
#             print(e)
#             print("-- error in rm requests", url)
#             continue
#     return -10

def apply_chat_template_qwen(system_prompt, user, assistant):
    return f"<|im_start|>system\n{system_prompt}.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"


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

# get_qwen_remote_reward_model_value(["http://172.20.65.240:8000/v1"], "What is the value of $x$ in the equation $2x + 3 = 7$?", "The value of $x$ in the equation $2x + 3 = 7$ is $x = 2$.")
def query_tgi_get_first_token(prompt, urls):
        first_token_list = []
        for i in range(16):
            try:
                url = random.choice(urls)
                rep = requests.post(
                    url,
                    json={
                        'inputs': prompt,
                        'stream': False,
                        "parameters": {
                            "best_of": 1,
                            "decoder_input_details": False,
                            "details": False,
                            "do_sample": True,
                            "max_new_tokens": 1,
                            "return_full_text": False,
                            "temperature": 30,
                            "top_k": 32,
                            "stop": ["<|user|>", "<|endoftext|>", "<|observation|>"],
                        }
                    },
                    headers={
                        'Content-Type': 'application/json'
                    },
                    timeout=360
                )
                rep = rep.json()
                ans = rep[0]['generated_text']
                # # 必须包含字母
                # if ans.strip() in ['', '<', '</'] or not any(c.isalpha() for c in ans):
                #     continue
                # 如果里面有中文，就跳过
                if any('\u4e00' <= char <= '\u9fff' for char in ans):
                    continue
                # if isinstance(rep, list):
                if ans:
                    first_token_list.append(ans)
                # else:
                #     return rep['generated_text'].strip().replace('<|user|>', '')
            except Exception as e:
                print(f"Exception: ", e)
                continue
        # 去重
        first_token_list = list(set(first_token_list))
        return first_token_list

