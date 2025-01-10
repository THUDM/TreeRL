from functools import partial
import json
import math
import random
from openai import OpenAI

import requests
import torch
from openrlhf.utils.utils import (
    map_with_progress,
    query_chatglm_platform,
    query_chatglm_tgi,
    query_vllm_platform,
    check_result,
    get_qwen_remote_reward_model_value,
)


def find_repeated_patterns(s, pattern_length=80, threshold=20):
    from collections import defaultdict
    pattern_counts = defaultdict(int)
    
    for i in range(len(s) - pattern_length + 1):
        pattern = s[i:i + pattern_length]
        pattern_counts[pattern] += 1
    
    repeated_patterns = {pattern: count for pattern, count in pattern_counts.items() if count >= threshold}
    return repeated_patterns


# def find_expected_patterns(s):
#     if "<answer>" not in s:
#         return 0

#     for tokens in ["<plan>", "<reflection>", "<action>"]:
#         if s.count(tokens) <= 1:
#             return 0
#     # if "<plan>" in s and "<reflection>" in s and "<answer>" in s and "<action>" in s:
#         # return 0.1
#     # else:
#     return 0.1


def find_expected_patterns(s):
    max_actions = 50
    
    if "<answer>" not in s:
        return 0

    action_count = {}
    for tokens in ["<plan>", "<reflection>", "<action>"]:
    #     if s.count(tokens) <= 1:
    #         return 0
        action_count[tokens] = s.count(tokens)
        if action_count[tokens] < 1:
            return 0

    num_actions = sum(action_count.values())
    
    return min(num_actions / max_actions, 1) * 0.1
    
    # if "<plan>" in s and "<reflection>" in s and "<answer>" in s and "<action>" in s:
        # return 0.1
    # else:
    return 0.1

def detect_repeated_patterns(responses, pattern_length=80, threshold=20):
    repeated_patterns = []
    if not isinstance(responses[0], str):
        responses = [x[1] for x in responses]
    for resp in responses:
        is_reptead = len(find_repeated_patterns(resp, pattern_length=pattern_length, threshold=threshold)) > 0
        repeated_patterns.append(is_reptead)
    return 1 - torch.tensor(repeated_patterns).float()


def _get_rule_base_reward(response, use_repeatted=False, use_expected_pattern=False):
    if not isinstance(response, str):
        # responses = [x[1] for x in responses]
        response = response[1]
    score = 0
    if use_repeatted:
        is_reptead = len(find_repeated_patterns(response, pattern_length=20, threshold=20)) > 0
        repeated_penalty = -0.2 if is_reptead else 0
        score += repeated_penalty
    if use_expected_pattern:
        pattern_reward = find_expected_patterns(response)
        score += pattern_reward
    return score


def get_rule_base_rewards(responses, use_repeatted=False, use_expected_pattern=False):
    rewards = []
    if not isinstance(responses[0], str):
        responses = [x[1] for x in responses]
    for resp in responses:
        # score = 0
        # if use_repeatted:
        #     is_reptead = len(find_repeated_patterns(resp, pattern_length=20, threshold=20)) > 0
        #     repeated_penalty = -0.2 if is_reptead else 0
        #     score += repeated_penalty
        # if use_expected_pattern:
        #     pattern_reward = find_expected_patterns(resp)
        #     score += pattern_reward
        score = _get_rule_base_reward(resp, use_repeatted=use_repeatted, use_expected_pattern=use_expected_pattern)
        rewards.append(score)
    return torch.tensor(rewards).float()
    

def _remote_binary_judge_evaluation(urls, queries, labels):
    if isinstance(urls, str):
        urls = [urls]

    queries = [(x[0], x[1], y) for x, y in zip(queries, labels)]
    results = map_with_progress(
        partial(check_result, urls), 
        queries, 
        num_threads=8
    )
    extracted_answer = [x[0] for x in results]
    results = [x[1] for  x in results]
    results = torch.tensor(results).float()
    return extracted_answer, results


##################
### consider reward models 
##################

# def apply_chat_template_qwen(system_prompt, user, assistant):
#     return f"<|im_start|>system\n{system_prompt}.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>\n"


# def get_qwen_remote_reward_model_value(urls, item):
#     url = random.choice(urls)
#     if len(item) == 3:
#         question, response, _ = item
#     else:
#         question, response = item
    
#     client = OpenAI(
#         api_key="EMPTY",
#         base_url=url,
#     )
    
#     system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
#     if len(question) + len(response) > 4096:
#         response = response[:4096 - len(question)]
        
#     conversation_str = apply_chat_template_qwen(system_prompt, question, response)

#     for _ in range(3):
#         try:
#             responses = client.embeddings.create(
#                 input=[conversation_str],
#                 model="Qwen2.5-Math-RM-72B",
#             )
    
#             # models = client.models.list()
#             # model = models.data[0].id

#             # responses = client.embeddings.create(
#             #     input=[conversation_str],
#             #     model=model,
#             # )
#             for data in responses.data:
#                 return float(data.embedding[-1])
#         except:
#             # print(e)
#             print("-- error in rm requests")
#             continue
#     return -10


def get_remote_reward_model_value_tgi(urls, item):
    url = random.choice(urls)
    if len(item) == 3:
        question, response, _ = item
    else:
        question, response = item
    
    # if len(question) + len(response) > 4096:
    #     response = response[:4096 - len(question)]
        
    conversation_str = f"<|user|>\n{question}<|assistant|>\n{response}<|user|>"

    payload = {
        "inputs": conversation_str,
        "stream": False,
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
    
    max_retry = 4
    for _ in range(max_retry):
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            response =  response.json()
            token_id = response["details"]["tokens"][0]["id"]
            if token_id == 0:
                logprob = response["details"]["tokens"][0]["logprob"]
            else:
                # for token in response["details"]["top_tokens"][0]:
                #     # print(token)
                #     if token["id"] == 0:
                #         logprob = token["logprob"]
                logprob = response["details"]["top_tokens"][0][-1]["logprob"]
            prob = math.exp(logprob)
            # print(token_id, logprob, prob)
            logits = math.log(31.0 * prob / (1.0 - prob))
            if random.randint(0, 10) == 10:
                print(f"reward is {logits}")
            return logits
        except Exception as e:
            print(f"Error occurred in request and parser: {e}")

    return -10

def _remote_reward_model_evaluation(urls, queries):
    # if isinstance(urls, list):
        # url = random.choice(urls)
        # url
    if isinstance(urls, str):
        urls = [urls]

    results = map_with_progress(
        partial(get_remote_reward_model_value_tgi, urls), 
        queries,
        num_threads=8
    )
    results = torch.tensor(results).float()
    return results












### TODO: Ongoing


def get_general_chat_eval(question, response, urls, tokenizer,):
    inp_tokenized = tokenizer.encode(question, add_special_tokens=False) 
    oup_tokenized =  tokenizer.encode(response, add_special_tokens=False)
    inp_len = len(inp_tokenized)
    oup_len = len(oup_tokenized)
    
    max_len = 8050
    if inp_len + oup_len > max_len:
        response = tokenizer.decode(oup_tokenized[-(max_len - inp_len):])

    result = get_remote_reward_model_value_tgi(
        urls, 
        (question, response)
    )
    return result


def get_stem_eval(
    query, 
    response,
    label, 
    remote_urls, 
    is_overlong=False,
    tokenizer=None,
    use_rule_based_reward=False,
    use_general_reward_for_reason=False
):
    remote_stem_urls = remote_urls["math"]
    
    extracted_answer, raw_remote_reward = check_result(remote_stem_urls, (query, response, label))
        
    # raw_remote_rewards = raw_remote_rewards.to(torch.cuda.current_device())
    binary_reward = raw_remote_reward
    assert binary_reward in (0, 1), f"binary_reward={binary_reward}"
    if use_general_reward_for_reason:
        # 附加一个，原本实现
        # remote_model_urls = remote_urls["math_RM"]
        # # _remote_model_rm_urls = load_reward_url(self.remote_reward_url[1])
        # # rm_based_reward = _remote_reward_model_evaluation(remote_model_urls, (query, response))
        # _rm_based_reward = get_general_chat_eval(query, response, remote_model_urls, tokenizer=tokenizer)
        # # close sigmoid
        # # rm_based_reward = _rm_based_reward
        # rm_based_reward = 1 / (1 + math.exp(-_rm_based_reward))
        # coeff = 0.3
        # raw_remote_reward = raw_remote_reward + coeff * rm_based_reward
        # print(f"stem_remote_reward={_rm_based_reward}")

        ## 新实现：只用rm
        import math
        remote_model_urls = remote_urls["math_RM"]
        _rm_based_reward = get_qwen_remote_reward_model_value(remote_model_urls, query, response)
        a = 0.5
        b = -2.898
        x = a*(_rm_based_reward-b)
        result = 1/(1+math.exp(-x))
        print("use pure rm-sigmoid in chain", result)
        raw_remote_reward = result
        # print(f"stem_remote_reward={_rm_based_reward}")
    else:
        print("use binary reward only")

    if use_rule_based_reward and not is_overlong:
        if binary_reward == 0:
            rule_reward = _get_rule_base_reward(response, use_expected_pattern=True, )
            # rule_reward = rule_rewards.to(torch.cuda.current_device())
            raw_remote_reward = raw_remote_reward + rule_reward

    return raw_remote_reward, binary_reward, extracted_answer


def get_code_eval(queries, remote_urls):
    return random.random(), 1, "[[Code; no answer]]"
    # raise NotImplementedError


def query_remote_reward_single_worker(
    urls, 
    tokenizer, 
    use_rule_based_reward,
    use_general_reward_for_reason,
    query,
):   
    QUESTOIN_KEY = "prompt"
    RESPONSE_KEY = "response"
    LABEL_KEY = "label"
    SOURCE_KEY = "data_type"
    OVER_LONG_KEY = "is_overlong"
    
    source = query[SOURCE_KEY] if SOURCE_KEY in query else "math"
    question = query[QUESTOIN_KEY]
    response = query[RESPONSE_KEY]
    label = query[LABEL_KEY] if LABEL_KEY in query else None
    is_overlong = query[OVER_LONG_KEY] if OVER_LONG_KEY in query else False
    
    # assert isinstance(question, str), f"question={question}"
    if isinstance(question, list):
        question = question[0]
    if isinstance(response, list):
        response = response[0]
    question = question.replace("[gMASK] <sop>", "").strip()
    
    assert label is not None or source in ("code", "chat"), f"Label must be provided for source: {source}"

    # _urls = urls[source]
    
    if source in ("math", "stem"):
        result, binary_result, extracted_answer = get_stem_eval(
            question, 
            response, 
            label,
            urls, 
            is_overlong=is_overlong,
            tokenizer=tokenizer,
            use_general_reward_for_reason=use_general_reward_for_reason,
            use_rule_based_reward=use_rule_based_reward
        )
    elif source in ("code"):
        result, binary_result, extracted_answer = get_code_eval(query, urls)
        # raise NotImplementedError
    elif source in ("chat"):
        _urls = urls["chat"]
        result = get_general_chat_eval(question, response, _urls, tokenizer)
        binary_result = None
        extracted_answer = "[[General chat; no answer]]"
    else:
        raise NotImplementedError

        # raise NotImplementedError
    return extracted_answer, result, binary_result


def get_remote_reward_entry(
    urls, 
    queries, 
    tokenizer,
    overlong_mask,
    use_general_reward_for_reason=False,
    use_rule_based_reward=False
):
    assert isinstance(urls, dict), f"urls must be a dict, found: {type(urls)}"
    # if isinstance(urls, str):
        # urls = [urls]
    assert isinstance(queries[0], dict), f"elements of queries must be dict, found {type(queries[0])}"

    # queries = [(x[0], x[1], y) for x, y in zip(queries, labels)]
    
    for item, ovl in zip(queries, overlong_mask):
        item["is_overlong"] = not bool(ovl)

    results = map_with_progress(
        partial(
            query_remote_reward_single_worker, 
            urls, 
            tokenizer, 
            use_rule_based_reward, 
            use_general_reward_for_reason,
            ), 
        queries, 
        num_threads=8
    )
    extracted_answers = [x[0] for x in results]
    binary_results = [x[2] for x in results]
    results = [x[1] for  x in results]
    
    results = torch.tensor(results).float()
    print("reward returned:",results)

    return extracted_answers, results, binary_results

def get_remote_reward_entry_mcts_mask(
    queries, 
    overlong_mask,
    use_rule_based_reward=False
):
    assert isinstance(queries[0], dict), f"elements of queries must be dict, found {type(queries[0])}"

    '''queries = [{
                "prompt": x[0],
                "response": x[1],
                "label": x[2],
                "reward": x[3], # [.....]token length
                "attention_mask": x[4], # [.....]token length
                "data_type": y
            }]
    '''
    
    for item, ovl in zip(queries, overlong_mask):
        item["is_overlong"] = not bool(ovl)

    results = []
    attention_mask = []
    threshold = 0.2
    for query in queries:
        raw_reward = query["reward"].tolist()
        if use_rule_based_reward and not query["is_overlong"]:
            if raw_reward < threshold:
                rule_reward = _get_rule_base_reward(query["response"], use_expected_pattern=True, )
                raw_remote_reward = raw_remote_reward + rule_reward
        results.append(raw_reward)
        attention_mask.append(query["attention_mask"].tolist())
    results = torch.tensor(results).float()
    #转为整数
    attention_mask = torch.tensor(attention_mask).float()
    print("reward returned:",results)
    print("attention_mask returned:",attention_mask.shape)
    return results, attention_mask

def get_remote_reward_entry_mcts(
    queries, 
    overlong_mask,
    use_rule_based_reward=False
):
    assert isinstance(queries[0], dict), f"elements of queries must be dict, found {type(queries[0])}"

    '''queries = [{
                "prompt": x[0],
                "response": x[1],
                "label": x[2],
                "reward": x[3], # [.....]token length
                "data_type": y
            }]
    '''
    
    for item, ovl in zip(queries, overlong_mask):
        item["is_overlong"] = not bool(ovl)

    results = []
    threshold = 0.2
    for query in queries:
        raw_reward = query["reward"].tolist()
        if use_rule_based_reward and not query["is_overlong"]:
            if raw_reward < threshold:
                rule_reward = _get_rule_base_reward(query["response"], use_expected_pattern=True, )
                raw_remote_reward = raw_remote_reward + rule_reward
        results.append(raw_reward)
    results = torch.tensor(results).float()
    print("reward returned:",results)
    return results
