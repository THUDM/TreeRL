from functools import partial
import random
from openai import OpenAI

import torch
from openrlhf.utils.utils import (
    map_with_progress,
    query_chatglm_platform,
    query_chatglm_tgi,
    query_vllm_platform,
    check_result
)


def find_repeated_patterns(s, pattern_length=80, threshold=20):
    from collections import defaultdict
    pattern_counts = defaultdict(int)
    
    for i in range(len(s) - pattern_length + 1):
        pattern = s[i:i + pattern_length]
        pattern_counts[pattern] += 1
    
    repeated_patterns = {pattern: count for pattern, count in pattern_counts.items() if count >= threshold}
    return repeated_patterns


def find_expected_patterns(s):
    if "<answer>" not in s:
        return 0

    for tokens in ["<plan>", "<reflection>", "<action>"]:
        if s.count(tokens) <= 1:
            return 0
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


def get_rule_base_rewards(responses, use_repeatted=False, use_expected_pattern=False):
    rewards = []
    if not isinstance(responses[0], str):
        responses = [x[1] for x in responses]
    for resp in responses:
        score = 0
        if use_repeatted:
            is_reptead = len(find_repeated_patterns(resp, pattern_length=20, threshold=20)) > 0
            repeated_penalty = -0.2 if is_reptead else 0
            score += repeated_penalty
        if use_expected_pattern:
            pattern_reward = find_expected_patterns(resp)
            score += pattern_reward
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


### TODO: Ongoing

def query_binary_remote_reward(urls, query):   
    QUESTOIN_KEY = "question"
    RESPONSE_KEY = "response"
    LABEL_KEY = "label"
    SOURCE_KEY = "source"
    
    source = query[SOURCE_KEY] if SOURCE_KEY in query else "math"
    question = query[QUESTOIN_KEY]
    response = query[RESPONSE_KEY]
    label = query[LABEL_KEY] if LABEL_KEY in query else None
    assert label is not None or source == "code"

    _urls = urls[source]
    if source in ("math", "stem"):
        extracted_answer, result = check_result(_urls, (question, response, label))
    elif source in ("code"):
        result = ...
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    
def _remote_binary_judge_evaluation_todo(urls, queries):
    assert isinstance(urls, dict), f"urls must be a dict, found: {type(urls)}"
    # if isinstance(urls, str):
        # urls = [urls]
    assert isinstance(queries[0], dict), f"elements of queries must be dict, found {type(queries[0])}"

    # queries = [(x[0], x[1], y) for x, y in zip(queries, labels)]
    
    results = map_with_progress(
        partial(query_binary_remote_reward, urls), 
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

def apply_chat_template_qwen(system_prompt, user, assistant):
    return f"<|im_start|>system\n{system_prompt}.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>\n"


def get_remote_reward_model_value(urls, item):
    url = random.choice(urls)
    if len(item) == 3:
        question, response, _ = item
    else:
        question, response = item
    
    client = OpenAI(
        api_key="EMPTY",
        base_url=url,
    )
    # chat = [
    #     {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
    #     {"role": "user", "content": question},
    #     {"role": "assistant", "content": response}
    # ]
    
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    if len(question) + len(response) > 4096:
        response = response[:4096 - len(question)]
        
    conversation_str = apply_chat_template_qwen(system_prompt, question, response)

    for _ in range(3):
        try:
            responses = client.embeddings.create(
                input=[conversation_str],
                model="Qwen2.5-Math-RM-72B",
            )
    
            # models = client.models.list()
            # model = models.data[0].id

            # responses = client.embeddings.create(
            #     input=[conversation_str],
            #     model=model,
            # )
            for data in responses.data:
                return float(data.embedding[-1])
        except:
            # print(e)
            print("-- error in rm requests")
            continue
    return -10


def _remote_reward_model_evaluation(urls, queries):
    # if isinstance(urls, list):
        # url = random.choice(urls)
        # url
    if isinstance(urls, str):
        urls = [urls]

    results = map_with_progress(
        partial(get_remote_reward_model_value, urls), 
        queries, 
        num_threads=4
    )
    # if torch.distributed.get_rank() == 0:
    #     results = [get_remote_rm_value(urls, item) for item in tqdm(queries)]
    # else:
    #     results = [get_remote_rm_value(urls, item) for item in queries]
    # extracted_answer = [x[0] for x in results]
    # results = [x for  x in results]
    results = torch.tensor(results).float()
    return results


def get_stem_eval(queries, labels, remote_urls, use_rule_based_reward=False):
    remote_binary_urls = remote_urls["binary"]
    remote_model_urls = remote_urls["model"] if "model" in remote_urls else None
    
    extracted_answer, raw_remote_rewards = _remote_binary_judge_evaluation(remote_binary_urls, queries, labels)
    raw_remote_rewards = raw_remote_rewards.to(torch.cuda.current_device())
    binary_rewards = raw_remote_rewards
    
    if remote_model_urls:
        # _remote_model_rm_urls = load_reward_url(self.remote_reward_url[1])
        rm_based_rewards = _remote_reward_model_evaluation(remote_model_urls, queries)
        rm_based_rewards = rm_based_rewards.to(torch.cuda.current_device())
        rm_based_rewards = torch.sigmoid(0.5 * rm_based_rewards)
        raw_remote_rewards = raw_remote_rewards + rm_based_rewards

    if use_rule_based_reward:
        rule_rewards = get_rule_base_rewards(queries, use_expected_pattern=True)
        rule_rewards = rule_rewards.to(torch.cuda.current_device())
        raw_remote_rewards = raw_remote_rewards + rule_rewards

    return raw_remote_rewards, binary_rewards


def get_remote_rewarwd(queries, sources, remote_urls):
    pass

