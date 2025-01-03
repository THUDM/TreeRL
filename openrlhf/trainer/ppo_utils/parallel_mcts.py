from __future__ import annotations
import Levenshtein

"""

Implements the MCTS + Self-Refine algorithm from
`Accessing GPT-4 level Mathematical Olympiad Solutions via Monte
Carlo Tree Self-refine with LLaMa-3 8B: A Technical Report`
by Zhang et. al.

The authors' [repo](https://github.com/trotsky1997/MathBlackBox) uses critiques,
refinements, and parent nodes' answers as conversation history.
I haven't tried it yet.

"""

from typing import Optional,Callable 
import random
import math
from collections import deque
from enum import Enum
from pydantic import BaseModel,PrivateAttr
from tqdm import tqdm
import json
import time
from functools import partial
from multiprocess import Queue, Process
import yaml
import torch
from typing import List
import re
from openrlhf.trainer.ppo_utils.evaluation import (
# from evaluation import (
    check_result,
    generate_logits,
    test_sglang_model,
    test_glm_model,
    get_qwen_remote_reward_model_value,
    query_local_vllm_completions_ids,
    QWEN_SYSTEM_PROMPT,
    UNDERSTANDING_PROMPT,
    QWEN_QA_PROMPT,
    GLM_QA_PROMPT,
    top_k_sampling,
)

from vllm import LLM,SamplingParams
import os

endoftexttoken = 151336
import numpy as np
import requests
import os

import cProfile
from datetime import datetime
from functools import wraps

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

ROOT_UCT_SCORE = 10_000
QUEUE_SIZE = 10000
NUM_PROCESS = 50
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/data/o1-cloud/checkpoints/sft/glm_9b_1102', trust_remote_code=True)

with open("/workspace/lurui/openrlhf-glm/openrlhf/trainer/ppo_utils/configs/api_config_llama3.1_max_request_1.json") as f:
    api_checker_config = json.load(f)

EVALUATOR_URLS = []
ips = api_checker_config['ip']
for key, value in ips.items():
    EVALUATOR_URLS.extend([key for _ in range(value)])

# with open("./configs/api_config_rm_general.json") as f:
#     rm_api_config = json.load(f)
# with open("/workspace/lurui/openrlhf-glm/openrlhf/trainer/ppo_utils/configs/api_qwen_rm.json") as f:
#     rm_api_config = json.load(f)

with open("/workspace/lurui/openrlhf-glm/openrlhf/trainer/ppo_utils/configs/api_qwen72_v6_rm.json") as f:
    rm_api_config = json.load(f)

RM_URLS = []
ips = rm_api_config['ip']
for key, value in ips.items():
    RM_URLS.extend([key for _ in range(value)])
    
    
with open("/workspace/lurui/openrlhf-glm/openrlhf/trainer/ppo_utils/configs/api_extractor.json") as f:
    api_extractor_config = json.load(f)

EXTRACTOR_URLS = []
ips = api_extractor_config['ip']
for key, value in ips.items():
    EXTRACTOR_URLS.extend([key for _ in range(value)])

eos_tokens_set = [151329,151336,151338]

# import cProfile
# import line_profiler
# def profile_with_time(output_dir="profile_results"):
#     """
#     装饰器：结合了line_profiler和cProfile，并保存结果到文件
#     """
#     def decorator(func):
#         # 创建line_profiler
#         profile = line_profiler.LineProfiler()
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             # 创建输出目录
#             os.makedirs(output_dir, exist_ok=True)
#             # 生成唯一的文件名（使用时间戳）
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             profile_output = os.path.join(output_dir, f"profile_{timestamp}.txt")
#             # 启动性能分析
#             profiler = cProfile.Profile()
#             try:
#                 # 使用line_profiler包装函数
#                 wrapped = profile(func)
#                 # 使用cProfile执行函数
#                 result = profiler.runcall(wrapped, *args, **kwargs)
#                 # 保存性能分析结果
#                 with open(profile_output, 'w') as f:
#                     # 写入line_profiler结果
#                     profile.print_stats(stream=f)
#                     f.write("\n\n" + "="*50 + "\n\n")
#                     # 写入cProfile结果
#                     import pstats
#                     stats = pstats.Stats(profiler, stream=f)
#                     stats.sort_stats('cumulative')
#                     stats.print_stats()
#                 return result
#             except Exception as e:
#                 print(f"Error during profiling: {e}")
#                 raise
#         return wrapper
#     return decorator

from collections import Counter
def find_repeated_patterns(s, pattern_length=50, threshold=20):
    from collections import defaultdict
    # pattern_counts = defaultdict(int)
    # for i in range(len(s) - pattern_length + 1):
    #     pattern = s[i:i + pattern_length]
    #     pattern_counts[pattern] += 1
    # repeated_patterns = {pattern: count for pattern, count in pattern_counts.items() if count >= threshold}
    # 生成 N-grams
    ngrams = [s[i:i + pattern_length]
              for i in range(len(s) - pattern_length + 1)]
    # 统计 N-gram 出现次数
    ngram_counts = Counter(ngrams)
    # 筛选出重复的 N-gram
    repeated_patterns = {gram: count for gram,
                         count in ngram_counts.items() if count > threshold}
    return repeated_patterns

def split_and_filter(text, split_chars):
    """
    根据指定的分割符分割字符串，并去除空白部分

    :param text: 要分割的字符串
    :param split_chars: 分割符列表
    :return: 分割后的子字符串列表
    """
    import re
    regex_pattern = '|'.join(map(re.escape, split_chars))
    parts = re.split(regex_pattern, text)
    return [part.strip() for part in parts if part.strip()]


def similarity(str1, str2, split_chars, threshold=0.3, min_proportion=0.65):
    """
    检查两个字符串的部分拼接是否过于相似

    :param str1: 第一个字符串
    :param str2: 第二个字符串
    :param split_chars: 分割符列表，用于分割字符串成片段
    :param threshold: 相似度阈值，表示Levenshtein距离除以较长字符串长度的比例，默认值为0.3
    :param min_proportion: 最小比例，表示拼接部分占原字符串长度的最小比例，默认值为0.5
    :return: 如果两个字符串的部分拼接过于相似，返回True；否则返回False
    """
    def calculate_similarity_ratio(s1, s2):
        """
        计算两个字符串的相似度比率

        :param s1: 第一个字符串
        :param s2: 第二个字符串
        :return: 相似度比率
        """
        len1 = len(s1)
        len2 = len(s2)

        # 计算Levenshtein距离
        lev_distance = Levenshtein.distance(s1, s2)

        # 计算相似比例，取两个字符串长度的较大值
        similarity_ratio = lev_distance / max(len1, len2)

        return similarity_ratio

    len1 = len(str1)
    len2 = len(str2)

    parts1 = split_and_filter(str1, split_chars)
    parts2 = split_and_filter(str2, split_chars)

    # 遍历第一组子串的所有可能组合
    for i in range(len(parts1)):
        for j in range(i+1, len(parts1)+1):
            substring1 = ''.join(parts1[i:j])
            length1_proportion = len(substring1) / len1
            for k in range(len(parts2)):
                for l in range(k+1, len(parts2)+1):
                    substring2 = ''.join(parts2[k:l])
                    length2_proportion = len(substring2) / len2

                    if length2_proportion >= min_proportion or length1_proportion >= min_proportion:
                        if calculate_similarity_ratio(substring1, substring2) < threshold:
                            # print("str1: ", substring1, "\nstr2: ", substring2, "\nratio1:", length1_proportion,
                            #       "\nratio2:", length2_proportion, calculate_similarity_ratio(substring1, substring2))
                            return True

    return False

def similarity_naive(str1,str2):
    if str1 == str2:
        with open("/workspace/lurui/openrlhf-mcts/data/similarity.log", "a") as f:
            f.write(f"similarity action\n")
    else:
        with open("/workspace/lurui/openrlhf-mcts/data/similarity.log", "a") as f:
            f.write(f"new action\n")
    return str1.strip() == str2.strip()


class MCTSNode(BaseModel):
    state: List[int]
    answer: str
    answer_token : List[int] = []
    aggregate_answer: str = ""
    aggregate_answer_token: List[int] = []
    parent: MCTSNode | None = None
    children: list[MCTSNode] = []
    visits: int = 0
    R : float = 0
    value : float = 0
    reward_samples: list[int] = []
    depth: int = 0
    main_chain: bool = False
    terminal: bool = False
    max_children: int = 3
    terminal_in_subtree: int = 0
    correct_terminal_in_subtree: int = 0
    accumulated_value: float = 0
    visited_terminal: int = 0
    repeat:bool = False
    node_id: int = 0

    def add_child(self, child_node: MCTSNode):
        self.children.append(child_node)

    def __repr__(self):
        return f"MCTSNode(answer={self.answer}, value={self.value:.2f}, visits={self.visits})"

    def __eq__(self, other):
        if isinstance(other, MCTSNode):
            # return self.state == other.state and self.answer == other.answer
            return self.node_id == other.node_id
        return False

    def __hash__(self):
        return hash((self.node_id))

    def add_reward(self, reward: int):
        self.reward_samples.append(reward)

class SelectionPolicy(Enum):
    GREEDY = 1
    IMPORTANCE_SAMPLING = 2
    PAIRWISE_IMPORTANCE_SAMPLING = 3


class InitializeStrategy(Enum):
    ZERO_SHOT = 1
    DUMMY_ANSWER = 2


class MCTSr(BaseModel):
    problem: str
    golden_answer: str
    max_nodes: int
    max_token_num: int
    max_time_use: int
    exploration_constant: float = 1.0
    epsilon: float = 1e-10
    reward_limit: int = 95
    excess_reward_penalty: int = 5
    max_depth: int = 40
    max_children: int = 3
    min_children: int = 2
    pass_k: int = 4
    path_num :int = 16
    total_token_num: int = 0
    backbone: str = "glm"
    passed_passktest : bool = False
    backprop: bool = True
    max_node_per_depth  :int = 16
    selection_policy: SelectionPolicy = SelectionPolicy.IMPORTANCE_SAMPLING
    first_token_temperature: bool = False
    selected_terminals: list[MCTSNode] = []
    # initialize_strategy: InitializeStrategy = InitializeStrategy.ZERO_SHOT

    root: MCTSNode = MCTSNode(
        state=[], answer="I don't know.", max_children=max_children
    )
    temperature: float = 0.9
    top_p: float = 0.9
    llms : List
    tokenize_fn :Callable 
    detokenize_fn :Callable

    # Logs
    rewards: list[float] = []
    selected_nodes: list[MCTSNode] = []

    depth_node_count: dict[int, int] = {}
    look_ahead :int = 0
    
    # parallel
    concurrent_num: int = 4  # 并行任务数
    node_number :int = 0
    leaf_num :int = 0
    terminal_flag :bool = False
    _lock: Optional[threading.Lock] = PrivateAttr(default=None)
    _childnum_lock: Optional[threading.Lock] = PrivateAttr(default=None)
    prompt_max_len:int = 1024
    leaves: List[int] = []
    step_level_norm: bool = True
    random_pick :bool = False
    parent_shift: bool = False
    use_orm_reward: bool = False
    select_correct_leaf: bool = False
    leaf_num_count: int = 1
    use_chain_reward: bool = False
    use_state_value_reward: bool = False
    use_pure_RM:bool = False
    use_pure_binary:bool = False
    shallow_enwide: bool = False
    system_prompt :str = ""

    # def __init__(self, temperature, top_p, model_name, stops=None):
    #     super().__init__()

    #     self.top_p = top_p
    #     self.model_name = model_name
    #     self.stops = stops

    def self_evaluate(self, node: MCTSNode, is_terminated: bool):
        if is_terminated:
            if node.repeat:
                reward = 0
            else:
                extracted_answer, result = check_result(
                    self.problem, node.answer, self.golden_answer,
                    # qwen_urls=qwen_urls,
                    extractor_urls=EXTRACTOR_URLS,
                    checker_urls=EVALUATOR_URLS)
                reward = result
            if self.use_pure_binary:
                print("use pure binary")
                node.accumulated_value = reward
            else:
                value = get_qwen_remote_reward_model_value(
                    urls= RM_URLS, question = self.problem, response = node.aggregate_answer)
                if self.use_pure_RM:
                    a = 0.5
                    b = -2.898
                    x = a*(value-b)
                    result = 1/(1+math.exp(-x))
                    print("rm_score",value, result)
                    node.accumulated_value = result
                else:
                    sigmoid_value = 1 / (1 + math.exp(-value))
                    coeff = 0.5
                    value = reward + coeff * sigmoid_value
                    node.accumulated_value = value
        else:
            # reward = generate_logits(
            #     urls=RM_URLS, user_query = self.problem, assistant_response = node.aggregate_answer,
            # )
            reward = get_qwen_remote_reward_model_value(
                urls= RM_URLS, question = self.problem, response = node.aggregate_answer)
        return reward
                        
    def backpropagate(self, node: MCTSNode,gamma=0.9,main_chain=False):
        print("backpropagate")
        parent = node.parent
        node.visits += 1
        if main_chain:
            node.main_chain = True
        # 遍历所有children
        if node.children:
            nume, deno = 0, 0
            for child in node.children:
                reward = child.R - node.R
                # print("R:", child.R, node.R)
                q_value = reward + gamma * child.value
                nume += q_value * child.visits
                # print("q_value", q_value, child.visits)
                deno += child.visits
            if deno:
                node.value = nume / deno
            else:
                print("Fail to process value", nume, deno)
        else:
            node.value = node.R
        if parent:
            self.backpropagate(parent,gamma,main_chain)
    
    def leaf_backpropagate(self, node: MCTSNode):
        if node.terminal and node.main_chain:
            node.terminal_in_subtree += 1
            node.correct_terminal_in_subtree += 1
            # 所有父亲的terminal_in_subtree和correct_terminal_in_subtree都加1
            parent = node.parent
            while parent:
                parent.terminal_in_subtree += 1
                parent.correct_terminal_in_subtree += 1
                parent.accumulated_value += node.accumulated_value
                parent = parent.parent
        elif node.terminal:
            node.terminal_in_subtree += 1
            # 所有父亲的terminal_in_subtree都加1
            parent = node.parent
            while parent:
                parent.terminal_in_subtree += 1
                parent.accumulated_value += node.accumulated_value
                parent = parent.parent

    def uct(self, node: MCTSNode,offset = 1):
        ## temp
        # return 1
        if not node.parent:
            # Using an arbitrarily high UCT score for the root node.
            # helps to prioritize breadth.
            return ROOT_UCT_SCORE
        # ans = (node.value+1)/2 + self.exploration_constant * math.sqrt(
        #     math.log(node.parent.visits + 1) / (node.visits + self.epsilon)
        # )
        # if ans<0:
        #     print("UCT score is negative",ans)
        # print((node.value+1+1e-8)/2,self.exploration_constant * math.sqrt(
        #     math.log(node.parent.visits + 1) / (node.visits + self.epsilon)
        # ))

        return (node.value+offset+1e-8)/2 + self.exploration_constant * math.sqrt(
            math.log(node.parent.visits + 1) / (node.visits + self.epsilon)
        )

    def is_fully_expanded(self, node: MCTSNode):
        # print(len(node.children), node.max_children, node.terminal)
        next_depth = node.depth + 1
        next_depth_count = self.depth_node_count.get(next_depth, 0)
        if next_depth_count >= self.max_node_per_depth:
            return True
        return len(node.children) >= node.max_children or node.terminal

    def weighted_sample_no_replacement(self, candidates, weights, k):
        total_candidates = len(candidates)
        k = min(total_candidates, k)  # 确保 k 不超过候选项数量

        if k >= total_candidates:
            return list(range(total_candidates))  # 如果 k 大于等于候选项数量，返回所有索引
        
        if k <= 0 or total_candidates == 0:
            return []  # 如果 k <= 0 或候选项为空，返回空列表

        indices = list(range(total_candidates))
        selected_indices = []

        for _ in range(k):
            # 根据权重随机选择一个索引
            try:
                chosen_index = random.choices(indices, weights=weights, k=1)[0]
            except ValueError:
                # 如果在选择索引时发生错误，重新计算权重再试一次
                uct_scores = [self.uct(node, 2) for node in candidates]
                chosen_index = random.choices(indices, weights=uct_scores, k=1)[0]
                
            selected_indices.append(chosen_index)
            
            # 移除已选择的索引和对应的权重
            remove_index = indices.index(chosen_index)
            indices.pop(remove_index)
            weights.pop(remove_index)
            candidates.pop(remove_index)
                
        return selected_indices


    def select_node(self, k=1,random_pick=False):
        """Select up to k non-fully expanded nodes with the highest UCT value.

        A node is fully expanded if either:
        1. It has reached the max number of children
        2. Any of its children have a Q value greater than its own
        """
        candidates = []
        to_consider = deque([self.root])

        while to_consider:
            current_node = to_consider.popleft()
            if not self.is_fully_expanded(current_node):
                candidates.append(current_node)
            to_consider.extend(current_node.children)

        if len(candidates) == 0:
            for node in to_consider:
                assert self.is_fully_expanded(node), "Not fully expanded"

        if not candidates:
            return None

        selected_nodes = []
        if random_pick:
            selected_nodes = random.sample(candidates, min(k, len(candidates)))
            return selected_nodes
        else:
            if self.selection_policy == SelectionPolicy.GREEDY:
                selected_nodes = sorted(candidates, key=self.uct, reverse=True)[:k]
            elif self.selection_policy == SelectionPolicy.IMPORTANCE_SAMPLING:
                uct_scores = [self.uct(node) for node in candidates]
                # 拷贝一份candidates，避免修改原列表
                candis = candidates.copy()
                selected_indices = self.weighted_sample_no_replacement(candis, uct_scores, k)
                selected_nodes = [candidates[i] for i in selected_indices]
            elif self.selection_policy == SelectionPolicy.PAIRWISE_IMPORTANCE_SAMPLING:
                uct_scores = [self.uct(node) for node in candidates]
                pairs = [
                    (i, j) for i in range(len(candidates)) for j in range(len(candidates))
                ]
                pair_weights = [
                    max(uct_scores[i], uct_scores[j]) -
                    min(uct_scores[i], uct_scores[j])
                    for i, j in pairs
                ]
                selected_pair_indices = random.choices(
                    range(len(pairs)), weights=pair_weights, k=min(k, len(pair_weights))
                )
                for pair_idx in selected_pair_indices:
                    selected_candidate_idx = max(
                        pairs[pair_idx], key=lambda x: uct_scores[x]
                    )
                    if candidates[selected_candidate_idx] not in selected_nodes:
                        selected_nodes.append(candidates[selected_candidate_idx])
                    if len(selected_nodes) >= k:
                        break
            else:
                raise ValueError(
                    f"Invalid selection policy: {self.selection_policy}")

            return selected_nodes

    def initialize(self, add_understanding=False):
        """Generate a zero-shot answer."""
        # init_prompt = QWEN_SYSTEM_PROMPT.format(
        # prompt=self.problem, examples=EXMAPLE_PROMPT)
        # print("concurrent_num:",self.concurrent_num)

        # if self.backbone == "glm":
        #     init_prompt = GLM_QA_PROMPT.format(
        #         prompt=self.problem, response="")
        # elif self.backbone == "qwen":
        #     init_prompt = QWEN_QA_PROMPT.format(
        #         prompt=self.problem, response="")
        #     # prompt=self.problem, example_answer=EXMAPLE_ANSWER, example_question=EXMAPLE_QUESTION)
        # else:
        #     raise NotImplementedError(
        #         f"Backbone {self.backbone} not implemented")
        
        # init_prompt = self.tokenize_fn([[self.problem],[None]],self.prompt_max_len, device="cpu")
        init_prompt = self.tokenize_fn([[self.problem],[None]],self.prompt_max_len, device="cpu",system_prompt=self.system_prompt)["input_ids"][0]

        self.root = MCTSNode(
            state=init_prompt,
            answer="",
            max_children=self.max_children,
            depth=0
        )
        self.depth_node_count[0] = 1  # Initialize for root node
    
    # @profile_with_time(output_dir="run_function_profile_yujiang")
    def run(self):
        self.initialize()
        node_number = 0
        leaf_num = 0
        terminal_flag = False
        start_time = time.time()

        while node_number < self.max_nodes and time.time() - start_time < self.max_time_use:
            nodes = self.select_node(k=self.concurrent_num,random_pick=self.random_pick)
            if not nodes:
                print("terminated because no node to expand")
                break

            try:
                results = self.expand(nodes)
            except Exception as e:
                print(f"Expansion failed with exception: {e}")
                break
            # results = self.expand(nodes)
            # print(results)

            self.total_token_num += results[1]

            for node, child_num in results[0]:
                if child_num == 0:
                    # print(f"Cannot expand the node {node}")
                    continue
                else:
                    node_number += child_num
                    self.depth_node_count[node.depth + 1] = self.depth_node_count.get(node.depth + 1, 0) + child_num
                    child_terminal_exits = False
                    for child in node.children:
                        is_terminated = child.terminal
                        if is_terminated and child.R > 0:
                            assert child.R == 1, "correct leaf reward is not 1"
                            child.main_chain = True
                            if self.backprop:
                                self.backpropagate(child, child.main_chain)
                        else:
                            if self.backprop:
                                self.backpropagate(child)
                        if is_terminated:
                            self.leaves.append(child)
                            # self.leaf_backpropagate(child)

                        if is_terminated and not child_terminal_exits:
                            leaf_num += 1
                            child_terminal_exits = True

            if leaf_num >= self.pass_k:
                print(f"terminated because reach {leaf_num} leaf nodes")
                terminal_flag = True
                break
        os.makedirs("/workspace/lurui/openrlhf-glm/logs/outputs", exist_ok=True)
        with open("/workspace/lurui/openrlhf-glm/logs/outputs/leaves.jsonl", "a") as f:
            #输出len(self.leaves)数量即可
            f.write(json.dumps({"leaf num": len(self.leaves)}))
            f.write("\n")
        
        if self.parent_shift:
            self.leaf_normalize(self.leaves)
            for leaf in self.leaves:
                self.leaf_backpropagate(leaf)
            self.select_terminal()
        else:
            self.select_terminal()

            for leaf in self.selected_terminals:
                self.leaf_backpropagate(leaf)

            if len(self.selected_terminals) > 1:
                self.leaf_normalize()
    
    def multi_language(self,text):
        """检查是否包含非英语内容（中文、日文、韩文、俄语）"""
        # 定义字符集范围
        chinese_range = re.compile(r'[\u4e00-\u9fff]')  # 中文
        japanese_range = re.compile(r'[\u3040-\u309f\u30a0-\u30ff]')  # 日文假名
        korean_range = re.compile(r'[\uac00-\ud7af]')  # 韩文
        russian_range = re.compile(r'[а-яА-ЯёЁ]')  # 俄语（包括大小写）
        # 检查是否包含非英语字符
        has_chinese = chinese_range.search(text)
        has_japanese = japanese_range.search(text)
        has_korean = korean_range.search(text)
        has_russian = russian_range.search(text)
        return has_chinese or has_japanese or has_korean or has_russian
        
    # @profile_with_time(output_dir="run_function_profile_yujiang")
    def expand(self, nodes):
        if len(nodes) == 0:
            return [], 0
        stops = get_stops()
        all_children_token_num = 0
        max_tokens_per_step = self.max_token_num
        max_attempts = 3
        token_threshold = 150
        children_map = {node: [] for node in nodes}  # 用于记录每个节点的孩子

        attempts = 0
        while attempts < max_attempts:
            prompts, node_prompts_map = [], []

            for node in nodes:
                num_current_children = len(children_map[node])
                prompts.extend([node.state] * (node.max_children - num_current_children))
                node_prompts_map.extend([node] * (node.max_children - num_current_children))

            if not prompts:
                break  # 如果没有需要生成的孩子，退出

            attempts += 1
            
            next_tokens = []
            next_strs = []
            if self.first_token_temperature and random.random() < 0.5:
                print("using first token tempeature")
                first_tokens = top_k_sampling(
                    llm = self.llms[0],
                    prompts = prompts,
                    top_p = self.top_p,
                    skip_special_tokens=False,
                    stops=stops,
                )
                full_strs = [self.detokenize_fn(first_token) for first_token in first_tokens]
                next_tokens = [random.choice(used_logprobs) for used_logprobs in first_tokens]
                next_strs = [self.detokenize_fn([next_token]) for next_token in next_tokens]
                # print("full_strs", full_strs)
                # with open("/workspace/lurui/openrlhf-mcts/data/first_tokens.jsonl", "a") as f:
                #     f.write(json.dumps({"prompts": prompts, "next_strs": full_strs}))
                #     f.write("\n")
                prompts = [prompt + [next_token] for prompt, next_token in zip(prompts, next_tokens)]

            responses_token, responses_str, finish_reasons, stop_tokens, token_nums = query_local_vllm_completions_ids(
                prompts,
                llm=self.llms[0],
                n=1,
                skip_special_tokens=True,
                max_tokens=max_tokens_per_step,
                stops=stops,
                temperature=self.temperature,
                top_p=self.top_p,
                model="glm",
                min_tokens = token_threshold
            )
            # print("stop_tokens", stop_tokens)

            for idx, (response_token_list,response_str_list, finish_reason_list, stop_token_list, token_num_list) in enumerate(zip(responses_token, responses_str, finish_reasons, stop_tokens, token_nums)):
                node = node_prompts_map[idx]
                response_token = response_token_list[0]
                response = response_str_list[0]
                finish_reason = finish_reason_list[0]
                stop_token = stop_token_list[0]
                if next_tokens:
                    assert len(next_strs) != 0 and len(next_tokens) != 0 and len(next_strs) == len(next_tokens), "next tokens and next strs should have the same length"
                    response = next_strs[idx] + response
                    
                    response_token = [next_tokens[idx]] + response_token
                if response is None:
                    print("response is None")
                    continue  # 如果响应为空或未结束，跳过

                action = response
                if not ((stop_token is None) or (stop_token in eos_tokens_set)):
                    # action += stop_token
                    stop_token_str = self.detokenize_fn([stop_token])[0]
                    action += stop_token_str

                new_action = action
                new_action_token = response_token

                # 过滤与当前 children_map[node] 中的项重复
                existing_actions = [
                    child.answer
                    for child in children_map[node]
                ]

                # if any(
                #     similarity(new_action, existing_action, split_chars=["\n\n", ". "])
                #     for existing_action in existing_actions
                # ):
                #     continue  # 如果新的动作与现有的孩子过于相似，跳过

                # if any(
                #     similarity_naive(new_action, existing_action)
                #     for existing_action in existing_actions
                # ):
                #     continue  # 如果新的动作与现有的孩子过于相似，跳过

                # expanded_state = node.state + response_token
                # expanded_state = torch.cat([node.state, response_token], dim=1)
                expanded_state = node.state + response_token
                new_aggregate_answer = node.aggregate_answer + new_action
                new_aggregate_answer_token = node.aggregate_answer_token + new_action_token
                all_children_token_num += token_num_list[0]  # 假设我们使用列表中的第一个标记数

                if (len(new_aggregate_answer_token) > self.max_token_num) or (find_repeated_patterns(new_aggregate_answer)):
                    repeat = True
                else:
                    repeat = False

                if (stop_token is None) or (stop_token in eos_tokens_set) or repeat:
                    if_finish = True
                else:
                    if_finish = False

                finished = self.judge_finished(
                    if_finish, node.depth + 1
                )
                if len(new_aggregate_answer_token) > self.max_token_num:
                    print("terminal because exceed max token num")
                elif find_repeated_patterns(new_aggregate_answer):
                    print("terminal because find repeated patterns")
                elif finished:
                    print("terminal because finished")
                if self.shallow_enwide:
                    print("shallow enwide")
                    max_children = max(node.max_children/2, self.min_children)
                else:
                    max_children = node.max_children
                child_node = MCTSNode(
                    state=expanded_state,
                    answer=new_action,
                    answer_token = new_action_token,
                    aggregate_answer=new_aggregate_answer,
                    aggregate_answer_token=new_aggregate_answer_token,
                    parent=node,
                    depth=node.depth + 1,
                    terminal=finished,
                    max_children=max_children,
                    repeat=repeat,
                    node_id = self.leaf_num_count
                )
                self.leaf_num_count += 1

                children_map[node].append(child_node)

            if all(len(children_map[node]) >= node.max_children for node in nodes):
                break  # 如果所有节点的孩子都生成完毕，退出
        # 输出children_map的长度之和
        # print("all childrens: ", sum(len(children) for children in children_map.values()))

        results = []

        for node, childrens in children_map.items():
            if self.random_pick:
                for i, child in enumerate(childrens):
                    if child.terminal:
                        child.R = self.self_evaluate(child, True)
                        child.value = child.R
            else:
                non_leaf_rewards = []
                non_leaf_indexes = []
                for i, child in enumerate(childrens):
                    if child.terminal:
                        child.R = self.self_evaluate(child, True)
                        child.value = child.R
                    else:
                        reward = self.self_evaluate(child, False)
                        non_leaf_rewards.append(reward)
                        non_leaf_indexes.append(i)

                if non_leaf_rewards:
                    non_leaf_rewards = np.array(non_leaf_rewards)
                    normalized_rewards = (non_leaf_rewards - np.mean(non_leaf_rewards)) / (np.std(non_leaf_rewards) + 1e-8)
                    # # 归一化到（0，1）之间
                    # normalized_rewards = normalized_rewards/2 + 0.5

                    for i, reward in zip(non_leaf_indexes, normalized_rewards):
                        childrens[i].R = reward
                        childrens[i].value = reward

            assert len(childrens) <= node.max_children, f"Too many children, {len(childrens)} > {node.max_children}"
            node.max_children = len(childrens)
            node.children = childrens

            results.append((node, len(node.children)))

        return results, all_children_token_num

    def print(self):
        print_tree(self.root)

    def judge_finished(self, is_stopped, depth):
        # find_answer = stop_token == "<|user|>"
        finished = is_stopped or depth > self.max_depth
        return finished

    def normalize_backprop(self):
        # 对所有叶子向上更新
        for node in self.selected_terminals:
            parent = node.parent
            while parent:
                parent.accumulated_value += node.accumulated_value
                parent.visited_terminal += 1
                if parent.visited_terminal == parent.terminal_in_subtree:
                    parent.accumulated_value = parent.accumulated_value / parent.terminal_in_subtree
                parent = parent.parent
        self.normalize_all_steps()

    def normalize_all_steps(self):
        # 从root开始遍历所有节点，对所有terminal_in_subtree！=0节点的accumulated_value进行归一化
        all_steps = []
        to_consider = deque([self.root])
        while to_consider:
            current_node = to_consider.popleft()
            if current_node.terminal_in_subtree != 0 or current_node.terminal:
                all_steps.append(current_node)
            to_consider.extend(current_node.children)

        print("all_step value",[node.accumulated_value for node in all_steps],len(all_steps))
        if self.step_level_norm:
            step_sum = 0
            step_num = 0
            for node in all_steps:
                step_sum += node.accumulated_value*node.terminal_in_subtree
                step_num += node.terminal_in_subtree
            if step_num == 0:
                mean = 0
            else:
                mean = step_sum/step_num
            print("mean:", mean,step_sum,step_num)
            for node in all_steps:
                node.accumulated_value = node.accumulated_value - mean
        else:
            print("token level normalization")
            step_sum = 0
            step_num = 0
            for node in all_steps:
                step_sum += node.accumulated_value*node.terminal_in_subtree*len(node.answer_token)
                step_num += node.terminal_in_subtree*len(node.answer_token)
            if step_num == 0:
                mean = 0
            else:
                mean = step_sum/step_num
            print("mean:", mean)
            for node in all_steps:
                node.accumulated_value = node.accumulated_value - mean
                

    def leaf_normalize(self,nodes):
        leaf_correctness = [leaf.accumulated_value for leaf in nodes]
        print("leaf_correctness",leaf_correctness)
        _sum = sum(leaf_correctness)
        num = len(leaf_correctness) - 1
        if num == 0:
            return
        else:
            mean = [(_sum - leaf_correctness[i]) / num for i in range(len(leaf_correctness))]
            for i, leaf in enumerate(nodes):
                leaf.accumulated_value = leaf.accumulated_value - mean[i]
        # self.normalize_backprop()

    # def select_terminal(self):
    #     # 从self.leaves中选择self.path_num 个叶子节点, 尽可能挑选同样数量的正确和错误的叶子，同一个父亲的叶子如果同对同错，只能选一个
    #     parent_to_children = {}
        
    #     # 分类出正确和错误的叶子节点
    #     correct_leaves = []
    #     incorrect_leaves = []

    #     if len(self.leaves) < 3:
    #         return False
        
    #     for leaf in self.leaves:
    #         parent = leaf.parent
    #         if parent not in parent_to_children.keys():
    #             parent_to_children[parent] = {'correct': [], 'incorrect': []}
                    
    #         if leaf.main_chain > 0:
    #             parent_to_children[parent]['correct'].append(leaf)
    #             correct_leaves.append(leaf)
    #         else:
    #             parent_to_children[parent]['incorrect'].append(leaf)
    #             incorrect_leaves.append(leaf)
        
    #     total_sum = sum([2 if (len(children["correct"]) and len(children["incorrect"])) else 1 for children in parent_to_children.values()])

    #     if total_sum == self.path_num:
    #         selected_terminals = []
    #         # 为每个父节点选择一个正确的孩子和一个错误的孩子（如果有的话）
    #         for parent, children in parent_to_children.items():
    #             if children['correct']:
    #                 selected_terminals.append(random.choice(children['correct']))
    #             if children['incorrect']:
    #                 selected_terminals.append(random.choice(children['incorrect']))
    #         self.selected_terminals = selected_terminals
    #         return True

    #     elif total_sum > self.path_num:
    #         selected_terminals_correct = []
    #         selected_terminals_incorrect = []

    #         # 为每个父节点选择一个正确的孩子和一个错误的孩子（如果有的话）
    #         for parent, children in parent_to_children.items():
    #             if children['correct']:
    #                 selected_terminals_correct.append(random.choice(children['correct']))
    #             if children['incorrect']:
    #                 selected_terminals_incorrect.append(random.choice(children['incorrect']))
    #         if len(selected_terminals_correct) <= self.path_num/2:
    #             num_correct = len(selected_terminals_correct)
    #             num_incorrect = self.path_num - num_correct
    #         elif len(selected_terminals_incorrect) <= self.path_num/2:
    #             num_incorrect = len(selected_terminals_incorrect)
    #             num_correct = self.path_num - num_incorrect
    #         else:
    #             num_correct = self.path_num//2
    #             num_incorrect = self.path_num - num_correct
            
    #         selected_terminals = random.sample(selected_terminals_correct, num_correct) + random.sample(selected_terminals_incorrect, num_incorrect)
    #         self.selected_terminals = selected_terminals
    #         return True
            
    #     else:
    #         selected_terminals = []
    #         # 为每个父节点选择一个正确的孩子和一个错误的孩子（如果有的话）
    #         for parent, children in parent_to_children.items():
    #             if children['correct']:
    #                 selected_terminals.append(random.choice(children['correct']))
    #                 children['correct'].remove(selected_terminals[-1])
    #             if children['incorrect']:
    #                 selected_terminals.append(random.choice(children['incorrect']))
    #                 children['incorrect'].remove(selected_terminals[-1])
    #         if len(selected_terminals) < self.path_num:
    #             k = 0
    #             while len(selected_terminals) < self.path_num:
    #                 added_in_this_round = False
    #                 for parent, children in parent_to_children.items():
    #                     if k < len(children['correct']):
    #                         selected_terminals.append(children['correct'][k])
    #                         added_in_this_round = True
    #                         if len(selected_terminals) >= self.path_num:
    #                             break
    #                     elif k < len(children['incorrect']):
    #                         selected_terminals.append(children['incorrect'][k])
    #                         added_in_this_round = True
    #                         if len(selected_terminals) >= self.path_num:
    #                             break
    #                 if not added_in_this_round:
    #                     break  # 如果这一轮没有添加新路径，则不能继续补全
    #                 k += 1
    #         while len(selected_terminals) < self.path_num:
    #             assert len(selected_terminals) > 0, "Not enough terminal nodes"
    #             for node in selected_terminals:
    #                 selected_terminals.append(node)
    #                 if len(selected_terminals) >= self.path_num:
    #                     break
    #         self.selected_terminals = selected_terminals
    #         return True

    def select_terminal(self):
        # 从self.leaves中选择self.path_num 个叶子节点, 尽可能挑选同样数量的正确和错误的叶子，同一个父亲的叶子如果同对同错，只能选一个
        parent_to_children = {}

        if len(self.leaves) < 3:
            return False
        
        correct_leaf_parent = None
        correct_leaf = None
        for leaf in self.leaves:
            if leaf.main_chain:
                correct_leaf_parent = leaf.parent
                correct_leaf = leaf
            parent = leaf.parent
            if parent not in parent_to_children.keys():
                parent_to_children[parent] = []
            parent_to_children[parent].append(leaf)
        
        total_sum = len(parent_to_children.keys())
        if correct_leaf_parent is not None:
            assert correct_leaf is not None, "correct leaf is None"
            print("got correct leaf!")
        
        if not self.select_correct_leaf:
            print("do not manually select correct leaf")
            correct_leaf = None
            correct_leaf_parent = None

        if total_sum == self.path_num:
            if correct_leaf is None:
                selected_terminals = []
                # 为每个父节点选择一个孩子
                for parent, children in parent_to_children.items():
                    selected_terminals.append(random.choice(children))
                self.selected_terminals = selected_terminals
                return True
            else:
                selected_terminals = []
                # 为每个父节点选择一个孩子
                for parent, children in parent_to_children.items():
                    if parent == correct_leaf_parent:
                        selected_terminals.append(correct_leaf)
                    else:
                        selected_terminals.append(random.choice(children))
                self.selected_terminals = selected_terminals
                return True

        elif total_sum > self.path_num:
            if correct_leaf is None:
                # 首先随机选self.path_num个父节点
                selected_parents = random.sample(set(parent_to_children.keys()), self.path_num)
                selected_terminals = []
                for parent in selected_parents:
                    selected_terminals.append(random.choice(parent_to_children[parent]))
                self.selected_terminals = selected_terminals
                return True
            else:
                other_parents = [parent for parent in parent_to_children.keys() if parent != correct_leaf_parent]
                selected_parents = random.sample(other_parents, self.path_num - 1)
                selected_terminals = [correct_leaf]
                for parent in selected_parents:
                    selected_terminals.append(random.choice(parent_to_children[parent]))
                self.selected_terminals = selected_terminals
                return True     
        else:
            if correct_leaf is None:
                selected_terminals = []
                # 为每个父节点选择一个正确的孩子和一个错误的孩子（如果有的话）
                for parent, children in parent_to_children.items():
                    selected_terminals.append(random.choice(children))
                if len(selected_terminals) < self.path_num:
                    k = 0
                    while len(selected_terminals) < self.path_num:
                        added_in_this_round = False
                        for parent, children in parent_to_children.items():
                            if k < len(children) and children[k] not in selected_terminals:
                                selected_terminals.append(children[k])
                                added_in_this_round = True
                                if len(selected_terminals) >= self.path_num:
                                    break     
                        if not added_in_this_round:
                            break  # 如果这一轮没有添加新路径，则不能继续补全
                        k += 1
                while len(selected_terminals) < self.path_num:
                    assert len(selected_terminals) > 0, "Not enough terminal nodes"
                    # 把selected_terminals shuffle一下，然后再从头开始添加
                    random.shuffle(selected_terminals)
                    for node in selected_terminals:
                        selected_terminals.append(node)
                        if len(selected_terminals) >= self.path_num:
                            break
                self.selected_terminals = selected_terminals
                return True
            else:
                selected_terminals = []
                for parent, children in parent_to_children.items():
                    if parent == correct_leaf_parent:
                        selected_terminals.append(correct_leaf)
                    else:
                        selected_terminals.append(random.choice(children))
                if len(selected_terminals) < self.path_num:
                    k = 0
                    while len(selected_terminals) < self.path_num:
                        added_in_this_round = False
                        for parent, children in parent_to_children.items():
                            if k < len(children) and children[k] not in selected_terminals:
                                selected_terminals.append(children[k])
                                added_in_this_round = True
                                if len(selected_terminals) >= self.path_num:
                                    break     
                        if not added_in_this_round:
                            break  # 如果这一轮没有添加新路径，则不能继续补全
                        k += 1
                while len(selected_terminals) < self.path_num:
                    assert len(selected_terminals) > 0, "Not enough terminal nodes"
                    # 把selected_terminals shuffle一下，然后再从头开始添加
                    random.shuffle(selected_terminals)
                    for node in selected_terminals:
                        selected_terminals.append(node)
                        if len(selected_terminals) >= self.path_num:
                            break
                self.selected_terminals = selected_terminals
                return True

                
###########################
# Functions for saving and loading the tree
###########################

def print_tree(node: MCTSNode | None, level: int = 0):
    if node is None:
        return
    indent = " " * level * 2
    node_str = repr(node)
    for line in node_str.split("\n"):
        print(indent + line)
    for child in node.children:
        print_tree(child, level + 1)


def convert_to_json(node: MCTSNode):
    if not node.children:
        return {
            "answer": node.answer,
            "aggregate_answer": node.aggregate_answer,
            "value": node.value,
            "R" : node.R,
            "visits": node.visits,
            "reward_samples": node.reward_samples,
            "depth": node.depth,
            "main_chain": node.main_chain,
            "terminal": node.terminal,
            # "selected_terminal": node.selected_terminal,
            "terminal_in_subtree": node.terminal_in_subtree,
            "correct_terminal_in_subtree": node.correct_terminal_in_subtree,
            "accumulated_value": node.accumulated_value,
            "visited_terminal": node.visited_terminal,
            "node_id": node.node_id,
            "max_children": node.max_children
        }
    else:
        return {
            "answer": node.answer,
            "aggregate_answer": node.aggregate_answer,
            "value": node.value,
            "R" : node.R,
            "visits": node.visits,
            "reward_samples": node.reward_samples,
            "depth": node.depth,
            "main_chain": node.main_chain,
            "terminal": node.terminal,
            # "selected_terminal": node.selected_terminal,
            "children": [convert_to_json(child) for child in node.children],
            "terminal_in_subtree": node.terminal_in_subtree,
            "correct_terminal_in_subtree": node.correct_terminal_in_subtree,
            "accumulated_value": node.accumulated_value,
            "visited_terminal": node.visited_terminal,
            "node_id": node.node_id,
            "max_children": node.max_children
        }


def build_tree_based_on_json(json_data):
    if "children" in json_data:
        node = MCTSNode(**json_data)
        node.children = [build_tree_based_on_json(
            child) for child in json_data["children"]]
        return node
    else:
        return MCTSNode(**json_data)

###########################
# multiprocess workers
###########################

def chain_worker(
    item,
    llm,
    init_prompt,
    prompt_key="problem",
    answer_key="golden_answer",
    args=None,
    first_token_temperature=0,
    detokenize_fn=None,
):
    pass_k = args["path_num"]
    stops = [151336, 151329,151338]
    max_attempts = 3
    attempts = 0
    paths = []
    while attempts < max_attempts:
        prompts = [init_prompt] * (pass_k - len(paths))
        if not prompts:
            break  # 如果没有需要生成的孩子，退出

        attempts += 1
        next_tokens = []
        next_strs = []
        if first_token_temperature:
            print("using first token tempeature")
            first_tokens = top_k_sampling(
                llm = llm,
                prompts = prompts,
                top_p = self.top_p,
                skip_special_tokens=False,
                stops=stops,
            )
            full_strs = [detokenize_fn(first_token) for first_token in first_tokens]
            next_tokens = [random.choice(used_logprobs) for used_logprobs in first_tokens]
            next_strs = [detokenize_fn([next_token]) for next_token in next_tokens]
            print("full_strs", full_strs)
            # with open("/workspace/lurui/openrlhf-mcts/data/first_tokens.jsonl", "a") as f:
            #     f.write(json.dumps({"prompts": prompts, "next_strs": full_strs}))
            #     f.write("\n")
            prompts = [prompt + [next_token] for prompt, next_token in zip(prompts, next_tokens)]

        responses_token, responses_str, finish_reasons, stop_tokens, token_nums = query_local_vllm_completions_ids(
            prompts,
            llm=llm,
            n=1,
            skip_special_tokens=True,
            max_tokens=4096,
            stops=stops,
            temperature=args["temperature"],
            top_p=args["top_p"],
            model="glm",
            min_tokens = 0
        )
        # print("stop_tokens", stop_tokens)

        for idx, (response_token_list,response_str_list, finish_reason_list, stop_token_list, token_num_list) in enumerate(zip(responses_token, responses_str, finish_reasons, stop_tokens, token_nums)):
            response_token = response_token_list[0]
            response = response_str_list[0]
            finish_reason = finish_reason_list[0]
            stop_token = stop_token_list[0]
            
            if next_tokens:
                assert len(next_strs) != 0 and len(next_tokens) != 0 and len(next_strs) == len(next_tokens), "next tokens and next strs should have the same length"
                response = next_strs[idx] + response
                response_token = [next_tokens[idx]] + response_token

            if (attempts != max_attempts) and (response is None):
                continue  # 如果响应为空或未结束，跳过

            action = response if response is not None else ""
            token_action = response_token if response_token is not None else []

            paths.append({"answer": action, "token_answer": token_action})
            if len(paths) >= pass_k:
                break

    assert len(paths) == pass_k, f"Failed to generate {pass_k} paths"

    results = []
    for path in paths:
        path["reward"] = get_qwen_remote_reward_model_value(urls= RM_URLS, question = item[prompt_key], response = path["answer"])
        path["pass_ratio"] = check_result(item[args["prompt_key"]], path["answer"], item[answer_key],checker_urls=EVALUATOR_URLS,extractor_urls=EXTRACTOR_URLS)[-1]
        results.append(path["pass_ratio"])
    _sum = sum(results)
    num = len(results) - 1
    if num == 0:
        return paths
    else:
        mean = [(_sum - results[i]) / num for i in range(len(results))]
        for i, path in enumerate(paths):
            path["value"] = path["pass_ratio"] - mean[i]
        paths = [[path] for path in paths]
        return paths

def mcts_worker(
    item,
    llm, 
    tokenize_fn,
    detokenize_fn,
    prompt_key="problem",
    answer_key="golden_answer",
    args=None,
    system_prompt=None,
):
    # 随机 sleep 一段时间，一分钟以内
    # time.sleep(random.randint(0, 60))
    pid = os.getpid()
    problem = item[prompt_key]
    answer = item[answer_key]
    mcts = MCTSr(
        temperature=args["temperature"],
        top_p=args["top_p"],
        problem=problem,
        golden_answer=answer,
        max_nodes=args["max_nodes"],
        exploration_constant=args["exploration_constant"],
        selection_policy=SelectionPolicy.IMPORTANCE_SAMPLING,
        backbone=args["backbone"],
        max_children=args["max_children"],
        min_children=args["min_children"],
        pass_k=args["pass_k"],
        max_depth=args["max_depth"],
        backprop=args["backprop"],
        max_node_per_depth = args["max_node_per_depth"],
        first_token_temperature=args["first_token_temperature"],
        look_ahead=args["look_ahead"],
        llms=[llm],
        tokenize_fn = tokenize_fn,
        detokenize_fn = detokenize_fn,
        concurrent_num=args["concurrent_num"],
        path_num = args["path_num"],
        prompt_max_len = args["prompt_max_len"],
        max_token_num = args["max_token_num"],
        max_time_use = args["max_time_use"],
        step_level_norm = args["step_level_norm"],
        random_pick = args["random_pick"],
        parent_shift = args["parent_shift"],
        use_orm_reward = args["use_orm_reward"],
        select_correct_leaf = args["select_correct_leaf"],
        use_chain_reward = args["use_chain_reward"],
        use_state_value_reward = args["use_state_value_reward"],
        use_pure_RM = args["use_pure_RM"],
        use_pure_binary = args["use_pure_binary"],
        shallow_enwide = args["shallow_enwide"],
        system_prompt=system_prompt,
    )
    # print(mcts.max_children)
    start_time = time.time()
    mcts.run()
    try:
        # mcts.run()
        root = mcts.root
        # with open("/workspace/lurui/openrlhf-glm/logs/outputs/trees_vine.jsonl", "a",encoding="utf-8") as f:
        # with open("/workspace/lurui/openrlhf-mcts/data/paths.jsonl", "a",encoding="utf-8") as f:
        #     tree_json = convert_to_json(root)
        #     tree_json["random_pick"] = args["random_pick"]
        #     # tree_json["time_used"] = time_used
        #     json.dump(tree_json, f)
        #     f.write("\n")
        # print("selected_terminals",mcts.selected_terminals[0])
        paths = gather_paths(mcts.selected_terminals,args["path_num"],parent_shift = mcts.parent_shift,use_orm_reward = mcts.use_orm_reward,use_chain_reward = mcts.use_chain_reward,step_level_norm = mcts.step_level_norm,use_state_value_reward = mcts.use_state_value_reward)
        time_used = time.time() - start_time
        pass_num = pass_rate(paths)
        os.makedirs("/workspace/lurui/openrlhf-glm/logs/outputs", exist_ok=True)
        with open("/workspace/lurui/openrlhf-glm/logs/outputs/trees_vine.jsonl", "a",encoding="utf-8") as f:
        # with open("/workspace/lurui/openrlhf-mcts/data/paths.jsonl", "a",encoding="utf-8") as f:
            tree_json = convert_to_json(root)
            tree_json["random_pick"] = args["random_pick"]
            tree_json["time_used"] = time_used
            tree_json["args"] = args
            tree_json["total_nodes"] = mcts.leaf_num_count
            tree_json["total_token_num"] = mcts.total_token_num
            tree_json["pass_num"] = pass_num
            json.dump(tree_json, f)
            f.write("\n")
    except Exception as e:
        # print(f"Error in MCTS: {e}")
        os.makedirs("/workspace/lurui/openrlhf-glm/logs/outputs", exist_ok=True)
        with open("/workspace/lurui/openrlhf-glm/logs/outputs/error.log", "a") as f:
            f.write(f"Error in MCTS: {e}")
        paths = None
        time_used = time.time() - start_time
    if paths is None:
        os.makedirs("/workspace/lurui/openrlhf-glm/logs/outputs", exist_ok=True)
        with open("/workspace/lurui/openrlhf-glm/logs/outputs/response_type.jsonl", "a",encoding="utf-8") as f:
            f.write("use chain_worker\n")
        # init_prompt = tokenize_fn([[problem],[None]],args["prompt_max_len"], device="cpu",system_prompt=system_prompt)
        init_prompt = tokenize_fn([[problem],[None]],args["prompt_max_len"], device="cpu",system_prompt=system_prompt)["input_ids"][0].tolist()
        paths = chain_worker(item, llm, init_prompt, prompt_key, answer_key, args)
        return paths, init_prompt
    else:
        with open("/workspace/lurui/openrlhf-glm/logs/outputs/response_type.jsonl", "a",encoding="utf-8") as f:
            f.write("use mcts_worker\n")
        return paths,root.state

# def get_stops():
#     return ["<|user|>", "<|endoftext|>", "<|observation|>","\n\n"]
def get_stops():
    return [271, 151336, 151329,151338, 2533, 382, 1447, 21467, 692]

def normalize_selected_terminals(selected_terminals: list[MCTSNode]):
    leaf_orm_value = [leaf.accumulated_value for leaf in selected_terminals]
    _sum = sum(leaf_orm_value)
    num = len(leaf_orm_value) - 1
    if num == 0:
        return leaf_orm_value
    else:
        mean = [(_sum - leaf_orm_value[i]) / num for i in range(len(leaf_orm_value))]
        orm_normalized = [leaf_orm_value[i] - mean[i] for i in range(len(leaf_orm_value))]
        return orm_normalized

def fill_in_paths(paths):
    # 对于每个路径，如果存在"value"=0，就用他的前一个节点的"value"填充
    for path in paths:
        for i in range(1,len(path)):
            epsilon = 1e-8
            if abs(path[i]["value"]) < epsilon: 
            # if path[i]["value"] == 0:
                assert i > 0, "value=0 in the first node"
                assert path[i]["value"] < epsilon  and path[i]["value"] > -epsilon, "value is not 0"
                # print("fill in value",path[i-1]["value"])
                path[i]["value"] = path[i-1]["value"]
    return paths

def normalize_all_paths(paths,step_level_norm = False):
    # 对所有路径进行归一化
    if step_level_norm:
        state_value_sum = 0
        state_value_num = 0
        for path in paths:
            for node in path:
                state_value_sum += node["state_value"]
                state_value_num += 1
        if state_value_num == 0:
            mean = 0
        else:
            mean = state_value_sum/state_value_num
        for path in paths:
            for node in path:
                node["state_value"] = node["state_value"] - mean
        return paths
    else:
        state_value_sum = 0
        state_value_num = 0
        for path in paths:
            for node in path:
                state_value_sum += node["state_value"]*len(node["token_answer"])
                state_value_num += len(node["token_answer"])
        if state_value_num == 0:
            mean = 0
        else:
            mean = state_value_sum/state_value_num
        for path in paths:
            for node in path:
                node["state_value"] = node["state_value"] - mean
        return paths

def path_from_root_to_node(node: MCTSNode,parent_shift:bool = False) -> List[Dict[str, Any]]:
    if parent_shift:
        path = []
        while node.parent is not None:
            parent_value = node.parent.accumulated_value/node.parent.terminal_in_subtree
            child_value = node.accumulated_value/node.terminal_in_subtree
            if node.terminal:
                assert node.terminal_in_subtree == 1, "terminal_in_subtree is not 1"
            # print("pass_ratio",parent_value,child_value)
            path.append({'answer': node.answer, 'token_answer':node.answer_token,'reward': node.value,"pass_ratio":node.correct_terminal_in_subtree/node.terminal_in_subtree,"value":child_value - parent_value,"state_value":child_value})
            # path.append({'answer': node.answer, 'token_answer':node.answer_token,'reward': node.value,"pass_ratio":node.correct_terminal_in_subtree/node.terminal_in_subtree,"value":child_value})
            node = node.parent
        return path[::-1]
    else:
        path = []
        while node is not None:
            print("pass_ratio",node.correct_terminal_in_subtree,node.terminal_in_subtree,node.accumulated_value)
            path.append({'answer': node.answer, 'token_answer':node.answer_token,'reward': node.value,"pass_ratio":node.correct_terminal_in_subtree/node.terminal_in_subtree,"value":node.accumulated_value})
            node = node.parent
        return path[::-1][1:]

def gather_paths(selected_terminals: list[MCTSNode], pass_k: int,parent_shift:bool = False,use_orm_reward:bool = False,use_chain_reward:bool=False,step_level_norm:bool=False,use_state_value_reward:bool=False) -> List[List[Dict[str, Any]]]:
    paths = []
    if len(selected_terminals) < pass_k:
        return None
    # terminal_values = normalize_selected_terminals(selected_terminals)
    terminal_values = [leaf.accumulated_value for leaf in selected_terminals]
    # 添加 selected_terminal 的叶子节点路径
    for terminal_node in selected_terminals:
        paths.append(path_from_root_to_node(terminal_node,parent_shift))
    assert len(paths) == pass_k, f"Failed to generate {pass_k} paths,{len(paths)} instead"
    paths = fill_in_paths(paths)
    if use_chain_reward:
        print("use chain reward in mcts!!")
        terminal_values = normalize_selected_terminals(selected_terminals)
        for path in paths:
            for node in path:
                node["value"] = terminal_values[paths.index(path)]
    elif use_orm_reward:
        print("use orm reward in mcts!!")
        terminal_values = normalize_selected_terminals(selected_terminals)
        for path in paths:
            for node in path:
                node["value"] = (node["value"] + terminal_values[paths.index(path)])/2
    elif use_state_value_reward:
        print("use state value reward in mcts!!")
        # paths = normalize_all_paths(paths,step_level_norm)
        for path in paths:
            for node in path:
                node["value"] = (node["value"] + node["state_value"])/2
    else:
        print("use pure advantage in mcts!!")
    print("path num",len(paths))
    return paths

def pass_rate(paths):
    pass_num = 0
    for path in paths:
        if path[-1]["pass_ratio"] ==1 :
            pass_num += 1
    return pass_num

# 封装为一个函数,输入为item,输出为paths
def parallel_mcts(item, llm, tokenize_fn, detokenize_fn, args,system_prompt=None):
    return mcts_worker(item, llm, tokenize_fn, detokenize_fn, args["prompt_key"], args["answer_key"], args,system_prompt)