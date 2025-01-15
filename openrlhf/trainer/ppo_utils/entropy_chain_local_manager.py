import time
import math
from typing import List, Dict, Any, Callable
try:
    from openrlhf.trainer.ppo_utils.tree_node import TreeNode, build_into_tree_format
    from openrlhf.trainer.ppo_utils.parallel_mcts import gather_paths
    from openrlhf.trainer.ppo_utils.evaluation import (
        check_result,
        query_local_vllm_completions_with_logprobs,
        query_local_vllm_ids_with_logprobs,
        GLM_QA_PROMPT,
        get_qwen_remote_reward_model_value
    )
except:
    from tree_node import TreeNode, build_into_tree_format
    from parallel_mcts import gather_paths
    from evaluation import (
        check_result,
        query_local_vllm_completions_with_logprobs,
        query_local_vllm_ids_with_logprobs,
        GLM_QA_PROMPT,
        get_qwen_remote_reward_model_value
    )
from IPython import embed

from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple


class EntropyGuidedChainLocalManager:
    def __init__(
        self,
        args: Dict[str, Any],
        llm: Any,
        encode_fn: Callable,
        decode_fn: Callable,
        evaluator_urls: List[str],
        extractor_urls: List[str],
        eos_tokens_set: List[int]
    ):
        """
        初始化管理器。

        :param args: 参数字典，包含 m, n, l 等。
        :param policy_urls: 模型接口 URL 列表。
        :param evaluator_urls: 评估器接口 URL 列表。
        :param eos_tokens_set: 结束 token 集合。
        """
        self.args = args
        self.llm = llm
        self.evaluator_urls = evaluator_urls
        self.extractor_urls = extractor_urls
        self.eos_tokens_set = eos_tokens_set
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.paths: Dict[str, Any] = {
            "M": args["m"],
            "N": args["n"],
            "L": args["l"],
            "T": args["t"],
            "pass_k_result": [],
            "time_use": 0,
            "tree_structures": []
        }

    def serialize_tree(self, node: TreeNode) -> Dict[str, Any]:
        """
        序列化树节点，用于存储。

        :param node: TreeNode 对象。
        :return: 字典形式的树结构。
        """
        return {
            'token_list': node.token_list,
            'log_prob_list': node.log_prob_list,
            'is_end': node.is_end,
            'children': [self.serialize_tree(child) for child in node.child_nodes]
        }

    def evaluate_node(self, args: Dict[str, Any], problem_str: str, node: TreeNode) -> Tuple[float, float]:
        """评估单个节点的分数

        Returns:
            Tuple[float, float]: (binary_score, final_score)
        """
        if node.is_end and node.finish_reason == "stop":
            binary_score = check_result(
                problem_str,
                node.total_str,
                self.answer_str,  # 需要将answer_str作为类属性存储
                checker_urls=self.evaluator_urls,
                extractor_urls=self.extractor_urls
            )[-1]
        else:
            binary_score = 0

        if args["use_pure_binary"]:
            return binary_score, binary_score

        # Get reward model score
        value = get_qwen_remote_reward_model_value(
            urls=args["entropy_rm_urls"],
            question=problem_str,
            response=node.total_str
        )

        if args["use_pure_RM"]:
            a, b = 0.5, -2.898
            x = a * (value - b)
            final_score = 1 / (1 + math.exp(-x))
        else:
            sigmoid_value = 1 / (1 + math.exp(-value))
            final_score = binary_score + 0.5 * sigmoid_value

        return binary_score, final_score

    def evaluate_trees(self, problem_str: str, answer_str: str, args: Dict[str, Any]) -> List[float]:
        """并发评估所有树中的节点"""
        self.answer_str = answer_str  # 临时存储供evaluate_node使用

        # 收集所有需要评估的节点
        evaluation_tasks = [
            (args, problem_str, node)
            for tree_list in self.tree_lists
            for node in tree_list
        ]

        # 使用线程池并发评估
        # with ThreadPoolExecutor(max_workers=min(32, len(evaluation_tasks))) as executor:
        with ThreadPoolExecutor(max_workers=min(8, len(evaluation_tasks))) as executor:
            results = list(executor.map(
                lambda params: self.evaluate_node(*params),
                evaluation_tasks
            ))

        # 更新节点分数并收集结果
        pass_k_result = []
        for (binary_score, final_score), (_, _, node) in zip(results, evaluation_tasks):
            node.binary_score = binary_score
            node.score = final_score
            pass_k_result.append(binary_score)

            if args["use_pure_RM"]:
                print("entropy rm_score", final_score)

        return pass_k_result

    def entropy_guided_chain(
        self,
        problem_str: str,
        answer_str: str,
        args: Dict[str, Any] = None,
        system_prompt=None,
        max_length=7144,
    ) -> Dict[str, Any]:
        """
        熵引导的链式推理。

        :param problem_str: 问题字符串。
        :param answer_str: 标准答案字符串。
        :return: 存储路径和结果的字典。
        """
        # init_prompt_with_template = GLM_QA_PROMPT.format(
        #     prompt=problem_str, response=""
        # )

        M = self.args["m"]
        N = self.args["n"]
        L = self.args["l"]
        T = self.args['t']

        init_prompt_ids_with_template = self.encode_fn(
            [[problem_str], [None]], 1024, device="cpu", system_prompt=system_prompt
        )["input_ids"][0].tolist()

        print(init_prompt_ids_with_template)

        paths = self.paths

        self.paths['init_prompt_ids_with_template'] = init_prompt_ids_with_template

        time_start = time.time()

        # 初始化 M 棵树
        self.tree_lists = []
        initial_prompt_ids = [init_prompt_ids_with_template] * M

        # 获取初始推理结果
        # initial_results = query_local_vllm_completions_with_logprobs(
        for _ in range(4):
            initial_results = query_local_vllm_ids_with_logprobs(
                initial_prompt_ids,
                llm=self.llm,
                skip_special_tokens=False,
                max_tokens=max_length,
                stops=self.eos_tokens_set,
                temperature=self.args["temperature"],
                top_p=self.args["top_p"],
            )
            if initial_results is None or initial_results[0] is None:
                continue
            break
        

        for idx, (content_token_ids, _, finish_reason, _, log_probs) in enumerate(zip(*initial_results)):
            root_node = TreeNode(
                tree_idx=idx,
                node_idx=0,
                decode_fn=self.decode_fn,
                token_id_list=content_token_ids,
                log_prob_list=log_probs,
                is_end=True,
                finish_reason=finish_reason
            )
            self.tree_lists.append([root_node])

        # 迭代扩展树
        for iteration in range(L):
            # print(f"第 {iteration + 1}/{L} 轮迭代")

            # 收集所有可扩展节点的熵 token 索引
            expansion_tasks = []
            for tree_idx, tree_list in enumerate(self.tree_lists):
                # 先在每个 Node 中取出 top-N 节点
                tree_entropy_tokens = []
                for node_idx, node in enumerate(tree_list):
                    if not all(node.mask):  # 节点未被完全mask
                        entropy_tokens = node.get_max_entropy_tokens(top_n=N)
                        for token_idx in entropy_tokens:
                            # 存储 (熵值, tree_idx, node_idx, node, token_idx)
                            entropy_value = - \
                                node.log_prob_list[token_idx]  # 负对数概率作为熵
                            tree_entropy_tokens.append(
                                (entropy_value, tree_idx,
                                 node_idx, node, token_idx)
                            )

                # 因为是同一道题目的，所以不需要考虑跨题目的熵值
                # 从中选择 top-N 个节点作为扩展任务
                tree_entropy_tokens.sort(reverse=True)  # 按熵值降序排序
                expansion_tasks.extend([
                    (tree_idx, node_idx, node, token_idx)
                    for _, tree_idx, node_idx, node, token_idx in tree_entropy_tokens[:N]
                ])

            if not expansion_tasks:
                print("没有可扩展的节点，提前终止迭代。")
                break

            # 准备推理
            m_tree_top_n_prompt_ids = []
            task_mapping = {}
            for i, (tree_idx, node_idx, node, split_idx) in enumerate(expansion_tasks * T):
                prefix_ids = node.get_prefix_ids(split_idx)
                prompt_ids = init_prompt_ids_with_template + prefix_ids
                m_tree_top_n_prompt_ids.append(prompt_ids)
                task_mapping[i] = (tree_idx, node_idx, node, split_idx)

            # 批量执行推理
            inference_results = query_local_vllm_ids_with_logprobs(
                m_tree_top_n_prompt_ids,
                llm=self.llm,
                skip_special_tokens=False,
                max_tokens=max_length,
                stops=self.eos_tokens_set,
                temperature=self.args["temperature"],
                top_p=self.args["top_p"],
            )
            if inference_results is None or inference_results[0] is None:
                continue

            # 处理结果，更新树结构
            for i, (content_token_ids, _, finish_reason, _, log_probs) in enumerate(zip(*inference_results)):
                tree_idx, node_idx, parent_node, split_idx = task_mapping[i]

                # 在 split_idx 处分裂当前节点
                new_node = TreeNode(
                    tree_idx=tree_idx,
                    node_idx=len(self.tree_lists[tree_idx]),
                    token_id_list=content_token_ids,
                    decode_fn=self.decode_fn,
                    log_prob_list=log_probs,
                    is_end=True,
                    parent_node=parent_node,
                    parent_node_idx=node_idx,
                    parent_node_split_idx=split_idx,
                    finish_reason=finish_reason
                )

                # 建立父子关系
                parent_node.add_child(new_node, split_idx)

                # 将新节点添加到对应的树列表中
                self.tree_lists[tree_idx].append(new_node)

        eval_time_start = time.time()

        # 评估结果
        # pass_k_result = []
        # for tree_list in self.tree_lists:
        #     for node in tree_list:
        #         if node.is_end and node.finish_reason == "stop":
        #             response_str = node.total_str
        #             # response_str = response_str.split("<|user|>")[0]
        #             score = check_result(
        #                 problem_str,
        #                 response_str,
        #                 answer_str,
        #                 checker_urls=self.evaluator_urls,
        #                 extractor_urls=self.extractor_urls
        #             )[-1]
        #             pass_k_result.append(score)
        #             node.binary_score = score
        #         else:
        #             pass_k_result.append(0)
        #             node.binary_score = 0
        #         if args["use_pure_binary"]:
        #             node.score = node.binary_score
        #         else:
        #             value = get_qwen_remote_reward_model_value(
        #                 urls=args["entropy_rm_urls"], question=problem_str, response=node.total_str)
        #             if args["use_pure_RM"]:
        #                 a = 0.5
        #                 b = -2.898
        #                 x = a*(value-b)
        #                 result = 1/(1+math.exp(-x))
        #                 print("entropy rm_score", value, result)
        #                 node.score = result
        #             else:
        #                 sigmoid_value = 1 / (1 + math.exp(-value))
        #                 coeff = 0.5
        #                 value = node.binary_score + coeff * sigmoid_value
        #                 node.score = value
        # paths['pass_k_result'] = pass_k_result
        # paths['eval_time_use'] = time.time() - eval_time_start
        # paths['time_use'] = time.time() - time_start

        # 以上为串行评估，以下为并发评估
        eval_time_start = time.time()
        paths['pass_k_result'] = self.evaluate_trees(
            problem_str, 
            answer_str, 
            args
        )
        paths['eval_time_use'] = time.time() - eval_time_start
        paths['time_use'] = time.time() - time_start

        print('eval_time_use: ',
              paths['eval_time_use'], '\ttime_use: ', paths['time_use'])

        # 序列化树结构
        paths['tree_structures'] = [
            self.serialize_tree_list(tree_list) for tree_list in self.tree_lists
        ]
        root, selected_terminals = build_into_tree_format(self.tree_lists,self.decode_fn,args['num_traces'],args["balance_ratio"],args["average_one_generation"],use_weighted_value = args["use_weighted_value"])
        paths = gather_paths(
            root = root,
            selected_terminals = selected_terminals,
            pass_k = args['num_traces'],
            use_orm_reward = args['use_orm_reward'],
            use_chain_reward = args["use_chain_reward"],
            step_level_norm = args["step_level_norm"],
            use_state_value_reward = args["use_state_value_reward"],
            use_value_only = args["use_value_only"],
            average_one_generation = args["average_one_generation"],
            advantage_mix_allancestor = args["advantage_mix_allancestor"]
        )
        return paths

    def serialize_tree_list(self, tree_list):
        """
        序列化单个树列表。
        """
        return [{
            'token_ids': node.token_id_list,
            'token_strs': node.token_str_list,
            'log_probs': node.log_prob_list,
            'is_end': node.is_end,
            'mask': node.mask,
            'finish_reason': node.finish_reason,
            'total_str': node.total_str,
            'parent_node_idx': node.parent_node_idx,
            'parent_node_split_idx': node.parent_node_split_idx
        } for node in tree_list]

    def process_single_item(self, item: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个数据项。

        :param item: 数据项，包含 'problem' 和 'golden_answer'。
        :return: 处理后的路径和结果。
        """
        problem = item["problem"]
        answer = item["golden_answer"]

        paths = self.entropy_guided_chain(problem, answer, args=args)
        result = {
            "problem": problem,
            "golden_answer": answer,
            "paths": paths,
        }
        return result
