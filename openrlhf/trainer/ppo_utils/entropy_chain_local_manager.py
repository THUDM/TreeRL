import time
from typing import List, Dict, Any
from tree_node import TreeNode
from evaluation import (
    check_result,
    query_local_vllm_completions_with_logprobs,
    GLM_QA_PROMPT
)

from IPython import embed


class EntropyGuidedChainLocalManager:
    def __init__(
        self,
        args: Dict[str, Any],
        llm: Any,
        evaluator_urls: List[str],
        eos_tokens_set: List[str]
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
        self.eos_tokens_set = eos_tokens_set
        self.paths: Dict[str, Any] = {
            "M": args["m"],
            "N": args["n"],
            "L": args["l"],
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

    def entropy_guided_chain(
        self,
        problem_str: str,
        answer_str: str
    ) -> Dict[str, Any]:
        """
        熵引导的链式推理。

        :param problem_str: 问题字符串。
        :param answer_str: 标准答案字符串。
        :return: 存储路径和结果的字典。
        """
        init_prompt_with_template = GLM_QA_PROMPT.format(
            prompt=problem_str, response=""
        )

        M = self.args["m"]
        N = self.args["n"]
        L = self.args["l"]

        paths = self.paths
        time_start = time.time()

        # 初始化 M 棵树
        self.tree_lists = []
        initial_prompts = [init_prompt_with_template] * M

        # 获取初始推理结果
        initial_results = query_local_vllm_completions_with_logprobs(
            initial_prompts,
            llm=self.llm,
            skip_special_tokens=False,
            max_tokens=4096,
            stops=self.eos_tokens_set,
            temperature=self.args["temperature"],
            top_p=self.args["top_p"],
        )

        for idx, (response_tokens, _, finish_reason, _, log_probs) in enumerate(zip(*initial_results)):
            root_node = TreeNode(
                tree_idx=idx,
                node_idx=0,
                token_list=response_tokens,
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
            m_tree_top_n_prompts = []
            task_mapping = {}
            for i, (tree_idx, node_idx, node, split_idx) in enumerate(expansion_tasks):
                prefix = node.get_prefix(split_idx)
                prompt = GLM_QA_PROMPT.format(
                    prompt=problem_str, response=prefix
                )
                m_tree_top_n_prompts.append(prompt)
                task_mapping[i] = (tree_idx, node_idx, node, split_idx)

            # 批量执行推理
            inference_results = query_local_vllm_completions_with_logprobs(
                m_tree_top_n_prompts,
                llm=self.llm,
                skip_special_tokens=False,
                max_tokens=4096,
                stops=self.eos_tokens_set,
                temperature=self.args["temperature"],
                top_p=self.args["top_p"],
            )

            # 处理结果，更新树结构
            for i, (response_tokens, _, finish_reason, _, log_probs) in enumerate(zip(*inference_results)):
                tree_idx, node_idx, parent_node, split_idx = task_mapping[i]

                # 在 split_idx 处分裂当前节点
                new_node = TreeNode(
                    tree_idx=tree_idx,
                    node_idx=len(self.tree_lists[tree_idx]),
                    token_list=response_tokens,
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
        pass_k_result = []
        for tree_list in self.tree_lists:
            for node in tree_list:
                if node.is_end:
                    response_str = node.total_str
                    response_str = response_str.split("<|user|>")[0]
                    score = check_result(
                        problem_str,
                        response_str,
                        answer_str,
                        urls=self.evaluator_urls
                    )[-1]
                    pass_k_result.append(score)
                else:
                    pass_k_result.append(0)

        paths['pass_k_result'] = pass_k_result
        paths['eval_time_use'] = time.time() - eval_time_start
        paths['time_use'] = time.time() - time_start

        # 序列化树结构
        paths['tree_structures'] = [
            self.serialize_tree_list(tree_list) for tree_list in self.tree_lists
        ]

        return paths

    def serialize_tree_list(self, tree_list):
        """
        序列化单个树列表。
        """
        return [{
            'tokens': node.token_list,
            'log_probs': node.log_prob_list,
            'is_end': node.is_end,
            'mask': node.mask,
            'finish_reason': node.finish_reason,
            'total_str': node.total_str,
            'parent_node_idx': node.parent_node_idx,
            'parent_node_split_idx': node.parent_node_split_idx
        } for node in tree_list]

    def process_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个数据项。

        :param item: 数据项，包含 'problem' 和 'golden_answer'。
        :return: 处理后的路径和结果。
        """
        problem = item["problem"]
        answer = item["golden_answer"]

        paths = self.entropy_guided_chain(problem, answer)
        result = {
            "problem": problem,
            "golden_answer": answer,
            "path": paths,
        }
        return result
