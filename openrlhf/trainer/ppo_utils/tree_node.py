from typing import List, Optional


class TreeNode:
    def __init__(
        self,
        tree_idx: int,
        node_idx: int,
        token_list: List[str],
        log_prob_list: List[float],
        finish_reason: Optional[str] = None,
        is_end: bool = False,
        parent_node: Optional['TreeNode'] = None,
        parent_node_idx: Optional[int] = None,
        parent_node_split_idx: Optional[int] = None,
        child_nodes: Optional[List['TreeNode']] = None,
        child_split_indices: Optional[List[int]] = None
    ):
        """
        树节点的信息
        """
        # --- 分组信息 ---
        self.tree_idx: int = tree_idx  # 树的索引
        self.node_idx: int = node_idx  # 节点的索引

        # --- 节点包含的文本信息 ---
        self.token_list: List[str] = token_list
        self.log_prob_list: List[float] = log_prob_list
        self.token_num: int = len(token_list)  # token 数量
        self.finish_reason: Optional[str] = finish_reason  # 结束原因
        self.is_end: bool = is_end  # 是否是叶子节点

        # --- 父亲节点信息 ---
        self.parent_node = parent_node  # 父节点对象
        self.parent_node_idx = parent_node_idx  # 父节点的索引
        self.parent_node_split_idx = parent_node_split_idx  # 从父节点分叉的 token 索引

        # --- 孩子节点信息 ---
        # 子节点列表
        self.child_nodes: List['TreeNode'] = child_nodes if child_nodes else []
        # 子节点分叉的 token 索引
        self.child_split_indices: List[int] = child_split_indices if child_split_indices else [
        ]

        # --- 截止到目前的 aggregate 字符串以及所有完整字符串（减少遍历时间，以及用于判断答案） ---
        self.aggregate_str: str = ""
        if parent_node is not None:
            parent_token_list = parent_node.token_list
            self.aggregate_str = parent_node.aggregate_str + \
                ''.join(parent_token_list[:parent_node_split_idx])
        self.total_str: str = self.aggregate_str + ''.join(token_list)

        # --- 掩码信息 ---
        self.mask: List[bool] = [False] * len(token_list)
        for i, token_str in enumerate(token_list):
            if "conclusion" in token_str or "answer" in token_str:
                # 掩盖后续 tokens
                for j in range(i + 1, len(self.mask)):
                    self.mask[j] = True
                self.is_end = True
                break

    def get_prefix(self, current_token_index: int) -> str:
        """
        给定截断的位置，获取前缀文本，通过迭代构建，从根节点到当前节点。

        :return: 拼接后的前缀字符串。
        """

        # # 存储从根节点到当前节点的路径信息
        # node_path = []
        # current_node = self

        # # 向上遍历到根节点，收集路径信息
        # while current_node is not None:
        #     node_path.append({
        #         'node': current_node,
        #         'split_idx': current_node.parent_node_split_idx
        #     })
        #     current_node = current_node.parent_node

        # # 从根节点开始构建字符串
        # result = ""
        # for i in range(len(node_path) - 1, -1, -1):  # 从后向前遍历（从根到叶）
        #     node_info = node_path[i]
        #     node = node_info['node']

        #     if i == 0:  # 当前节点
        #         result += ''.join(node.token_list[:current_token_index])
        #     else:  # 父节点们
        #         split_idx = node_info['split_idx']
        #         result += ''.join(node.token_list[:split_idx])

        # expected = self.aggregate_str + \
        #     ''.join(self.token_list[:current_token_index])
        # assert result == expected, f"Prefix mismatch:\nExpected: {expected}\nGot: {result}"
        # print("you pass!")

        parent_tokens = self.aggregate_str
        return parent_tokens + ''.join(self.token_list[:current_token_index])

    def add_child(self, child_node: 'TreeNode', split_index: int) -> None:
        """
        添加子节点。

        :param child_node: TreeNode 对象。
        :param split_index: 分裂的token索引。
        """
        self.child_nodes.append(child_node)
        self.child_split_indices.append(split_index)
        child_node.parent_node = self
        child_node.parent_split_index = split_index

    def get_max_entropy_tokens(self, top_n: int = 1) -> List[int]:
        """
        获取最高熵的token索引，返回top_n个。
        只考虑未被 mask 的 token。

        :param top_n: 需要返回的最高熵token数量。
        :return: 最高熵token的索引列表。
        """
        # 计算每个 token 位置的熵
        entropies = []
        for i, log_prob in enumerate(self.log_prob_list):
            if not self.mask[i]:  # 只考虑未被 mask 的 token
                entropy = -log_prob  # 简单地用负对数概率作为熵
                entropies.append((entropy, i))

        # 按熵值排序并返回前 top_n 个索引
        sorted_indices = sorted(entropies, key=lambda x: x[0], reverse=True)
        result = [idx for _, idx in sorted_indices[:top_n]]

        # 如果不够 top_n 个，复制若干份，确保鲁棒性
        while len(result) < top_n:
            result += result[:top_n - len(result)]

        return result
