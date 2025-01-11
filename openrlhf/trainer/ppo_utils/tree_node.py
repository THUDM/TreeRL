from __future__ import annotations
from typing import List, Optional, Callable
from pydantic import BaseModel
import json
import random


class MCTSNode(BaseModel):
    answer: str
    answer_token : List[int]
    parent: MCTSNode | None = None
    children: list[MCTSNode] = []
    R : float = 0
    depth: int = 0
    main_chain: bool = False
    terminal: bool = False
    terminal_in_subtree: int = 0
    correct_terminal_in_subtree: int = 0
    accumulated_value: float = 0
    repeat:bool = False
    value: float = 0

class TreeNode:
    def __init__(
        self,
        tree_idx: int,
        node_idx: int,
        decode_fn: Callable,
        token_id_list: List[int],
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
        self.token_id_list: List[int] = token_id_list
        self.token_str_list: List[str] = [
            decode_fn([token_id]) for token_id in token_id_list]

        self.log_prob_list: List[float] = log_prob_list
        self.token_num: int = len(token_id_list)  # token 数量
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

        # --- 孩子正确率信息（分段） ---
        self.child_correct_num: List[int] = []
        self.child_total_num: List[int] = []

        # --- 截止到目前的 aggregate 字符串以及所有完整字符串（减少遍历时间，以及用于判断答案） ---
        self.aggregate_str: str = ""
        if parent_node is not None:
            parent_token_str_list = parent_node.token_str_list
            self.aggregate_str = parent_node.aggregate_str + \
                ''.join(parent_token_str_list[:parent_node_split_idx])
        self.total_str: str = self.aggregate_str + ''.join(self.token_str_list)

        self.aggregate_token_ids: List[int] = []
        if parent_node is not None:
            self.aggregate_token_ids = parent_node.aggregate_token_ids + \
                parent_node.token_id_list[:parent_node_split_idx]

        # --- 掩码信息 ---
        self.mask: List[bool] = [False] * len(self.token_str_list)
        for i, token_str in enumerate(self.token_str_list):
            if "conclusion" in token_str.lower() or "answer" in token_str.lower():
                # 掩盖后续 tokens
                for j in range(i + 1, len(self.mask)):
                    self.mask[j] = True
                self.is_end = True
                break

        # --- 节点的分数 ---
        self.binary_score: Optional[float] = None
        self.score: Optional[float] = None

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
        return parent_tokens + ''.join(self.token_str_list[:current_token_index])

    def get_prefix_ids(self, current_token_index: int) -> List[int]:
        """
        给定截断的位置，获取前缀 token_ids

        :return: 拼接后的前缀 token_ids 列表。
        """

        parent_token_ids = self.aggregate_token_ids
        return parent_token_ids + self.token_id_list[:current_token_index]

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
        # 但这样可能会导致重复，所以目前不使用
        # while len(result) < top_n:
        #     result += result[:top_n - len(result)]

        return result

def build_into_tree_format(tree_lists,decode_fn,num_traces,balance_ratio=0,average_one_generation=False) -> MCTSNode:
    # from IPython import embed
    # embed()
    all_leaves = []
    try:
        def convert_to_json(node: MCTSNode):
            if not node.children:
                print("answer_token: ",node.answer_token)
                return {
                    "answer": node.answer,
                    "answer_token": node.answer_token,
                    "R" : node.R,
                    "depth": node.depth,
                    "main_chain": node.main_chain,
                    "terminal": node.terminal,
                    "terminal_in_subtree": node.terminal_in_subtree,
                    "correct_terminal_in_subtree": node.correct_terminal_in_subtree,
                    "accumulated_value": node.accumulated_value
                }
            else:
                print("answer_token: ",node.answer_token)
                return {
                    "answer": node.answer,
                    "answer_token": node.answer_token,
                    "R" : node.R,
                    "depth": node.depth,
                    "main_chain": node.main_chain,
                    "terminal": node.terminal,
                    "children": [convert_to_json(child) for child in node.children],
                    "terminal_in_subtree": node.terminal_in_subtree,
                    "correct_terminal_in_subtree": node.correct_terminal_in_subtree,
                    "accumulated_value": node.accumulated_value,
                }
        def build_tree_node(decode_fn,tree_node: TreeNode, parent_mcts_node: Optional[MCTSNode] = None) -> MCTSNode:
            # 对 child_nodes 按照 parent_node_split_idx 进行排序
            tree_node.child_nodes.sort(key=lambda x: x.parent_node_split_idx)

            # 存储所有孩子节点的 parent_node_split_idx
            child_split_indices = [child.parent_node_split_idx for child in tree_node.child_nodes]
            
            # 如果没有孩子节点，设置end_idx为整个token_id_list的长度
            is_terminal = False
            R = 0
            main_chain = False
            if not child_split_indices:
                first_child_split_idx = len(tree_node.token_id_list)
                is_terminal = True
                R = tree_node.score
                if tree_node.binary_score == 1:
                    main_chain = True
            else:
                first_child_split_idx = child_split_indices[0]

            # 初始节点段，从0到第一个孩子的分割位置
            root_node = MCTSNode(
                answer=''.join([decode_fn([token_id]) for token_id in tree_node.token_id_list[:first_child_split_idx]]),
                answer_token=tree_node.token_id_list[:first_child_split_idx],
                parent=parent_mcts_node,
                depth=(parent_mcts_node.depth + 1) if parent_mcts_node else 0,
                terminal=is_terminal,
                R = R,
                main_chain = main_chain
            )
            
            if root_node.terminal:
                all_leaves.append(root_node)

            # 递归构建子树
            def add_segments_and_children(current_mcts_node: MCTSNode, start_idx: int):
                i = 0
                while i < len(tree_node.child_nodes):
                    child_nodes_group = []
                    current_split_idx = child_split_indices[i]
                    
                    # 收集所有具有相同 parent_node_split_idx 的孩子节点
                    while i < len(tree_node.child_nodes) and child_split_indices[i] == current_split_idx:
                        child_nodes_group.append(tree_node.child_nodes[i])
                        i += 1
                    is_terminal = False
                    R = 0
                    main_chain = False
                    if i < len(tree_node.child_nodes):
                        next_split_idx = child_split_indices[i]
                    else:
                        next_split_idx = len(tree_node.token_id_list)
                        is_terminal = True
                        R = tree_node.score
                        if tree_node.binary_score == 1:
                            main_chain = True
                    
                    # 创建当前段后的子段
                    segment_node = MCTSNode(
                        answer=''.join([decode_fn([token_id]) for token_id in tree_node.token_id_list[start_idx:next_split_idx]]),
                        answer_token=tree_node.token_id_list[start_idx:next_split_idx],
                        parent=current_mcts_node,
                        depth=current_mcts_node.depth + 1,
                        terminal=is_terminal,
                        R = R,
                        main_chain = main_chain
                    )
                    current_mcts_node.children.append(segment_node)
                    if segment_node.terminal:
                        all_leaves.append(segment_node)
                    
                    # 为每一个子节点组添加子树, 并将子树挂载到segment_node
                    for child_node in child_nodes_group:
                        child_mcts_node = build_tree_node(decode_fn,child_node, current_mcts_node)
                        current_mcts_node.children.append(child_mcts_node)
                    
                    start_idx = next_split_idx
                    # 更新当前父节点
                    current_mcts_node = segment_node

            # 初始调用，为根节点添加子段
            add_segments_and_children(root_node, first_child_split_idx)
            
            return root_node

        # 构建根节点
        root = MCTSNode(
            answer="",
            answer_token=[]
        )
        
        # 根的所有孩子是所有tree_lists[i][0]
        for i, tree_list in enumerate(tree_lists):
            if len(tree_list) > 0:
                root.children.append(build_tree_node(decode_fn,tree_list[0], root))
        
        leaf_normalize(all_leaves,root,average_one_generation)
        selected_terminals = select_terminal(all_leaves,num_traces,balance_ratio)
        # with open("/workspace/lurui/openrlhf-mcts/data/tree_local.jsonl","a") as f:
        #     f.write(json.dumps(convert_to_json(root)))
        #     f.write("\n")
        
        return root, selected_terminals
    except Exception as e:
        print(e)
        from IPython import embed
        embed()
        
def leaf_normalize(nodes,root,average_one_generation:bool = False):
    leaf_correctness = [leaf.R for leaf in nodes]
    print("leaf_correctness",leaf_correctness)
    _sum = sum(leaf_correctness)
    num = len(leaf_correctness) - 1
    if num == 0:
        # return
        assert False, "entropy num_traces == 0"
    else:
        mean = [(_sum - leaf_correctness[i]) / num for i in range(len(leaf_correctness))]
        for i, leaf in enumerate(nodes):
            leaf.R = leaf.R - mean[i]
            leaf.accumulated_value = leaf.R
            leaf_backpropagate(leaf)
        if average_one_generation:
            update_accumulated_values(root)
            
def leaf_backpropagate(node: MCTSNode):
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
        
def compute_accumulated_value(node: MCTSNode):
    if not node.children:  # If the node is a leaf node
        return node.accumulated_value

    # Post-order traversal: first process all children
    total_value = 0
    terminal_children = 0
    for child in node.children:
        if child.terminal_in_subtree > 0:
            terminal_children += 1
            total_value += compute_accumulated_value(child)

    # Calculate the average accumulated value for the current node
    node.accumulated_value = total_value / terminal_children if terminal_children > 0 else 0
    return node.accumulated_value

# Helper function to initialize calculation from the root node
def update_accumulated_values(root):
    compute_accumulated_value(root)

def select_terminal(nodes, num_traces, balance_ratio = 0):
    if balance_ratio == 0:
        random.shuffle(nodes)
        selected_terminals = []
        remaining_terminals = []

        # Traverse the shuffled paths and select the first pass_ratio == 1 path
        for node in nodes:
            if node.main_chain and len(selected_terminals) == 0:
                selected_terminals.append(node)
            else:
                remaining_terminals.append(node)

        # Calculate how many additional paths we need
        remaining_num_traces = num_traces - len(selected_terminals)

        # Randomly select remaining_num_traces paths from the remaining_paths if possible
        if remaining_num_traces > 0:
            selected_terminals.extend(random.sample(remaining_terminals, min(
                remaining_num_traces, len(remaining_terminals))))

        # Shuffle the selected paths to ensure they are returned in random order
        random.shuffle(selected_terminals)
        assert len(
            selected_terminals) == num_traces, f"len(selected_paths) = {len(selected_terminals)} != num_traces = {num_traces}"

        return selected_terminals
    else:
        random.shuffle(nodes)
        num_correct_needed = int(num_traces * balance_ratio)
        num_incorrect_needed = int(num_traces * balance_ratio)

        selected_correct = []
        selected_incorrect = []
        remaining_terminals = []

        # Traverse the shuffled nodes and select correct and incorrect nodes
        for node in nodes:
            if node.main_chain and len(selected_correct) < num_correct_needed:
                selected_correct.append(node)
            elif not node.main_chain and len(selected_incorrect) < num_incorrect_needed:
                selected_incorrect.append(node)
            else:
                remaining_terminals.append(node)
        
        # Calculate how many additional terminals we need
        num_selected = len(selected_correct) + len(selected_incorrect)
        remaining_num_traces = num_traces - num_selected
        print(f"num_correct = {len(selected_correct)}, num_incorrect = {len(selected_incorrect)},remaining_num_traces = {remaining_num_traces}")

        selected_terminals = selected_correct + selected_incorrect

        # Randomly select remaining_num_traces paths from remaining_terminals if possible
        if remaining_num_traces > 0:
            selected_terminals.extend(random.sample(remaining_terminals, min(remaining_num_traces, len(remaining_terminals))))

        # Shuffle the selected terminals to ensure they are returned in random order
        random.shuffle(selected_terminals)
        assert len(selected_terminals) == num_traces, f"len(selected_terminals) = {len(selected_terminals)} != num_traces = {num_traces}"

        return selected_terminals