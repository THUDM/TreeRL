from vllm import LLM

from transformers import AutoTokenizer
from entropy_chain_local_manager import EntropyGuidedChainLocalManager
from IPython import embed


def tokenize_fn(
    texts,
    tokenizer,
    max_length=4096,
):
    sample_input_ids = tokenizer.encode(texts, add_special_tokens=False)
    sample_input_ids = sample_input_ids[-max_length:]
    return sample_input_ids


def normalize_selected_terminals(paths):
    leaf_orm_value = [leaf["value"] for leaf in paths]
    _sum = sum(leaf_orm_value)
    num = len(leaf_orm_value) - 1
    mean = [(_sum - leaf_orm_value[i]) /
            num for i in range(len(leaf_orm_value))]
    orm_normalized = [leaf_orm_value[i] - mean[i]
                      for i in range(len(leaf_orm_value))]
    for i in range(len(orm_normalized)):
        paths[i]["value"] = orm_normalized[i]
    return paths


def parallel_entropy_guided_tree(
    item,
    llm,
    args=None,
    tokenizer=None,
):
    manager = EntropyGuidedChainLocalManager(
        args=args,
        llm=llm,
        evaluator_urls=args['evaluator_urls'],
        eos_tokens_set=args['eos_tokens'],
    )

    result = manager.process_single_item(item)

    trees = result['path']['tree_structures']
    pass_k_result = result['path']['pass_k_result']

    contexts = [node['total_str'].split("<|user|>")[0]
                for tree in trees for node in tree]

    assert len(contexts) == len(pass_k_result)

    paths = []
    for context, pass_k in zip(contexts, pass_k_result):
        paths.append({
            "token_answer": tokenize_fn(context, tokenizer),
            "pass_ratio": pass_k,
            "value": pass_k,
        })

    paths = normalize_selected_terminals(paths)
    embed()
    return [paths]


if __name__ == '__main__':
    item = {
        "problem": "The graph of $$x^4=x^2 y^2$$ is a union of $$n$$ different lines. What is the value of $$n$$ ?",
        "golden_answer": "3"
    }
    llm = LLM(
        model="/workspace/reason_data/checkpoint/glm-o1-2w-sft",
        tensor_parallel_size=1,
        trust_remote_code=True,
        seed=3407
    )
    args = {
        "temperature": 1.2,
        "top_p": 0.9,
        "m": 4,
        "n": 2,
        "l": 1,
        "evaluator_urls": ["http://172.18.74.40:8000/v1"],
        "eos_tokens": ["<|user|>", "<|endoftext|>", "<|observation|>"],
    }
    tokenizer = AutoTokenizer.from_pretrained(
        '/workspace/reason_data/checkpoint/glm-o1-2w-sft',
        trust_remote_code=True
    )
    parallel_entropy_guided_tree(item, llm, args, tokenizer)
