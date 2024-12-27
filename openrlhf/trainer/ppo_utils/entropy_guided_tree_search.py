from vllm import LLM

from transformers import AutoTokenizer
from entropy_chain_local_manager import EntropyGuidedChainLocalManager

from IPython import embed

tokenizer = AutoTokenizer.from_pretrained(
    '/workspace/reason_data/checkpoint/glm-o1-2w-sft',
    trust_remote_code=True
)


def tokenize_fn(texts, max_length=4096):
    sample_input_ids = tokenizer.encode(texts, add_special_tokens=False)
    sample_input_ids = sample_input_ids[-max_length:]
    return sample_input_ids


def parallel_entropy_guided_tree(
    item,
    llm,
    args=None
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
    finish_reasons = [node['finish_reason'] for tree in trees for node in tree]

    assert len(contexts) == len(pass_k_result) == len(finish_reasons)

    paths = []
    for context, pass_k, finish_reason in zip(contexts, pass_k_result, finish_reasons):
        paths.append({
            "token_answer": tokenize_fn(context),
            "value": pass_k,
            "finish_reason": finish_reason
        })

    return paths


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
    parallel_entropy_guided_tree(item, llm, args)
