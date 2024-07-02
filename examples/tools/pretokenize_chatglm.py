import argparse
import json
from multiprocessing import Pool
from transformers import AutoTokenizer
from tqdm import tqdm
from functools import partial


def load_data(filepath):
    with open(filepath) as f:
        data = [json.loads(x) for x in filepath]
    return data


def build_conversation_from_sft(item):
    conversations = []
    if item.get("reference"):
        conversations.append({"role": "system", "value": f"不一定有用的参考信息：\n\n{item['reference']}"})
    for hist_item in item["history"]:
        conversations.append({"role": "user", "value": hist_item["prompt"]})
        conversations.append({"role": "assistant", "value": hist_item["response"], "loss": False})
    conversations.append({"role": "user", "value": item["prompt"]})
    conversations.append({"role": "assistant", "value": item["response"], "loss": True})
    return conversations


def build_single_message(tokenizer, role, metadata, message=None):
    assert role in ["<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"], role
    role_tokens = [tokenizer.get_command(role)] + tokenizer.encode(f"{metadata}\n", add_special_tokens=False)
    message_tokens = tokenizer.encode(message, add_special_tokens=False)
    tokens = role_tokens + message_tokens

    return tokens

def get_data(item, tokenizer):
    conversations = build_conversation_from_sft(item)
    messages = []
    for conv in conversations:
        if conv["role"] == "system":
            messages.append(build_single_message(tokenizer, "<|system|>", "", conv["value"]))
        elif conv["role"] == "user":
            messages.append(build_single_message(tokenizer, "<|user|>", "", conv["value"]))
        elif conv["role"] == "assistant":
            messages.append(build_single_message(tokenizer, "<|assistant|>", "", conv["value"]))
    prompt_tokens = []
    for message in messages[:-1]:
        prompt_tokens.extend(message)
    prompt_attention_mask = [0] * len(prompt_tokens)
    response_tokens = messages[-1]
    response_attention_mask = [1] * len(response_tokens)
    return {"prompt": prompt_tokens, "response": response_tokens}
    # return {"prompt": prompt_tokens, "prompt_attention_mask": prompt_attention_mask, "response": response_tokens, "response_attention_mask": response_attention_mask}



def main(input_file, output_file, model_path):
    data = load_data(input_file)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenize_data_func = partial(get_data, tokenizer)
    with Pool(20) as p:
        result = list(tqdm(p.imap(tokenize_data_func, data), total=len(data)))
    with open(output_file, "w") as f:
        f.writelines([json.dumps(x, ensure_ascii=False) + "\n" for x in result])
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    
    if args.output_file is None:
        args.output_file = args.input_file.replace(".jsonl", "_tokenized.jsonl")
    main(args.input_file, args.output_file, args.model_path)