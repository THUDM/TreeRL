import os
import json
import argparse
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

# from reward_model import RewardModel
from openrlhf.models import Actor, get_llm_for_sequence_regression

import torch.distributed as dist

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
device = rank % torch.cuda.device_count()
torch.cuda.set_device(device)



def build_inputs_for_response(tokenizer, prompt, response, history):
    contents = []
    for idx, r in enumerate(history):
        if type(r) == dict:
            contents.append({'role': 'user', 'content': r['prompt']})
            contents.append({'role': 'assistant', 'content': r['response']})
        elif type(r) == list:
            contents.append({'role': 'user', 'content': r[0]})
            contents.append({'role': 'assistant', 'content': r[1]})
        else:
            raise ValueError(f"history should be list of dict/list, but get list of {type(r)}")
    contents.append({'role': 'user', 'content': prompt})
    
    inputs = tokenizer.build_chat_input(response, history=contents, role="assistant")

    input_ids = inputs['input_ids'].tolist()[0][:-1] + [tokenizer.eos_token_id]
    
    return input_ids


def build_inputs_for_item(tokenizer, item, response):
    input_ids = build_inputs_for_response(tokenizer, item['prompt'], response, item['history'])

    return input_ids


def build_inputs_for_batch(args, tokenizer, item):
    reply_ids = []
    input_ids = []
    # for reply_id, response in item[args.response_key]:
    input_ids = build_inputs_for_item(tokenizer, item, item[args.response_key])

        # if args.max_seq_len > 0 and len(_input_ids) > args.max_seq_len:
            # continue
        # reply_ids.append(reply_id)
        # input_ids.append(_input_ids)
    if reply_ids == []:
        return None, None
    
    max_length = max([len(_input_ids) for _input_ids in input_ids])
    attention_mask = [[0] * (max_length - len(_input_ids)) + [1] * len(_input_ids) for _input_ids in input_ids]
    position_ids = [[0] * (max_length - len(_input_ids)) + list(range(len(_input_ids))) for _input_ids in input_ids]
    input_ids = [[0] * (max_length - len(_input_ids)) + _input_ids for _input_ids in input_ids]

    inputs = {
        'input_ids': torch.tensor(input_ids).long().cuda(),
        'attention_mask': torch.tensor(attention_mask).float().cuda(),
        'position_ids': torch.tensor(position_ids).long().cuda(),
    }

    return inputs


def predict_reward(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # model = RewardModel(args.base_model_name_or_path, args.model_name_or_path).to(torch.bfloat16)
    model = get_llm_for_sequence_regression(
        args.model_name_or_path,
        "reward",
        normalize_reward=True,
        use_flash_attention_2=False,
        load_in_4bit=False,
        bf16=args.bf16,
    )
    model.cuda()
    model.eval()
    print(f'load model to {device}')

    data = []
    with open(args.input_file, 'r') as r:
        for line in r:
            data.append(json.loads(line))
    if rank == 0:
        print(f'load {len(data)} examples from {args.input_file}')
    data = data[rank::world_size]

    with open(args.output_file, 'w') as w:
        for item in tqdm(data, desc=f'device={device}'):
            # item[args.response_key] = [
            #     [
            #         reply_id,
            #         response[len('<|assistant|>'):] if response.startswith('<|assistant|>') else response
            #     ] for response in item[args.response_key]
            # ]

            assert 'rewards' not in item
            # item['rewards'] = {}
            # num_batch = len(item[args.response_key]) // args.batch_size
            # if num_batch * args.batch_size < len(item[args.response_key]):
                # num_batch += 1

            # for i in range(num_batch):
            with torch.no_grad():
                # begin, end = i * args.batch_size, (i + 1) * args.batch_size
                inputs = build_inputs_for_batch(args, tokenizer, item)
                # if reply_ids is None:
                    # continue
                rewards = model(**inputs).tolist()
                assert len(rewards) == 1
                # assert len(reply_ids) == len(rewards)
                # for reward in zip(reply_ids, rewards):
                item['rewards'] = rewards[0]
                    
            w.write(json.dumps(item, ensure_ascii=False) + '\n')
            w.flush()
    print(f'task on {device} finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--response_key", type=str, default="response")
    parser.add_argument("--output_file", type=str, default='')

    # parser.add_argument("--base_model_name_or_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=-1, help="-1 means no limit.")
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--bf16", action='store_true')
    args = parser.parse_args()

    final_output_file = args.output_file
    if final_output_file == '':
        final_output_file = f'{args.input_file[:-6]}.reward.jsonl'

    args.output_file = f'{args.input_file}.part_{rank}' # temp output file

    if rank == 0:
        print(f'model file  : {args.model_name_or_path}')
        print(f'input file  : {args.input_file}')
        print(f'response key: {args.response_key}')
        print(f'output file : {final_output_file}')
        print(f'GPU number  : {world_size}')
        # assert world_size == 8
        
    predict_reward(args)

    dist.barrier()
    if rank == 0:
        num_save = 0
        with open(final_output_file, 'w', encoding='utf-8') as w:
            for i in range(world_size):
                with open(f'{args.input_file}.part_{i}', 'r', encoding='utf-8') as r:
                    for line in r:
                        w.write(line)
                        num_save += 1
                os.remove(f'{args.input_file}.part_{i}')
        print(f'save {num_save} examples to {final_output_file}')
    dist.barrier()
