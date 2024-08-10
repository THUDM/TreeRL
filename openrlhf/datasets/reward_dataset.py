import random
from typing import Callable
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import re
from .utils import exist_and_not_none, zero_pad_sequences


def preprocess_data(data, input_template=None, prompt_key=None, chosen_key=None, rejected_key=None, source_key=None) -> str:
    # custom dataset
    if chosen_key and rejected_key:
        if prompt_key:
            prompt = data[prompt_key]
        else:
            prompt = ""
            input_template = None  # do not modified with input template again
        chosen = data[chosen_key]
        reject = data[rejected_key]
        source_type = data[source_key] if source_key else None
    else:
        # Anthropic/hh-rlhf
        # tasksource/oasst1_pairwise_rlhf_reward
        if exist_and_not_none(data, "chosen") and exist_and_not_none(data, "rejected"):
            prompt = data["prompt"] if exist_and_not_none(data, "prompt") else ""
            if prompt.startswith("prompter:"):
                prompt = (
                    prompt.replace("prompter:", "\nHuman: ").replace("assistant:", "\nAssistant: ") + "\nAssistant: "
                )
            chosen = data["chosen"]
            reject = data["rejected"]
            input_template = None  # do not modified with input template again
        # lvwerra/stack-exchange-paired
        elif exist_and_not_none(data, "response_j"):
            prompt = data["question"]
            chosen = data["response_j"]
            reject = data["response_k"]
        # lmsys/chatbot_arena_conversations
        elif exist_and_not_none(data, "winner") and exist_and_not_none(data, "conversation_a"):

            def process_chatbot_arena_conversations(lll):
                result = []
                for l in lll:
                    if "user" in l["role"]:
                        result.append(input_template.format(l["content"]))
                    else:
                        result.append(l["content"])
                return "\n".join(result)

            prompt = ""
            chosen = data["conversation_a"] if data["winner"] == "model_a" else data["conversation_b"]
            reject = data["conversation_b"] if data["winner"] == "model_a" else data["conversation_a"]
            chosen = process_chatbot_arena_conversations(chosen)
            reject = process_chatbot_arena_conversations(reject)
            input_template = None  # do not modified with input template again
        # openai/webgpt_comparisons
        elif exist_and_not_none(data, "answer_0") and exist_and_not_none(data, "answer_1"):
            prompt = data["question"]["full_text"]
            chosen = data["answer_0"] if data["score_0"] > data["score_1"] else data["answer_1"]
            reject = data["answer_1"] if data["score_0"] > data["score_1"] else data["answer_0"]
        # damo/CValues-Comparison https://www.modelscope.cn/datasets/damo/CValues-Comparison/quickstart
        elif exist_and_not_none(data, "pos_resp") and exist_and_not_none(data, "neg_resp"):
            prompt = data["prompt"]
            chosen = data["pos_resp"]
            reject = data["neg_resp"]
        else:
            raise ValueError("Unknown reward dataset")
        source_type = None

    history = data.get("history", None)        

    # margin loss
    margin = data["margin"] if exist_and_not_none(data, "margin") else 0

    # input template
    if input_template:
        prompt = input_template.format(prompt)
    return prompt, chosen, reject, history, margin, source_type


class RewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template="Human: {}\nAssistant: ",
    ) -> None:
        super().__init__()
        self.prompts = []
        self.chosens = []
        self.rejects = []
        self.margins = []
        self.history = []
        self.data_source = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        self.current_model = strategy.args.pretrain
        if "glm" in self.current_model:
            self.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|user|>")
        else:
            self.eos_token_id = None

        prompt_key = getattr(self.strategy.args, "prompt_key", None)
        chosen_key = getattr(self.strategy.args, "chosen_key", None)
        rejected_key = getattr(self.strategy.args, "rejected_key", None)
        source_key = getattr(self.strategy.args, "source_key", "type")

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, chosen, reject, history, margin, source_type = preprocess_data(
                data, input_template, prompt_key, chosen_key, rejected_key, source_key
            )
            if source_type is None:
                source_type = "unknown"
                
            self.prompts.append(prompt)
            self.chosens.append(chosen)
            self.rejects.append(reject)
            self.margins.append(margin)
            self.history.append(history)
            self.data_source.append(source_type)

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject, history, margin = self.prompts[idx], self.chosens[idx], self.rejects[idx], self.history[idx], self.margins[idx]
        source_type = self.data_source[idx]
        
        if "glm" in self.current_model.lower():
            return self.tokenize_func_chatglm(prompt, chosen, reject, history, margin, source_type)

        assert False, "Unknown model, not ChatGLM"

        chosen = prompt + chosen + " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        reject = prompt + reject + " " + self.tokenizer.eos_token
        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            margin,
        )
    
    def tokenize_func_chatglm(self, prompt, chosen, reject, history, margin, source_type):
        # item["role"], item.get("metadata", ""), content)
        history_input_ids = []
        if history is not None:
            for item in history:
                history_input_ids.extend(self.tokenizer.build_single_message("user", "", item["prompt"]))
                history_input_ids.extend(self.tokenizer.build_single_message("assistant", "", item["response"]))
        
        def conjecture(response):
            sample_input_ids = self.tokenizer.build_single_message("user", "", prompt) + self.tokenizer.build_single_message("assistant", "", response)

            sample_input_ids = history_input_ids + sample_input_ids
            sample_input_ids = sample_input_ids[:self.max_length-3] + [self.tokenizer.convert_tokens_to_ids("<|user|>")]
            
            sample = self.tokenizer.batch_encode_plus([sample_input_ids], return_tensors="pt", is_split_into_words=True)
            return sample["input_ids"], sample["attention_mask"]

        if source_type in ("math", "code"):
            assert len(chosen) == 0 or len(reject) == 0, "Math source should have only one response"
            label = 1 if chosen else -1
            sample = conjecture(chosen) if chosen else conjecture(reject)
            # chosen_sample = conjecture(chosen)
            return (
                sample[0], sample[1], sample[0], sample[1], margin, label
            )
        else:
            chosen_sample = conjecture(chosen)
            reject_sample = conjecture(reject)
            label = 0
            return (
                chosen_sample[0],
                chosen_sample[1],
                reject_sample[0],
                reject_sample[1],
                margin,
                label
            )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        margins = []
        labels = []
        pairwise_list = [x for x in item_list if x[-1] == 0]
        instancewise_list = [x for x in item_list if x[-1] != 0]
        
        for chosen_id, chosen_mask, reject_id, rejects_mask, margin, label in pairwise_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            margins.append(margin)
            labels.append(label)
        for chosen_id, chosen_mask, _, _, margin, label in instancewise_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            # reject_ids.append(reject_id)
            # rejects_masks.append(rejects_mask)
            margins.append(margin)
            labels.append(label)

        chosen_ids = zero_pad_sequences(chosen_ids, value=self.tokenizer.pad_token_id, side="left")
        chosen_masks = zero_pad_sequences(chosen_masks, side="left")
        if len(reject_ids) > 0:
            reject_ids = zero_pad_sequences(reject_ids, side="left", value=self.tokenizer.pad_token_id)
            rejects_masks = zero_pad_sequences(rejects_masks, side="left")
        else:
            reject_ids = torch.tensor([])
            rejects_masks = torch.tensor([])
            
        labels = torch.tensor(labels, dtype=torch.float32)

        return chosen_ids, chosen_masks, reject_ids, rejects_masks, torch.tensor(margins, dtype=torch.float32), labels


SPLIT_MAP = {
    # "\n": "к",
    "\n\n": "и",
    # "\n\n": "<eop>",
}

# SPLIT_MAP = {
#     "\n": "<|begin_of_video|>",
#     "\n\n": "<|end_of_video|>",
# }


# def revert_special_tokens(tokenizer, input_ids):
#     use_nk = "\n" in SPLIT_MAP    

#     dnk = SPLIT_MAP["\n\n"]
    
#     dnk_token_id = tokenizer.encode(dnk, add_special_tokens=False)[0]
#     dni_token_id = tokenizer.encode("\n\n", add_special_tokens=False)[0]
    
#     input_ids[input_ids == dnk_token_id] = dni_token_id

#     if use_nk:
#         nk = SPLIT_MAP["\n"]
#         nk_token_id = tokenizer.encode(nk, add_special_tokens=False)[0]
#         ni_token_id = tokenizer.encode("\n", add_special_tokens=False)[0]
#         input_ids[input_ids == nk_token_id] = ni_token_id

#     return input_ids


# def reformat_response_into_steps(response):
#     dnk = SPLIT_MAP["\n\n"]

#     use_nk = "\n" in SPLIT_MAP
#     if use_nk:
#         nk = SPLIT_MAP["\n"]
    
#     # response = response.replace("\n\n", dnk).replace("\n", nk)
#     response = re.sub(r"\n\n+", dnk, response)

#     if use_nk:
#         response = re.sub(r"\n+", nk, response)
    
#     # num_steps = re.split(f"{dnk}|{nk}", response)
#     # assert len(num_steps) == target_num_steps, f"expected_steps={target_num_steps}, split_steps={len(num_steps)}"
    
#     return response


def reformat_response_into_steps(text, use_nk=False, flag=""):
    # 使用正则表达式分割文本，保留分隔符
    if use_nk:
        parts = re.split(r'(\n+)', text)
    else:
        parts = re.split(r'(\n\s*?\n+)', text)

    # 将分隔符与前一句合并
    result = []
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            result.append(parts[i] + parts[i+1])
        else:
            result.append(parts[i])
    
    result = flag.join(result) + flag
    return result


def get_process_flag_tokens(tokenizer):
    judge_tokens = [tokenizer.encode(x, add_special_tokens=False)[0] for x in SPLIT_MAP.values()]
    # judge_tokens.append(tokenizer.convert_tokens_to_ids("<|user|>"))
    return judge_tokens


def get_process_flag():
    return list(SPLIT_MAP.values())


def process_finegrained_data(data, prompt_key, response_key, label_key, tokenizer, source_key=None):
    prompt = data[prompt_key]
    if source_key is None or source_key not in data:
        source_type = "unknown"
    else:
        source_type = data[source_key]
    # response = data[response_key]
    # tokenizer.encode(response, add_special_tokens=False)
    label = data[label_key]

    # for item in label:
    #     if isinstance(item[1], str):
    #         item[1] = float(item[1])
    
    # formatted_response = split_response_into_steps(response, len(label))
    steps = [x["text"] for x in label]
    
    def random_interleaved_concat(strings):
        result = strings[0]
        # separators = list(SPLIT_MAP.keys())
        text_op, separator = random.choice(list(SPLIT_MAP.items()))
        for sentence in strings[1:]:
            # separator = random.choice(separators)
            result += text_op + separator + sentence
        result += separator
        return result
    
    # formatted_response = SPLIT_MAP["\n"].join(steps)
    formatted_response = random_interleaved_concat(steps)

    history = data.get("history", None)        
    return prompt, formatted_response, label, history, source_type
    

class RewardProcessDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template="Human: {}\nAssistant: ",
    ) -> None:
        super().__init__()
        self.prompts = []
        self.responses = []
        self.labels = []
        self.margins = []
        self.history = []
        self.data_source = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        self.current_model = strategy.args.pretrain
        if "glm" in self.current_model:
            self.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|user|>")
        else:
            self.eos_token_id = None

        prompt_key = getattr(self.strategy.args, "prompt_key", None)
        response_key = getattr(self.strategy.args, "response_key", None)
        source_key = getattr(self.strategy.args, "source_key", "type")
        label_key = getattr(self.strategy.args, "label_key", "labels")

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, response, label, history, source_type = process_finegrained_data(
                data, prompt_key, response_key, label_key, source_key
            )
            # if source_type is None:
                # source_type = "unknown"
                
            self.prompts.append(prompt)
            self.responses.append(response)
            self.labels.append(label)
            self.history.append(history)
            self.data_source.append(source_type)
        
        # self.judge_tokens = [self.tokenizer.encode(x, add_special_tokens=False)[0] for x in SPLIT_MAP.values()]
        # self.judge_tokens.append(self.tokenizer.convert_tokens_to_ids("<|user|>"))
        self.judge_tokens = get_process_flag_tokens(self.tokenizer)
        assert len(self.judge_tokens) == 1

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt, response, history, label = self.prompts[idx], self.responses[idx], self.history[idx], self.labels[idx]
        source_type = self.data_source[idx]
        
        if "glm" in self.current_model.lower():
            return self.tokenize_func_chatglm(prompt, response, history, label, source_type)

        assert False, "Unknown model, not ChatGLM"
    
    def tokenize_func_chatglm(self, prompt, response, history, labels, source_type):
        # item["role"], item.get("metadata", ""), content)
        history_input_ids = []
        if history is not None:
            for item in history:
                history_input_ids.extend(self.tokenizer.build_single_message("user", "", item["prompt"]))
                history_input_ids.extend(self.tokenizer.build_single_message("assistant", "", item["response"]))
        
        def conjecture(response):
            sample_input_ids = history_input_ids + self.tokenizer.build_single_message("user", "", prompt)
            prompt_len = len(sample_input_ids) + 2
            
            sample_input_ids = sample_input_ids + self.tokenizer.build_single_message("assistant", "", response)

            sample_input_ids = sample_input_ids[:self.max_length-3] + [self.tokenizer.convert_tokens_to_ids("<|user|>")]
            
            sample = self.tokenizer.batch_encode_plus([sample_input_ids], return_tensors="pt", is_split_into_words=True)

            return sample["input_ids"], sample["attention_mask"], prompt_len

        input_ids, attention_mask, prompt_len = conjecture(response)
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        input_ids = input_ids.view(-1)
        attention_mask = attention_mask.view(-1)

        process_labels = torch.zeros_like(input_ids)
        positions = torch.zeros_like(input_ids).bool()

        # for flag in self.judge_tokens:
            # positions = positions | (input_ids == flag)
        positions = positions | (input_ids == self.judge_tokens[0])
        positions = positions.float()
                
        positions[:prompt_len] = 0
        
        step_labels = torch.tensor([x["tag"] for x in labels]).to(positions.dtype)
        assert step_labels.abs().sum() > 0, step_labels

        process_labels = process_labels.to(step_labels.dtype)
        scatter_index = torch.where(positions != 0)[0]
        assert len(scatter_index) == len(step_labels), f"{len(scatter_index)} != {len(step_labels)}, {scatter_index} != {step_labels}, prompt_len={prompt_len}"
        
        process_labels = process_labels.scatter_(0, scatter_index, step_labels)

        # input_ids = revert_special_tokens(self.tokenizer, input_ids)

        input_ids = input_ids.view(1, -1)
        attention_mask = attention_mask.view(1, -1)
        process_labels = process_labels.view(1, -1)
        # assert process_labels[0].abs().sum() > 0, process_labels

        return (
            input_ids,
            attention_mask,
            process_labels
        )

    def collate_fn(self, item_list):
        input_ids = []
        attention_masks = []
        process_labels = []
        
        for input_id, attention_mask, process_label in item_list:
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            process_labels.append(process_label)

        input_ids = zero_pad_sequences(input_ids, value=self.tokenizer.pad_token_id, side="left")
        attention_masks = zero_pad_sequences(attention_masks, side="left")
        process_labels = zero_pad_sequences(process_labels, side="left")

        return input_ids, attention_masks, process_labels


def process_finegrained_data_for_inference(data, prompt_key, response_key, source_key=None):
    prompt = data[prompt_key]
    if source_key is None or source_key not in data:
        source_type = "unknown"
    else:
        source_type = data[source_key]

    response = data[response_key]
    if "\n\n" in response and response.count("\n\n") > 2:
        steps = response.split("\n\n")
    else:
        steps = response.split("\n")
        steps = [x for x in steps if len(x) > 0]
    
    def random_interleaved_concat(strings):
        result = strings[0]
        separators = list(SPLIT_MAP.values())
        for s in strings[1:]:
            separator = random.choice(separators)
            result += separator + s
        return result
    
    # formatted_response = SPLIT_MAP["\n"].join(steps)
    formatted_response = random_interleaved_concat(steps)

    history = data.get("history", None)        
    return prompt, formatted_response, history, source_type
    

class RewardProcessDatasetInference(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template="Human: {}\nAssistant: ",
    ) -> None:
        super().__init__()
        self.prompts = []
        self.responses = []
        self.labels = []
        self.margins = []
        self.history = []
        self.data_source = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        self.current_model = strategy.args.pretrain
        if "glm" in self.current_model:
            self.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|user|>")
        else:
            self.eos_token_id = None

        prompt_key = getattr(self.strategy.args, "prompt_key", None)
        response_key = getattr(self.strategy.args, "response_key", None)
        source_key = getattr(self.strategy.args, "source_key", "type")
        label_key = getattr(self.strategy.args, "label_key", "labels")

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, response, label, history, source_type = process_finegrained_data_for_inference(
                data, prompt_key, response_key, source_key
            )
            # if source_type is None:
                # source_type = "unknown"
                
            self.prompts.append(prompt)
            self.responses.append(response)
            self.history.append(history)
            self.data_source.append(source_type)
        
        # self.judge_tokens = [self.tokenizer.encode(x, add_special_tokens=False)[0] for x in SPLIT_MAP.values()]
        # self.judge_tokens.append(self.tokenizer.convert_tokens_to_ids("<|user|>"))
        self.judge_tokens = get_process_flag_tokens(self.tokenizer)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt, response, history, label = self.prompts[idx], self.responses[idx], self.history[idx], self.labels[idx]
        source_type = self.data_source[idx]
        
        if "glm" in self.current_model.lower():
            return self.tokenize_func_chatglm(prompt, response, history, label, source_type)

        assert False, "Unknown model, not ChatGLM"
    
    def tokenize_func_chatglm(self, prompt, response, history, labels, source_type):
        # item["role"], item.get("metadata", ""), content)
        history_input_ids = []
        if history is not None:
            for item in history:
                history_input_ids.extend(self.tokenizer.build_single_message("user", "", item["prompt"]))
                history_input_ids.extend(self.tokenizer.build_single_message("assistant", "", item["response"]))
        
        def conjecture(response):
            sample_input_ids = history_input_ids + self.tokenizer.build_single_message("user", "", prompt)
            prompt_len = len(sample_input_ids)
            sample_input_ids = sample_input_ids + self.tokenizer.build_single_message("assistant", "", response)

            sample_input_ids = sample_input_ids[:self.max_length-3] + [self.tokenizer.convert_tokens_to_ids("<|user|>")]
            
            sample = self.tokenizer.batch_encode_plus([sample_input_ids], return_tensors="pt", is_split_into_words=True)

            return sample["input_ids"], sample["attention_mask"], prompt_len

        input_ids, attention_mask, prompt_len = conjecture(response)
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        input_ids = input_ids.view(-1)
        attention_mask = attention_mask.view(-1)

        # input_ids = revert_special_tokens(self.tokenizer, input_ids)

        input_ids = input_ids.view(1, -1)
        attention_mask = attention_mask.view(1, -1)

        return (
            input_ids,
            attention_mask,
        )

    def collate_fn(self, item_list):
        input_ids = []
        attention_masks = []

        for input_id, attention_mask, process_label in item_list:
            input_ids.append(input_id)
            attention_masks.append(attention_mask)

        input_ids = zero_pad_sequences(input_ids, value=self.tokenizer.pad_token_id, side="left")
        attention_masks = zero_pad_sequences(attention_masks, side="left")

        return input_ids, attention_masks