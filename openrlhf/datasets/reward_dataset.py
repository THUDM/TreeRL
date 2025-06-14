import random
from typing import Callable
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import re
from .utils import exist_and_not_none, zero_pad_sequences


""" 
{
    "prompt": str,
    "chosen_response": str,
    "rejected": str,
    "history": list[Dict] or None,
    "labels": list[Dict] or None,
    "type": str
}
"""

def preprocess_data(data, input_template=None, prompt_key=None, chosen_key=None, rejected_key=None, source_key=None, label_key=None) -> str:
    # custom dataset
    source_type = data[source_key] if source_key else None
    if source_type in ('math', 'code'):
        prompt = data[prompt_key]
        if label_key is not None and label_key in data:
            chosen = data[label_key]
            reject = ""
        else:
            chosen = data[chosen_key]
            reject = data[rejected_key]
    elif chosen_key and rejected_key:
        if prompt_key:
            prompt = data[prompt_key]
        else:
            prompt = ""
            input_template = None  # do not modified with input template again
        chosen = data[chosen_key]
        reject = data[rejected_key]
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
    # if input_template:
    #     prompt = input_template.format(prompt)
    return prompt, chosen, reject, history, margin, source_type

def preprocess_data_mult(data, input_template=None, prompt_key=None, chosen_key=None, rejected_key=None, source_key=None, label_key=None,choice_type_key=None,chosen_step_key=None,rejected_step_key=None
            ) -> str:
    # custom dataset
    source_type = data[source_key] if source_key else None

    if prompt_key:
        prompt = data[prompt_key]
    else:
        prompt = ""
        input_template = None  # do not modified with input template again
    chosen = data[chosen_key]
    reject = data[rejected_key]
    choice_type = data[choice_type_key]
    chosen_step = data[chosen_step_key]
    reject_step = data[rejected_step_key]

    history = data.get("history", None)        

    # margin loss
    margin = data["margin"] if exist_and_not_none(data, "margin") else 0

    # input template
    # if input_template:
    #     prompt = input_template.format(prompt)
    return prompt, chosen, reject, choice_type, chosen_step, reject_step, history, margin, source_type


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
        if "glm" in self.current_model.lower():
            self.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|user|>")
        else:
            self.eos_token_id = None

        prompt_key = getattr(self.strategy.args, "prompt_key", None)
        chosen_key = getattr(self.strategy.args, "chosen_key", None)
        rejected_key = getattr(self.strategy.args, "rejected_key", None)
        source_key = getattr(self.strategy.args, "source_key", "type")
        label_key = getattr(self.strategy.args, "label_key", "labels")
        self.chosen_eos = []
        self.rejected_eos = []

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, chosen, reject, history, margin, source_type = preprocess_data(
                data, input_template, prompt_key, chosen_key, rejected_key, source_key, label_key
            )
            if source_type is None:
                source_type = "unknown"
                
            self.prompts.append(prompt)
            self.chosens.append(chosen)
            self.rejects.append(reject)
            self.margins.append(margin)
            self.history.append(history)
            self.data_source.append(source_type)
            self.chosen_eos.append(data.get("chosen_eos", True))
            self.rejected_eos.append(data.get("rejected_eos", True))       

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject, history, margin = self.prompts[idx], self.chosens[idx], self.rejects[idx], self.history[idx], self.margins[idx]
        source_type = self.data_source[idx]
        
        chosen_eos = self.chosen_eos[idx]
        rejeced_eos = self.rejected_eos[idx]
        
        if "glm" in self.current_model.lower():
            return self.tokenize_func_chatglm(prompt, chosen, reject, history, margin, source_type, chosen_eos, rejeced_eos)
        
        if "qwen" in self.current_model.lower() or "llama" in self.current_model.lower():
            return self.tokenize_func_llama(prompt, chosen, reject, history, margin, source_type)

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
        
    def tokenize_func_llama(self, prompt, chosen, rejected, history, margin, source_type, chosen_eos=True, rejected_eos=True):
        def conjecture(response):
            conversation = []
            for x in history:
                conversation.extend[{"role": "user ", "metadata": "", "content": x["prompt"]}, {"role": "assistant", "content": x["response"]}]
            conversation.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
            # rejected_conv = [{"role": "user", "metadata": "", "content": x["prompt"]}, {"role": "assistant", "content": x["response"]} for x in history] + [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}]
            conversation = self.tokenizer.apply_chat_template(conversation)
            # with open("/workspace/lurui/test/llama_conversation.txt", "w") as f:
            #     f.write(conversation)
            #     f.write("\n")
            print(conversation)
            # rejected_conv = self.tokenizer.apply_chat_template(rejected_conv)
            attention_mask = [1] * conversation

            return torch.tensor(conversation), torch.tensor(attention_mask)

        if source_type in ("math", "code"):
            assert len(chosen) == 0 or len(rejected) == 0, "Math source should have only one response"
            label = 1 if chosen else -1
            sample = conjecture(chosen) if chosen else conjecture(rejected)
            return (
                sample[0], sample[1], sample[0], sample[1], margin, label
            )
        else:
            chosen_sample = conjecture(chosen)
            reject_sample = conjecture(rejected)
            label = 0
            return (
                chosen_sample[0],
                chosen_sample[1],
                reject_sample[0],
                reject_sample[1],
                margin,
                label
            )

    def tokenize_func_chatglm(self, prompt, chosen, reject, history, margin, source_type, chosen_eos=True, rejected_eos=True):
        # item["role"], item.get("metadata", ""), content)
        history_input_ids = []
        if history is not None:
            for item in history:
                history_input_ids.extend(self.tokenizer.build_single_message("user", "", item["prompt"]))
                history_input_ids.extend(self.tokenizer.build_single_message("assistant", "", item["response"]))
        
        def conjecture(response, eos=True):
            sample_input_ids = self.tokenizer.build_single_message("user", "", prompt) + self.tokenizer.build_single_message("assistant", "", response)

            sample_input_ids = history_input_ids + sample_input_ids
            sample_input_ids = sample_input_ids[:self.max_length-3] 
            
            if eos:
                sample_input_ids += [self.tokenizer.convert_tokens_to_ids("<|user|>")]
                
            # print("Sample tokens:", self.tokenizer.convert_ids_to_tokens(sample_input_ids))
            # print("Sample string:", self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(sample_input_ids)))
            
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
            chosen_sample = conjecture(chosen, chosen_eos)
            reject_sample = conjecture(reject, rejected_eos)
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
        pairwise_list = [x for x in item_list if not torch.is_tensor(x[-1]) and x[-1] == 0]
        instancewise_list = [x for x in item_list if torch.is_tensor(x[-1]) or (not torch.is_tensor(x[-1]) and x[-1] != 0)]
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
            
        if torch.is_tensor(labels[0]):
            labels = torch.stack(labels).squeeze()
        else:
            labels = torch.tensor(labels, dtype=torch.float32)

        return chosen_ids, chosen_masks, reject_ids, rejects_masks, torch.tensor(margins, dtype=torch.float32), labels


SPLIT_MAP = {
    # "\n": "к",
    "\n\n": "и",
    # "\n\n": "<eop>",
}


def reformat_response_into_steps(text, use_nk=False, flag="", join=True, return_num_splits=False):
    # 使用正则表达式分割文本，保留分隔符
    if use_nk:
        parts = re.split(r'(\n+)', text)
    else:
        parts = re.split(r'(\n+\s*?\n+)', text)

    # print(parts)
    # 将分隔符与前一句合并
    result = []
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            result.append(parts[i] + parts[i+1])
        else:
            if len(parts[i]) > 0:
                result.append(parts[i])
            break
    
    num_steps = len(result)
    # print(result)
    if join:
        result = flag.join(result) + flag
    else:
        result = [x + flag for x in result]

    if return_num_splits:
        return result, num_steps
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


class RewardMixProcessDataset(RewardDataset):
    def __init__(self, dataset, tokenizer, max_length: int, strategy, input_template="Human: {}\nAssistant: ") -> None:
        super().__init__(dataset, tokenizer, max_length, strategy, input_template)
        self.judge_tokens = get_process_flag_tokens(self.tokenizer)
        assert len(self.judge_tokens) == 1

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt, chosen, reject, history, margin = self.prompts[idx], self.chosens[idx], self.rejects[idx], self.history[idx], self.margins[idx]
        source_type = self.data_source[idx]
        
        chosen_eos = self.chosen_eos[idx]
        rejeced_eos = self.rejected_eos[idx]
        
        if "glm" in self.current_model.lower():
            return self.tokenize_func_chatglm(prompt, chosen, reject, history, margin, source_type, chosen_eos, rejeced_eos)
        
        if "qwen" in self.current_model.lower() or "llama" in self.current_model.lower():
            return self.tokenize_func_llama(prompt, chosen, reject, history, margin, source_type)

        assert False, "Unknown model, not ChatGLM"

    def tokenize_func_chatglm(self, prompt, chosen, reject, history, margin, source_type, chosen_eos=True, rejected_eos=True):
        # item["role"], item.get("metadata", ""), content)
        history_input_ids = []
        if history is not None:
            for item in history:
                history_input_ids.extend(self.tokenizer.build_single_message("user", "", item["prompt"]))
                history_input_ids.extend(self.tokenizer.build_single_message("assistant", "", item["response"]))
        
        def conjecture_orm(response, eos=True):
            sample_input_ids = self.tokenizer.build_single_message("user", "", prompt) + self.tokenizer.build_single_message("assistant", "", response)

            sample_input_ids = history_input_ids + sample_input_ids
            sample_input_ids = sample_input_ids[:self.max_length-3] 
            
            if eos:
                sample_input_ids += [self.tokenizer.convert_tokens_to_ids("<|user|>")]
            
            sample = self.tokenizer.batch_encode_plus([sample_input_ids], return_tensors="pt", is_split_into_words=True)
            return sample["input_ids"], sample["attention_mask"]

        def random_interleaved_concat(strings):
            strings = [x.strip() for x in strings]
            result = strings[0]
            # separators = list(SPLIT_MAP.keys())
            text_op, separator = random.choice(list(SPLIT_MAP.items()))
            if len(strings) > 1:
                for sentence in strings[1:]:
                    # separator = random.choice(separators)
                    result += text_op + separator + sentence
            result += separator
            return result
        
        def conjecture_prm(response, eos=True):
            steps = [x["text"] for x in response]
            formatted_response = random_interleaved_concat(steps)
            
            sample_input_ids = history_input_ids + self.tokenizer.build_single_message("user", "", prompt)
            prompt_len = len(sample_input_ids) + 2
            # sample_input_ids = sample_input_ids 
            response_ids = self.tokenizer.build_single_message("assistant", "", formatted_response)
            prompt_len = min(prompt_len, self.max_length - len(response_ids) - 1)
            sample_input_ids = sample_input_ids[-prompt_len:] + response_ids
            sample_input_ids = sample_input_ids + [self.tokenizer.convert_tokens_to_ids("<|user|>")]
            # sample_input_ids = sample_input_ids[:self.max_length-3] + [self.tokenizer.convert_tokens_to_ids("<|user|>")]
            # sample_input_ids = sample_input_ids[-(self.max_length-3):] + [self.tokenizer.convert_tokens_to_ids("<|user|>")]
            sample = self.tokenizer.batch_encode_plus([sample_input_ids], return_tensors="pt", is_split_into_words=True)
            
            input_ids = sample["input_ids"].view(-1)
            attention_mask = sample["attention_mask"].view(-1)
            
            process_labels = torch.zeros_like(input_ids)
            positions = torch.zeros_like(input_ids).bool()
            positions = positions | (input_ids == self.judge_tokens[0])
            positions = positions.float()
            assert (positions.sum(-1) > 0).all(), f"positions={positions}, response={formatted_response}"
            positions[:prompt_len] = 0
            
            # formatted_response = SPLIT_MAP["\n"].join(steps)
            step_labels = torch.tensor([x["tag"] for x in response]).to(positions.dtype)
            process_labels = process_labels.to(step_labels.dtype)
            scatter_index = torch.where(positions != 0)[0]
            #TODO How to clip out of max_lenth
            # step_labels = step_labels[:len(scatter_index)]
            
            assert len(scatter_index) == len(step_labels), f"{formatted_response}, {scatter_index} != {step_labels}, prompt_len={prompt_len}"
            assert len(step_labels) > 0, step_labels
            process_labels = process_labels.scatter_(0, scatter_index, step_labels)       
            
            input_ids = input_ids.view(1, -1)
            attention_mask = attention_mask.view(1, -1)
            process_labels = process_labels.view(1, -1)     
            return input_ids, attention_mask, process_labels

        if source_type in ("math", "code"):
            assert len(chosen) == 0 or len(reject) == 0, "Math source should have only one response"
            if isinstance(chosen, str):
                label = 1 if chosen else -1
                sample = conjecture_orm(chosen) if chosen else conjecture_orm(reject)
            else:
                sample = conjecture_prm(chosen) if chosen else conjecture_prm(reject)
                sample, label = (sample[0], sample[1]), sample[2]
            # chosen_sample = conjecture(chosen)
            return (
                sample[0], sample[1], sample[0], sample[1], margin, label
            )
        else:
            chosen_sample = conjecture_orm(chosen, chosen_eos)
            reject_sample = conjecture_orm(reject, rejected_eos)
            label = 0
            return (
                chosen_sample[0],
                chosen_sample[1],
                reject_sample[0],
                reject_sample[1],
                margin,
                label
            )


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
        if "glm" in self.current_model.lower():
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
        if "glm" in self.current_model.lower():
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


class RewardMultiTaskDataset(Dataset):
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
        self.choice_types = []
        self.chosen_steps = []
        self.rejected_steps = []
        self.margins = []
        self.history = []
        self.data_source = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        self.current_model = strategy.args.pretrain
        if "glm" in self.current_model.lower():
            self.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|user|>")
        else:
            self.eos_token_id = None

        prompt_key = getattr(self.strategy.args, "prompt_key", None)
        chosen_key = getattr(self.strategy.args, "chosen_key", None)
        rejected_key = getattr(self.strategy.args, "rejected_key", None)
        choice_type_key = getattr(self.strategy.args, "choice_type_key", None)
        chosen_step_key = getattr(self.strategy.args, "chosen_step_key", None)
        rejected_step_key = getattr(self.strategy.args, "rejected_step_key", None)
        source_key = getattr(self.strategy.args, "source_key", "type")
        label_key = getattr(self.strategy.args, "label_key", "labels")
        self.use_multi_task = getattr(self.strategy.args, "use_multi_task", False)
        print("Use multi task:", self.use_multi_task)
        self.chosen_eos = []
        self.rejected_eos = []

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, chosen, reject, choice_type, chosen_step, reject_step, history, margin, source_type = preprocess_data_mult(
                data, input_template, prompt_key, chosen_key, rejected_key, source_key, label_key, choice_type_key,chosen_step_key,rejected_step_key
            )
            if source_type is None:
                source_type = "unknown"
                
            self.prompts.append(prompt)
            self.chosens.append(chosen)
            self.rejects.append(reject)
            self.choice_types.append(choice_type)
            self.chosen_steps.append(chosen_step)
            self.rejected_steps.append(reject_step)
            self.margins.append(margin)
            self.history.append(history)
            self.data_source.append(source_type)
            self.chosen_eos.append(data.get("chosen_eos", True))
            self.rejected_eos.append(data.get("rejected_eos", True))       

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject, choice_type, chosen_step, rejected_step, history, margin = self.prompts[idx], self.chosens[idx], self.rejects[idx], self.choice_types[idx], self.chosen_steps[idx], self.rejected_steps[idx], self.history[idx], self.margins[idx]
        source_type = self.data_source[idx]
        
        chosen_eos = self.chosen_eos[idx]
        rejeced_eos = self.rejected_eos[idx]
        
        if "glm" in self.current_model.lower():
            return self.tokenize_func_chatglm(prompt, chosen, reject, choice_type, chosen_step, rejected_step, history, margin, source_type, chosen_eos, rejeced_eos, self.use_multi_task)
        
        if "qwen" in self.current_model.lower() or "llama" in self.current_model.lower():
            return self.tokenize_func_llama(prompt, chosen, reject, history, margin, source_type)

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
        
    def tokenize_func_llama(self, prompt, chosen, rejected, history, margin, source_type, chosen_eos=True, rejected_eos=True):
        def conjecture(response):
            conversation = []
            for x in history:
                conversation.extend[{"role": "user ", "metadata": "", "content": x["prompt"]}, {"role": "assistant", "content": x["response"]}]
            conversation.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
            # rejected_conv = [{"role": "user", "metadata": "", "content": x["prompt"]}, {"role": "assistant", "content": x["response"]} for x in history] + [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}]
            conversation = self.tokenizer.apply_chat_template(conversation)
            # with open("/workspace/lurui/test/llama_conversation.txt", "w") as f:
            #     f.write(conversation)
            #     f.write("\n")
            print(conversation)
            # rejected_conv = self.tokenizer.apply_chat_template(rejected_conv)
            attention_mask = [1] * conversation

            return torch.tensor(conversation), torch.tensor(attention_mask)

        if source_type in ("math", "code"):
            assert len(chosen) == 0 or len(rejected) == 0, "Math source should have only one response"
            label = 1 if chosen else -1
            sample = conjecture(chosen) if chosen else conjecture(rejected)
            return (
                sample[0], sample[1], sample[0], sample[1], margin, label
            )
        else:
            chosen_sample = conjecture(chosen)
            reject_sample = conjecture(rejected)
            label = 0
            return (
                chosen_sample[0],
                chosen_sample[1],
                reject_sample[0],
                reject_sample[1],
                margin,
                label
            )

    def tokenize_func_chatglm(self, prompt, chosen, reject, choice_type, chosen_step, reject_step, history, margin, source_type, chosen_eos=True, rejected_eos=True,use_multi_task = False):
        # item["role"], item.get("metadata", ""), content)
        if use_multi_task:
            
            def conjecture(full, step, choice_type, eos=True):
                sample_input_ids = self.tokenizer.build_single_message("user", "", prompt) + self.tokenizer.build_single_message("assistant", "", full)

                if choice_type == "step":
                    sample_input_ids = sample_input_ids + self.tokenizer.build_single_message("user", "", "Evaluate this step: "+ step)
                else:
                    sample_input_ids = sample_input_ids + self.tokenizer.build_single_message("user", "", "Evaluate the solution.")

                sample_input_ids = sample_input_ids[:self.max_length-3] 
                
                if eos:
                    sample_input_ids += [self.tokenizer.convert_tokens_to_ids("<|assistant|>")]
                    
                # print("Sample tokens:", self.tokenizer.convert_ids_to_tokens(sample_input_ids))
                # print("Sample string:", self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(sample_input_ids)))
                
                sample = self.tokenizer.batch_encode_plus([sample_input_ids], return_tensors="pt", is_split_into_words=True)
                return sample["input_ids"], sample["attention_mask"]

            chosen_sample = conjecture(chosen, chosen_step, choice_type, chosen_eos)
            reject_sample = conjecture(reject, reject_step, choice_type, rejected_eos)
            label = 0
            return (
                chosen_sample[0],
                chosen_sample[1],
                reject_sample[0],
                reject_sample[1],
                margin,
                label
            )
        else:
            history_input_ids = []
            if history is not None:
                for item in history:
                    history_input_ids.extend(self.tokenizer.build_single_message("user", "", item["prompt"]))
                    history_input_ids.extend(self.tokenizer.build_single_message("assistant", "", item["response"]))
            
            def conjecture(response, eos=True):
                sample_input_ids = self.tokenizer.build_single_message("user", "", prompt) + self.tokenizer.build_single_message("assistant", "", response)

                sample_input_ids = history_input_ids + sample_input_ids
                sample_input_ids = sample_input_ids[:self.max_length-3] 
                
                if eos:
                    sample_input_ids += [self.tokenizer.convert_tokens_to_ids("<|user|>")]
                
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
                chosen_sample = conjecture(chosen, chosen_eos)
                reject_sample = conjecture(reject, rejected_eos)
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
        pairwise_list = [x for x in item_list if not torch.is_tensor(x[-1]) and x[-1] == 0]
        instancewise_list = [x for x in item_list if torch.is_tensor(x[-1]) or (not torch.is_tensor(x[-1]) and x[-1] != 0)]
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
            
        if torch.is_tensor(labels[0]):
            labels = torch.stack(labels).squeeze()
        else:
            labels = torch.tensor(labels, dtype=torch.float32)

        return chosen_ids, chosen_masks, reject_ids, rejects_masks, torch.tensor(margins, dtype=torch.float32), labels


SPLIT_MAP = {
    # "\n": "к",
    "\n\n": "и",
    # "\n\n": "<eop>",
}


def reformat_response_into_steps(text, use_nk=False, flag="", join=True, return_num_splits=False):
    # 使用正则表达式分割文本，保留分隔符
    if use_nk:
        parts = re.split(r'(\n+)', text)
    else:
        parts = re.split(r'(\n+\s*?\n+)', text)

    # print(parts)
    # 将分隔符与前一句合并
    result = []
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            result.append(parts[i] + parts[i+1])
        else:
            if len(parts[i]) > 0:
                result.append(parts[i])
            break
    
    num_steps = len(result)
    # print(result)
    if join:
        result = flag.join(result) + flag
    else:
        result = [x + flag for x in result]

    if return_num_splits:
        return result, num_steps
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