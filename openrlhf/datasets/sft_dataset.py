from typing import Callable
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none, zero_pad_sequences


def preprocess_data(data, input_template=None, input_key=None, output_key=None):
    # custom dataset
    if input_key and output_key:
        prompt = data[input_key]
        target = data[output_key]
    else:
        # Dahoas/full-hh-rlhf
        # iamketan25/open-assistant-instructions
        if exist_and_not_none(data, "prompt") and exist_and_not_none(data, "chosen"):
            prompt = data["prompt"]
            target = data["chosen"]
            input_template = None  # do not modified with input template again
        # pvduy/sharegpt_alpaca_oa_vicuna_format
        elif exist_and_not_none(data, "prompt") and exist_and_not_none(data, "label"):
            prompt = data["prompt"].replace("USER:", "").replace("ASSISTANT:", "")
            target = data["label"].replace("</s>", "")
        # BelleGroup/train_0.5M_CN
        # LLMs/Alpaca-ShareGPT
        # yahma/alpaca-cleaned
        # QingyiSi/Alpaca-CoT
        elif exist_and_not_none(data, "instruction") and exist_and_not_none(data, "output"):
            input = " " + data["input"] if exist_and_not_none(data, "input") else ""
            prompt = data["instruction"] + input
            target = data["output"]
        # Open-Orca/OpenOrca
        elif exist_and_not_none(data, "system_prompt") and exist_and_not_none(data, "response"):
            prompt = data["system_prompt"] + "\n" + data["question"]
            target = data["response"]
        # crumb/gpt4all-clean
        # nomic-ai/gpt4all-j-prompt-generations
        elif exist_and_not_none(data, "prompt") and exist_and_not_none(data, "response"):
            prompt = data["prompt"]
            target = data["response"]
        # EleutherAI/pile [pretrain !!!]
        elif exist_and_not_none(data, "text") and exist_and_not_none(data, "meta"):
            assert input_template is None  # pretrain_mode
            prompt = ""
            target = data["text"]
        # for batch_inference.py
        elif exist_and_not_none(data, "input") and exist_and_not_none(data, "output"):
            prompt = data["input"]
            target = data["output"]
            input_template = None
        else:
            raise ValueError("Unknown SFT dataset")

    # input template
    if input_template:
        prompt = input_template.format(prompt)
    return prompt, target



class SFTDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template="Human: {}\nAssistant: ",
        pretrain_mode=False,
        prompt_key: str = "prompt",
        response_key: str = "response",
    ) -> None:
        super().__init__()
        self.prompts = []
        self.targets = []
        self.prompt_ids_lens = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.current_model = "chatglm" if "glm" in strategy.args.pretrain else ""
        self.history = []
        self.prompt_data_indices = []
        
        input_key = getattr(self.strategy.args, "input_key", None)
        output_key = getattr(self.strategy.args, "output_key", None)

        if "glm" in self.current_model.lower() or "llama" in self.current_model.lower() or "qwen" in self.current_model.lower():
            for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
                # self.prompts.append({"input_ids": data["prompt"], "attention_mask": data["prompt_attention_mask"]})
                # self.targets.append({"input_ids": data["response"], "attention_mask": data["response_attention_mask"]})
                if len(data[prompt_key]) <= 1:
                    continue
                if len(data[response_key]) <= 1:
                    continue

                if "history" in data:
                    if data["history"] is not None:
                        for item in data["history"]:
                            if len(item["prompt"]) <= 1 or len(item["response"]) <= 1:
                                self.history.append(None)
                                break
                        else:
                            self.history.append(data["history"])
                else:
                    self.history.append(None)
                if "_id" in data or "id" in data:
                    if "_id" in data:
                        self.prompt_data_indices.append(data["_id"])
                    else:
                        self.prompt_data_indices.append(data["id"])
                else:
                    self.prompt_data_indices.append(None)

                self.prompts.append(data[prompt_key])
                self.targets.append(data[response_key])
                self.prompt_ids_lens.append(len(data[prompt_key]))
        else:
            for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
                prompt, target = preprocess_data(data, None if pretrain_mode else input_template, input_key, output_key)

                if not self.pretrain_mode:
                    prompt_token = self.tokenizer(
                        prompt,
                        max_length=self.max_length,
                        padding=False,
                        truncation=True,
                        return_tensors="pt",
                    )
                    prompt_ids_len = prompt_token["attention_mask"].sum().item()
                else:
                    prompt_ids_len = 0

                if not self.pretrain_mode:
                    # filter the sample whose length is greater than max_length (2 for answer length)
                    if prompt_ids_len >= self.max_length - 2:
                        continue
                    if not prompt or not target:
                        continue

                self.prompt_ids_lens.append(prompt_ids_len)
                self.prompts.append(prompt)
                self.targets.append(target)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        if "glm" in self.current_model.lower() or "qwen" in self.current_model.lower() or "llama" in self.current_model.lower():
            # prompt_ids_len = self.prompt_ids_lens[idx]
            # input_ids = torch.Tensor(self.prompts[idx]["input_ids"] + self.targets[idx]["input_ids"]).long()
            # attention_mask = torch.ones(input_ids.size(0)).bool()
            prompt = self.prompts[idx]
            response = self.targets[idx]
            history = self.history[idx]
            
            if "glm" in self.current_model.lowewr():
                input_ids, attention_mask, action_mask = self.tokenize_func_chatglm(prompt, response, history)
                prompt_ids_len = 1
            else:
                input_ids, attention_mask, action_mask, prompt_ids_len = self.tokenize_func_llama(prompt, response, history)

            # action_mask = torch.ones(input_ids.size(0))
            # action_mask[:prompt_ids_len] = 0
            if attention_mask.dtype == torch.bool:
                action_mask = action_mask.bool()
                
            data_id = self.prompt_data_indices[idx]

            return (
                prompt_ids_len, 
                input_ids, 
                attention_mask, 
                {"input": prompt, "output": response, "action_mask": action_mask, "history": history, "_id": data_id}
            )
        # elif "qwen" or "llama" in self.current_model:
            
        # others        
        prompt_ids_len = self.prompt_ids_lens[idx]
        prompt = self.prompts[idx]
        target = self.targets[idx]

        input_token = self.tokenizer(
            prompt + target + " " + self.tokenizer.eos_token,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        info = {"input": prompt, "output": target}
        # to avoid EOS_token truncation
        input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input_token["attention_mask"][0][-1] = True
        return prompt_ids_len, input_token["input_ids"], input_token["attention_mask"], info

    def tokenize_func_llama(self, prompt, response, history):
        messages = []
        if history:
            for item in history:
                for item in history:
                    messages.append({"role": "user", "content": item["prompt"]})
                    messages.append({"role": "assistant", "content": item["response"]})
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": response})

        input_ids = self.tokenizer.apply_chat_template(messages)
        prompt_ids = self.tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)
        prompt_len = len(prompt_ids)
        input_ids = torch.Tensor(input_ids)
        attention_masks = torch.ones_like(input_ids)
        action_masks = torch.ones_like(input_ids)
        action_masks[:prompt_len] = 0
        return input_ids, attention_masks, action_masks, prompt_len

    def tokenize_func_chatglm(self, prompt, response, history):
        # item["role"], item.get("metadata", ""), content)
        input_ids = []
        if history is not None:
            for item in history:
                input_ids.extend(self.tokenizer.build_single_message("user", "", item["prompt"]))
                input_ids.extend(self.tokenizer.build_single_message("assistant", "", item["response"]))
        
        def conjecture(response):
            prompt_input_ids = self.tokenizer.build_single_message("user", "", prompt)
            response_input_ids = self.tokenizer.build_single_message("assistant", "", response)
            
            prompt_input_ids = input_ids + prompt_input_ids
            if len(response_input_ids) > self.max_length - 1:
                response_input_ids = response_input_ids[:int(self.max_length*0.9)]
            if len(prompt_input_ids) + len(response_input_ids) > self.max_length:
                prompt_input_ids = prompt_input_ids[-(self.max_length - len(response_input_ids)):]
                
            sample_input_ids = prompt_input_ids + response_input_ids
            # + [self.tokenizer.convert_tokens_to_ids("<|user|>")]
            
            # sample = self.tokenizer.([sample_input_ids], return_tensors="pt", is_split_into_words=True)
            sample_input_ids = torch.tensor(sample_input_ids).long()
            attention_mask = torch.ones_like(sample_input_ids)
            action_mask = torch.ones_like(sample_input_ids)
            action_mask[:len(prompt_input_ids)] = 0
            
            return sample_input_ids, attention_mask, action_mask

        sample = conjecture(response)

        return sample[0], sample[1], sample[2]
            
    def collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids = []
        attention_masks = []
        response_ids_lens = []
        action_masks = []
        infos = {"input": [], "output": [], "history": [], "_id": []}

        for prompt_ids_len, input_id, attention_mask, info in item_list:
            prompt_ids_lens.append(prompt_ids_len)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            response_ids_lens.append(len(input_ids) - prompt_ids_len)
            # if "action_mask" in info:
            # action_masks.append(info["action_mask"])
            # else:
                # action_masks.append(attention_mask)
            # infos["input"].append(info["input"])
            # infos["output"].append(info["output"])
            # infos["history"].append(info["history"])
            # infos["_id"].append(info["_id"])


        input_ids = zero_pad_sequences(input_ids, "left", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "left", 0)
        # action_masks = zero_pad_sequences(action_masks, "right")
        # infos["action_mask"] = action_masks
        length = input_ids.shape[1]
        prompt_ids_lens = [length - x for x in response_ids_lens]
        return prompt_ids_lens, input_ids, attention_masks, "hi"
        # return prompt_ids_lens, input_ids, attention_masks, infos
