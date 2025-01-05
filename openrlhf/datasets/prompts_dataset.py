from torch.utils.data import Dataset
from tqdm import tqdm
import json
from .utils import exist_and_not_none


def preprocess_data(data, input_template=None, input_key=None) -> str:
    # custom dataset
    if input_key:
        prompt = data[input_key]
    else:
        # Dahoas/full-hh-rlhf
        if exist_and_not_none(data, "prompt"):
            prompt = data["prompt"]
            # tasksource/oasst1_pairwise_rlhf_reward
            if prompt.startswith("prompter:"):
                prompt = (
                    prompt.replace("prompter:", "\nHuman: ").replace("assistant:", "\nAssistant: ") + "\nAssistant: "
                )
            input_template = None  # do not modified with input template again
        # Open-Orca/OpenOrca
        elif exist_and_not_none(data, "system_prompt") and exist_and_not_none(data, "response"):
            prompt = data["system_prompt"] + "\n" + data["question"]
        # BelleGroup/train_0.5M_CN
        # LLMs/Alpaca-ShareGPT
        # yahma/alpaca-cleaned
        # QingyiSi/Alpaca-CoT
        elif exist_and_not_none(data, "instruction") and exist_and_not_none(data, "output"):
            input = " " + data["input"] if exist_and_not_none(data, "input") else ""
            prompt = data["instruction"] + input
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

            prompt = data["conversation_a"][:-1]
            prompt = process_chatbot_arena_conversations(prompt)
            input_template = None  # do not modified with input template again
        # openai/webgpt_comparisons
        elif exist_and_not_none(data, "question") and exist_and_not_none(data, "answer_1"):
            prompt = data["question"]["full_text"]
        else:
            raise ValueError("Unknown prompts dataset")

    history = data.get("history", None)
    if history is None:
        history = []
    
    # input template
    # if input_template:
        # prompt = input_template.format(prompt)
    return prompt, history


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template="Human: {}\nAssistant: ",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        # self.input_template = input_template
        self.input_template = None
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        source_key = getattr(self.strategy.args, "source_key", None)
        
        self.current_model = strategy.args.pretrain

        # assert label_key is not None, f"label_key={label_key}"

        self.prompts = []
        self.history = []
        self.labels = []
        self.sources = []
        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, history = preprocess_data(data, input_template, input_key)
            if label_key is not None:
                label = data[label_key]
            else:
                label = None
                
            if source_key is not None:
                source = data[source_key]
            else:
                source = None
                
            self.prompts.append(prompt)
            self.history.append(history)
            self.labels.append(label)
            self.sources.append(source)

    def __len__(self):
        length = len(self.prompts)
        return length

    # def __getitem__(self, idx):
    #     if "glm" in self.current_model or "llama" in self.current_model or "qwen" in self.current_model:
    #         return self.prompts[idx], json.dumps(self.history[idx], ensure_ascii=False)
    #     else:
    #         return self.prompts[idx]

    def __getitem__(self, idx):
        print("current model",self.current_model)
        if "glm" in self.current_model.lower() or "llama" in self.current_model.lower() or "qwen" in self.current_model.lower():
            history = self.history[idx]
            prompt = self.prompts[idx]
            label = self.labels[idx]
            source = self.sources[idx]
            if history is None:
                history = []
            
            if label is not None:
                output = history + [{"prompt": prompt, self.strategy.args.label_key: label}]
            else:
                output = history + [{"prompt": prompt}]

            if source is not None:
                output[-1][self.strategy.args.source_key] = source

            return json.dumps(output)
            # return self.prompts[idx], json.dumps(self.history[idx], ensure_ascii=False)
        else:
            return self.prompts[idx]