import json
import copy
from tqdm import tqdm


data_path = "/workspace/zhenyu/data/pairwise_data/reward_data/0613/20240613v2-rm_path-0625.jsonl"


def merge_data(data_list, original_data_list):
    # merge identical prefix
    sample_id_dict = {}
    for item in data_list:
        if item["_id"] not in sample_id_dict:
            sample_id_dict[item["_id"]] = []
        sample_id_dict[item["_id"]].append(item)
    samples = []
    for item in sample_id_dict.values():
        sample = item[0]
        rewards = [x["reward"] for x in item]
        reward = sum(rewards) / len(rewards)
        sample["prefix_reward"] = reward   
        samples.append(sample)

    original_samples = {x["id"]: x for x in original_data_list}

    merged_samples = {}
    for sample in samples:
        sample_id = sample["_id"].split("_")[0] 
        sample["prefix"] = original_samples[sample["_id"]]["prefix"]
        sample["history"] = original_samples[sample["_id"]]["history"]
        sample["source"] = original_samples[sample["_id"]]["source"]
        if sample_id not in merged_samples:
            merged_samples[sample_id] = []
        merged_samples[sample_id].append(sample)

    output = []
    samples = list(merged_samples.values())
    for sample in samples:
        sample = sorted(sample, key=lambda x: len(x["prefix"]))
        out_sample = []
        for idx, item in enumerate(sample):
            if idx == 0:
                out_sample.append((item["prefix"], item["prefix_reward"]))
            else:
                out_sample.append((item["prefix"][len(out_sample[-1][0]):], item["prefix_reward"]))
        out_sample.append((item["response"][len(out_sample[-1][0]):], item["reward"]))
        response = "\n\n".join([x[0] for x in out_sample])
        out_reward = [x[1] for x in out_sample]
        labels = out_sample
        
        output.append({
            "response": response, 
            # "reward": out_reward,
            "prompt": sample[0]["prompt"],
            "history": sample[0]["history"],
            "labels": labels,
        })
        
    return output


def filter_func(item):
    # not (len(item["generated_paths"]) == 1 and len("".join(x["generated_paths"][0]["extension"])) <= 1) and x["generated_paths"][0]["step"] != x["response_chosen"]
    generated_paths = item["generated_paths"]
    generated_paths = [x for x in generated_paths if len("".join(x["extension"])) > 0]

    if len(generated_paths) == 0:
        return None
    generated_paths = [x for x in generated_paths if x["step"] != item["response_chosen"] and x["step"][-1] != "\n"]
    if len(generated_paths) == 0:
        return None

    item["generated_paths"] = generated_paths
    return item

print(f"start loading data...")
data = [json.loads(x) for x in open(data_path)]
print(f"start filtering data...")
data_used = [filter_func(x) for x in tqdm(data, desc="Filtering data")]
data_used = [x for x in data_used if x is not None]
for i, item in enumerate(data_used):
    item["id"] = "sample-" + str(i)


print(f"Total data: {len(data)}, used data: {len(data_used)}")

output = []
avg_steps = []
for item in tqdm(data_used, desc="Processing data"):
    item.pop("response_chosen")
    item.pop("response_rejected")
    generated_paths = item["generated_paths"]
    # generated_paths = list(set([x.strip() for x in generated_paths]))

    avg_steps.append(len(generated_paths))
    item.pop("generated_paths")
    for step_id, step in enumerate(generated_paths):
        pie = copy.deepcopy(item)
        prefix = step["step"]
        extensions = step["extension"]
        extensions = list(set([x.strip() for x in extensions]))

        pie["prefix"] = prefix
        for ext in extensions:
            pie["step"] = prefix + "\n" + ext
            pie["id"] = item["id"] + "_" + str(step_id)
            output.append(copy.deepcopy(pie))

avg_steps = sum(avg_steps) / len(avg_steps)
print(f"Average steps: {avg_steps}")

with open("/workspace/zhenyu/data/pairwise_data/reward_data/0613/20240613v2-rm_path_filtered_steps.jsonl", "w") as f:
    for item in output:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
    f.close()

print(f"Total steps: {len(output)}")

