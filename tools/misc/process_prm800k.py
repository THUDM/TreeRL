def func(x):
    problem = x["question"]["problem"]
    steps = x["question"]["pre_generated_steps"]
    label_steps = x["label"]["steps"]
    out_steps = []
    for step in label_steps:
        if len(step["completions"]) == 1:
            out_steps.append(step["completions"][0])
            if step["completions"][0]["rating"] == -1:
                break
        else:
            sst = [item for item in step["completions"] if item["rating"] == -1]
            if len(sst) > 0:
                out_steps.append(sst[0])
                break
            else:
                out_steps.append(step["completions"][0])

    finish_reason = x["label"]["finish_reason"] if "finish_reason" in x["label"] else "unknown"

    if len(out_steps) < 1:
        print(x)
        return None
    
    if out_steps[-1]["text"] != steps[len(out_steps)-1]:
        steps = [item["text"] for item in out_steps]
            
    out_steps_map = {item["text"]: item["rating"] for item in out_steps}
    labels = []
            
    for x in steps:
        lbl = out_steps_map.get(x, 0)
        labels.append((x, lbl))
    
    return {
        "prompt": problem,
        "response": "\n".join(steps),
        "labels": labels,
        "finish_reason": finish_reason
    }
