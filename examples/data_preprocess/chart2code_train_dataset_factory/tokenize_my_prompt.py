

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("/data/MY_USERNAME/hf_dl/Qwen2.5-VL-7B-Instruct")


def parse(tokenizer, system_prompt, user_prompt):    
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


if __name__ == "__main__":
    rl_system_prompt_chart2code_cot_v1 = """You are an expert Python developer specializing in generating matplotlib code to reproduce a given chart. Please think through the reasoning process in your mind and then provides the user with the matplotlib code that can reproduce the picture. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., 
<think>
reasoning process here
</think>
<answer>
```python
matplotlib code here
```
</answer>
"""

    user_prompt = "Please generate matplotlib code according the given chart image."

    tokens = parse(tokenizer, rl_system_prompt_chart2code_cot_v1, user_prompt)

    print(len(tokens))
