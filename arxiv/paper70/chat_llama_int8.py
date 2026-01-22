#!/usr/bin/env python3
"""
Llama-3.3-70B-Instruct-INT8 交互式聊天

在 Docker 容器内运行：
    python /workspace/ai-theorys-study/arxiv/paper70/chat_llama_int8.py

依赖（必须用旧版本）：
    pip install transformers==4.51.3 compressed-tensors==0.9.0 accelerate

命令：
    - 输入问题开始聊天
    - 输入 'clear' 清空对话历史
    - 输入 'quit' 或 'exit' 退出

Author: Zero + Suzaku (Claude Code)
Date: 2026-01-22
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/workspace/models/Llama-3.3-70B-Instruct-INT8"

def load_model():
    """加载模型"""
    print("=" * 60)
    print("Loading Llama-3.3-70B-Instruct-INT8...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    model.eval()

    print(f"Model loaded. Device: {next(model.parameters()).device}")
    return model, tokenizer


def chat_stream(model, tokenizer, messages, max_new_tokens=512):
    """流式生成回复，一个字一个字输出"""
    from transformers import TextIteratorStreamer
    from threading import Thread

    # 使用 Llama 3 的 chat 模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 创建流式输出器
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

    # 在后台线程中生成
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 流式输出
    response = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        response += new_text

    thread.join()
    print()  # 换行

    return response.strip()


def main():
    model, tokenizer = load_model()

    print("\n" + "=" * 60)
    print("Llama-3.3-70B-Instruct-INT8 Chat")
    print("Commands: 'clear' = 清空历史, 'quit' = 退出")
    print("=" * 60 + "\n")

    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit']:
            print("Bye!")
            break

        if user_input.lower() == 'clear':
            messages = []
            print(">>> 对话历史已清空\n")
            continue

        # 添加用户消息
        messages.append({"role": "user", "content": user_input})

        # 生成回复（流式输出）
        print("Llama: ", end="", flush=True)
        response = chat_stream(model, tokenizer, messages)

        # 添加助手回复到历史
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
