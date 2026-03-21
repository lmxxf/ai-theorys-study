"""
Doc-to-LoRA 消融实验: 准确率 vs 块数曲线
红楼梦英译本 (Hung Lou Meng, Book I) 前 8 章, 24 道问答题

实验设计:
- 只截取前 8 章文本 (~85K tokens, ~166 块)
- 只用前 8 章的 24 道题
- Baseline + N=1, 5, 20, 50, 100, ALL
"""

import gc
import json
import sys
import time

import torch

sys.path.insert(0, "/workspace/doc-to-lora/src")

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel, apply_lora_to_layers
from ctx_to_lora.modeling.lora_merger import combine_lora
from ctx_to_lora.data.processing import tokenize_ctx_text

# ============================================================
# 配置
# ============================================================
CHECKPOINT = "/workspace/models/doc-to-lora/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin"
MODEL_PATH = "/workspace/models/gemma-2-2b-it"
DOC_PATH = "/workspace/ai-theorys-study/arxiv/wechat110/story_of_the_stone.txt"
QA_PATH = "/workspace/ai-theorys-study/arxiv/wechat110/qa_dataset.json"
OUTPUT_PATH = "/workspace/ai-theorys-study/arxiv/wechat110/results.json"
MAX_CHUNK_LEN = 512
MAX_NEW_TOKENS = 128
MAX_CHAPTER = 8  # 只用前 8 章

# 测试的块数梯度 (最后一个 "all" 在运行时替换为实际总块数)
CHUNK_COUNTS = [1, 5, 20, 50, 100, "all"]


def load_model():
    print("Loading model...")
    state_dict = torch.load(CHECKPOINT, weights_only=False)
    state_dict["base_model_name_or_path"] = MODEL_PATH
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False
    )
    model.reset()
    tokenizer = get_tokenizer(MODEL_PATH)
    print("Model loaded.")
    return model, tokenizer


def extract_first_n_chapters(doc_text, n_chapters):
    """截取前 N 章的文本"""
    # 找到第 N+1 章的开头作为截断点
    import re
    # 匹配 CHAPTER IX. 这种格式
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
             "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
             "XXI", "XXII", "XXIII", "XXIV"]
    if n_chapters < len(roman):
        next_chapter = f"CHAPTER {roman[n_chapters]}."
        pos = doc_text.find(next_chapter)
        if pos > 0:
            return doc_text[:pos]
    return doc_text


def tokenize_document(doc_text, tokenizer):
    ctx_ids = tokenize_ctx_text(dict(context=[doc_text]), tokenizer)["ctx_ids"]
    return ctx_ids[0]


def chunk_tokens(token_ids, chunk_len):
    chunks = []
    for i in range(0, len(token_ids), chunk_len):
        chunk = token_ids[i : i + chunk_len]
        if len(chunk) > 10:
            chunks.append(chunk)
    return chunks


def internalize_chunks(model, chunks):
    """
    多块 internalize:
    1. 每块分别过超网络得到 rank-8 的 LoRA
    2. 中间结果存 CPU 省显存
    3. 最后搬回 GPU 合并 + 挂载
    """
    model.reset()
    model.patch_lora_forward()
    all_loras = []

    for i, chunk in enumerate(chunks):
        ctx_ids = torch.tensor([chunk], device=model.device)
        ctx_attn_mask = torch.ones_like(ctx_ids)
        with torch.no_grad():
            loras, _ = model.generate_weights(ctx_ids, ctx_attn_mask, None)
        # 移到 CPU 省显存
        cpu_lora = {}
        for module_name in loras:
            cpu_lora[module_name] = {}
            for k in ("A", "B"):
                cpu_lora[module_name][k] = loras[module_name][k].cpu()
        all_loras.append(cpu_lora)
        del loras, ctx_ids, ctx_attn_mask
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()
            print(f"  Processed {i+1}/{len(chunks)} chunks...")

    print(f"  Processed {len(chunks)}/{len(chunks)} chunks. Merging...")

    # 沿 dim=0 拼接, 搬回 GPU
    combined = {}
    for module_name in all_loras[0]:
        combined[module_name] = {}
        for matrix_key in ("A", "B"):
            tensors = [lora[module_name][matrix_key] for lora in all_loras]
            combined[module_name][matrix_key] = torch.cat(tensors, dim=0).to(model.device)
    del all_loras
    gc.collect()
    torch.cuda.empty_cache()

    # combine_lora 合并
    n_chunks_tensor = torch.tensor([len(chunks)], device=model.device)
    combined = combine_lora(
        combined,
        n_chunks_tensor,
        lora_bias=model.hypernet.get_head_bias()
        if model.hypernet.config.use_bias
        else None,
    )

    # 挂载到模型
    n_queries = torch.ones(1, dtype=torch.int32, device=model.device)
    apply_lora_to_layers(
        model.base_model,
        model.hypernet.layer_indices,
        combined,
        n_queries,
        None,
    )


def ask_question(model, tokenizer, question):
    chat = [{"role": "user", "content": question}]
    chat_ids = tokenizer.apply_chat_template(
        chat,
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        outputs = model.base_model.generate(input_ids=chat_ids, max_new_tokens=MAX_NEW_TOKENS)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "model\n" in response:
        response = response.split("model\n")[-1].strip()
    return response


def evaluate(model, tokenizer, qa_data, condition_name):
    results = []
    for i, qa in enumerate(qa_data):
        question = qa["question"]
        expected = qa["answer"]
        response = ask_question(model, tokenizer, question)

        expected_lower = expected.lower()
        response_lower = response.lower()
        keywords = [w for w in expected_lower.split() if len(w) > 3]
        if not keywords:
            keywords = expected_lower.split()
        match = any(kw in response_lower for kw in keywords)

        results.append({
            "chapter": qa["chapter"],
            "question": question,
            "expected": expected,
            "response": response,
            "match": match,
        })

    correct = sum(1 for r in results if r["match"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    print(f"  [{condition_name}] {correct}/{total} = {accuracy:.1%}")
    return results, accuracy


def main():
    model, tokenizer = load_model()

    # 加载文档, 截取前 8 章
    with open(DOC_PATH, "r") as f:
        doc_text_full = f.read()
    doc_text = extract_first_n_chapters(doc_text_full, MAX_CHAPTER)
    print(f"Extracted first {MAX_CHAPTER} chapters: {len(doc_text)} chars, ~{len(doc_text.split())} words")

    # 只用前 8 章的题
    with open(QA_PATH, "r") as f:
        qa_all = json.load(f)
    qa_data = [q for q in qa_all if q["chapter"] <= MAX_CHAPTER]
    print(f"Questions: {len(qa_data)} (chapters 1-{MAX_CHAPTER})")

    # Tokenize
    print("Tokenizing...")
    doc_tokens = tokenize_document(doc_text, tokenizer)
    total_tokens = len(doc_tokens)
    all_chunks = chunk_tokens(doc_tokens, MAX_CHUNK_LEN)
    total_chunks = len(all_chunks)
    print(f"Tokens: {total_tokens}, Chunks: {total_chunks}")

    # 替换 "all"
    chunk_counts = [n if n != "all" else total_chunks for n in CHUNK_COUNTS]
    # 去掉超过总块数的
    chunk_counts = [n for n in chunk_counts if n <= total_chunks]
    # 如果最后一个不是 total_chunks, 加上
    if chunk_counts[-1] != total_chunks:
        chunk_counts.append(total_chunks)

    print(f"Chunk counts to test: {chunk_counts}")
    print()

    all_results = {}

    # ============================================================
    # Baseline
    # ============================================================
    print("=" * 60)
    print("Baseline (no LoRA)")
    print("=" * 60)
    model.reset()
    results_bl, acc_bl = evaluate(model, tokenizer, qa_data, "baseline")
    all_results["baseline"] = {
        "n_chunks": 0, "rank": 0, "accuracy": acc_bl, "results": results_bl
    }

    # ============================================================
    # 逐步增加块数
    # ============================================================
    for n in chunk_counts:
        chunks = all_chunks[:n]
        rank = 8 * (n + 1)
        tokens_covered = sum(len(c) for c in chunks)
        pct = tokens_covered / total_tokens * 100

        print()
        print("=" * 60)
        print(f"N={n} (rank-{rank}, {tokens_covered} tokens, {pct:.0f}% of text)")
        print("=" * 60)

        t0 = time.time()
        internalize_chunks(model, chunks)
        t1 = time.time()
        print(f"  Internalization: {t1-t0:.1f}s")

        results, acc = evaluate(model, tokenizer, qa_data, f"N={n}")
        t2 = time.time()
        print(f"  Evaluation: {t2-t1:.1f}s")

        all_results[f"n{n}"] = {
            "n_chunks": n,
            "rank": rank,
            "tokens_covered": tokens_covered,
            "pct_text": round(pct, 1),
            "accuracy": acc,
            "results": results,
        }

        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # 汇总
    # ============================================================
    print()
    print("=" * 60)
    print("SUMMARY: Accuracy vs Chunks")
    print("=" * 60)
    print(f"{'Condition':<12} {'Chunks':>7} {'Rank':>7} {'Text%':>7} {'Accuracy':>10}")
    print("-" * 47)
    print(f"{'Baseline':<12} {'0':>7} {'0':>7} {'0%':>7} {acc_bl:>10.1%}")
    for n in chunk_counts:
        r = all_results[f"n{n}"]
        label = f"N={n}" if n != total_chunks else f"N=ALL({n})"
        print(f"{label:<12} {r['n_chunks']:>7} {r['rank']:>7} {str(r['pct_text'])+'%':>7} {r['accuracy']:>10.1%}")

    # 保存
    summary = {
        "document": f"Hung Lou Meng, Book I, Chapters 1-{MAX_CHAPTER}",
        "document_tokens": total_tokens,
        "total_chunks": total_chunks,
        "max_chunk_len": MAX_CHUNK_LEN,
        "n_questions": len(qa_data),
        "chunk_counts_tested": chunk_counts,
        "conditions": {
            k: {key: v for key, v in val.items() if key != "results"}
            for k, val in all_results.items()
        },
        "detailed_results": {
            k: val["results"] for k, val in all_results.items()
        },
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
