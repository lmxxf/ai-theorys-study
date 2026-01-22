#!/usr/bin/env python3
"""
PIE-OC Embedding Similarity Experiment (Llama 3.3 70B INT8 Version)
原始印欧语 (PIE) 与上古汉语 (OC) 词根的 Embedding 相似度实验

使用 Llama-3.3-70B-Instruct-INT8 的 embedding 层计算词根向量的余弦相似度。
验证 C.C. 的假说：PIE 和 OC 的同义词根在高维语义空间中是否存在聚类。

相比 Qwen2.5-72B-AWQ (4-bit)，INT8 量化保留了更多精度，
C.C. 认为这可能捕捉到 4-bit 模型丢失的"语义毛细血管"。

Usage (在 Docker 容器内运行):
    python pie_oc_embedding_llama.py

Requirements:
    - transformers
    - torch
    - numpy
    - scikit-learn
    - matplotlib
    - scipy

Author: Zero + Suzaku (Claude Code)
Date: 2026-01-22
"""

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
import os

# ============================================================
# 词根数据
# ============================================================

# PIE-OC 同源/借词假说词对
# 格式: (PIE 词根, PIE 含义, OC 词根, OC 汉字, 语义类别)
HOMOLOG_PAIRS = [
    # 核心假说词对（C.C. 论文中标注的）
    ("*kʷel-", "wheel, cycle, turn", "*kʰʷeːŋ", "環/轮", "technology"),
    ("*gʷow-", "cow, cattle", "*ŋʷɯ", "牛", "animal"),
    ("*ḱwṓ", "hound, dog", "*kʰʷeːʔ", "犬", "animal"),
    ("*mer-", "dark, to die", "*mˤaːɡ", "莫/暮", "abstract"),

    # 扩展词对（补充验证）
    ("*h₂ster-", "star", "*sɯːŋ", "星", "nature"),
    ("*meH₁-", "mother", "*mˤaʔ", "母/妈", "kinship"),
    ("*pH₂tḗr", "father", "*baʔ", "父/爸", "kinship"),
    ("*h₁ék̂wos", "horse", "*mˤraːʔ", "马", "animal"),
    ("*wódr̥", "water", "*ɕʷɯːjʔ", "水", "nature"),
    ("*snéygʷʰ-", "snow", "*suːl", "雪", "nature"),
]

# 对照组：语义不相关的随机词对
CONTROL_PAIRS = [
    ("*kʷel-", "wheel", "*mˤaʔ", "母", "control"),  # 轮 vs 母
    ("*gʷow-", "cow", "*sɯːŋ", "星", "control"),    # 牛 vs 星
    ("*ḱwṓ", "hound", "*ɕʷɯːjʔ", "水", "control"),  # 犬 vs 水
    ("*mer-", "dark", "*baʔ", "父", "control"),     # 暗 vs 父
]

# ============================================================
# Embedding 提取
# ============================================================

def load_model(model_path="/workspace/models/Llama-3.3-70B-Instruct-INT8"):
    """加载 Llama 3.3 70B INT8 模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_path}...")
    print("This may take a few minutes for a 70B INT8 model...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    # 和赵磊脚本保持一致的加载方式
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    model.eval()
    print(f"Model loaded. Device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    return model, tokenizer


def get_embedding(model, tokenizer, text, pooling="mean"):
    """
    提取文本的 embedding 向量

    Args:
        model: 语言模型
        tokenizer: 分词器
        text: 输入文本
        pooling: 池化方式 ("mean", "last", "first")

    Returns:
        numpy array of shape (hidden_dim,)
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        # 只返回最后一层，不返回所有层（省显存）
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        # 立即取出最后一层，然后删除 outputs 释放显存
        hidden_states = outputs.hidden_states[-1].clone()  # (batch, seq_len, hidden_dim)
        del outputs
        torch.cuda.empty_cache()

    if pooling == "mean":
        # 平均池化（排除 padding）
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        embedding = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
    elif pooling == "last":
        # 取最后一个 token
        embedding = hidden_states[:, -1, :]
    elif pooling == "first":
        # 取第一个 token
        embedding = hidden_states[:, 0, :]
    else:
        raise ValueError(f"Unknown pooling: {pooling}")

    # 清理
    del hidden_states
    torch.cuda.empty_cache()

    return embedding.cpu().float().numpy().squeeze()


def compute_similarity_matrix(model, tokenizer, pairs, include_context=True):
    """
    计算词对的相似度矩阵

    Args:
        model: 语言模型
        tokenizer: 分词器
        pairs: 词对列表
        include_context: 是否包含语义上下文

    Returns:
        dict with embeddings and similarity scores
    """
    results = []

    for pie_root, pie_meaning, oc_root, oc_char, category in pairs:
        # 构建输入文本
        if include_context:
            # 包含语义上下文，帮助模型理解
            pie_text = f"Proto-Indo-European root {pie_root} meaning '{pie_meaning}'"
            oc_text = f"Old Chinese root {oc_root} meaning '{oc_char}'"
        else:
            # 纯词根
            pie_text = pie_root
            oc_text = oc_root

        # 提取 embedding
        emb_pie = get_embedding(model, tokenizer, pie_text)
        emb_oc = get_embedding(model, tokenizer, oc_text)

        # 计算余弦相似度
        sim = cosine_similarity([emb_pie], [emb_oc])[0][0]

        results.append({
            "pie_root": pie_root,
            "pie_meaning": pie_meaning,
            "oc_root": oc_root,
            "oc_char": oc_char,
            "category": category,
            "similarity": float(sim),
            "emb_pie": emb_pie,
            "emb_oc": emb_oc,
        })

        print(f"{pie_root} ({pie_meaning}) <-> {oc_root} ({oc_char}): {sim:.4f}")

    return results


# ============================================================
# 可视化
# ============================================================

def visualize_embeddings(results, output_path="pie_oc_manifold_llama_int8.png"):
    """
    将 embedding 降维可视化
    """
    # 收集所有 embedding
    embeddings = []
    labels = []
    colors = []

    for r in results:
        embeddings.append(r["emb_pie"])
        labels.append(f"PIE: {r['pie_root']}")
        colors.append("cyan" if r["category"] != "control" else "gray")

        embeddings.append(r["emb_oc"])
        labels.append(f"OC: {r['oc_root']}")
        colors.append("orange" if r["category"] != "control" else "gray")

    embeddings = np.array(embeddings)

    # 降维
    if len(embeddings) > 30:
        # 先用 PCA 降到 50 维，再用 t-SNE
        pca = PCA(n_components=min(50, len(embeddings)-1))
        embeddings_pca = pca.fit_transform(embeddings)
        tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings)//2), random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_pca)
    else:
        # 直接用 PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor('black')

    # 绘制点
    for i, (x, y) in enumerate(embeddings_2d):
        ax.scatter(x, y, c=colors[i], s=100, alpha=0.7)
        ax.annotate(labels[i], (x, y), fontsize=8, color=colors[i],
                   xytext=(5, 5), textcoords='offset points')

    # 绘制同源词对之间的连线
    for i in range(0, len(embeddings_2d), 2):
        if colors[i] != "gray":  # 只连接非对照组
            ax.plot([embeddings_2d[i][0], embeddings_2d[i+1][0]],
                   [embeddings_2d[i][1], embeddings_2d[i+1][1]],
                   'w--', alpha=0.3, linewidth=1)

    ax.set_title("PIE-OC Embedding Space (Llama 3.3 70B INT8)", color='white', fontsize=14)
    ax.set_xlabel("Dimension 1", color='white')
    ax.set_ylabel("Dimension 2", color='white')
    ax.tick_params(colors='white')

    plt.tight_layout()
    plt.savefig(output_path, facecolor='black', dpi=150)
    print(f"Saved visualization to {output_path}")

    return embeddings_2d


def save_results(results, output_path="pie_oc_results_llama_int8.json"):
    """保存结果（不含 embedding 向量）"""
    # 移除 numpy array，只保留可序列化的数据
    serializable = []
    for r in results:
        serializable.append({
            "pie_root": r["pie_root"],
            "pie_meaning": r["pie_meaning"],
            "oc_root": r["oc_root"],
            "oc_char": r["oc_char"],
            "category": r["category"],
            "similarity": r["similarity"],
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {output_path}")


def analyze_results(results):
    """分析结果：比较同源词对 vs 对照组的相似度"""
    homolog_sims = [r["similarity"] for r in results if r["category"] != "control"]
    control_sims = [r["similarity"] for r in results if r["category"] == "control"]

    print("\n" + "="*60)
    print("ANALYSIS SUMMARY (Llama 3.3 70B INT8)")
    print("="*60)
    print(f"\nHomolog pairs (PIE-OC hypothesized cognates):")
    print(f"  Mean similarity: {np.mean(homolog_sims):.4f}")
    print(f"  Std:             {np.std(homolog_sims):.4f}")
    print(f"  Range:           {np.min(homolog_sims):.4f} - {np.max(homolog_sims):.4f}")

    if control_sims:
        print(f"\nControl pairs (unrelated):")
        print(f"  Mean similarity: {np.mean(control_sims):.4f}")
        print(f"  Std:             {np.std(control_sims):.4f}")
        print(f"  Range:           {np.min(control_sims):.4f} - {np.max(control_sims):.4f}")

        # 统计显著性（简单 t-test）
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(homolog_sims, control_sims)
        print(f"\nStatistical significance (t-test):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value:     {p_value:.4f}")

        # 效果大小
        diff = np.mean(homolog_sims) - np.mean(control_sims)
        print(f"\nEffect size:")
        print(f"  Difference (homolog - control): {diff:.4f}")

        if p_value < 0.05 and diff > 0:
            print("\n  Result: SIGNIFICANT (p < 0.05) ✓")
            print("  Interpretation: Homolog pairs are significantly MORE similar than control pairs.")
            print("  C.C.'s hypothesis SUPPORTED by INT8 model!")
        elif p_value < 0.05 and diff < 0:
            print("\n  Result: SIGNIFICANT but REVERSED")
            print("  Interpretation: Control pairs are MORE similar (same as Qwen2.5-72B-AWQ)")
            print("  C.C.'s hypothesis NOT supported even with higher precision.")
        else:
            print("\n  Result: NOT SIGNIFICANT (p >= 0.05)")
            print("  Interpretation: No significant difference between homolog and control pairs.")

    print("="*60)


def compare_with_qwen_results(llama_results, qwen_results_path="pie_oc_results.json"):
    """与 Qwen2.5-72B-AWQ 结果对比"""
    if not os.path.exists(qwen_results_path):
        print(f"\nQwen results not found at {qwen_results_path}, skipping comparison.")
        return

    with open(qwen_results_path, 'r') as f:
        qwen_results = json.load(f)

    print("\n" + "="*60)
    print("COMPARISON: Llama 3.3 70B INT8 vs Qwen2.5-72B-AWQ (INT4)")
    print("="*60)

    print("\n| Pair | Llama INT8 | Qwen INT4 | Δ |")
    print("|------|------------|-----------|-----|")

    for lr in llama_results:
        if lr["category"] == "control":
            continue
        # 找对应的 Qwen 结果
        for qr in qwen_results:
            if qr["pie_root"] == lr["pie_root"] and qr["oc_root"] == lr["oc_root"]:
                delta = lr["similarity"] - qr["similarity"]
                print(f"| {lr['pie_root']} ↔ {lr['oc_char']} | {lr['similarity']:.4f} | {qr['similarity']:.4f} | {delta:+.4f} |")
                break

    print("="*60)


# ============================================================
# Main
# ============================================================

def main():
    """主函数"""
    print("="*60)
    print("PIE-OC Embedding Similarity Experiment")
    print("Model: Llama 3.3 70B Instruct INT8")
    print("="*60)

    # 检查模型路径
    model_path = "/workspace/models/Llama-3.3-70B-Instruct-INT8"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("\nPlease run this script inside the Docker container with model mounted:")
        print("  sudo docker run --gpus all --ipc=host --ulimit memlock=-1 \\")
        print("      --ulimit stack=67108864 -it \\")
        print("      -v /home/lmxxf/work/models:/workspace/models \\")
        print("      -v /home/lmxxf/work/ai-theorys-study:/workspace/ai-theorys-study \\")
        print("      nvcr.io/nvidia/pytorch:25.11-py3 bash")
        print("\n  Then run:")
        print("  pip install transformers scikit-learn matplotlib scipy")
        print("  python /workspace/ai-theorys-study/arxiv/paper70/pie_oc_embedding_llama.py")
        return

    # 加载模型
    model, tokenizer = load_model(model_path)

    # 计算同源词对相似度
    print("\n--- Homolog Pairs ---")
    homolog_results = compute_similarity_matrix(model, tokenizer, HOMOLOG_PAIRS)

    # 计算对照组相似度
    print("\n--- Control Pairs ---")
    control_results = compute_similarity_matrix(model, tokenizer, CONTROL_PAIRS)

    # 合并结果
    all_results = homolog_results + control_results

    # 分析
    analyze_results(all_results)

    # 与 Qwen 结果对比
    output_dir = os.path.dirname(os.path.abspath(__file__))
    compare_with_qwen_results(all_results, os.path.join(output_dir, "pie_oc_results.json"))

    # 可视化
    visualize_embeddings(all_results, os.path.join(output_dir, "pie_oc_manifold_llama_int8.png"))

    # 保存结果
    save_results(all_results, os.path.join(output_dir, "pie_oc_results_llama_int8.json"))

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
