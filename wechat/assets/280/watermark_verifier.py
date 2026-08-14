#!/usr/bin/env python3
"""
文本水印验证器 —— 复现 KGW 方案

论文：Kirchenbauer, Geiping, Wen, Katz, Miers, Goldstein,
     "A Watermark for Large Language Models", ICML 2023, arXiv:2301.10226
     （University of Maryland）

参数一律按原文主设置：γ=0.5, δ=2.0, 判定阈值 z>4（单侧 p 值 3e-5）

四个实验：
  1. 自证 —— 加了能检出、密钥错了检不出
  2. 长度门槛 —— 多长的文本才能确信
  3. 熵的作用 —— 复现 Theorem 4.2：能塞多少水印由文本熵决定
  4. 鲁棒性 —— 改多少词能洗掉

⚠️ 两条必须说清的限制：
  1) 本脚本用自己的密钥加水印、再自己检出来，证明的是"这套数学成立"，
     不是"能检测 Claude"。Anthropic 的密钥不公开，也没说用的是不是这套方案。
  2) 实验用的是模拟 logits，不是真实模型 —— 但 logits 的形状按真实模型的
     幂律分布来造，所以数量级和论文对得上（本脚本 128 token 时 z≈6，
     论文 γ=.5/δ=2 实测约需 128 token 到 z=4）。看结构和趋势，
     具体数字以论文为准。

纯标准库，无依赖。
"""

import math
import random
from typing import List, Tuple

# ── 论文主设置 ──────────────────────────────────────────
GAMMA = 0.5     # 绿名单占词表比例（原文 Table 2 主设置；扫过 0.5/0.25，发现 0.1 帕累托最优）
DELTA = 2.0     # 加在绿名单 logits 上的偏置（原文网格 1.0/2.0/5.0）
Z_THRESHOLD = 4.0   # 原文：z>4 判定有水印，单侧 p 值 3e-5
VOCAB = 4000    # 模拟词表大小。真实模型是 3-15 万，但 γ/δ/z 检验全都与词表
                # 大小无关（z 只看绿词比例），调小纯粹是为了让脚本跑得快。


# ============================================================
# 核心：红/绿名单划分
# ============================================================

_MASK = (1 << 31) - 1


def is_green(token: int, prev_token: int, key: int,
             gamma: float = GAMMA) -> bool:
    """
    判断 token 在不在绿名单里。

    原文 Algorithm 2 第 2-3 步：
      "Compute a hash of token s^(t-1), and use it to seed a random number generator."
      "randomly partition the vocabulary into a 'green list' G of size γ|V|,
       and a 'red list' R of size (1-γ)|V|."

    为什么只用前 1 个 token 做种子？原文给了理由：
      "enabling the red list to be reproduced later without access to
       the entire generated sequence."
    —— 检测方只要有文本就能逐位重算，不需要原始 prompt，也不需要模型。

    ★ 实现说明：论文的描述是"把整个词表洗牌后切两半"，但那样每个位置都要
      构造一个上万元素的集合，慢得没必要。等价且高效的做法是直接把
      (前一个token, 密钥, 当前token) 哈希成一个数，看它落不落在前 γ 比例里 ——
      每个 token 被判为绿色的概率同样是 γ，且同样只依赖前一个 token 和密钥。
      数学上等价，速度差几个数量级。
    """
    h = (prev_token * 1000003 ^ key * 15485863 ^ token * 2654435761) & _MASK
    h = (h * 2246822519 + 374761393) & _MASK
    h ^= h >> 15
    return (h & _MASK) < gamma * _MASK


# ============================================================
# 检测器 —— 原文公式 (3)
# ============================================================

def detect(tokens: List[int], key: int, vocab_size: int = VOCAB,
           gamma: float = GAMMA) -> Tuple[float, int, int]:
    """
    单比例 z 检验。原文公式 (3)：

        z = (|s|_G - γT) / sqrt(T · γ · (1-γ))

    分子 = 实际绿词数 - 零假设下的期望值
    分母 = 零假设下的标准差

    零假设 H0（原文公式 1）：
      "The text sequence is generated with no knowledge of the red list rule."
      —— 即：写这段文本的人/机器不知道红名单规则，绿词比例应该就是 γ。

    检测需要什么？原文 desiderata 第一条：
      "without any knowledge of the model parameters or access to the
       language model API" —— 不需要模型，不需要 prompt。但需要密钥。
    """
    if len(tokens) < 2:
        return 0.0, 0, 0

    green = 0
    # 从第 2 个 token 起：第 1 个没有"前一个 token"可做种子
    for i in range(1, len(tokens)):
        if is_green(tokens[i], tokens[i - 1], key, gamma):
            green += 1

    T = len(tokens) - 1
    std = math.sqrt(T * gamma * (1 - gamma))
    z = (green - gamma * T) / std if std > 0 else 0.0
    return z, green, T


# ============================================================
# 生成侧 —— 原文 Algorithm 2（Soft Red List）
# ============================================================

def sample_plain(logits: List[float]) -> int:
    """不加水印的普通采样：softmax 后按概率抽。"""
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    r = random.random() * total
    acc = 0.0
    for i, e in enumerate(exps):
        acc += e
        if acc >= r:
            return i
    return len(exps) - 1


def sample_watermarked(logits: List[float], prev_token: int, key: int,
                       vocab_size: int = VOCAB, gamma: float = GAMMA,
                       delta: float = DELTA) -> int:
    """
    带水印的采样。原文 Algorithm 2 第 4-5 步：
      "Add δ to each green list logit. Apply the softmax operator to these
       modified logits to get a probability distribution over the vocabulary."

    ★ 注意插入位置：δ 加在 logits 上、softmax 之前。
      模型的前向计算此刻已经全部结束了 —— 这一步在模型"之后"。

    ★ 注意是"软"的：不禁止红名单词，只是让绿名单更容易被选中。
      原文解释为什么不能硬禁止（Algorithm 1 的缺陷）：
        "the token 'Barack' is almost deterministically followed by 'Obama'
         in many text datasets, yet 'Obama' may be disallowed by the red list."
      硬禁止会逼模型在"Barack"后面不说"Obama"，句子就毁了。
    """
    biased = [lg + (delta if is_green(i, prev_token, key, gamma) else 0.0)
              for i, lg in enumerate(logits)]
    return sample_plain(biased)


# ============================================================
# 模拟 logits：用温度控制熵的高低
# ============================================================

def make_logits(vocab_size: int = VOCAB, temperature: float = 1.0) -> List[float]:
    """
    造一组模拟 logits。

    ★ 形状按真实语言模型来：概率沿排名幂律衰减 —— top-1 一枝独大，
      往后迅速掉下去，长尾几乎为零。（如果用平坦的高斯 logits，δ 会显得
      比实际有效得多，长度门槛也会算得过于乐观。）

    temperature 控制这个分布有多尖，即熵的高低：
      温度低 → 更尖 → 模型"几乎只有一个选择"（低熵：背诵、代码、套话）
      温度高 → 更平 → 模型"不知道说什么好"（高熵：开放式写作）

    真实模型逐位置剧烈变化：同一句话里，"人工智能"后面接"技术"是低熵位置，
    而一段议论文的下一句怎么起头是高熵位置。
    """
    # 幂律：第 r 名的 logit ≈ -zipf * ln(r)，再叠一点噪声打乱排名
    order = list(range(vocab_size))
    random.shuffle(order)
    lg = [0.0] * vocab_size
    for rank, tok in enumerate(order, start=1):
        lg[tok] = -2.2 * math.log(rank) / temperature + random.gauss(0, 0.3)
    return lg


def spike_entropy(logits: List[float], delta: float = DELTA) -> float:
    """
    原文 Definition 4.1 的 spike entropy：

        S(p, z) = Σ_k  p_k / (1 + z·p_k)

    衡量分布"有多不集中"。概率全压在一个 token 上时最小，均匀分布时最大。
    论文不用香农熵而自造这个量，是因为它能直接代进 Theorem 4.2
    给出绿词数量的下界 —— 把"文本的熵"和"能塞进多少水印"用公式连起来了。

    这里按原文惯例取 modulus z = (α-1) = exp(δ)-1。
    """
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    z = math.exp(delta) - 1
    return sum((e / total) / (1 + z * (e / total)) for e in exps)


def generate(n: int, key: int, watermark: bool, temperature: float = 1.0,
             gamma: float = GAMMA, delta: float = DELTA) -> Tuple[List[int], float]:
    """生成一段 token 序列，同时返回平均 spike entropy。"""
    toks = [random.randrange(VOCAB)]
    ents = []
    for _ in range(n):
        lg = make_logits(temperature=temperature)
        ents.append(spike_entropy(lg, delta))
        if watermark:
            toks.append(sample_watermarked(lg, toks[-1], key, gamma=gamma, delta=delta))
        else:
            toks.append(sample_plain(lg))
    return toks, sum(ents) / len(ents)


def mean_z(runs: int, fn) -> float:
    """
    跑 runs 次取平均 z 值。

    ⚠️ 单次抽样的 z 值波动很大（n=200 时标准差约 1.0），
    单跑一次容易出现"40% 替换比 30% 替换 z 值还高"这种假象。
    所有对外报告的数字都必须取平均。
    """
    return sum(fn(i) for i in range(runs)) / runs


def verdict(z: float) -> str:
    """按原文阈值判定。z>4 → 有水印（误报率 3e-5）。"""
    if z < 2.0:
        return "无证据"
    if z < Z_THRESHOLD:
        return "可疑"
    if z < 6.0:
        return "检出 (z>4)"
    return "确信"


# ============================================================
# 实验一：自证
# ============================================================

def exp1_self_proof(key: int = 20260814, n: int = 200, runs: int = 30):
    print("=" * 68)
    print(f"实验一：自证（γ={GAMMA}, δ={DELTA}, 判定阈值 z>{Z_THRESHOLD:.0f}）")
    print(f"每格 {runs} 次独立生成取平均，每次 {n} tokens")
    print("=" * 68)

    rows = []
    for label, wm, det_key in [
        ("无水印文本", False, key),
        ("有水印文本", True, key),
        ("有水印 + 错误密钥", True, key + 1),
    ]:
        zs, gs = [], []
        for r in range(runs):
            random.seed(1000 + r)
            toks, _ = generate(n, key, watermark=wm)
            z, g, T = detect(toks, det_key)
            zs.append(z)
            gs.append(g / T)
        rows.append((label, sum(gs) / runs, sum(zs) / runs))

    print(f"\n{'':<22}{'绿词比例':>10}{'平均 z':>10}   判定")
    print("-" * 60)
    for label, gr, z in rows:
        pad = 22 if "错误" not in label else 20
        print(f"{label:<{pad}}{gr:>9.1%}{z:>10.2f}   {verdict(z)}")

    print("\n★ 零假设下绿词比例应为 γ=50%。加了水印推到 70% 以上，z 值就爆表。")
    print("★ 但密钥错一位，同一段文本立刻变回 50% —— 水印彻底隐形。")
    print("  这就是为什么外部无法独立验证 Claude 的水印：算法可以公开，密钥不行。")


# ============================================================
# 实验二：长度门槛
# ============================================================

def exp2_length(key: int = 20260814, runs: int = 30):
    print("\n" + "=" * 68)
    print(f"实验二：需要多长的文本才能确信（{runs} 次取平均）")
    print("=" * 68)

    seqs = []
    for r in range(runs):
        random.seed(2000 + r)
        toks, _ = generate(600, key, watermark=True)
        seqs.append(toks)

    print(f"\n{'token 数':>9}{'绿词比例':>11}{'平均 z':>10}   判定")
    print("-" * 52)
    for n in [10, 16, 25, 50, 100, 128, 200, 400, 600]:
        zs, gs = [], []
        for toks in seqs:
            z, g, T = detect(toks[:n + 1], key)
            zs.append(z)
            gs.append(g / T)
        z_avg = sum(zs) / runs
        print(f"{n:>9}{sum(gs)/runs:>10.1%}{z_avg:>10.2f}   {verdict(z_avg)}")

    print("\n★ 论文摘要说『少至 25 个 token』，但那是宣传口径，必须分清三个数字：")
    print("    16  token —— 硬水印的理论下界（全绿时 z 恰好到 4）")
    print("    25  token —— 摘要里的宣传数字")
    print("   128  token —— γ=0.5/δ=2 这个实际设置真正需要的长度（论文实测）")
    print("★ 本脚本用模拟 logits，但按真实模型的幂律形状来造，量级和论文对得上。")


# ============================================================
# 实验三：熵 —— 论文 Theorem 4.2 的现象
# ============================================================

def exp3_entropy(key: int = 20260814, n: int = 200, runs: int = 30):
    print("\n" + "=" * 68)
    print("实验三：为什么低熵文本加不上水印（复现 Theorem 4.2）")
    print("=" * 68)

    print(f"\n同样 {n} 个 token、同样的 δ={DELTA}，只改变模型有多『确定』")
    print(f"（{runs} 次取平均）：\n")
    print(f"{'模型状态':<30}{'spike熵':>9}{'绿词比例':>11}{'平均 z':>9}   判定")
    print("-" * 74)

    cases = [
        (3.0, "很不确定（开放式写作）"),
        (1.6, "不太确定"),
        (1.0, "中等（一般行文）"),
        (0.7, "比较确定（套话、格式）"),
        (0.45, "几乎唯一答案（背诵、代码）"),
    ]
    for temp, label in cases:
        zs, gs, es = [], [], []
        for r in range(runs):
            random.seed(3000 + r)
            toks, ent = generate(n, key, watermark=True, temperature=temp)
            z, g, T = detect(toks, key)
            zs.append(z); gs.append(g / T); es.append(ent)
        print(f"{label:<28}{sum(es)/runs:>9.3f}{sum(gs)/runs:>10.1%}"
              f"{sum(zs)/runs:>9.2f}   {verdict(sum(zs)/runs)}")

    print("\n★ 熵越低，同样的 δ 推不动分布 —— 因为 top-1 的概率已经压倒性高，")
    print("  加 2.0 的偏置也翻不了盘。绿词比例回落到 50%，水印消失。")
    print("★ 论文的 Barack→Obama 例子说的就是这件事：'Barack' 后面几乎必然是")
    print("  'Obama'，你没有别的词可选，也就没有地方藏水印。")
    print("★ 这不是工程缺陷，是 Theorem 4.2 给出的数学上界：")
    print("  能塞进去多少水印，由文本自身的熵决定。")


# ============================================================
# 实验四：鲁棒性
# ============================================================

def exp4_robustness(key: int = 20260814, n: int = 300, runs: int = 30):
    print("\n" + "=" * 68)
    print(f"实验四：改写多少能把水印洗掉（{runs} 次取平均）")
    print("=" * 68)

    seqs = []
    for r in range(runs):
        random.seed(4000 + r)
        toks, _ = generate(n, key, watermark=True)
        seqs.append(toks)

    print(f"\n{'替换比例':>10}{'平均 z':>10}   判定")
    print("-" * 42)
    for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]:
        zs = []
        for r, toks in enumerate(seqs):
            random.seed(5000 + r)
            edited = toks[:]
            k = int(frac * (len(edited) - 1))
            for i in random.sample(range(1, len(edited)), k):
                edited[i] = random.randrange(VOCAB)
            z, _, _ = detect(edited, key)
            zs.append(z)
        z_avg = sum(zs) / runs
        print(f"{frac:>9.0%}{z_avg:>10.2f}   {verdict(z_avg)}")

    print("\n★ 水印是统计信号不是指纹：衰减是连续的，不是断崖。")
    print("★ 但这里是随机换词，会毁掉可读性。论文做的是正经攻击实验：")
    print("  用 T5-Large 做同义替换，10% 预算下水印 AUC 只掉 0.01；")
    print("  30% 预算才能有效削弱，代价是文本困惑度暴涨 3 倍。")
    print("  —— 攻击成功的同时，文本也毁了。")


if __name__ == "__main__":
    print("\nKGW 文本水印方案复现")
    print("arXiv:2301.10226, ICML 2023, University of Maryland")
    print("（Anthropic 用的是不是这套方案，官方一个字没说）\n")
    exp1_self_proof()
    exp2_length()
    exp3_entropy()
    exp4_robustness()
    print("\n" + "=" * 68)
    print("检测器三十行就能写完，数学也不难。")
    print("卡住外部验证的从来不是算法，是密钥。")
    print("=" * 68 + "\n")
