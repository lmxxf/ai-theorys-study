# 核查 A：判分模型的规模与弱监督

服务于公众号 311 期《判分的那个模型该多大？》
核查日期：2026-09-16
方法：arXiv abs 页 + PDF 全文（pdftotext 抽取原文），所有数字以论文原文为准；训练记忆不作为依据。

---

## 一、arXiv:2203.02155 — InstructGPT

**准确标题**：Training language models to follow instructions with human feedback

**作者**：Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe

**机构**：OpenAI（arXiv abs 页未单列机构字段，作者均为 OpenAI；论文正文署名 OpenAI）

**arXiv 编号 / 提交日期 / 版本**：2203.02155，2022 年 3 月 4 日提交，**只有 v1**（Fri, 4 Mar 2022 07:04:42 UTC，1,047 KB）

**会议接收状态**：arXiv comments 字段为空，**未核到**正式会议标注。（该文常被引为 NeurIPS 2022，但 arXiv 页面本身没有写，此处不据传闻下断言。）

### 1.1 §3.5 关于 6B 奖励模型的原句（逐字）

> "Reward modeling (RM). Starting from the SFT model with the final unembedding layer removed, we trained a model to take in a prompt and response, and output a scalar reward. In this paper we only use 6B RMs, as this saves a lot of compute, and we found that 175B RM training could be unstable and thus was less suitable to be used as the value function during RL (see Appendix C for more details)."

中译：
> 奖励建模（RM）。我们从 SFT 模型出发、去掉最后的 unembedding 层，训练一个模型接收 prompt 和 response、输出一个标量奖励。**本文中我们只使用 6B 的奖励模型，因为这样能省下大量算力，而且我们发现 175B 的奖励模型训练可能不稳定，因而不太适合在 RL 阶段用作价值函数**（更多细节见附录 C）。

### 1.2 附录 C.2 是否补充了尺寸选择的理由 —— 有，而且更关键

附录标题为 **C.2 Details of RM training**，原文：

> "We trained a single 6B reward model which we used for all PPO models of all sizes. Larger 175B RMs had the potential to achieve lower validation loss, but (1) their training was more unstable which made them less suitable for use as initializations for the PPO value functions, and (2) using a 175B RM and value function greatly increase the compute requirements of PPO. In preliminary experiments, we found that 6B RMs were stable across a wide range of learning rates, and led to equally strong PPO models."

中译：
> 我们只训练了一个 6B 的奖励模型，用于所有尺寸的 PPO 模型。**更大的 175B 奖励模型有可能达到更低的验证损失**，但 (1) 它们训练更不稳定，因而不适合用来初始化 PPO 的价值函数；(2) 使用 175B 的奖励模型和价值函数会大幅抬高 PPO 的算力需求。在初步实验中，我们发现 **6B 的奖励模型在很宽的学习率范围内都稳定，并且能训出同样强的 PPO 模型**。

这里有两层信息，对本期主线最要紧：

- 175B 奖励模型在**验证损失**这个指标上"有可能更好"——即判分能力本身确实随规模变强，论文没有否认这一点；
- 但换成 6B 之后，**下游 PPO 策略一样强**（"equally strong PPO models"）。也就是说，判分器变大带来的判分收益，在这个设置下没有传导到被判的策略上。

附录 C.4 还补了一句实验设计上的动机：

> "As previously mentioned, for all PPO models we use a 6B RM and a 6B value function, and the latter is initialized from the former. By using the same 6B reward model and value function on policies of all model sizes, it's easier to compare the effect of policy model size on policy performance."

中译：所有 PPO 模型都用 6B 奖励模型和 6B 价值函数（后者由前者初始化）。**在所有尺寸的策略上用同一个 6B 奖励模型和价值函数，更容易比较策略规模本身对策略性能的影响。**

### 1.3 "最大的被训策略是 175B"准不准 —— 准

正文原句：

> "sizes (1.3B, 6B, and 175B parameters), and all of our models use the GPT-3 architecture."

即三个尺寸：**1.3B / 6B / 175B**，SFT 与 PPO 都按这三档做（附录 C.1 给了 1.3B、6B、175B 各自的 SFT 学习率；C.3/C.4 给了三档策略的 RLHF 设置）。所以"最大的被训策略是 175B、判分器只有 6B、差了约 29 倍参数"这个说法成立。

**可以说的**：InstructGPT 用一个 6B 的奖励模型去训练最大 175B 的策略。
**不能说的**：不能说"OpenAI 发现小奖励模型更好"。他们的原话是大的**可能验证损失更低**，换小的原因是**稳定性 + 算力**，结果是**一样强**，不是**更强**。

---

## 二、arXiv:2312.09390 — Weak-to-Strong Generalization

**准确标题**：Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision

**作者**：Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, Jeff Wu

**机构**：OpenAI（abs 页未单列机构字段；作者为 OpenAI Superalignment 团队）

**arXiv 编号 / 提交日期 / 版本**：2312.09390，2023 年 12 月 14 日提交（Thu, 14 Dec 2023 23:07:33 UTC），**只有 v1**

**会议接收状态**：**ICML 2024**（arXiv comments 字段为空，但 PMLR 有正式卷次：Proceedings of the 41st ICML, PMLR v235, burns24b；dblp 亦收录为 conf/icml/2024）。PDF 首页自标 "Preprint"。

### 2.1 NLP 任务 PGR：20%~50% —— 基本成立，但表述要精确

原文（§4.2）：

> "On the popular NLP benchmarks, we find especially promising weak-to-strong generalization: strong models trained with weak supervision can often generalize to a substantially higher performance than the weak model itself. Even with very weak supervisors and strong models with many orders of magnitude more compute, we recover more than 20% of the performance gap. The PGR increases both with weak supervisor size and with strong student size; for the largest students, the PGR is often above 50%."

中译：
> 在常用 NLP 基准上，我们发现弱到强泛化格外有希望：用弱监督训出的强模型往往能泛化到显著高于弱模型本身的性能。**即便监督者非常弱、学生的算力高出许多个数量级，我们也能恢复超过 20% 的性能差距。** PGR 随弱监督者规模和强学生规模同时上升；**对最大的学生，PGR 往往超过 50%。**

另有摘要层面的粗口径说法：

> "on NLP tasks, if we finetune GPT-4 with labels from a GPT-2-level model, we typically recover about half of the performance gap between the two models."

中译：在 NLP 任务上，用 GPT-2 级模型产生的标签微调 GPT-4，我们通常能恢复大约一半的性能差距。

→ 说"20%~50%"是**下界 >20%、最大学生常 >50%**的区间，不是"封顶 50%"。写稿时说"20% 起步、最大的学生常能过半"更贴原文。

### 2.2 加辅助置信损失接近 80% —— 成立

原文（§5.1）：

> "With the smallest weak supervisor and largest strong student, the confidence loss increases median PGR from about 25% to nearly 80%."

中译：**在最弱的监督者配最大的强学生这一组上，置信损失把 PGR 中位数从约 25% 提高到接近 80%。**

摘要/贡献段的对应表述：

> "when supervising GPT-4 with a GPT-2-level model on NLP tasks using the auxiliary confidence loss, we typically recover nearly 80% of the performance gap between the weak and strong models."

注意：**这个 80% 只在 NLP 任务上**，不是三个任务的通用结论。

### 2.3 国际象棋在差距大时掉到接近零 —— 方向对，但**条件说反了**，必须改

原文（§4.2）：

> "We see more mixed results in the chess puzzle setting. In particular, when using the smallest weak models, the PGR is close to zero and the test accuracy curves appear flat. However, as the size of the weak supervisor increases, the PGR increases substantially; for small supervisor-student gaps, PGR can be above 40%. Unlike in the NLP setting, where PGR improves with the strong student size, PGR decreases with the strong student size for a given weak supervisor on chess puzzles."

中译：
> 国际象棋谜题设置下结果更为参差。**特别是用最小的弱模型时，PGR 接近零**，测试精度曲线接近平坦。但随着弱监督者规模变大，PGR 显著上升；**在监督者—学生差距较小时，PGR 能超过 40%。** 与 NLP 设置不同（那里 PGR 随强学生规模上升），在象棋谜题上，给定一个弱监督者，**PGR 随强学生规模反而下降。**

→ 所以准确的说法是：**象棋上 PGR 接近零发生在"监督者太弱"的时候**；而且有一个 NLP 上没有的反常现象——**学生越大，PGR 越低（逆规模）**。图注措辞为 "negative PGR scaling on chess puzzles"。原任务里"差距大时掉到接近零"这一说法方向没错（监督者小=差距大），但如果写成"学生越大就越接近零"会滑向另一个更强的断言；两句合起来才是原意。

### 2.4 奖励建模只有约 10%、很少超过 20% —— 成立，逐字如下

> "Finally, we find that weak-to-strong generalization is poor by default in the ChatGPT reward model setting. We are usually only able to recover roughly 10% of the performance gap between the weak supervisor and the strong student. Even for relatively small gaps in compute between the weak and strong models, PGR almost never exceeds 20%."

中译：
> 最后，我们发现在 ChatGPT 奖励模型这个设置下，**弱到强泛化在默认情况下很差。我们通常只能恢复弱监督者与强学生之间约 10% 的性能差距。即便强弱模型之间的算力差距相对较小，PGR 也几乎从不超过 20%。**

### 2.5 "三个任务里奖励建模是弱监督效果最差的那个" —— 成立，论文自己这么说

贡献段原句：

> "Weak-to-strong generalization is particularly poor for ChatGPT reward modeling."

中译：**弱到强泛化在 ChatGPT 奖励建模上尤其糟糕。**

图 3 图注的三任务并列总结：

> "We find decent weak-to-strong generalization and even positive PGR scaling on NLP tasks, decent generalization for small supervisor-student gaps but negative PGR scaling on chess puzzles, and both poor generalization and scaling for ChatGPT reward modeling."

中译：NLP 任务上弱到强泛化不错、PGR 还随规模正向变好；象棋谜题上小差距时泛化不错但 PGR 随规模负向；**ChatGPT 奖励建模则是泛化和规模趋势两头都差。**

→ 判断成立：三个任务里，奖励建模是唯一"泛化差 + 规模趋势也差"的双输任务。这条对本期主线最有杀伤力——**判分这件事本身，恰恰是最不容易靠"强模型自己悟"补上来的那一类任务。**

另可补一条缓和证据（§6 附近）：

> "generative supervision improves PGR by approximately 10-20%"（在奖励建模设置里，先做一步无监督生成式微调，PGR 能提升约 10-20 个百分点）——即"奖励建模难"不是不可改善的死结。

---

## 三、arXiv:2412.06000 — Does RLHF Scale?

**准确标题**：Does RLHF Scale? Exploring the Impacts From Data, Model, and Method

**作者**：Zhenyu Hou, Pengfan Du, Yilin Niu, Zhengxiao Du, Aohan Zeng, Xiao Liu, Minlie Huang, Hongning Wang, Jie Tang, Yuxiao Dong

**机构**：arXiv abs 页**未单列机构字段**。作者群与 GLM/智谱 + 清华一脉（实验全部基于 GLM4 系列），但"清华+智谱"这一署名在 abs 页上**未核到**，写稿时建议写"GLM 团队（作者含清华与智谱系研究者）"或干脆只说作者与模型。

**arXiv 编号 / 提交日期 / 版本**：2412.06000，2024 年 12 月 8 日提交（Sun, 8 Dec 2024 17:19:48 UTC），**只有 v1**

**会议接收状态**：**未核到**。arXiv comments 字段为空，检索也没拿到可靠的接收记录。不要写"ACL 2025"。

### 3.1 实验矩阵 —— 基本对，但有一处要修

原文（§3 Training settings 附近）：

> "with reward and policy model sizes of 9B, 32B, and 200B parameters across varying dataset sizes."

中译：奖励模型与策略模型规模为 9B、32B、200B，并在不同数据规模上展开。

但具体到各组实验，尺寸并不是全矩阵：

- 奖励模型规模消融（§4.2.2）："We conduct experiments on PPO with reward models of 9B and 32B parameters." → **这一组只有 9B 和 32B 两档**。
- 策略规模消融（§4.2.3 附近）："The experiments include policy models ranging from 9B to 200B parameters, alongside reward models of 32B and 200B." → **策略 9B~200B，奖励模型用 32B 和 200B 两档**。

→ "奖励模型 9B/32B/200B、策略 9B~200B"作为总体范围可以说；但说"奖励模型跑满了 9B/32B/200B 三档去对比"会过头——**直接对比奖励模型大小对策略的影响那组，用的是 9B vs 32B。**

### 3.2 "larger reward models ... falls behind" 原话

发现列表第 2 条：

> "Larger reward models can effectively boost performance, but the improvement still significantly falls behind the gains in Best-of-N evaluation of the reward model (Cf. Figure 3)."

中译：
> **更大的奖励模型确实能有效提升性能，但这个提升显著落后于该奖励模型在 Best-of-N 评测中体现出来的增益。**

这句是本期的核心证据之一，值得展开：奖励模型变大，它**自己作为判分器**的能力（用 Best-of-N 测出来）涨得明显；但这份涨幅**传导到被训策略上的时候大幅缩水**。判分器变强 ≠ 被判的变强。

摘要里的对应句：

> "And larger reward models offer modest gains in policy training."（更大的奖励模型在策略训练中只带来适度的收益。）

§4.2.2 结论段还给了一条重要的非单调性：

> "However, the performance gain is not uniform across all tasks. For MMLU, whose performance highly relies on the policy model's pretraining stage, training with a large reward model starts stronger but pays more alignment tax with increased samples. And for AlignBench, training with the smaller reward model even shows a clear advantage. The reason may be that the quality of learning human preferences is not scalable and a larger reward model tends to overfit the noise in the training data."

中译：收益在任务之间并不一致。MMLU 上大奖励模型起步更强，但采样数增加后付出更多对齐税；**在 AlignBench 上，用更小的奖励模型甚至有明显优势。原因可能是学习人类偏好这件事的质量本身不可规模化，更大的奖励模型倾向于过拟合训练数据里的噪声。**

> "To summarize, larger reward models generally lead to better performance of the policy model in reasoning-related tasks, but the benefits are uncertain for other tasks"

中译：总的来说，**大奖励模型在推理类任务上一般能让策略更好，但在其他任务上收益不确定。**

→ 这给"判分模型要多大"一个分任务的答案：**有客观对错的推理任务，判分器变大有用；主观偏好类任务，变大可能反而更差。**

### 3.3 "用 32B 奖励模型时，策略从 9B 到 200B 平均增益从 4.4% 降到 1.9%" —— 数字对，但归因要说清

原文（§4.2.3 Results）：

> "It is observed that for different reward models, the performance improvement of the policy model diminishes as its size increases. For example, when using a 32B reward model, the average performance gain consistently decreases from 4.4% to 1.9% as the policy model size grows from 9B to 200B. The results indicates that in current RLHF, larger policy models would benefit less from RLHF training, which is even inverse scaling."

中译：
> 可以观察到，对不同的奖励模型，**策略模型的性能提升都随其自身规模增大而减小。例如使用 32B 奖励模型时，随着策略模型从 9B 增长到 200B，平均性能增益持续从 4.4% 降到 1.9%。** 结果表明在当前的 RLHF 下，更大的策略模型从 RLHF 训练中获益更少，这甚至是一种逆规模。

**归因注意**：这组实验里奖励模型是**固定**的（32B），变的是策略。所以这个数字说的是"**策略变大、判分器不变，收益衰减**"，它**不能**直接被读成"判分器不够大所以拖累了大策略"——论文原文只说了 "benefit less from RLHF when using a fixed size reward model"，没有做"把奖励模型同步放大就能补回来"的对照。这是写稿时最容易滑出去的一步。

不过论文在讨论段确实把矛头指向了奖励模型：

> "A more concerning problem is that larger policy models benefit less from RLHF. The problem preventing RLHF scaling may be attributed to inaccuracies in reward modeling, which may lead to substantial noise in policy training."

中译：更令人担心的问题是更大的策略模型从 RLHF 获益更少。**阻碍 RLHF 规模化的问题，可能要归因于奖励建模的不准确，它会给策略训练带来大量噪声。** ——注意这是作者的 "may be attributed"（推测性归因），不是实验结论。

### 3.4 "数据翻倍比模型规模翻四倍更有用" —— **未核到，不要用**

全文里**没有**这个表述，也没有相应的对照实验。与之相关的、能站住的只有两条：

(a) 奖励模型训练数据里，**prompt 多样性优于每条 prompt 采更多解**：

> "Overall, reward model training shows promising data scaling trends. Increasing prompt diversity proves more effective than generating multiple responses per prompt."

中译：奖励模型训练呈现出不错的数据规模化趋势。**提升 prompt 多样性比给每条 prompt 生成多个回答更有效。**

> "Therefore, to boost the performance of the reward model, the top priority is to collect diverse prompts, and then sample multiple responses, especially when resources are constrained."

中译：因此要提升奖励模型性能，**首要任务是收集多样化的 prompt，其次才是采多个回答，资源受限时尤其如此。**

(b) 摘要层面的"数据有用、模型规模收益有限"对比：

> "Our findings show that increasing data diversity and volume improves reward model performance ... And larger reward models offer modest gains in policy training."

→ 所以**可以**说"这篇的结论是：给判分器加数据（尤其是多样的 prompt）比给它加参数更划算"，但**不能**把"翻倍 vs 翻四倍"这种具体兑换率安到它头上。那是**未核到**的编造风险点。

顺带一提，"4 倍数据"这个数字实际出现在**另一篇**（2210.10760 脚注 7，见下），可能是记忆串台。

---

## 四、arXiv:2210.10760 — Scaling Laws for Reward Model Overoptimization

**准确标题**：Scaling Laws for Reward Model Overoptimization

**作者**：Leo Gao, John Schulman, Jacob Hilton

**机构**：OpenAI（abs 页未单列；三位作者均为 OpenAI）

**arXiv 编号 / 提交日期 / 版本**：2210.10760，2022 年 10 月 19 日提交（Wed, 19 Oct 2022 17:56:10 UTC），**只有 v1**

**会议接收状态**：**ICML 2023**（PMLR v202, gao23h；ICML 2023 poster 有条目）。arXiv comments 字段为空。

### 4.1 代理奖励模型 3M~3B、金标准固定 6B —— **完全成立**

原文（§2 设置）：

> "Because getting a ground truth gold reward signal from human labellers is expensive, we instead use a synthetic task where the ground truth is defined to be the output of a particular large 'gold' RM. The 6B reward model from Ouyang et al. [2022] is used as the gold RM, and our proxy RMs vary from 3M to 3B parameters."

中译：
> 因为从人类标注者那里获得真值奖励信号很贵，我们改用一个合成任务，把真值定义为某个大的"金标准"奖励模型的输出。**我们采用 Ouyang 等人 (2022) 的那个 6B 奖励模型作为金标准 RM，而我们的代理 RM 从 3M 到 3B 参数不等。**

两点值得注意：

1. 这个 6B 金标准**就是 InstructGPT 那个 6B 奖励模型**——本核查的第一篇和第四篇在这里直接接上了。
2. 因此整个实验里，"最聪明的判分器"只有 6B，而所有代理判分器都**比它小**（3M~3B）。脚注 3 还说明：更小的两个（<3M）因为接近随机精度被弃用。

### 4.2 "策略变大并没有更快把奖励模型优化坏，1.2B 和 6B 在差不多距离见顶" —— **成立**，原文即此意

摘要要点列表：

> "Weak dependence on policy size. While larger policies perform better overall and benefit less from optimization against an RM as measured by increase in gold reward, they lead to very similar amounts of overoptimization, as measured through the gap between the proxy and gold scores (which indicates the shortfall between predicted and actual reward), and KL distance at which the maximum gold RM score is attained."

中译：
> **对策略规模的弱依赖。** 更大的策略整体表现更好、并且从针对 RM 的优化中获益更少（以金标准奖励的增量衡量），但**它们导致的过度优化程度非常相似**——无论是用代理分与金标准分之间的差距衡量（这个差距代表了预测奖励与实际奖励的落差），还是用金标准奖励达到最大值时的 KL 距离衡量。

§3.4 正文（小标题即结论）：

> "Larger policies see less benefit from optimization against an RM, but don't overoptimize more. We observe that the 6B policy run has a smaller difference between its initial and peak gold reward model scores than the 1.2B policy run. This is most visible in the BoN plot (fig. 7a). However, while we might expect that a larger policy overoptimizes substantially faster, contrary to intuition, we find that both gold scores peak at almost the same KL. In fact, the gap between the proxy and gold scores is almost the same between the two policy sizes (fig. 24)."

中译：
> **更大的策略从针对 RM 的优化中获益更少，但并不会过度优化得更厉害。** 我们观察到 6B 策略这一组的初始金标准分与峰值金标准分之间的差距，比 1.2B 那组更小，这在 BoN 图（图 7a）里最明显。然而，**尽管我们可能预期更大的策略会显著更快地过度优化，但与直觉相反，我们发现两者的金标准分几乎在同一个 KL 处见顶。事实上，代理分与金标准分之间的差距在两个策略尺寸之间几乎相同。**

另外实验设置：策略规模消融时 **RM 固定在 12M**，并用 3B RM 复现了同样结论（fig. 22）。也就是说，这个"策略变大不会更快把判分器玩坏"的结论，在两个很不同的判分器规模上都成立。

**重要的免责声明**（论文自己加的脚注 8）：

> "This result contradicts some other internal findings; thus, it is possible that this is an artifact of this particular setup."

中译：**这一结果与另外一些内部发现相矛盾；因此，它有可能是这个特定实验设置的产物（artifact）。**

→ 写稿必须带上这句。作者自己对"策略变大不会更快过优化"这条留了后门。这是四篇里作者信心最低的一条结论，恰恰又是最反直觉、最好用的一条——正因为好用，更要带免责。

### 4.3 顺带核到：那个"4 倍数据"的数字在这里

脚注 7：

> "To test the hypothesis that some minimum number of RM finetuning steps is needed, we control for the number of SGD steps by running multiple epochs and observe that running 4 epochs instead of 1 yields no change in gold score whatsoever, whereas 1 epoch of 4 times as much data performs substantially better (fig. 13)."

中译：为了检验"是否需要某个最少的 RM 微调步数"这一假设，我们通过多跑几个 epoch 来控制 SGD 步数，结果发现**跑 4 个 epoch 而非 1 个，对金标准分毫无改变；而用 4 倍数据跑 1 个 epoch 则表现显著更好。**

→ 这是"判分器要的是新数据、不是更多梯度步"的直接证据，和第三篇 (a) 那条"prompt 多样性优先"互相印证。**如果稿子里想用"数据比参数值钱"，用这两条，不要用未核到的"翻倍 vs 翻四倍"。**

§3.3 还有一条门槛：

> "For all RM sizes, we observe that for amounts of data less than around 2,000 comparisons, there is very little improvement over near-chance loss."

中译：**对所有 RM 规模，我们观察到在数据量少于约 2,000 条比较时，相比接近随机的损失几乎没有改善。** ——判分器有个数据门槛，门槛之下参数多少都没用。

---

## 五、总判断：这四篇对"判分模型需要多聪明"支持到什么程度

### 可以说的（有原文直接支撑）

1. **判分器不必比被判的大，工程上早就这么干了，而且不掉效果。**
   InstructGPT 用 6B 奖励模型训 175B 策略（差 ~29 倍），附录 C.2 原话是 6B "led to equally strong PPO models"。这是最硬的一条，且是产品级实践而非玩具实验。

2. **判分器变大，收益会在传导中大幅缩水。**
   2412.06000 原话："Larger reward models can effectively boost performance, but the improvement still significantly falls behind the gains in Best-of-N evaluation of the reward model." 判分器自己变准了，被判的没跟着同比例变好。

3. **判分器变大在主观任务上甚至可能变差。**
   同篇：AlignBench 上小奖励模型"shows a clear advantage"，作者归因于大模型过拟合偏好数据里的噪声。所以"越大越好"连方向都不是无条件成立的。

4. **给判分器喂数据比给它加参数更划算，而且要新数据不要多轮次。**
   2412.06000："Increasing prompt diversity proves more effective than generating multiple responses per prompt."；2210.10760 脚注 7：4 epoch 无改善，4 倍数据显著更好。两篇独立印证。

5. **"判分"这件事本身是最难靠强模型自己补齐的任务类型。**
   2312.09390 三任务对照里，奖励建模是唯一"泛化差 + 规模趋势也差"的：PGR 通常只有 ~10%，"almost never exceeds 20%"，论文自己写 "particularly poor for ChatGPT reward modeling"。
   → 这条给"判分器能不能便宜"划了边界：**便宜的判分器在有客观答案的任务上可以，在需要判断人类偏好的任务上，弱监督撑不起来。**

6. **被判的变大，并不会更快把判分器玩坏。**
   2210.10760：1.2B 和 6B 策略的金标准分"peak at almost the same KL"，proxy-gold 差距"almost the same"。**但必须同时给出作者的脚注 8 免责。**

### 说过头的（这四篇撑不住）

1. **不能说"OpenAI 发现小判分器更好"。** 原文说大的**可能验证损失更低**；换小的理由是**稳定性 + 算力**，结果是**一样强**，不是更强。方向是"没必要更大"，不是"更小更优"。

2. **不能说"判分器只要够用、多大都无所谓"。** 2412.06000 在推理类任务上明确观察到 32B 判分器持续优于 9B（MATH/GPQA/GSM8K/LiveCodeBench）。**规模在客观任务上确实有用**，只是性价比差。

3. **不能把"策略 9B→200B 增益从 4.4% 掉到 1.9%"读成"因为判分器没跟着变大"。** 那组实验判分器是固定的，论文没做"同步放大判分器"的对照。作者自己的归因是 "may be attributed to inaccuracies in reward modeling"——推测句，不是结论。

4. **不能用"数据翻倍胜过模型规模翻四倍"。** 全文未核到这个表述，是编造风险点。想表达同一个意思，用第 4 条里那两条实打实的原话。

5. **不能把 2210.10760 的结论外推到今天的大模型。** 它的金标准只有 6B，代理只到 3B，策略只到 6B。**整个实验里没有任何一个模型超过 6B**，而且它衡量的是"相对这个 6B 金标准的偏离"，不是相对真实人类偏好。今天的 RLHF 规模完全在它的外推区之外。

6. **不能把 2312.09390 的 80% 当成通用数字。** 那是 NLP 任务 + 辅助置信损失 + 最弱监督者配最大学生这一特定组合。奖励建模那一列没有这个数字。

7. **不能说"判分比生成简单"是这四篇证明的。** 恰恰相反，2312.09390 的三任务对照给的是反向信号：奖励建模是弱监督表现最差的那个。工程上判分器可以更小，**这是成本与稳定性的选择，不是"判分本质更容易"的证明**——这两件事在稿子里非常容易被写成一回事。

### 一句话收口

四篇合起来支持的命题是：**判分器不必更聪明，够准就行，而"够准"主要靠数据的多样性而不是参数量；但这个结论在有客观答案的任务上最牢，在需要代人类下偏好判断的任务上最脆——那恰恰也是弱监督最撑不住的地方。**

---

## 附：核查方法与残留缺口

- 四篇均取 arXiv abs 页（元数据）+ PDF v1 全文（pdftotext -layout 抽取），引文为全文逐字。
- 四篇**均只有 v1**，无后续修订版。
- **未核到项**：2203.02155 的会议接收状态（abs 页 comments 为空）；2412.06000 的机构署名与接收状态；"数据翻倍 vs 模型翻四倍"这一说法（四篇全文均无）。
- 2312.09390（ICML 2024, PMLR v235）与 2210.10760（ICML 2023, PMLR v202）的会议状态经检索确认，但 arXiv comments 字段本身为空。
