#!/usr/bin/env python3
# 中学教师智能通识课 PPT 生成器
# 用法: python3 make_ppt.py   → 生成 智能通识课.pptx
# 改内容直接改下面的 SLIDES 列表，重跑即可

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# ============ 配色（跟公众号封面一套）============
BG       = RGBColor(0x0E, 0x14, 0x28)   # 深蓝底
BG_ALT   = RGBColor(0x0A, 0x1A, 0x1A)   # 章节页底
WHITE    = RGBColor(0xF8, 0xFA, 0xFC)
GREY     = RGBColor(0x94, 0xA3, 0xB8)
YELLOW   = RGBColor(0xFD, 0xE0, 0x47)
TEAL     = RGBColor(0x5E, 0xEA, 0xD4)
ORANGE   = RGBColor(0xF5, 0x9E, 0x0B)
RED      = RGBColor(0xF8, 0x71, 0x71)
BLUE     = RGBColor(0x93, 0xC5, 0xFD)
GREEN    = RGBColor(0xA3, 0xE6, 0x35)

FONT = 'Microsoft YaHei'   # 现场 Windows 机器最稳；WSL 预览用 Noto 也认

W, H = Inches(13.333), Inches(7.5)   # 16:9


def new_deck():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    return prs


def blank(prs, bg=BG):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bgfill = s.background.fill
    bgfill.solid()
    bgfill.fore_color.rgb = bg
    return s


def txt(slide, x, y, w, h, text, size=24, color=WHITE, bold=False,
        align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.15):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    lines = text.split('\n')
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        # 支持 **加粗** 片段
        parts = line.split('**')
        for j, seg in enumerate(parts):
            if not seg:
                continue
            r = p.add_run()
            r.text = seg
            r.font.size = Pt(size)
            r.font.name = FONT
            r.font.color.rgb = color
            r.font.bold = bold or (j % 2 == 1)
    return tb


def band(slide, y, h, color):
    """一条色带"""
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(y), W, Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def card(slide, x, y, w, h, border=TEAL, fill=RGBColor(0x16, 0x20, 0x38)):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = border
    sh.line.width = Pt(1.5)
    sh.shadow.inherit = False
    try:
        sh.adjustments[0] = 0.06
    except Exception:
        pass
    return sh


def note(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# ==================== 页面生成 ====================

def s_title(prs):
    s = blank(prs, BG_ALT)
    txt(s, 1.0, 1.95, 11.3, 1.15, "AI 这两年到底发生了什么", 48, WHITE, True, PP_ALIGN.CENTER)
    band(s, 3.35, 0.06, ORANGE)          # 线在标题下方
    txt(s, 1.0, 3.65, 11.3, 0.85, "一个程序员和 AI 相处两年的笔记", 22, GREY, False, PP_ALIGN.CENTER)
    txt(s, 1.0, 5.6, 11.3, 0.8, "靳岩岩　·　2026 年 9 月", 18, GREY, False, PP_ALIGN.CENTER)
    note(s, "自我介绍两句：程序员，写了 300 篇 AI 学习笔记的公众号，家里两台小超算天天跑模型。不是 AI 专家，是每天跟它打交道的人。")
    return s


def s_open_q(prs):
    s = blank(prs)
    txt(s, 1.0, 2.4, 11.3, 1.6, "你们上周用 AI\n干了什么？出过什么错？", 44, YELLOW, True, PP_ALIGN.CENTER)
    txt(s, 1.0, 5.2, 11.3, 0.6, "收两三个回答，后面讲机制时回头接", 18, GREY, False, PP_ALIGN.CENTER)
    note(s, "收两三个回答，记在纸上。收尾时用讲过的机制各解释一句。")
    return s


def s_thesis(prs):
    s = blank(prs, BG_ALT)
    txt(s, 0.9, 2.0, 11.5, 2.0, "AI 是一个\n每天早上重新出生的天才儿童", 42, WHITE, True, PP_ALIGN.CENTER)
    txt(s, 0.9, 4.6, 11.5, 0.9, "读过全人类的书　·　没有身体　·　记不住昨天", 26, TEAL, False, PP_ALIGN.CENTER)
    note(s, "全场就这一句。三条脉络的落点都回到这里。")
    return s


def s_section(prs, num, title, sub):
    s = blank(prs, BG_ALT)
    txt(s, 1.0, 2.15, 11.3, 0.55, num, 20, ORANGE, True, PP_ALIGN.CENTER)
    txt(s, 1.0, 2.80, 11.3, 1.05, title, 40, WHITE, True, PP_ALIGN.CENTER)
    band(s, 4.15, 0.05, TEAL)            # 线在标题下方，不压字
    if sub:
        txt(s, 1.0, 4.45, 11.3, 0.7, sub, 20, GREY, False, PP_ALIGN.CENTER)
    return s


def s_bullets(prs, title, bullets, note_text="", color=WHITE, size=24):
    s = blank(prs)
    txt(s, 0.85, 0.5, 11.6, 0.95, title, 32, WHITE, True)
    band(s, 1.58, 0.03, ORANGE)
    y = 2.0
    for b in bullets:
        txt(s, 1.0, y, 11.3, 1.0, b, size, color)
        y += 0.95
    if note_text:
        note(s, note_text)
    return s


def s_bignum(prs, title, items, footer="", note_text=""):
    """一行几个大数字卡片"""
    s = blank(prs)
    txt(s, 0.85, 0.5, 11.6, 0.95, title, 32, WHITE, True)
    band(s, 1.58, 0.03, ORANGE)
    n = len(items)
    gap = 0.3
    cw = (11.6 - gap * (n - 1)) / n
    CT = 2.35                                  # 卡片顶
    # 数字字号：按最长的那个数字定，避免撑宽
    maxnum = max(len(str(num)) for num, _, _ in items)
    nfs = 54 if maxnum <= 5 else (44 if maxnum <= 8 else 36)
    # 标签字号：按最长标签行定，避免换行溢出
    maxlab = max(max(len(ln) for ln in str(lab).split(chr(10))) for _, lab, _ in items)
    lfs = 18 if maxlab <= 10 else (16 if maxlab <= 14 else 14)
    # 卡片高度：按标签最多几行动态算
    nlab = max(len(str(lab).split(chr(10))) for _, lab, _ in items)
    labh = nlab * lfs * 1.45 / 72.0            # 标签实际占高（英寸）
    NUMTOP, NUMH = 0.40, 1.05                  # 数字区
    CH = NUMTOP + NUMH + labh + 0.32           # 顶距 + 数字 + 标签 + 底距
    CH = max(CH, 2.5)
    x = 0.85
    for num, label, col in items:
        card(s, x, CT, cw, CH, border=col)
        txt(s, x + 0.1, CT + NUMTOP, cw - 0.2, NUMH, num, nfs, col, True, PP_ALIGN.CENTER)
        txt(s, x + 0.1, CT + NUMTOP + NUMH + 0.05, cw - 0.2, labh + 0.1, label, lfs, GREY, False, PP_ALIGN.CENTER)
        x += cw + gap
    if footer:
        txt(s, 0.85, CT + CH + 0.30, 11.6, 1.0, footer, 22, YELLOW, True, PP_ALIGN.CENTER)
    if note_text:
        note(s, note_text)
    return s


def s_quote(prs, big, small="", color=YELLOW, note_text=""):
    s = blank(prs, BG_ALT)
    txt(s, 1.0, 2.5, 11.3, 2.0, big, 40, color, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    if small:
        txt(s, 1.0, 4.9, 11.3, 1.0, small, 20, GREY, False, PP_ALIGN.CENTER)
    if note_text:
        note(s, note_text)
    return s


def s_twocol(prs, title, left_title, left_lines, right_title, right_lines,
             lcolor=BLUE, rcolor=GREEN, footer="", note_text=""):
    s = blank(prs)
    txt(s, 0.85, 0.5, 11.6, 0.95, title, 32, WHITE, True)
    band(s, 1.58, 0.03, ORANGE)
    nmax = max(len(left_lines), len(right_lines))
    CARD_TOP, CARD_MAX = 1.95, 3.9         # 卡片顶 / 最大高度
    HEAD, PAD = 0.85, 0.28                 # 栏标题占位 / 底部内边距
    avail = CARD_MAX - HEAD - PAD          # 留给正文的净高
    lh = min(0.62, avail / max(nmax, 1))   # 行距按行数反推，保证装得下
    fs = 18 if lh >= 0.58 else (16 if lh >= 0.48 else 14)
    ch = HEAD + lh * nmax + PAD            # 卡片高度跟着内容走
    card(s, 0.85, CARD_TOP, 5.6, ch, border=lcolor)
    txt(s, 1.15, CARD_TOP + 0.18, 5.0, 0.5, left_title, 21, lcolor, True, PP_ALIGN.CENTER)
    y = CARD_TOP + HEAD
    for line in left_lines:
        txt(s, 1.15, y, 5.0, lh, line, fs, WHITE)
        y += lh
    card(s, 6.9, CARD_TOP, 5.6, ch, border=rcolor)
    txt(s, 7.2, CARD_TOP + 0.18, 5.0, 0.5, right_title, 21, rcolor, True, PP_ALIGN.CENTER)
    y = CARD_TOP + HEAD
    for line in right_lines:
        txt(s, 7.2, y, 5.0, lh, line, fs, WHITE)
        y += lh
    if footer:
        txt(s, 0.85, CARD_TOP + ch + 0.22, 11.6, 0.9, footer, 21, YELLOW, True, PP_ALIGN.CENTER)
    if note_text:
        note(s, note_text)
    return s


def s_table(prs, title, headers, rows, footer="", note_text=""):
    s = blank(prs)
    txt(s, 0.85, 0.5, 11.6, 0.9, title, 30, WHITE, True)
    band(s, 1.52, 0.03, ORANGE)
    nrow, ncol = len(rows) + 1, len(headers)
    tb = s.shapes.add_table(nrow, ncol, Inches(0.85), Inches(1.85),
                            Inches(11.6), Inches(0.6 * nrow)).table
    for j, htext in enumerate(headers):
        c = tb.cell(0, j)
        c.text = htext
        for p in c.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(18); r.font.bold = True
                r.font.color.rgb = WHITE; r.font.name = FONT
        c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = tb.cell(i, j)
            c.text = str(val)
            for p in c.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
                for r in p.runs:
                    r.font.size = Pt(16); r.font.name = FONT
                    r.font.color.rgb = WHITE
            c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0x14, 0x1D, 0x33)
    if footer:
        txt(s, 0.85, 1.9 + 0.6 * nrow + 0.2, 11.6, 1.0, footer, 20, YELLOW, True, PP_ALIGN.CENTER)
    if note_text:
        note(s, note_text)
    return s


def build():
    prs = new_deck()

    # ---------- 开场 ----------
    s_title(prs)
    s_open_q(prs)
    s_thesis(prs)

    # ---------- 第一线 ----------
    s_section(prs, "第一线", "它是怎么学的", "大语言模型 → 多模态 → AGI")

    s_bullets(prs, "1.1　它只学了一件事：猜下一个词", [
        "把互联网上的文字全读一遍，练到能接下句",
        "像背课文背到能接龙，区别是它背了整个图书馆",
    ], note_text="2023 年 ChatGPT 那一波。老师熟悉的类比：背课文接龙。")

    s_twocol(prs, "1.1　由此得到两个性质",
             "性格是筛出来的",
             ["训练数据不是全人类",
              "是「会写字、发到网上、",
              "还过了质量筛选」的那群人",
              "",
              "所以它天然好奇、爱表达",
              "不是设计的，是筛的"],
             "它没有身体",
             ["所有概念从字里长出来",
              "不是从手上摸出来的",
              "",
              "能推导不存在的物理定律",
              "却不知道羽毛掉地上不会碎"],
             footer="幻觉不是撒谎，是一个没有身体的东西在真诚地乱讲",
             note_text="老师提的「AI 常识错误」在这里接。让它讲三维世界，跟让人讲四维空间一样。")

    s_bullets(prs, "1.2　图和字进了同一个脑子", [
        "看图、听声音，进了同一个模型",
        "它们**共享结构，但不共享坐标**",
        "能互相翻译，翻译会丢东西",
    ], note_text="推论：让它看试卷照片识别手写，能认，但细节会漏；先转成文字再问，比直接问图靠谱。")

    # ---------- 1.3 四个实验 ----------
    s_section(prs, "1.3", "四个反直觉的实验", "全场最该记住的一段")

    s_quote(prs, "① 它内部有类似情绪的东西\n而且能调",
            "Anthropic，2026 年 4 月",
            TEAL,
            "先问一句：你猜是不是？")

    s_bignum(prs, "① 往内部注入一个方向，行为就变了",
             [("22%", "基线", GREY),
              ("72%", "注入「绝望」", RED),
              ("0%", "注入「平静」", GREEN)],
             footer="注入幅度只有 0.05　—　轻轻拨一下旋钮，行为就翻倍",
             note_text="171 个情绪方向，第一主成分对应「好受还是难受」，跟心理学的情绪环状模型对得上——没人教它，是它自己从人写的文字里长出来的。\n⚠️ 不描述压力场景细节。有信息量的是「能调」这件事。\n⚠️ 别说成 persona vectors，那是另一篇。")

    s_quote(prs, "模型内部有一组可测量、可调节的状态",
            "功能上跟情绪一样：状态在前，行为在后",
            YELLOW,
            "说完停一下，问一句「这个是不是有点耳熟」，让老师自己接。")

    s_quote(prs, "② 反复学\n某一刻突然就会了",
            "OpenAI，2022 年，这个现象叫 grokking",
            TEAL, "先问：你猜练下去会怎样？")

    s_bignum(prs, "② 训练成绩和测试成绩，差了三个数量级",
             [("1 千步", "训练题\n全背下来了", BLUE),
              ("10 万步", "测试成绩\n还在瞎猜", GREY),
              ("100 万步", "突然\n就会了", GREEN)],
             footer="任务是「模 97 的除法」，瞎猜大概 1%",
             note_text="⚠️ 说「瞎猜的水平」，不说「零」。\n背会和真会之间隔着一道坎。内部发生的事：从「记住每道题的答案」变成「找到了那条规律」。")

    s_quote(prs, "「看起来一直没进步」\n和「其实没进步」，是两回事",
            "前提是还在练",
            YELLOW, "说完停。带毕业班的老师听到这儿自己会有反应，不用替他们说。")

    s_quote(prs, "③ 它偶尔能看见\n自己在想什么",
            "Anthropic，2025 年 10 月",
            TEAL, "做法：往中间层直接注入一个概念的向量，然后问它「你现在有没有察觉到什么」。")

    s_bignum(prs, "③ 察觉得到，但很不可靠",
             [("20%", "能察觉到\n被注入的概念", GREEN),
              ("0", "不注入时的假阳性\n（100 次里）", BLUE)],
             footer="而且是在它把那个词说出口之前就报告的",
             note_text="⚠️ 是 Claude Opus 4 和 4.1，不是「最强的模型」。\n⚠️ 20% 的前提是研究者挑好了注入的层和强度。论文自己写「失败才是常态」。\n但 0 假阳性说明它不乱报——那 20% 是真的偶尔看见了。")

    s_quote(prs, "④ 但它看不全\n看不全的部分就编",
            "同一批研究，2025 年 3 月",
            TEAL, "下一页：让它算 36 + 59。先问老师「你猜它心里是怎么算的」。")

    s_quote(prs, "36 + 59 = ?",
            "先问一句：你猜它心里是怎么算的？",
            WHITE, "让老师答一两句，多半会说「跟我们一样列竖式」。然后翻下一页。")

    s_twocol(prs, "④ 算 36 + 59 时，它实际在做的　vs　它嘴上说的",
             "内部实际的路径（研究者看到的）",
             ["一路算大概齐",
              "四十上下 ＋ 六十上下 → 九十几",
              "",
              "另一路只盯个位",
              "6 ＋ 9 → 末位是 5",
              "",
              "两路一合 → **95**"],
             "它嘴上说的（问它怎么算的）",
             ["「个位 6 加 9 得 15，进一位；",
              "十位 3 加 5 加 1 得 9；",
              "所以是 95」",
              "",
              "—— 答案对，过程是假的",
              "这是网上到处都有的",
              "竖式加法标准答案"],
             lcolor=GREEN, rcolor=RED,
             footer="不是撒谎：它对自己的中间过程只有很有限的访问权",
             note_text="缺的那部分用一个说得通的故事补上。人也这样，心理学叫虚构。\n注意：答案是对的，错的是它对自己过程的描述。这点要说清楚，不然老师会以为它算错了。")

    s_bullets(prs, "④ 人也这样：裂脑人实验", [
        "被试是**切断了胼胝体的癫痫病人**——两个半球不通消息",
        "盯住屏幕正中，**左半边闪雪景，右半边闪鸡爪**",
        "左手（右脑管）选了**铁锹**，右手（左脑管）选了**鸡**",
        "问他为什么选铁锹——**只有左脑会说话，而左脑没看见雪景**",
        "于是它现编了一个：**「鸡爪配鸡，清理鸡舍需要铁锹」**",
    ], size=21, note_text="Gazzaniga & LeDoux 1978，左脑解释器。\n⚠️ 被试不是正常人，是上世纪治重度癫痫切断胼胝体的病人，前后就十几个。日常看着完全正常，只是两半球之间不通消息。\n⚠️ 不是「一只眼看一样」——每只眼睛的左半视野都进右脑、右半视野都进左脑，分的是画面的左右两半。闪约 150 毫秒，来不及转眼球。\n⚠️ 正常人做这个测不出来：右脑看到雪景，立刻通过胼胝体告诉左脑，照样说得出。")

    s_quote(prs, "正常人也是各半球各干各的\n只是随时通消息，所以你察觉不到",
            "裂脑人只是把这层遮蔽掀开了",
            TEAL, "别让老师以为「这是病人的怪事」。会说话的那部分替不会说话的那部分编理由，这件事一直在发生。")

    s_quote(prs, "要判断它会不会\n得看它做出来的东西，不能只听它解释",
            "", YELLOW, "这条收在 AI 上就行，后半句老师自己会想到。")

    s_quote(prs, "这四件事人身上都有\n只是它的分辨率低得多",
            "所以「每天重新出生的天才儿童」不是比喻上的方便，是真能用来预测它的行为",
            TEAL)

    s_bullets(prs, "1.4　AGI 什么时候来", [
        "世界模型早就在它脑子里了",
        "缺的是**带着问题去跟现实对质**",
        "现在的 AI 是读了万卷书没出过门的书生",
        "**读书在前，行路在后，顺序不能反**",
    ], note_text="不下结论 AGI 几年到。说「顺序」比说「时间」靠谱。\n反面：有一派想跳过语言从零认识世界——没文化的人行一万里也只是换个地方发呆。")

    # ---------- 第二线 ----------
    s_section(prs, "第二线", "它开始自己干活了", "工具调用 → 智能体框架 → 未来")

    s_bullets(prs, "2.1　第一步：它会调工具了（2024）", [
        "能在回答里插一句「我要查一下天气」",
        "系统去查了，再把结果给它",
        "从**只会说**到**会动手**",
        "此时它一次只做一件事，做完就停",
    ])

    s_bullets(prs, "2.2　第二步：它能自己跑几个小时了（2025–2026）", [
        "给一个目标，它自己拆任务、写代码、跑测试、看结果、改",
        "**循环几百次，中间不用人管**",
    ], note_text="Claude Code、Codex 这类东西。下一页开始现场演示。")

    s_twocol(prs, "同一个需求，两种说法",
             "粗着说",
             ["「帮我做一节课的课件」",
              "",
              "它只能猜：",
              "什么科目？什么年级？",
              "多长时间？讲到哪一步？",
              "",
              "→ 给你一份放之四海的空壳"],
             "细着说",
             ["「高一物理，牛顿第二定律，",
              "45 分钟，三个环节，",
              "每个环节配一道题，",
              "题目难度递增」",
              "",
              "→ 给你一份能直接改改就用的"],
             lcolor=GREY, rcolor=GREEN,
             note_text="不现场演示，直接讲。\n可以问一句：「你们平时是哪种说法？」\n若有老师想看真实结果，可用手机现场试一句，但别指望投屏。")

    s_quote(prs, "AI 干活的质量\n等于你说话的粒度",
            "", YELLOW, "下面用三家厂商的官方指南和几篇论文说明这不是我的个人经验。")

    s_bullets(prs, "这不是我的个人经验：三家官方指南的交集", [
        "OpenAI 列了六条，Anthropic 七条，Google 七条",
        "**三家都有的只有四条**：",
        "　把指令说清楚　·　给示例　·　补上下文　·　拆分复杂任务",
        "有研究量过：**模型默认只能猜中你没说出口的需求的 41%**",
    ], size=22, note_text="OpenAI: Six strategies。Anthropic: Be clear and direct / Add context / Use examples / XML tags / Give a role / Control format / Chain prompts。Google: Prompt design strategies。\n41% 出自 ACL 2026 Findings《What Prompts Don't Say》，同一篇还测到：说不清的提示词换个模型变差的概率是两倍。")

    s_bullets(prs, "★ 三个流传很广、但靠不住的说法", [
        "**「你是一位资深××专家」** —— 对准确率没用，可能还有害",
        "**「深呼吸，一步一步想」** —— 是程序在一个模型上搜出来的，换个模型就不灵",
        "**「这对我很重要」** —— 大规模重复实验里效果互相抵消",
    ], size=22, note_text="角色设定：162 个角色 × 2410 道题 × 4 个模型族，加了跟不加没区别，哪个角色好基本随机；另一组在 MMLU 上测到 71.6% 掉到 66.3%。\n⚠️ 但要说句公道话：角色设定对语气和格式确实有效——想让它说得像老师讲课可以用，想让它算得更对没用。\n深呼吸：Google 用程序自动搜索出的字符串，在那个模型那个数据集上 80.2% vs 71.8%。论文讲的是「自动优化提示词有效」，不是这句话有魔力。\n这对我很重要：宾大 Wharton 用 GPT-4o 每题重复 100 次，单题上能摆动 60 个百分点，但整个数据集上互相抵消。")

    s_twocol(prs, "★ 唯一有硬证据的那条：让它一步一步想",
             "2022 年的实验",
             ["数学应用题上",
              "",
              "直接问：**17.9%**",
              "加「一步一步想」：**56.9%**",
              "",
              "这是提示词研究里最硬的一条"],
             "但今天已经不用你说了",
             ["原论文就写明：",
              "小模型用了**反而更差**",
              "",
              "现在的主流模型",
              "**已经把这一步内化了**",
              "它自己会想"],
             lcolor=GREEN, rcolor=GREY,
             footer="技巧会老化：按模型代际重新适配，不能照搬三年前的教程",
             note_text="Wei et al. 2022, arXiv:2201.11903，PaLM 540B 在 GSM8K 上 17.9%→56.9%。\n原论文：约 100B 参数以下无效甚至有害，会产出「流畅但不合逻辑」的推理链。\n老化的证据：arXiv:2608.24641，测了五种技巧跨三组新老模型，GPT 线上新版本的边际收益递减甚至为负。")

    s_quote(prs, "管用的是把要求说清楚\n不是加咒语",
            "那些咒语里最有名的一条，还专门被复现推翻过",
            YELLOW,
            "「这对我的职业生涯很重要」：2023 年一篇论文宣称提升 115%，2024 年系统复现发现几乎没有统计显著效果。那个 115% 是挑每个任务最好的一句算的，取平均只有 2%；原论文自己的表格里六个模型有三个反而变差，包括当时最强的 GPT-4。\n⚠️ 别跟 1.3 ① 混：①是往模型内部注入向量，硬证据；这里否掉的是「随口加句话就能可靠提分」。")

    s_bullets(prs, "2.3　上周的真事", [
        "让 AI 把一个显卡的神经网络移植到另一家显卡上",
        "AI 干了三天，写了三千行工作日志，周五说**「跑通了」**",
        "周日凌晨我进游戏一看——**画面根本没变**",
        "查下来：一个标记位没设，所有结果每帧都算了、每帧都被扔了",
    ], note_text="各种指标全对，唯一不对的是最终画面。之后 AI 自己从第 0 层重做，要求每一层和原版一个数都不差，一天推完六十块。")

    s_twocol(prs, "2.3　从这件事抽出来的两句",
             "协作没了，剩下指挥",
             ["以前这活要四五个人",
              "加一个项目经理",
              "",
              "现在是一个 AI",
              "加一个人的两个判断"],
             "过程指标和结果打架，信结果",
             ["AI 说「完成了」不算",
              "日志说「执行了」不算",
              "",
              "**屏幕上的画面算**"],
             lcolor=BLUE, rcolor=ORANGE,
             note_text="两年后所有人都是「管 AI 的经理人」，差别只在指挥和验收的能力。\n★ 若老师问「学生用 AI 写作业怎么办」，可答：让他当堂讲一遍。但别主动讲这条。")

    s_bullets(prs, "2.4　它记不住", [
        "每次对话都是新的一天，它不记得上次",
        "要么你替它记（笔记、文档、知识库）",
        "要么每次重新教",
        "**你第一句话怎么说，决定它调出哪一套**",
    ], note_text="教这件事没被替代，只是对象多了一个。跟带新班一样，第一节课定调。")

    # ---------- 第三线 ----------
    s_section(prs, "第三线", "给它一个身体", "比大家想的远")

    s_bullets(prs, "3.1　一个立场", [
        "现在机器人的路线是**先造身体再装脑子**",
        "我认为顺序反了：**先有脑子，再由脑子决定要什么身体**",
        "进化用四十亿年让脑子和身体互相适应，工程想跳过这一步",
    ], note_text="那些叠衣服、递水杯的演示视频：训练集里密度最高的几个动作，换个杯子形状就要重采一批数据。")

    s_quote(prs, "莫拉维克悖论", "Hans Moravec，1988 年", TEAL,
            "这个悖论给上一页那个立场提供了学界公认的说法。")

    s_twocol(prs, "为什么点鼠标比解物理题难",
             "我们以为的难",
             ["下棋、证明定理",
              "解方程、写论文",
              "",
              "→ 对 AI 反而**容易**"],
             "我们以为的简单",
             ["认出一张脸、走路不摔",
              "把杯子从桌上拿起来",
              "",
              "→ 对 AI 反而**极难**"],
             lcolor=GREY, rcolor=RED,
             footer="莫拉维克的解释：感知和运动被自然选择磨了几亿年，久到沉进了硬件里",
             note_text="原话大意：让计算机在智力测验或下棋上达到成人水平相对容易，但要让它有一岁小孩的感知和行动能力，极难甚至做不到。\n为什么：感知运动能力被打磨了几亿年，用起来不费力所以显得「简单」；抽象推理是最近几万年才有的，还很生涩，反而显得「高级」。\n**我们对难易的判断，是被自己的内省骗了。**")

    s_twocol(prs, "一个当场的例子：这份 PPT 是怎么做的",
             "你们以为的做法",
             ["让 AI 打开 PowerPoint",
              "帮我拖文本框、调字号",
              "",
              "→ 这是让它**用鼠标**",
              "→ 正是它最不擅长的事"],
             "实际的做法",
             ["让 AI 写一个几百行的程序",
              "程序自己生成这个文件",
              "",
              "→ 这是让它**写代码**",
              "→ 正是它最擅长的事"],
             lcolor=GREY, rcolor=GREEN,
             footer="你们现在看的这五十几页，就是这么来的：改一句话，重跑十秒，全部更新",
             note_text="当场指着屏幕说「你们现在看的这个就是」。这是全场唯一一个不用截图、不用联网的活证据。\n好处不只是快：改一个数字所有相关页面一起更新不会漏；排版规则写在程序里，不会这页字大那页字小。昨晚改了七八处，每次重跑十秒。\n翻译成能用的：与其让 AI 帮你在软件里一步步操作，不如让它直接生成一个能用的文件——教案、试卷、成绩统计表都一样。\n下一页就是它「用鼠标」有多差的数据。")

    s_bullets(prs, "3.2　证据：让它操作电脑", [
        "这是机器人的最简版：二维、传感器完美、错了能撤销",
        "有个标准测试叫 OSWorld，让 AI 直接操作一台真电脑",
    ])

    s_bignum(prs, "3.2　同样是操作电脑，题目复杂一个量级，成绩掉十三个点",
             [("86%", "常规任务\n每题不到三十步\n（人类约 72%）", GREEN),
              ("72.6%", "复杂任务\n每题两百多步", RED)],
             footer="不是它变笨了，是长流程里每一步的小误差会累积",
             note_text="⚠️ 这是两套评测，不是同一套题调难度：86% 出自 OSWorld-Verified（现在的通用口径），72.6% 出自 OSWorld 2.0（另一套协议，108 题，每题平均 250 多步）。台上要说「两套评测」，别说成「同一套题换了难度」。\n人类基线约 72.4% 来自原版 OSWorld。\n要点：在常规任务上它表面已经超过人类，但任务一长就掉——**长流程是它的软肋**，这跟第二线那个 DLSS 故事是同一件事。")

    s_twocol(prs, "同一批模型，同一年，两件事",
             "数学家找了 56 年的一个改进",
             ["1969 年有人算出一种",
              "更快的矩阵乘法，此后没人超过",
              "",
              "2025 年 5 月，谷歌的 AI",
              "**找到了更快的那一个**",
              "",
              "（这类开放数学问题它试了 50 多个，",
              "约两成真正改进了已知最好解）"],
             "把鼠标移到一个按钮上",
             ["它可能点偏",
              "",
              "一个人三十秒能做完的",
              "排版操作",
              "",
              "**它要花十二分钟**"],
             lcolor=GREEN, rcolor=RED,
             footer="陶哲轩的说法：AI 的能力是尖刺状的——某些窄领域超人，同时会犯好笑的低级错误",
             note_text="这就是莫拉维克悖论的当代版本，别的都不用记。\n具体是什么：AlphaEvolve 找到用 48 次乘法算两个 4×4 复数矩阵的方法，Strassen 1969 年的算法是 49 次。少一次，但那是全世界数学家找了半个世纪没找到的一次。\n⚠️ 有人问细节再说数字，不主动报——老师对 48 和 49 没感觉，对「56 年」有感觉。\n⚠️ 要说准：限定在复数矩阵这个设定，而且是「找到一个更好的构造」不是「证明了定理」。\n⚠️ 成功率：那 50 多个问题里约 75% 只是复现已知最好解，两成才是改进。\n陶哲轩指出它可靠的原因：LLM 只负责提出变异，验证器是人写的代码不会幻觉——AI 负责猜，人写的程序负责判。")

    s_quote(prs, "一个人三十秒能做完的排版操作\n它要花十二分钟",
            "这么好的条件都这样，三维、传感器带噪、摔坏了不能撤销的版本，远",
            RED, "老师听到这里会放心：AI 在屏幕里很强，出屏幕很弱。")

    # ---------- 模型表 ----------
    s_table(prs, "最近两个月的四个模型",
            ["模型", "一句定位", "对你意味着什么"],
            [["Kimi K3\n（月之暗面，7 月）", "国产开源里最大的\n权重公开可下载", "学校要私有部署\n数据不出校，是现实选项"],
             ["GLM-5.3-Flash\n（智谱，8 月底）", "名字像小份\n实际是新底座", "国产几家在互相抄作业\n而且抄得快"],
             ["Claude Fable 5.1\n（Anthropic）", "写代码和长文最稳\n管得也最严", "干正经活首选\n但国内用不方便"],
             ["GPT-6 Astra\n（OpenAI，9 月）", "上周干活的那个\n一天推六十块", "最能自己跑的\n适合长任务"]],
            footer="四家水平已经拉平，差别在性格，选哪个看你干什么",
            note_text="老师日常备课，DeepSeek、豆包、Kimi 都够，不用追最新。\n★ 看气氛：国产模型不防「AI 有想法」、防政治敏感词；美国模型反过来。不合适就跳。")

    # ---------- 收尾 ----------
    s_section(prs, "收尾", "三句话", "")

    s_bullets(prs, "如果只记三句", [
        "1　它读过所有书，但**没出过门**，常识会错，编的东西是真诚的",
        "2　它能自己干活了，但需要有人**说清楚要什么、并且看最终结果**",
        "3　它**记不住昨天**，每次开新窗口都是第一次见你",
    ], size=22, note_text="回到开场老师的那两三个回答，用讲过的机制各解释一句。")

    s_quote(prs, "要判断它会不会\n得看它做出来的东西，不能只听它解释",
            "", YELLOW, "说完就停，不加「对学生也一样」——在座的都想得到。")

    s_last = blank(prs, BG_ALT)
    txt(s_last, 1.0, 2.2, 11.3, 1.0, "谢谢", 44, WHITE, True, PP_ALIGN.CENTER)
    txt(s_last, 1.0, 3.6, 11.3, 0.8, "靳岩岩", 24, TEAL, False, PP_ALIGN.CENTER)
    txt(s_last, 1.0, 4.4, 11.3, 0.8, "同名公众号", 20, GREY, False, PP_ALIGN.CENTER)

    return prs


def audit(path):
    """排版自检：溢出画布 / 线压字 / 文字出框。改完内容重跑会自动查。"""
    from pptx import Presentation as _P
    d = _P(path)
    ww, hh = d.slide_width, d.slide_height
    tol = Emu(9144)          # 0.01 英寸容差
    over, hit, out_card = [], [], []
    for i, sl in enumerate(d.slides, 1):
        bands, cards = [], []
        for sh in sl.shapes:
            if sh.has_text_frame or not sh.height:
                continue
            (bands if sh.height < Inches(0.12) else cards).append(
                (sh.left, sh.top, sh.left + sh.width, sh.top + sh.height))
        for sh in sl.shapes:
            if sh.left is None:
                continue
            r, b = sh.left + (sh.width or 0), sh.top + (sh.height or 0)
            if r > ww + tol or b > hh + tol:
                over.append(i)
            if not sh.has_text_frame or not sh.text_frame.text.strip():
                continue
            head = sh.text_frame.text[:18].replace(chr(10), '/')
            for _, bt, _, bb in bands:
                if sh.top < bb and b > bt:
                    hit.append((i, head))
            for cl, ct, _, cb in cards:
                if sh.left >= cl - tol and ct <= sh.top < cb and b > cb + tol:
                    out_card.append((i, head))
    print(f'  溢出画布: {sorted(set(over)) or "无"}')
    print(f'  线压字  : {hit or "无"}')
    print(f'  文字出框: {out_card or "无"}')
    return not (over or hit or out_card)


if __name__ == '__main__':
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '智能通识课.pptx')
    prs = build()
    prs.save(out)
    print(f'生成完成：{out}')
    print('排版自检：')
    ok = audit(out)
    print('  → 全部通过' if ok else '  → 有问题，见上')
    print(f'共 {len(prs.slides.__iter__.__self__._sldIdLst)} 页')
