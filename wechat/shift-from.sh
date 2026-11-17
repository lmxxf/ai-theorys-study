#!/usr/bin/env bash
# 期号顺延 / 回缩 / 单篇调号工具
# 用法:
#   1) 连续期号平移 (遇到第一个缺号停止):
#      ./shift-from.sh 376          → 376 起连续期号全部 +1 (腾出 376)
#      ./shift-from.sh 376 +20      → 376 起连续期号全部 +20 (写 20 或 +20 均可)
#      ./shift-from.sh 396 -20      → 396 起连续期号全部 -20 (向前收拢填坑)
#   2) 单篇定点挪号 (把单篇从 A 挪到空号 B):
#      ./shift-from.sh "407-->376"  → 支持 --> / -> / => (需加引号防 shell 重定向)
#      ./shift-from.sh 407:376      → 免引号简写
#      ./shift-from.sh 407 to 376   → 免引号语义写法
#
# 同步处理对象: <n>.md, assets/<n>/, cover<n>.svg, cover<n>.png, 以及 <n>.md 内的本号链接
# 用 git mv 保 git history; 非 git 仓库自动回退到 mv; 执行前预检目标空位防覆盖

set -euo pipefail

usage() {
    cat <<EOF
用法:
  连续平移:
    $0 <起始期号> [偏移量]      (偏移量默认 +1，支持 +20 / 20 / -20)
    例: $0 376 +20              → 376 及之后连续期号全部 +20
        $0 396 -20              → 396 及之后连续期号全部 -20

  单篇挪号:
    $0 "<源期号>--> <目标期号>"  (支持 --> / -> / =>，建议加引号)
    $0 <源期号>:<目标期号>       (免引号简写，如 407:376)
    $0 <源期号> to <目标期号>    (免引号写法，如 407 to 376)
EOF
    exit 1
}

[[ $# -ge 1 && $# -le 3 ]] || usage

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    MV="git mv"
else
    MV="mv"
fi

# 检查某个期号是否存在任何关联文件/目录
slot_occupied() {
    local k="$1"
    [[ -e "${k}.md" || -e "assets/${k}" || -e "cover${k}.svg" || -e "cover${k}.png" ]]
}

# 单篇搬移 + 正文内本号链接同步
move_issue() {
    local old="$1"
    local new="$2"

    [[ -f "${old}.md" ]] && $MV "${old}.md" "${new}.md" && echo "  ${old}.md → ${new}.md"
    [[ -d "assets/${old}" ]] && $MV "assets/${old}" "assets/${new}" && echo "  assets/${old}/ → assets/${new}/"
    [[ -f "cover${old}.svg" ]] && $MV "cover${old}.svg" "cover${new}.svg" && echo "  cover${old}.svg → cover${new}.svg"
    [[ -f "cover${old}.png" ]] && $MV "cover${old}.png" "cover${new}.png" && echo "  cover${old}.png → cover${new}.png"

    if [[ -f "${new}.md" ]]; then
        python3 - "${new}.md" "${old}" "${new}" <<'PY'
import re, sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, "r", encoding="utf-8") as f:
    txt = f.read()
txt_new, c1 = re.subn(rf"(?<![0-9A-Za-z_-])assets/{re.escape(old)}(?=/|[^0-9A-Za-z_-]|$)", f"assets/{new}", txt)
txt_new, c2 = re.subn(rf"(?<![0-9A-Za-z_-])cover{re.escape(old)}(?=\.(?:svg|png)\b)", f"cover{new}", txt_new)
total = c1 + c2
if total:
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt_new)
    print(f"    ↳ 已更新 {path} 内 {total} 处本号链接 ({old} → {new})")
PY
    fi
}

# ========== 解析模式 ==========
MODE=""
SRC=""
DST=""
START=""
DELTA=""

if [[ $# -eq 1 ]]; then
    if [[ "$1" =~ ^([0-9]+)[[:space:]]*(-+>|=>|:)[[:space:]]*([0-9]+)$ ]]; then
        MODE="single"
        SRC="${BASH_REMATCH[1]}"
        DST="${BASH_REMATCH[3]}"
    elif [[ "$1" =~ ^[0-9]+$ ]]; then
        MODE="range"
        START="$1"
        DELTA=1
    else
        usage
    fi
elif [[ $# -eq 2 ]]; then
    if [[ "$1" =~ ^[0-9]+$ && "$2" =~ ^([+-]?)([0-9]+)$ ]]; then
        MODE="range"
        START="$1"
        DELTA=$(( ${BASH_REMATCH[1]}10#${BASH_REMATCH[2]} ))
    else
        usage
    fi
elif [[ $# -eq 3 ]]; then
    if [[ "$1" =~ ^[0-9]+$ && ( "$2" == "to" || "$2" == "->" || "$2" == "-->" || "$2" == "=>" ) && "$3" =~ ^[0-9]+$ ]]; then
        MODE="single"
        SRC="$1"
        DST="$3"
    else
        usage
    fi
fi

# ========== 执行单篇挪号 ==========
if [[ "$MODE" == "single" ]]; then
    SRC=$((10#$SRC))
    DST=$((10#$DST))
    if [[ "$SRC" -eq "$DST" ]]; then
        echo "源期号与目标期号相同 (${SRC})，无需移动"
        exit 0
    fi
    if [[ "$DST" -lt 1 ]]; then
        echo "错误: 目标期号必须 >= 1，收到 ${DST}"
        exit 1
    fi
    if ! [[ -f "${SRC}.md" ]]; then
        echo "错误: ${SRC}.md 不存在"
        exit 1
    fi
    if slot_occupied "$DST"; then
        echo "错误: 目标期号 ${DST} 已被占用 (${DST}.md / assets/${DST} / cover${DST}.*)，拒绝覆盖"
        exit 1
    fi

    echo "单篇挪号: ${SRC} → ${DST} (用 $MV)"
    move_issue "$SRC" "$DST"
    echo "完成。已将 ${SRC} 挪至 ${DST}"
    exit 0
fi

# ========== 执行连续平移 ==========
START=$((10#$START))
if [[ "$DELTA" -eq 0 ]]; then
    echo "偏移量为 0，无需顺延"
    exit 0
fi

nums=()
n="$START"
while [[ -f "${n}.md" ]]; do
    nums+=("$n")
    n=$((n + 1))
done

if [[ ${#nums[@]} -eq 0 ]]; then
    echo "${START}.md 不存在，无需移动"
    exit 0
fi

first_new=$((nums[0] + DELTA))
last_new=$((nums[-1] + DELTA))
if [[ "$first_new" -lt 1 ]]; then
    echo "错误: 移动后期号将小于 1 (${nums[0]} + ${DELTA} = ${first_new})"
    exit 1
fi

# 预检冲突: 目标号若不在本次搬移源区间 [nums[0], nums[-1]] 内，必须为空槽位
min_src="${nums[0]}"
max_src="${nums[-1]}"
for old in "${nums[@]}"; do
    target=$((old + DELTA))
    if (( target < min_src || target > max_src )); then
        if slot_occupied "$target"; then
            echo "错误: 目标期号 ${target} 已被占用，无法将 ${old} 平移至 ${target} (已中止，未改动任何文件)"
            exit 1
        fi
    fi
done

sign_str="${DELTA}"
(( DELTA > 0 )) && sign_str="+${DELTA}"
echo "将平移 ${#nums[@]} 期 (每期 ${sign_str}): ${nums[0]}..${nums[-1]} → ${first_new}..${last_new} (用 $MV)"

if (( DELTA > 0 )); then
    # 向后顺延: 从大到小倒序挪，避免覆盖自身
    for ((i=${#nums[@]}-1; i>=0; i--)); do
        old="${nums[i]}"
        move_issue "$old" "$((old + DELTA))"
    done
    if (( DELTA <= ${#nums[@]} )); then
        if (( DELTA == 1 )); then
            echo "完成。腾出 ${START} 号"
        else
            echo "完成。腾出 ${START}–$((START + DELTA - 1)) 共 ${DELTA} 个号"
        fi
    else
        echo "完成。腾出 ${START}–$((first_new - 1)) 共 ${DELTA} 个号"
    fi
else
    # 向前收拢: 从小到大正序挪，避免覆盖自身
    for ((i=0; i<${#nums[@]}; i++)); do
        old="${nums[i]}"
        move_issue "$old" "$((old + DELTA))"
    done
    echo "完成。已向前收拢 ${#nums[@]} 期 (${nums[0]}..${nums[-1]} → ${first_new}..${last_new})"
fi
