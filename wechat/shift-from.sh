#!/usr/bin/env bash
# 把 <start> 及以后的连续期号全部 +<count>，腾出 <count> 个空号
# 用法: ./shift-from.sh 236        → 腾出 236 一个空号
#       ./shift-from.sh 236 3      → 腾出 236/237/238 三个空号
# 顺延对象: <n>.md, assets/<n>/, cover<n>.svg, cover<n>.png
# 用 git mv 保 git history; 非 git 仓库回退到 mv

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "用法: $0 <起始期号> [腾出几个]"
    echo "例如: $0 236    → 把 236.md 及之后的连续期号全部 +1，腾出 236"
    echo "      $0 236 3  → 全部 +3，腾出 236/237/238"
    exit 1
fi

START="$1"
COUNT="${2:-1}"

if [[ ! "$COUNT" =~ ^[0-9]+$ ]] || [[ "$COUNT" -lt 1 ]]; then
    echo "错误: 腾出个数必须是不小于 1 的整数，收到 '$COUNT'"
    exit 1
fi
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# 用 git 还是 mv
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    MV="git mv"
else
    MV="mv"
fi

# 找出从 START 起的连续期号(遇到第一个缺号就停)
nums=()
n="$START"
while [[ -f "${n}.md" ]]; do
    nums+=("$n")
    n=$((n + 1))
done

if [[ ${#nums[@]} -eq 0 ]]; then
    echo "${START}.md 不存在,无需顺延"
    exit 0
fi

echo "将顺延 ${#nums[@]} 期(每期 +${COUNT}): ${nums[0]} → $((nums[-1] + COUNT))"
echo "用 $MV"

# 从大到小倒序挪,避免覆盖
for ((i=${#nums[@]}-1; i>=0; i--)); do
    old="${nums[i]}"
    new=$((old + COUNT))

    [[ -f "${old}.md" ]] && $MV "${old}.md" "${new}.md" && echo "  ${old}.md → ${new}.md"
    [[ -d "assets/${old}" ]] && $MV "assets/${old}" "assets/${new}" && echo "  assets/${old}/ → assets/${new}/"
    [[ -f "cover${old}.svg" ]] && $MV "cover${old}.svg" "cover${new}.svg" && echo "  cover${old}.svg → cover${new}.svg"
    [[ -f "cover${old}.png" ]] && $MV "cover${old}.png" "cover${new}.png" && echo "  cover${old}.png → cover${new}.png"
done

if [[ "$COUNT" -eq 1 ]]; then
    echo "完成。腾出 ${START} 号"
else
    echo "完成。腾出 ${START}–$((START + COUNT - 1)) 共 ${COUNT} 个号"
fi
