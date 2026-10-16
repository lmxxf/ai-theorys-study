Antigravity CLI（agy）x86 Linux 修正版
更新日期：2026-10-16

【适用环境】

Linux x86_64，包括 Windows 里的 x86_64 WSL。
这是可直接使用的完整程序压缩包，不需要编译，也不需要先安装原版。
DGX Spark 的 ARM64 不能用这一份。

【怎么安装／误升级后怎么恢复】

1. 退出正在运行的 agy。

2. 在 WSL 终端运行：

python3 /home/lmxxf/work/ai-theorys-study/wechat/assets/340-A/x86-linux/restore-agy.py

如果仓库放在其他位置，改成你自己的路径即可。
也可以先进入本目录，再运行：

python3 restore-agy.py

脚本会自动解压同目录的 agy.gz，检查 SHA256，然后安装到：

~/.local/bin/agy

如果那里已有其他版本，会先保存到 ~/.local/share/agy-patches/，再替换。
如果已经是这一份修正版，会提示 Already installed，不重复操作。
需要 Python 3；解压、校验和安装只用标准库，不需要另外装依赖。

3. 进入你要工作的项目目录，启动：

agy --model gemini-4-argon-low

请新开会话，让项目的 AGENTS.md 重新完整加载。
已经运行的进程不会自动换成新版；旧会话此前丢失的历史也不会自动补回来。

以后如果 agy 自动升级，导致这些限制重新出现，再运行一次恢复脚本即可。
恢复的是本目录归档的版本，不是给升级后的新版本动态打补丁。

【改了什么】

- Gemini 4 Argon Low（模型编号 1419）：在最终配置合并后，将上下文
  max_token_limit 设为 1000000。实际请求扣除输出预留后为 983040 tokens。
- 关闭自动 checkpoint 的触发判断。
- 三处规则／技能文件读取上限：从 24000 字节提高到 1048576 字节。
- 规则与自定义内容的 token 预算：两项都提高到 131072 tokens，
  避免长 AGENTS.md 超预算后被整份排除。

上下文窗口修改只针对 Argon Low；其他模型的窗口没有改。
文件读取上限和规则预算修改适用于整个客户端。
手动 /compact 未验证；这份补丁不意味着所有上下文裁剪都被关闭。

【这台 WSL 上验证了什么】

新建 Argon Low 会话，真实请求正常返回 OK。
SQLite 会话数据库记录：

- max_context_tokens = 983040
- estimated_tokens_used = 56707
- user_rules 完整包含本地 AGENTS.md 的 119127 字节原文
- 没有“超过规则 token 预算”的排除提示

这台 x86 机器没有重复做超过 256K 的长会话压力测试。
具体记录见 verification.txt。不要只凭 /context 面板的 1M 判断是否生效。

【目录里的文件】

agy.gz                         修好的完整可执行程序，解压后 217768144 字节
restore-agy.py                 日常安装／恢复用这个脚本
SHA256SUMS                     本目录各文件的 SHA256 校验清单
agy-x86-patches.json           原版与修正版哈希、每处修改的偏移和字节
fix-argon-context-x86.py       从指定原版构建制作补丁的脚本
argon-x86-trampoline.s         x86_64 跳板汇编源码
trampoline.bin                跳板机器码，制作补丁时与脚本放在同一目录
functions.json                本次定位的 Go 函数与地址信息
verify-argon-x86.py            读取本机测试会话数据库的核查脚本
verification.txt              本次实请求核查结果

普通使用只需要 agy.gz 和 restore-agy.py。
verify-argon-x86.py 使用原测试机的会话路径，其他机器没有那份数据库，
不能直接照跑；会话数据库没有上传到这里。
fix-argon-context-x86.py 只接受指定原版的 SHA256，不能拿固定偏移套新版本。
本目录没有打包原版程序。

【程序哈希（解压后）】

修正版 SHA256：
4701d840f20eb78319153f177f8814932052259d418de32b54f5bcb9ad71b34a

制作补丁时使用的原版 SHA256：
e14253aa79d4e910cbf8bcb40b68e80f9e85476725e130ed085122c98face92b
