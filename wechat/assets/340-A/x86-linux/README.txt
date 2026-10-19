Antigravity CLI（agy）Linux x86_64 1.4.2 修正版
更新日期：2026-10-19

【怎么恢复】

退出 agy，在终端运行：

python3 /home/lmxxf/work/ai-theorys-study/wechat/assets/340-A/x86-linux/restore-agy.py

仓库换了位置就改路径。脚本解压同目录 agy.gz、检查 SHA256，
将当前安装版本备份到 ~/.local/share/agy-patches/，再原子替换 ~/.local/bin/agy。
已是这一份修改版则不重复操作。只需 Python 3 标准库。
这是 Linux x86_64 完整程序，ARM64 不能用。

然后重启 agy，新建会话，让完整 AGENTS.md 重新加载。
旧会话已经丢失的历史不会自动恢复。
自动升级覆盖补丁后，再运行恢复脚本可恢复本目录归档的 1.4.2，
不是为未来新版本自动打补丁。

【改了什么】

- Argon High/Medium/Low（1417/1418/1419）在最终配置合并出口
  设置 max_token_limit=1000000，实际请求上限为983040 tokens。
- 自动 checkpoint 判断返回 false,nil（同时清零 AX/BX/CX）。
- 三处规则/技能单文件读取上限：24000 → 1048576 字节。
- rules/customizations 两项最终 token 预算分别改为131072。

其他模型的窗口不修改。规则读取与预算修改对整个客户端生效。
自动 checkpoint 关闭不等于所有裁剪关闭；手动 /compact 未验证。

【验证】

Low、Medium、High 三档分别新开会话发送小请求，均正常回复 OK。
每档 executor_metadata 与 gen_metadata 实际上限均为983040。
user_rules 逐字包含本地 AGENTS.md 的118430字节全文，没有预算排除提示。
完整记录见 verification.txt。未重跑超过256K的长会话压力测试。
启动、补丁重复运行、归档解压与安装版逐字节一致、逆向补丁还原原版哈希均已检查。

【文件】

agy.gz                       修改版完整可执行文件，解压后221286656字节
restore-agy.py               安装/恢复归档
SHA256SUMS                   本目录文件校验清单
agy-x86-patches.json         原版/修改版哈希、每处修改前后字节与偏移
fix-argon-context-x86.py     对指定原版构建打补丁；支持 --target
argon-x86-trampoline.s      跳板汇编源码
trampoline.bin              跳板机器码
functions.json              新版 pclntab 定位结果，含实际虚拟地址
verify-argon-x86.py          只读核查会话数据库
verification.txt            三档真实请求的验证记录

核查自己新建的会话：
python3 verify-argon-x86.py <conversation_id> --agents /项目路径/AGENTS.md

原版只备份在本机 ~/.local/share/agy-patches/，不放入仓库。
原版 SHA256：085f43e0d0628a8a8dabe1e547067818fa2c0adc86cfac98a3c4492918f455ef
修改版 SHA256：2a58516fdd1275208425a78bbfb4b2b43ec0e71d7d4f2664d64c09382b0f5b7c

【新版本维护】

未知哈希必须重新定位，不能删除哈希检查或沿用固定偏移。
本次 pclntab 位于0x1512750，Go text基址0x61d4080；通过三个getter
核对 planner指针+8、plan_model+64、max_token_limit+16。
最终配置出口0x78c1213，原指令为mov [rax+16],rcx（AT&T：mov %rcx,16(%rax)）
及xor ebx,ebx，替换为跳往0x7ee74e0；跳板重放两条指令后返回0x78c1219。
checkpointNeeded入口0x7ee74c0短路后，借用其函数体容纳跳板。
规则/技能读取点：0x7f3fc49、0x7f40bc5、0x7f4534e。
最终预算参数读取点：0x7f46cd4、0x7f46cde，覆盖配置值和默认值两条路径。

下次适配参照 ../arm-linux/readme.md 的定位思路，指令/ABI必须按x86重新核对。
恢复或修改后必须验证真实请求，不以UI显示1M或模型自述为依据。
