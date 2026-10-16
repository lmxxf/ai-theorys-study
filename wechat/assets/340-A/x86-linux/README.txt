Antigravity CLI x86_64 WSL backup — 2026-10-16

agy.gz: patched executable (217768144 bytes uncompressed).
restore-agy.py: verifies archive hash, preserves the currently installed binary,
then atomically restores ~/.local/bin/agy. No downloads or build tools needed.

Restore patched version after an accidental upgrade:
python3 /home/lmxxf/work/ai-theorys-study/wechat/assets/340-A/x86-linux/restore-agy.py

Restart agy after restoring. Existing running processes retain the old executable.
Start a new conversation to load the complete AGENTS.md; previously lost history
is not recovered by this patch.

Changes:
- Argon Low only (model 1419): override final checkpoint max_token_limit=1000000.
- Disable automatic checkpoint predicate, returning false,nil via AX/BX/CX.
- Three rule/skill file read caps: 24000 -> 1048576 bytes.
- Both final rules/customizations token-budget arguments: 131072 tokens.
  This bypasses configured and default budget paths at their shared consumer.
- Other models' context limits are unchanged. Manual compact was not tested.

Verification:
A real new Argon Low request returned OK. SQLite request metadata records
max_context_tokens=983040 and estimated_tokens_used=56707.
The user rules payload contains all 119127 bytes of the local AGENTS.md;
no rules-token-budget exclusion warning. Details: verification.txt.
A >256K pressure test was NOT repeated on this x86 machine.

agy-x86-patches.json contains original/patched hashes and exact byte changes.
fix-argon-context-x86.py is restricted to the exact original hash, is idempotent,
and requires trampoline.bin beside it. Do not apply its offsets to new builds.
argon-x86-trampoline.s and functions.json document the architecture-specific work.
verify-argon-x86.py reads the local test conversation DB (DB not included here).
SHA256SUMS checks the archive and companion files.

Patched SHA256 (uncompressed):
4701d840f20eb78319153f177f8814932052259d418de32b54f5bcb9ad71b34a
Original SHA256 (uncompressed):
e14253aa79d4e910cbf8bcb40b68e80f9e85476725e130ed085122c98face92b
