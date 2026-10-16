.text
.global _start
_start:
 mov 8(%rax), %rdx
 test %rdx, %rdx
 jz done
 cmpl $1419, 64(%rdx)
 jne done
 test %rcx, %rcx
 jz done
 movl $1000000, 16(%rcx)
done:
 mov %rcx, 16(%rax)
 xor %ebx, %ebx
 jmp 0x765b965
