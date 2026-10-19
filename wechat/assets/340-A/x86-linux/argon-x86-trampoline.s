.text
.global _start
_start:
 mov 8(%rax), %rdx
 test %rdx, %rdx
 jz done
 mov 64(%rdx), %edx
 sub $1417, %edx
 cmp $2, %edx
 ja done
 test %rcx, %rcx
 jz done
 movl $1000000, 16(%rcx)
done:
 mov %rcx, 16(%rax)
 xor %ebx, %ebx
 jmp 0x78c1219
