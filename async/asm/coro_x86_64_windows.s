# See Microsoft x86-64 calling convention
# https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention
.global libcoro_stack_swap
libcoro_stack_swap:
# Store Windows stack information
pushq %gs:0x10
pushq %gs:0x08

# Store caller registers
pushq %rbp
pushq %rbx
pushq %rdi
pushq %rsi
pushq %r12
pushq %r13
pushq %r14
pushq %r15

# Store caller simd/float registers
subq $0xa0, %rsp
movups %xmm6,  0x00(%rsp)
movups %xmm7,  0x10(%rsp)
movups %xmm8,  0x20(%rsp)
movups %xmm9,  0x30(%rsp)
movups %xmm10, 0x40(%rsp)
movups %xmm11, 0x50(%rsp)
movups %xmm12, 0x60(%rsp)
movups %xmm13, 0x70(%rsp)
movups %xmm14, 0x80(%rsp)
movups %xmm15, 0x90(%rsp)

# Modify stack pointer of current coroutine (rcx, first argument)
movq %rsp, (%rcx)

# Load stack pointer from target coroutine (rdx, second argument)
movq (%rdx), %rsp

# Restore target simd/float registers
movups 0x00(%rsp), %xmm6
movups 0x10(%rsp), %xmm7
movups 0x20(%rsp), %xmm8
movups 0x30(%rsp), %xmm9
movups 0x40(%rsp), %xmm10
movups 0x50(%rsp), %xmm11
movups 0x60(%rsp), %xmm12
movups 0x70(%rsp), %xmm13
movups 0x80(%rsp), %xmm14
movups 0x90(%rsp), %xmm15
addq $0xa0, %rsp

# Restore target registers
popq %r15
popq %r14
popq %r13
popq %r12
popq %rsi
popq %rdi
popq %rbx
popq %rbp

# Restore Windows stack information
popq %gs:0x08
popq %gs:0x10

retq
