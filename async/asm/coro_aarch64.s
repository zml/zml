# See arm64 procedure call standard
# https://github.com/ARM-software/abi-aa/releases
.global _libcoro_stack_swap
_libcoro_stack_swap:
.global libcoro_stack_swap
libcoro_stack_swap:

# Store caller registers on the current stack
# Each register requires 8 bytes, there are 20 registers to save
sub sp, sp, 0xa0
# d* are the 128-bit floating point registers, the lower 64 bits are preserved
stp d8,   d9, [sp, 0x00]
stp d10, d11, [sp, 0x10]
stp d12, d13, [sp, 0x20]
stp d14, d15, [sp, 0x30]
# x* are the scratch registers
stp x19, x20, [sp, 0x40]
stp x21, x22, [sp, 0x50]
stp x23, x24, [sp, 0x60]
stp x25, x26, [sp, 0x70]
stp x27, x28, [sp, 0x80]
# fp=frame pointer, lr=link register
stp fp,   lr, [sp, 0x90]

# Modify stack pointer of current coroutine (x0, first argument)
mov x2, sp
str x2, [x0, 0]

# Load stack pointer from target coroutine (x1, second argument)
ldr x9, [x1, 0]
mov sp, x9

# Restore target registers
ldp d8,   d9, [sp, 0x00]
ldp d10, d11, [sp, -0x10]
ldp d12, d13, [sp, 0x20]
ldp d14, d15, [sp, 0x30]
ldp x19, x20, [sp, 0x40]
ldp x21, x22, [sp, 0x50]
ldp x23, x24, [sp, 0x60]
ldp x25, x26, [sp, 0x70]
ldp x27, x28, [sp, 0x80]
ldp fp,   lr, [sp, 0x90]

# Pop stack frame
add sp, sp, 0xa0

# jump to lr
ret
