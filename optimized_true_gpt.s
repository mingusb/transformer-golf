	.file	"optimized_true_gpt.c"
	.text
	.globl	predict_next_token_optimized    # -- Begin function predict_next_token_optimized
	.type	predict_next_token_optimized,@function
predict_next_token_optimized:           # @predict_next_token_optimized
	.cfi_startproc
# %bb.0:
	cmpq	$2, %rsi
	pushq	$1
	.cfi_adjust_cfa_offset 8
	popq	%rcx
	.cfi_adjust_cfa_offset -8
	cmovlq	%rcx, %rsi
	xorl	%eax, %eax
.LBB0_1:                                # =>This Inner Loop Header: Depth=1
	cmpq	%rcx, %rsi
	je	.LBB0_2
# %bb.3:                                #   in Loop: Header=BB0_1 Depth=1
	movl	-8(%rdi,%rcx,8), %r8d
	xorl	%edx, %r8d
	movl	%r8d, %r9d
	movl	%r8d, %r10d
	shrl	$3, %r10d
	orl	%r8d, %r10d
	shrl	%r8d
	shrl	$2, %r9d
	orl	%r8d, %r9d
	orl	%r9d, %r10d
	orq	$-2, %r10
	incq	%r10
	andq	(%rdi,%rcx,8), %r10
	orq	%r10, %rax
	incq	%rcx
	jmp	.LBB0_1
.LBB0_2:
	retq
.Lfunc_end0:
	.size	predict_next_token_optimized, .Lfunc_end0-predict_next_token_optimized
	.cfi_endproc
                                        # -- End function
	.ident	"Ubuntu clang version 21.1.8 (6ubuntu1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
