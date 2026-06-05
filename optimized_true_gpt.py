import torch

def optimized_true_gpt(x, wq1, wk1, wv1, wo1, wq2, wk2, wv2, wo2, lm_head_w):
    _Tensor_1 = x + (x @ wq1).flash_attention(
        x @ wk1, x @ wv1, 4.0 ** -1.0
    ) @ wo1
    return (
        _Tensor_1
        + (_Tensor_1 @ wq2).flash_attention(_Tensor_1 @ wk2, _Tensor_1 @ wv2, 4.0 ** -1.0) @ wo2
    ) @ lm_head_w