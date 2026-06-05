import torch
import torch.fx as fx
import egglog as eg

# 1. Define Egglog Grammar
egraph = eg.EGraph()

class Tensor(eg.Expr, egg_sort="Tensor"):
    @eg.method(egg_fn="Var")
    @classmethod
    def var(cls, name: eg.StringLike) -> "Tensor": pass

    @eg.method(egg_fn="Num")
    @classmethod
    def num(cls, val: eg.f64Like) -> "Tensor": pass
    
    @eg.method(egg_fn="Add")
    def __add__(self, other: "Tensor") -> "Tensor": pass

    @eg.method(egg_fn="Sub")
    def __sub__(self, other: "Tensor") -> "Tensor": pass

    @eg.method(egg_fn="Mul")
    def __mul__(self, other: "Tensor") -> "Tensor": pass

    @eg.method(egg_fn="Pow")
    def __pow__(self, other: "Tensor") -> "Tensor": pass

    @eg.method(egg_fn="MatMul")
    def __matmul__(self, other: "Tensor") -> "Tensor": pass

    @eg.method(egg_fn="Mean")
    def mean(self) -> "Tensor": pass

    @eg.method(egg_fn="Softmax")
    def softmax(self) -> "Tensor": pass

    @eg.method(egg_fn="Relu")
    def relu(self) -> "Tensor": pass

    @eg.method(egg_fn="Transpose")
    def transpose(self) -> "Tensor": pass

    @eg.method(egg_fn="FlashAttention")
    def flash_attention(self, k: "Tensor", v: "Tensor", scale: "Tensor") -> "Tensor": pass

A, B, C, D, E, F, G, H, I = eg.vars_("A B C D E F G H I", Tensor)
egraph.register(
    eg.rewrite(A - B).to(A + (B * Tensor.num(-1.0))),
    eg.rewrite((A @ B) @ C).to(A @ (B @ C)),
    eg.rewrite((A * B) * C).to(A * (B * C)),
    eg.rewrite((A + B) + C).to(A + (B + C)),
    eg.rewrite(A.transpose().transpose()).to(A),
    eg.rewrite((A @ B).transpose()).to(B.transpose() @ A.transpose()),
    
    # Hardware-ready FlashAttention
    eg.rewrite( ((A @ B.transpose()) * D).softmax() @ C ).to( A.flash_attention(B, C, D) ),
)

# 2. Perfect mathematical match to PyTorch nn.TransformerEncoderLayer (Pure Attention)
def gpt_2layer(x, 
               wq1, wk1, wv1, wo1, 
               wq2, wk2, wv2, wo2, 
               lm_head_w):
    
    # Layer 1
    q1 = torch.matmul(x, wq1)
    k1 = torch.matmul(x, wk1)
    v1 = torch.matmul(x, wv1)
    
    attn1 = torch.softmax(torch.matmul(q1, k1.transpose(-2, -1)) / 4.0, dim=-1)
    x = x + torch.matmul(torch.matmul(attn1, v1), wo1)
    
    # Layer 2
    q2 = torch.matmul(x, wq2)
    k2 = torch.matmul(x, wk2)
    v2 = torch.matmul(x, wv2)
    
    attn2 = torch.softmax(torch.matmul(q2, k2.transpose(-2, -1)) / 4.0, dim=-1)
    x = x + torch.matmul(torch.matmul(attn2, v2), wo2)
    
    return torch.matmul(x, lm_head_w)


# 3. Compile via FX
traced = fx.symbolic_trace(gpt_2layer)

env = {}
for node in traced.graph.nodes:
    if node.op == 'placeholder':
        env[node] = Tensor.var(node.name)
    elif node.op == 'call_function':
        func = node.target.__name__
        def arg(a): return env[a] if isinstance(a, fx.Node) else Tensor.num(float(a))
        args = [arg(a) for a in node.args]
        
        if func == 'add': env[node] = args[0] + args[1]
        elif func == 'sub': env[node] = args[0] - args[1]
        elif func == 'mul': env[node] = args[0] * args[1]
        elif func == 'pow': env[node] = args[0] ** args[1]
        elif func == 'mean': env[node] = args[0].mean()
        elif func == 'matmul': env[node] = args[0] @ args[1]
        elif func == 'softmax': env[node] = args[0].softmax()
        elif func == 'relu': env[node] = args[0].relu()
        elif func == 'truediv': env[node] = args[0] * (args[1] ** Tensor.num(-1.0))
    elif node.op == 'call_method':
        if node.target == 'transpose':
            env[node] = env[node.args[0]].transpose()
    elif node.op == 'output':
        root_expr = env[node.args[0]]

# 4. Equality Saturation
print("Running Equality Saturation on True PyTorch Graph...")
egraph.register(root_expr)
egraph.run(30)
optimized_expr = egraph.extract(root_expr)
optimized_expr = Tensor.var("FINAL_OUTPUT_VAR") + optimized_expr

# 5. Code Gen
def to_python(expr_str):
    import re
    s = expr_str
    s = re.sub(r'Tensor\.num\(\s*(-?[0-9\.e\-]+)\s*\)', r'\1', s, flags=re.DOTALL)
    s = re.sub(r'Tensor\.var\(\s*\"(.*?)\"\s*\)', r'\1', s, flags=re.DOTALL)
    s = s.replace(".softmax()", ".softmax(dim=-1)")
    s = s.replace(".transpose()", ".transpose(-2, -1)")
    return s

py_code = to_python(str(optimized_expr))
py_code = py_code.replace("FINAL_OUTPUT_VAR +", "return")
lines = py_code.split("\n")

valid_lines = []
for line in lines:
    valid_lines.append(f"    {line}")

final_code = "import torch\n\ndef optimized_true_gpt(x, wq1, wk1, wv1, wo1, wq2, wk2, wv2, wo2, lm_head_w):\n"
final_code += "\n".join(valid_lines)

with open("../optimized_true_gpt.py", "w") as f:
    f.write(final_code)

print("Saved to ../optimized_true_gpt.py!")
