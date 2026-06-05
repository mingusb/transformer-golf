import ast
import re

def to_subscript(name):
    mapping = {
        'x': r'\mathbf{X}',
        'wq1': r'\mathbf{W}_{q1}', 'wk1': r'\mathbf{W}_{k1}', 'wv1': r'\mathbf{W}_{v1}', 'wo1': r'\mathbf{W}_{o1}',
        'wq2': r'\mathbf{W}_{q2}', 'wk2': r'\mathbf{W}_{k2}', 'wv2': r'\mathbf{W}_{v2}', 'wo2': r'\mathbf{W}_{o2}',
        'lm_head_w': r'\mathbf{W}_E^T'
    }
    if name in mapping: return mapping[name]
    if name.startswith("_Tensor_"): return r'\mathbf{T}_{' + name.split('_')[-1] + '}'
    return name

class LatexVisitor(ast.NodeVisitor):
    def visit_Name(self, node): return to_subscript(node.id)
    def visit_Constant(self, node): return str(node.value)
    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.left, ast.BinOp) and not isinstance(node.op, (ast.Add, ast.Sub)):
            left = f"\\left({left}\\right)"
        if isinstance(node.right, ast.BinOp) and not isinstance(node.op, (ast.Add, ast.Sub)):
            right = f"\\left({right}\\right)"
        if isinstance(node.op, ast.Add): return f"{left} + {right}"
        if isinstance(node.op, ast.Sub): return f"{left} - {right}"
        if isinstance(node.op, ast.Mult): return f"{left} \\cdot {right}"
        if isinstance(node.op, ast.MatMult): return f"{left} \\mathbf{{\\times}} {right}"
        if isinstance(node.op, ast.Div): return f"\\frac{{{left}}}{{{right}}}"
        if isinstance(node.op, ast.Pow): 
            if not isinstance(node.left, (ast.Name, ast.Constant)): left = f"\\left({left}\\right)"
            return f"{left}^{{{right}}}"
        return f"{left} ? {right}"
    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub): return f"-{operand}"
        return f"?{operand}"
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            base = self.visit(node.func.value)
            attr = node.func.attr
            if attr == "mean": return f"\\mathbb{{E}}\\left[{base}\\right]"
            if attr == "softmax": return f"\\text{{softmax}}\\left({base}\\right)"
            if attr == "relu": return f"\\text{{ReLU}}\\left({base}\\right)"
            if attr == "transpose": 
                if not isinstance(node.func.value, (ast.Name, ast.Constant)): base = f"\\left({base}\\right)"
                return f"{base}^T"
            if attr == "flash_attention":
                x = base; w_qk = self.visit(node.args[0])
                w_ov = self.visit(node.args[1])
                return f"\\text{{FlashAttention}}\\left({x}, {w_qk}, {w_ov}\\right)"
            if attr == "transformer_block":
                x = base; w_qk = self.visit(node.args[0])
                w_ov = self.visit(node.args[1])
                return f"\\text{{TransformerBlock}}\\left({x}, {w_qk}, {w_ov}\\right)"
        return "CALL"
    def visit_Assign(self, node):
        target = self.visit(node.targets[0])
        val = self.visit(node.value)
        return f"{target} &= {val} \\\\"
    def visit_Return(self, node):
        val = self.visit(node.value)
        return f"\\mathbf{{Y}} &= {val}"

class LatexUnrolledVisitor(ast.NodeVisitor):
    def __init__(self):
        self.env = {}
    def visit_Name(self, node):
        name = node.id
        if name in self.env: return self.env[name]
        return to_subscript(name)
    def visit_Constant(self, node): return str(node.value)
    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add): return f"{left} + {right}"
        if isinstance(node.op, ast.Sub): return f"{left} - {right}"
        if isinstance(node.op, ast.Mult): return f"{left} \\cdot {right}"
        if isinstance(node.op, ast.MatMult): return f"{left} \\mathbf{{\\times}} {right}"
        if isinstance(node.op, ast.Div): return f"\\frac{{{left}}}{{{right}}}"
        if isinstance(node.op, ast.Pow): 
            if not isinstance(node.left, (ast.Name, ast.Constant)): left = f"({left})"
            return f"{left}^{{{right}}}"
        return f"{left} ? {right}"
    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub): return f"-{operand}"
        return f"?{operand}"
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            base = self.visit(node.func.value)
            attr = node.func.attr
            if attr == "mean": return f"\\mathbb{{E}}[{base}]"
            if attr == "softmax": return f"\\text{{softmax}}({base})"
            if attr == "relu": return f"\\text{{ReLU}}({base})"
            if attr == "transpose": 
                if not isinstance(node.func.value, (ast.Name, ast.Constant)): base = f"({base})"
                return f"{base}^T"
            if attr == "flash_attention":
                x = base; w_qk = self.visit(node.args[0])
                w_ov = self.visit(node.args[1])
                return f"\\text{{FlashAttention}}({x}, {w_qk}, {w_ov})"
            if attr == "transformer_block":
                x = base; w_qk = self.visit(node.args[0])
                w_ov = self.visit(node.args[1])
                return f"\\text{{TransformerBlock}}({x}, {w_qk}, {w_ov})"
        return "CALL"
    def visit_Assign(self, node):
        val = self.visit(node.value)
        target_name = node.targets[0].id
        self.env[target_name] = val
        return None
    def visit_Return(self, node):
        val = self.visit(node.value)
        return f"\\mathbf{{Y}} &= {val}"

def colorize_brackets(text):
    colors = ["red", "blue", "teal", "magenta", "orange", "cyan", "purple", "brown"]
    depth = 0; result = []
    for char in text:
        if char in "([":
            color = colors[depth % len(colors)]
            result.append(f"\\textcolor{{{color}}}{{{char}}}")
            depth += 1
        elif char in ")]":
            depth -= 1
            color = colors[depth % len(colors)]
            result.append(f"\\textcolor{{{color}}}{{{char}}}")
        else: result.append(char)
    return "".join(result)

with open("../optimized_true_gpt.py", "r") as f: source = f.read()
tree = ast.parse(source)

# 1. Rolled Equation
visitor1 = LatexVisitor()
rolled_lines = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        for stmt in node.body:
            rolled_lines.append(visitor1.visit(stmt))

# 2. Unrolled Equation
visitor2 = LatexUnrolledVisitor()
final_equation = ""
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        for stmt in node.body:
            res = visitor2.visit(stmt)
            if res is not None: final_equation = res

rolled_wrapped = []
for line in rolled_lines:
    l = line.replace(r"\left(", "(").replace(r"\right)", ")").replace(r"\left[", "[").replace(r"\right]", "]")
    l = l.replace(r"+ \text{FlashAttention}", r"\\&\quad + \text{FlashAttention}")
    l = l.replace(r".\text{TransformerBlock}", r"\\&\quad .\text{TransformerBlock}")
    l = l.replace(r", ((", r", \\&\quad\quad\quad ((")
    l = colorize_brackets(l)
    rolled_wrapped.append(l)

final_equation = final_equation.replace(r"\left(", "(").replace(r"\right)", ")").replace(r"\left[", "[").replace(r"\right]", "]")
final_equation = final_equation.replace(r"+ \text{FlashAttention}", r"\\&\quad + \text{FlashAttention}")
final_equation = final_equation.replace(r".\text{TransformerBlock}", r"\\&\quad .\text{TransformerBlock}")
final_equation = final_equation.replace(r", \mathbf{X}", r", \\&\quad\quad\quad \mathbf{X}")
final_equation = final_equation.replace(r", \textcolor", r", \\&\quad\quad\quad \textcolor")
final_equation = final_equation.replace(r"\mathbf{\times} \mathbf{W}_E^T", r"\\&\quad \mathbf{\times} \mathbf{W}_E^T")
final_equation = colorize_brackets(final_equation)

print(r'''\documentclass[10pt,fleqn]{article}
\usepackage{amsmath, amssymb, amsfonts, xcolor}
\setlength{\mathindent}{2em}
\setlength{\parindent}{0pt}
\usepackage[margin=0.2in, paperwidth=12in, paperheight=18in]{geometry}
\begin{document}
\thispagestyle{empty}
\textbf{\huge Fully Optimized Algorithmic Forward Pass}
\vspace{1em}

\textbf{\Large 0. Compiler Intrinsics (Definitions)}
\begin{align*}
&\text{FlashAttention}(\mathbf{X}, \mathbf{W}_{QK}^{scaled}, \mathbf{W}_{OV}) \\
&\quad = \text{softmax}\left( \mathbf{X}\mathbf{W}_{QK}^{scaled}\mathbf{X}^T \right) \mathbf{X}\mathbf{W}_{OV} \\
&\text{TransformerBlock}(\mathbf{X}, \mathbf{W}_{QK}^{scaled}, \mathbf{W}_{OV}) \\
&\quad = \mathbf{X} + \text{FlashAttention}\left(\mathbf{X}, \mathbf{W}_{QK}^{scaled}, \mathbf{W}_{OV}\right)
\end{align*}

\vspace{2em}

\textbf{\Large 1. The Rolled (Block-wise) Forward Pass}
\begin{align*}''')
for line in rolled_wrapped:
    print(line)
print(r'''\end{align*}

\vspace{2em}

\textbf{\Large 2. The Unrolled (Graph-wise) Forward Pass}
\begin{align*}''')
print(final_equation)
print(r'''\end{align*}
\end{document}''')
