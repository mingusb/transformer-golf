def XNOR(x, y):
    return ~(x ^ y) & 1

def induction_match(Context_Token, Query_Token, Z):
    A = (Context_Token >> 0) & 1
    B = (Query_Token >> 0) & 1
    C = (Context_Token >> 1) & 1
    D = (Query_Token >> 1) & 1
    E = (Context_Token >> 2) & 1
    F = (Query_Token >> 2) & 1
    G = (Context_Token >> 3) & 1
    H = (Query_Token >> 3) & 1
    return Z & XNOR(A, B) & XNOR(C, D) & XNOR(E, F) & XNOR(G, H)

def predict_next_token_sympy(context, query):
    y = 0
    for j in range(5):
        Z = (context[j+1] >> 0) & 1
        y |= induction_match(context[j], query, Z)
    return y
