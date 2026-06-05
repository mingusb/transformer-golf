def bitwise_match_logic(a_bit, b_bit):
    """
    Define the fundamental boolean operation required to match a context bit with a query bit.
    For standard Induction Heads (Exact Equality Match), this is the XNOR logic.
    Change this to whatever boolean function you want the network to learn!
    """
    return (a_bit and b_bit) or (not a_bit and not b_bit)

def sequence_target_offset():
    """
    If the context token successfully matches the query, return the offset of the target token.
    1 means "return the token immediately after the match".
    """
    return 1
