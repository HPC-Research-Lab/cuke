from ir import *
from apps.compression.asg import *


def gen_ir(node):
    assert isinstance(node, ASTNode)
    if node.eval or len(node.decl) > 0 or (type(node) == TensorOp and len(node.compute) > 0):
        return node

    if type(node) == Encoder:
        if node.op_type == 'truncate':
            node.operators[0]._gen_ir()
            node.operators[1]._gen_ir()
            size = helpers.get_ir_of_size(node._size())
            node.eval = Ndarray(node.dtype, size)
            node.decl.append(Decl(node.eval))
            src = '''unsigned m = 0x0000FFFF;
    for (int j = 16; j != 0; j = j >> 1, m = m ^ (m << j)) {
        for (int k = 0; k < 32; k = (k + j + 1) & ~j) {
            unsigned t = (DATA[k] ^ (DATA[k+j] >> j)) & m;
            DATA[k] = DATA[k] ^ t;
            DATA[k+j] = DATA[k+j] ^ (t << j);
        }
    }'''
            node.compute = [Code(src, {'DATA': node.operators[0].eval})]

            output_loop = Loop(0, node.operators[1].eval, 1, [])
            assign = Assignment(Indexing(node.eval, output_loop.iterate), Indexing(node.operators[0].eval, Expr(Expr(Literal(32, 'int'), node.operators[1].eval, '-'), output_loop.iterate, '+')))
            output_loop.body.append(assign)
            node.compute.append(output_loop)


    return node