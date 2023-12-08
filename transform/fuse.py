from asg import *
from ir import *
import codegen
from helpers import get_obj, get_val, ASGTraversal, rebind_iterate, ir_defs, ir_uses, remove_decl, clear_compute, \
    ir_find_defs, same_object


class fuser:
    def __init__(self):
        self.rules = []

    def register(self, rule):
        self.rules.append(rule)

    def fuse(self, node):
        def action(node, res):
            for r in self.rules:
                r(node, res)

        t = ASGTraversal(action)
        t(node)
        return node


# TODO: reimplement this with IRTraversal
def _replace_arrindex_with_scalar(ir, old, new):
    if type(ir) == list or type(ir) == tuple:
        for l in ir:
            _replace_arrindex_with_scalar(l, old, new)
    elif type(ir) == Loop:
        _replace_arrindex_with_scalar(ir.body, old, new)
    elif type(ir) == FilterLoop:
        if type(ir.cond) in (Indexing, Scalar):
            obj = get_obj(ir.cond)
            if obj.dobject_id == old.dobject_id:
                ir.cond = new
        else:
            _replace_arrindex_with_scalar(ir.cond, old, new)
        _replace_arrindex_with_scalar(ir.cond_body, old, new)
        _replace_arrindex_with_scalar(ir.body, old, new)
    elif type(ir) == Expr:
        if type(ir.left) in (Indexing, Scalar):
            obj = get_obj(ir.left)
            if obj.dobject_id == old.dobject_id:
                ir.left = new
        else:
            _replace_arrindex_with_scalar(ir.left, old, new)
        if type(ir.right) in (Indexing, Scalar):
            obj = get_obj(ir.right)
            if obj.dobject_id == old.dobject_id:
                ir.right = new
        else:
            _replace_arrindex_with_scalar(ir.right, old, new)
    elif type(ir) == Assignment:
        if type(ir.lhs) in (Indexing, Scalar):
            obj = get_obj(ir.lhs)
            if obj.dobject_id == old.dobject_id:
                ir.lhs = new
        else:
            _replace_arrindex_with_scalar(ir.lhs, old, new)
        if type(ir.rhs) in (Indexing, Scalar):
            obj = get_obj(ir.rhs)
            if obj.dobject_id == old.dobject_id:
                ir.rhs = new
        else:
            _replace_arrindex_with_scalar(ir.rhs, old, new)
    elif type(ir) == Slice:
        if type(ir.start) in (Indexing, Scalar):
            obj = get_obj(ir.start)
            if obj.dobject_id == old.dobject_id:
                ir.start = new
        else:
            _replace_arrindex_with_scalar(ir.start, old, new)

        if type(ir.stop) in (Indexing, Scalar):
            obj = get_obj(ir.stop)
            if obj.dobject_id == old.dobject_id:
                ir.stop = new
        else:
            _replace_arrindex_with_scalar(ir.stop, old, new)

        if type(ir.step) in (Indexing, Scalar):
            obj = get_obj(ir.step)
            if obj.dobject_id == old.dobject_id:
                ir.step = new
        else:
            _replace_arrindex_with_scalar(ir.step, old, new)

    elif type(ir) == Math:
        if type(ir.val) in (Indexing, Scalar):
            obj = get_obj(ir.val)
            if obj.dobject_id == old.dobject_id:
                ir.val = new
        else:
            _replace_arrindex_with_scalar(ir.val, old, new)
    elif type(ir) == Code:
        if type(ir.output[1]) in (Indexing, Scalar):
            obj = get_obj(ir.output[1])
            if obj.dobject_id == old.dobject_id:
                ir.output = (ir.output[0], new)
        # TODO: replace inputs


def match_orders(order1, order2):
    if len(order1) == len(order2):
        for i in range(len(order1)):
            x1 = get_val(order1[i][1].start)
            y1 = get_val(order1[i][1].end)
            z1 = get_val(order1[i][1].step)
            x2 = get_val(order2[i][1].start)
            y2 = get_val(order2[i][1].end)
            z2 = get_val(order2[i][1].step)
            if x1 == None or not (x1 == x2 or same_object(x1, x2)):
                return False
            if y1 == None or not (y1 == y2 or same_object(y1, y2)):
                return False
            if z1 == None and not (z1 == z2 or same_object(z1, z2)):
                return False
        return True
    else:
        return False


def merge_loops(order1, order2, data, this_node, input_node):
    if match_orders(order1, order2):
        for i in range(len(order1)):
            nl = order1[i][1]
            ol = order2[i][1]
            rebind_iterate(order2[-1][1], ol.iterate, nl.iterate)
            if i < len(order1) - 1:
                nl.body[0:0] = [s for s in ol.body if s != order2[i + 1][1]]
            if 'loop_ofs' in ol.attr:
                if 'loop_ofs' in nl.attr:
                    nl.attr['loop_ofs'] = max(nl.attr['loop_ofs'], ol.attr['loop_ofs'])
                else:
                    nl.attr['loop_ofs'] = ol.attr['loop_ofs']

        dfs = ir_find_defs(order2[-1][1].body, data)
        if len(dfs) > 0:
            if ir_uses(dfs[-1], data):
                df = Scalar(data.dtype)
                this_node.decl.append(Decl(df))
            else:
                df = dfs[-1].rhs
                order2[-1][1].body = [s for s in order2[-1][1].body if not ir_defs(s, data)]

            if type(order1[-1][1]) == FilterLoop and data.dobject_id == get_obj(order1[-1][1].cond).dobject_id:
                order1[-1][1].cond_body.extend(order2[-1][1].body)
                _replace_arrindex_with_scalar(order1[-1][1], data, df)
                clear_compute(input_node)
                remove_decl(input_node, input_node.eval)
            else:
                j = len(order1[-1][1].body)
                for i in range(len(order1[-1][1].body)):
                    if ir_uses(order1[-1][1].body[i], data):
                        j = i
                        break
                order1[-1][1].body[j:j] = order2[-1][1].body
                _replace_arrindex_with_scalar(order1[-1][1].body, data, df)
                clear_compute(input_node)
                remove_decl(input_node, input_node.eval)
        else:
            if type(order1[-1][1]) == FilterLoop and data == get_obj(order1[-1][1].cond):
                order1[-1][1].cond_body.extend(order2[-1][1].body)
                clear_compute(input_node)
            else:
                j = len(order1[-1][1].body)
                for i in range(len(order1[-1][1].body)):
                    if ir_uses(order1[-1][1].body[i], data):
                        j = i
                        break
                order1[-1][1].body[j:j] = order2[-1][1].body
                clear_compute(input_node)


def fuse_operators(op1, order1, op2):
    if len(order1) > 0:
        merge_loops(order1, op2.output_order, op2.eval, op1, op2)


def basic_rule(node, res):
    if type(node) == TensorOp and node.op_type in elementwise_op:
        if type(node.operators[0]) == TensorOp and node.operators[0].op_type in (
                elementwise_op + ['apply', 'einsum', 'setval']) and len(
                node.operators[0].ref_by) == 1:
            fuse_operators(node, node.input_orders[0], node.operators[0])

        if node.op_type in binary_elw:
            if type(node.operators[1]) == TensorOp and node.operators[1].op_type in (
                    elementwise_op + ['apply', 'einsum', 'setval']) and len(
                    node.operators[1].ref_by) == 1:
                fuse_operators(node, node.input_orders[1], node.operators[1])

    elif type(node) == TensorOp and node.op_type == 'apply':
        cond = node.operators[2 + 2 * node.nparams]
        if cond != None:
            this_loop = node.output_order[0]
            fuse_operators(node, [this_loop], cond)

    elif type(node) == TensorOp and node.op_type == 'index':
        if type(node.operators[1]) == TensorOp and node.operators[1].op_type in (
                elementwise_op + ['setval']) and len(
                node.operators[1].ref_by) == 1:
            assert len(node.operators[1]._size()) == 0
            # TODO
            # dfs = ir_find_defs(node.operators[1].compute, node.operators[1].eval)
            # if len(dfs) > 0:
            #     df = dfs[-1].rhs
            #     rebind_iterate(node.compute, node.opertors[1].eval, df)
            #     remove_decl(node.operators[1], node.operators[1].eval)
            # fuse_operators(node, node.input_orders[1], node.operators[1])


def test1():
    A = Tensor('a', (10, 20))
    B = Tensor('b', (10, 20))
    C = Tensor('c', (10, 20))
    D = Tensor('d', (10, 20))

    A = setval(A, 1)
    t1 = A + B
    t2 = (C - D).abs()
    res1 = t1 + t2
    ir1 = res1._gen_ir()
    f = fuser()
    f.register(basic_rule)
    code = codegen.cpu.print_cpp(f.fuse(ir1))
    print(code)


def test2():
    A = Tensor('a', (10, 20))
    B = Tensor('b', (20, 30))
    C = Tensor('c', (10, 30))
    D = Tensor('d', (10, 30))
    t1 = (A @ B).abs()
    t2 = (C - D).abs()
    res1 = t1 + t2
    ir1 = res1._gen_ir()
    f = fuser()
    f.register(basic_rule)
    code = codegen.cpu.print_cpp(f.fuse(ir1))
    print(code)


def test3():
    A = Tensor('a', (10, 20))
    B = Tensor('b', (20, 30))
    C = Tensor('c', (10, 20))
    D = Tensor('d', (20, 30))
    t1 = (A @ B).abs()
    t2 = (C @ D).round()
    res1 = t1 + t2
    ir1 = res1._gen_ir()
    f = fuser()
    f.register(basic_rule)
    code = codegen.cpu.print_cpp(f.fuse(ir1))
    print(code)


if __name__ == "__main__":
    test1()
    test2()
    test3()
