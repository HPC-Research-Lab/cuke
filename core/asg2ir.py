from core.asg import *
from core.ir import *
from helpers import get_ir_of_size, ASGTraversal, replace_all_ref, rebind_iterate, remove_decl


def num_unbind(index):
    if type(index) == Indexing:
        return num_unbind(index.dobject) + num_unbind(index.idx)
    elif type(index) == Literal and index.val == -1:
        return 1
    else:
        return 0


# def get_first_unbind(index: (Indexing, Ndarray, Slice)):
#     if type(index) == Indexing:
#         x = get_first_unbind(index.dobject)
#         if x != None:
#             return x
#         else:
#             if type(index.idx) == Literal and index.idx.val == -1:
#                 return index
#             else:
#                 y = get_first_unbind(index.idx)
#                 return y
#     return None


def bind(object: Indexing | Ndarray | Slice, subscripts: list | tuple, attrs = None):
    new_index = copy.deepcopy(object)
    if attrs == None:
        attrs = [{} for _ in range(len(subscripts))]
    j = 0
    if type(new_index) == Indexing:
        indices = [new_index]
        while type(indices[-1].dobject) == Indexing:
            indices.append(indices[-1].dobject)
        indices.reverse()
        i = 0
        while i < len(indices) and j < len(subscripts):
            index = indices[i]
            i += 1
            while type(index.idx) == Indexing:
                index = index.idx
            assert type(index.idx) in (Scalar, Literal)
            if type(index.idx) == Scalar or (type(index.idx) == Literal and index.idx.val != -1):
                continue
            idx = subscripts[j]
            if type(idx) in (Scalar, Literal, Indexing):
                index.idx = idx
            elif type(idx) in (Ndarray, Slice):
                index.idx = Indexing(idx, Literal(-1, 'int'))
            else:
                raise TypeError('idx type error when binding')
            index.attr.update(attrs[j])
            j += 1

    while j < len(subscripts):
        idx = subscripts[j]
        if type(idx) in (Scalar, Literal, Indexing):
            new_index = Indexing(new_index, idx)
        elif type(idx) in (Ndarray, Slice):
            new_index = Indexing(new_index, Indexing(idx, Literal(-1, 'int')))
        else:
            raise TypeError('incorrect idx type!')
        new_index.attr.update(attrs[j])
        j += 1

    return new_index

    # x = get_first_unbind(index)
    # if x == None:
    #     res = Indexing(index, idx)
    #     res.attr.update(attr)
    #     return res
    # else:
    #     old = copy.copy(x.idx)
    #     old_attr = copy.copy(x.attr)
    #     x.idx = idx
    #     x.attr.update(attr)
    #     new_index = copy.deepcopy(index)
    #     x.idx = old
    #     x.attr = old_attr
    #     return new_index


def get_slice(index: (Indexing, Ndarray, Slice)):
    if type(index) == Indexing:
        x = get_slice(index.dobject)
        if x != None:
            return x
        else:
            y = get_slice(index.idx)
            if y != None:
                return y
            else:
                if type(index.dobject) == Slice and type(index.idx) == Literal and index.idx.val == -1:
                    return index.dobject
    return None


def replace_output(ir, old, new):
    if type(ir) == list or type(ir) == tuple:
        for l in ir:
            replace_output(l, old, new)
    elif type(ir) == Loop:
        replace_output(ir.body, old, new)
    elif type(ir) == Assignment:
        if ir.lhs == old:
            ir.lhs = new
        else:
            replace_output(ir.lhs, old, new)
    elif type(ir) == Indexing:
        if ir.dobject == old:
            ir.dobject = new
        else:
            replace_output(ir.dobject, old, new)


def has_same_iteration_space(l1, l2):
    return has_same_value(l1.start, l2.start) and has_same_value(l1.end, l2.end) and has_same_value(l1.step, l2.step)


def gen_ir(node):
    assert isinstance(node, ASTNode)
    if node.eval or len(node.decl) > 0 or (type(node) == TensorOp and len(node.compute) > 0):
        return node
    if type(node) == Const:
        if node.dtype != 'slice':
            assert type(node.val) == int or type(node.val) == float
            node.eval = Literal(node.val, node.dtype)
        else:
            node.val.start._gen_ir()
            node.val.stop._gen_ir()
            node.val.step._gen_ir()
            node.eval = Slice(node.val.start.eval, node.val.stop.eval, node.val.step.eval)


    elif type(node) == Var or (type(node) == Tensor and len(node._size()) == 0):
        node.eval = Scalar(node.dtype, node.name, node.is_arg)
        node.decl = [Decl(node.eval)]

    elif type(node) == Tensor and len(node._size()) > 0:
        # convert AST sizes to IR sizes
        size = get_ir_of_size(node._size())
        node.eval = Ndarray(node.dtype, size, node.name, node.is_arg)
        node.decl = [Decl(node.eval)]

    elif type(node) == TensorOp:
        if node.op_type in arith_op or node.op_type in cmp_op:
            # arith_op and cmp_op are binary operations, we generate the two operands first
            node.operators[0]._gen_ir()
            node.operators[1]._gen_ir()
            assert isinstance(node.operators[0], Tensor) and isinstance(node.operators[1], Tensor)

            if node.op_type in arith_op:
                op = arith_op[node.op_type]
            else:
                op = node.op_type

            if len(node._size()) > 0:  # if output has >=1 dimensions, it should be stored in an Ndarray
                size = get_ir_of_size(node._size())
                node.eval = Ndarray(node.dtype, size)
            else:  # otherwise, it is a scalar
                size = []
                node.eval = Scalar(node.dtype)
            node.decl = [Decl(node.eval)]

            left_levels = len(node.operators[0]._size())
            right_levels = len(node.operators[1]._size())
            max_levels = max(left_levels, right_levels)
            assert max_levels == len(size)

            lhs = node.operators[0].eval
            rhs = node.operators[1].eval
            res = node.eval
            ir = node.compute

            for level in range(max_levels):

                # handle out of bound slicing
                left_slice = get_slice(lhs)
                right_slice = get_slice(rhs)
                left_attr = {}
                if left_slice != None and type(left_slice.start) == Literal:
                    if left_slice.start.val < 0:
                        left_ofs = -left_slice.start.val
                        left_attr['slice_ofs'] = left_ofs
                    else:
                        left_ofs = 0
                else:
                    left_ofs = 0
                right_attr = {}
                if right_slice != None and type(right_slice.start) == Literal:
                    if right_slice.start.val < 0:
                        right_ofs = -right_slice.start.val
                        right_attr['slice_ofs'] = right_ofs
                    else:
                        right_ofs = 0
                else:
                    right_ofs = 0

                pre_loop = Loop(0, size[level], 1, [])
                loop_ofs = max(left_ofs, right_ofs)
                if loop_ofs > 0:
                    pre_loop.attr['loop_ofs'] = loop_ofs

                if level < left_levels:
                    lhs = bind(lhs, [pre_loop.iterate], [left_attr])
                    node.input_orders[0].append((level, pre_loop))
                if level < right_levels:
                    rhs = bind(rhs, [pre_loop.iterate], [right_attr])
                    node.input_orders[1].append((level, pre_loop))
                res = bind(res, [pre_loop.iterate])
                node.output_order.append((level, pre_loop))
                pre_loop.attr['output_axis'] = level
                ir.append(pre_loop)
                ir = pre_loop.body

            ir.append(Assignment(res, Expr(lhs, rhs, op)))

        elif node.op_type in math_op:
            node.operators[0]._gen_ir()

            if len(node._size()) > 0:
                size = get_ir_of_size(node._size())
                node.eval = Ndarray(node.dtype, size)
            else:
                size = []
                node.eval = Scalar(node.dtype)

            node.decl = [Decl(node.eval)]

            res = node.eval
            val = node.operators[0].eval
            levels = len(size)
            ir = node.compute

            for level in range(levels):
                slice = get_slice(val)
                attr = {}
                if slice != None and type(slice.start) == Literal:
                    if slice.start.val < 0:
                        ofs = -slice.start.val
                        attr['slice_ofs'] = ofs
                    else:
                        ofs = 0
                else:
                    ofs = 0

                pre_loop = Loop(0, size[level], 1, [])
                if ofs > 0:
                    pre_loop.attr['loop_ofs'] = ofs

                val = bind(val, [pre_loop.iterate], [attr])
                node.input_orders[0].append((level, pre_loop))
                res = bind(res, [pre_loop.iterate])
                node.output_order.append((level, pre_loop))
                pre_loop.attr['output_axis'] = level
                ir.append(pre_loop)
                ir = pre_loop.body

            ir.append(Assignment(res, Math(val, node.op_type)))

        elif node.op_type == 'setval':
            if type(node.operators[0]) == Tensor:
                node.operators[0].is_arg = False

            node.operators[0]._gen_ir()
            node.operators[1]._gen_ir()

            node.eval = node.operators[0].eval
            # remove_decl(node.operators[0], node.eval)
            # node.decl.append(Decl(node.eval))

            if is_scalar(node.operators[1]):
                val = node.operators[1].eval

                if len(node.ref_size) > 0:
                    size = get_ir_of_size(node.ref_size)
                    pre_loop = Loop(0, size[0], 1, [])
                    node.compute = [pre_loop]
                    res = bind(node.eval, [pre_loop.iterate])
                    for i in range(1, len(size)):
                        loop = Loop(0, size[i], 1, [])
                        pre_loop.body.append(loop)
                        pre_loop = loop
                        res = bind(res, [pre_loop.iterate])

                    assign = Assignment(res, val)
                    pre_loop.body.append(assign)
                else:
                    node.compute = [Assignment(node.eval, val)]

                l = node.compute[0]
                for i in range(len(node.eval.size)):
                    node.output_order.append((i, l))
                    l.attr['output_axis'] = i
                    l = l.body[0]
            else:
                node.operators[1].decl = [d for d in node.operators[1].decl if d.dobject != node.operators[1].eval]
                node.operators[1].eval = node.eval
                replace_output(node.operators[1].compute, node.operators[1].eval, node.eval)


        elif node.op_type == 'einsum':
            node.operators[0]._gen_ir()
            if node.operators[1] != None:
                node.operators[1]._gen_ir()
            node.input_orders[0] = []
            node.input_orders[1] = []

            exp = node.operators[2]
            inputs, output = exp.split('->')
            input1, input2 = inputs.split(',')
            all_indices = ''.join(sorted(set(input1 + input2)))
            all_loops = []
            mapping = {}

            reduce_begins = len(output)

            for i in output:
                pos1 = input1.find(i)
                pos2 = input2.find(i)
                if (pos1 >= 0 and pos2 < 0):
                    mapping[i] = len(all_loops)
                    l = Loop(0, node.operators[0].eval.ref_size(pos1), 1, [])
                    all_loops.append(l)
                    node.input_orders[0].append((len(node.input_orders[0]), l))
                elif (pos1 < 0 and pos2 >= 0):
                    mapping[i] = len(all_loops)
                    l = Loop(0, node.operators[1].eval.ref_size(pos2), 1, [])
                    all_loops.append(l)
                    node.input_orders[1].append((len(node.input_orders[1]), l))

            for i in all_indices:
                if i in output:
                    continue
                pos1 = input1.find(i)
                pos2 = input2.find(i)
                if (pos1 >= 0 and pos2 < 0):
                    mapping[i] = len(all_loops)
                    l = Loop(0, node.operators[0].eval.ref_size(pos1), 1, [])
                    all_loops.append(l)
                    node.input_orders[0].append((len(node.input_orders[0]), l))
                elif (pos1 < 0 and pos2 >= 0):
                    mapping[i] = len(all_loops)
                    l = Loop(0, node.operators[1].eval.ref_size(pos2), 1, [])
                    all_loops.append(l)
                    node.input_orders[1].append((len(node.input_orders[1]), l))

            for i in all_indices:
                pos1 = input1.find(i)
                pos2 = input2.find(i)
                if pos1 >= 0 and pos2 >= 0:
                    mapping[i] = len(all_loops)
                    l = Loop(0, node.operators[0].eval.ref_size(pos1), 1, [], 'reduction')
                    all_loops.append(l)
                    node.input_orders[0].append((len(node.input_orders[0]), l))
                    node.input_orders[1].append((len(node.input_orders[1]), l))

            for i in all_indices:
                pos1 = input1.find(i)
                pos2 = input2.find(i)
                if pos1 < 0 and pos2 < 0:
                    raise IndexError('index not found!')

            op1 = node.operators[0].eval
            for i in input1:
                op1 = bind(op1, [all_loops[mapping[i]].iterate])

            if node.operators[1] != None:
                op2 = node.operators[1].eval
                for i in input2:
                    op2 = bind(op2, [all_loops[mapping[i]].iterate])
            else:
                op2 = None

            size = get_ir_of_size(node._size())
            if len(size) > 0:
                node.eval = Ndarray(node.dtype, size)
            else:
                node.eval = Scalar(node.dtype)
            node.decl = [Decl(node.eval)]
            res = node.eval
            for i in output:
                res = bind(res, [all_loops[mapping[i]].iterate])

            if op2 != None:
                expr = Expr(op1, op2, '*')
            else:
                expr = op1
            if reduce_begins == len(all_loops):
                body = Assignment(res, expr)
            else:
                body = Assignment(res, expr, '+')
            init = Assignment(res, 0)
            if reduce_begins == 0:
                node.compute.append(init)
            pre_loop = all_loops[0]
            node.compute.append(pre_loop)
            for i in range(1, len(all_loops)):
                if reduce_begins == i:
                    pre_loop.body.append(init)
                loop = all_loops[i]
                pre_loop.body.append(loop)
                pre_loop = loop
            pre_loop.body.append(body)

            l = node.compute[0]
            for i in range(len(node.eval.size)):
                node.output_order.append((i, l))
                l.attr['output_axis'] = i
                l = l.body[0]


        elif node.op_type == 'index':
            node.operators[0]._gen_ir()
            subscripts = []
            for op in node.operators[1:]:
                op._gen_ir()
                subscripts.append(op.eval)

            node.eval = bind(node.operators[0].eval, subscripts)

        elif node.op_type == 'apply':
            func = node.operators[0]

            # operators: func, data (node.nparams), axis (node.nparams), out_ofs, cond, items (node.nparams), ret
            for i in range(1, 3 + 2 * node.nparams):
                if node.operators[i] != None:
                    node.operators[i]._gen_ir()

            primary_axis = node.operators[1 + node.nparams].eval.val

            # this is the loop that iterates over the axis of the primary (first) tensor input
            cond = node.operators[2 + 2 * node.nparams]
            if cond == None:
                outer_loop = Loop(0, node.operators[1].eval.size[primary_axis], 1, [])
            else:
                outer_loop = FilterLoop(0, node.operators[1].eval.size[primary_axis], 1,
                                        cond.eval, [], [])
                # gen ir for the counter
                node.operators[-1]._gen_ir()


            nn = []
            for i in range(node.nparams):
                item = node.operators[3 + 2 * node.nparams + i]
                item.eval = node.operators[1 + i].eval
                axis = node.operators[1 + node.nparams + i].eval.val
                n = num_unbind(item.eval)
                nn.append(n)
                for i in range(n, axis):
                    item.eval = Indexing(item.eval, Literal(-1, 'int'))
                if axis >= n:
                    item.eval = Indexing(item.eval, outer_loop.iterate)
                else:
                    item.eval = bind(item.eval, [outer_loop.iterate])

            # since input items of func has been generated and indexed, we can generate the IR of the func
            ret = node.operators[-2]
            ret._gen_ir()

            # get the input orders
            for i in range(min(len(ret.input_orders), node.nparams)):
                n = nn[i]
                l = node.input_orders[1 + i]
                if axis >= n:
                    for j in range(axis):
                        l.append((len(l), ret.input_orders[i][j][1]))
                    l.append((len(l), outer_loop))
                    for j in range(axis, len(ret.input_orders[i])):
                        l.append((len(l), ret.input_orders[i][j][1]))
                else:
                    l.append((len(l), outer_loop))
                    for j in range(len(ret.input_orders[i])):
                        l.append((len(l), ret.input_orders[i][j][1]))

            def action(n, res):
                if isinstance(n, Tensor):
                    for s in n.compute:
                        s.attr['asgnode'] = n
                    res.extend(n.compute)
                    n.compute.clear()
                    n.attr['scope'] = node

            t = ASGTraversal(action)
            ret_compute = t(ret)

            # if there is no compute in the func, we simply assign the result to itself, so that later the lhs of the assignment will be changed to the output array
            if len(ret_compute) == 0:
                ret_compute.append(Assignment(ret.eval, copy.copy(ret.eval)))

            # the computation of func are added in the outer_loop
            outer_loop.body.extend(ret_compute)
            size = get_ir_of_size(node._size())
            node.eval = Ndarray(ret.eval.dtype, size)
            node.decl.append(Decl(node.eval))
            node.compute = [outer_loop]

            out_ofs = node.operators[1 + 2 * node.nparams]
            res = bind(node.eval, [outer_loop.iterate]) if out_ofs == None else node.eval
            replace_all_ref(node.compute, ret.eval, res)
            remove_decl(ret, ret.eval)
            # if there is an offset for output storage
            if out_ofs != None:
                assert type(ret_compute[-1]) in (Loop, Assignment)
                l = ret_compute[-1]
                while (type(l) == Loop):
                    l = l.body[-1]
                # But the index to the node.eval in res is incorrect, we need to change it according to the offset
                rebind_iterate(l.lhs, ret_compute[-1].iterate,
                               Expr(Indexing(out_ofs.eval, outer_loop.iterate), ret_compute[-1].iterate, '+'))
            # ret.eval is removed from the decl
            node.decl = [d for d in node.decl if d.dobject != ret.eval]

            if cond != None:
                counter = node.operators[-1].eval
                outer_loop.body.append(Assignment(counter, 1, '+'))
                assert type(ret_compute[-1]) in (Loop, Assignment)
                l = ret_compute[-1]
                while (type(l) == Loop):
                    l = l.body[-1]
                rebind_iterate(l.lhs, outer_loop.iterate, counter)


            node.output_order = [(0, outer_loop)]
            outer_loop.attr['output_axis'] = 0
            if hasattr(ret, 'output_order'):
                for i in range(len(ret.output_order)):
                    node.output_order.append((i + 1, ret.output_order[i][1]))
                    ret.output_order[i][1].attr['output_axis'] = i + 1


        elif node.op_type == 'reduce':
            # TODO: add input_orders for reduce, and aggr
            node.operators[0]._gen_ir()  # input data
            node.operators[3]._gen_ir()  # axis
            axis = node.operators[3].eval.val

            size = get_ir_of_size(node._size())
            if len(size) > 0:
                node.eval = Ndarray(node.dtype, size)
            else:
                node.eval = Scalar(node.dtype)

            node.operators[2]._gen_ir()  # init
            # the decl of node.eval should be added to the init
            node.operators[2].decl.append(Decl(node.eval))

            outer_loop = Loop(0, node.operators[0].eval.size[axis], 1, [], 'reduction')

            item1 = node.operators[4]
            item2 = node.operators[5]
            item1.eval = node.eval
            item2.eval = node.operators[0].eval
            n = num_unbind(item2.eval)
            for i in range(n, axis):
                item2.eval = Indexing(item2.eval, Literal(-1, 'int'))
            if axis > n:
                item2.eval = Indexing(item2.eval, outer_loop.iterate)
            else:
                item2.eval = bind(item2.eval, [outer_loop.iterate])
            item2.decl = []
            item1.decl = []

            ret = node.operators[-1]
            ret._gen_ir()

            compute = ret.output_order[-1][1].body if len(ret.output_order) > 0 else ret.compute
            outer_loop.body = compute[:]
            compute.clear()

            # merge init into node.compute
            init = node.operators[2].output_order[-1][1].body if len(node.operators[2].output_order) > 0 else \
            node.operators[2].compute
            # assert len(node.operators[2].output_order) == len(ret.output_order)
            for i in range(len(node.operators[2].output_order)):
                # assert has_same_iteration_space(node.operators[2].output_order[i][1], ret.output_order[i][1])
                rebind_iterate(init, node.operators[2].output_order[i][1].iterate, ret.output_order[i][1].iterate)
                node.output_order.append((i, ret.output_order[i][1]))
                ret.output_order[i][1].attr['output_axis'] = i
            compute.extend(init)
            node.operators[2].compute.clear()
            compute.append(outer_loop)

            def action(node, res):
                if isinstance(node, Tensor):
                    res.extend(node.compute)
                    node.compute.clear()

            t = ASGTraversal(action)
            ret_compute = t(ret)

            node.compute.extend(ret_compute)

            replace_output(node.compute, ret.eval, node.eval)
            node.decl = [d for d in node.decl if d.dobject != ret.eval]


        elif node.op_type == 'aggr':
            node.operators[0]._gen_ir()  # input tensor
            node.operators[3]._gen_ir()  # indices
            node.operators[4]._gen_ir()  # axis
            axis = node.operators[4].eval.val
            size = get_ir_of_size(node._size())
            node.eval = Ndarray(node.dtype, size)
            node.operators[2]._gen_ir()  # init
            node.operators[2].decl.append(Decl(node.eval))

            # compute
            outer_loop = Loop(0, node.operators[0].eval.size[axis], 1, [], 'reduction')

            item1 = node.operators[6]
            item2 = node.operators[7]
            item1.eval = Indexing(node.eval, Indexing(node.operators[3].eval, outer_loop.iterate))
            item2.eval = node.operators[0].eval
            for i in range(axis):
                item2.eval = Indexing(item2.eval, Literal(-1, 'int'))
            item2.eval = Indexing(item2.eval, outer_loop.iterate)
            item2.decl = []
            item1.decl = []

            ret = node.operators[-1]
            ret._gen_ir()

            def action(node, res):
                if isinstance(node, Tensor):
                    res.extend(node.compute)
                    node.compute.clear()

            t = ASGTraversal(action)
            ret_compute = t(ret)

            outer_loop.body.extend(ret_compute)
            node.compute.append(outer_loop)

            replace_output(node.compute, ret.eval, item1.eval)
            node.decl = [d for d in node.decl if d.dobject != ret.eval]

            node.output_order = [(0, outer_loop)]
            for i in range(len(ret.output_order)):
                node.output_order.append((i + 1, ret.output_order[i][1]))
                ret.output_order[i][1].attr['output_axis'] = i + 1

        elif node.op_type == 'inline':
            src = node.operators[0]
            keyvalue = []
            for i in range(2, len(node.operators), 2):
                node.operators[i]._gen_ir()
                keyvalue.append((node.operators[i-1], node.operators[i].eval))

            node.eval = node.operators[2].eval
            # remove_decl(node.operators[2], node.eval)
            # node.decl.append(Decl(node.eval))
            node.compute = [Code(src, keyvalue[0], dict(keyvalue[1:]))]

        elif node.op_type == 'getmember':
            node.operators[0]._gen_ir()
            member = getattr(node.operators[0], node.operators[1])
            node.eval = member.eval



    return node
