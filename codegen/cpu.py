from cset.ast2ir import *
from helpers import collect_ir, get_input_nodes
import random
import string
from codegen.oob import lower_bound_padding
import os


def to_string(ir):
    match ir.__class__.__name__:
        case 'Expr':
            if ir.op in arith_op.values():
                return f"({to_string(ir.left)}" + f" {ir.op} " + f"{to_string(ir.right)})"
            elif ir.op == 'bigger':
                return f"({to_string(ir.left)} > {to_string(ir.right)} ? ({to_string(ir.left)}) : ({to_string(ir.right)}))"
            elif ir.op == 'smaller':
                return f"({to_string(ir.left)} < {to_string(ir.right)} ? ({to_string(ir.left)}) : ({to_string(ir.right)}))"
            else:
                return f"({to_string(ir.left)}" + f" {ir.op} " + f"{to_string(ir.right)})"
        case 'Assignment':
            if ir.op is None:
                return f"{to_string(ir.lhs)} = {to_string(ir.rhs)};\n"
            else:
                return f"{to_string(ir.lhs)} {ir.op}= {to_string(ir.rhs)};\n"
        case 'Loop':
            code = ''
            if ir.attr['ptype'] == 'naive' and 'plevel' in ir.attr and 'nprocs' in ir.attr:
                code += f"#pragma omp parallel for num_threads({ir.attr['nprocs'][ir.attr['plevel']][0]})\n"
            code += f"for (int {to_string(ir.iterate)} = {to_string(ir.start)}; {to_string(ir.iterate)} < {to_string(ir.end)}; {to_string(ir.iterate)} += {to_string(ir.step)}) {{\n"
            for e in ir.body:
                if e:
                    code += to_string(e)
            code += "} \n"
            return code
        case 'Scalar' | 'Ndarray' | 'Ref':
            return ir.name()
        case 'Literal':
            return str(ir.val)
        case 'Indexing':
            if type(ir.dobject) == Slice:
                if ir.dobject.step == 1 or (type(ir.dobject.step) == Literal and ir.dobject.step.val == 1):
                    if ir.dobject.start == 0 or (type(ir.dobject.start) == Literal and ir.dobject.start.val == 0):
                        return f'({to_string(ir.idx)})'
                    else:
                        return f'(({to_string(ir.dobject.start)})+({to_string(ir.idx)}))'
                else:
                    if ir.dobject.start == 0 or (type(ir.dobject.start) == Literal and ir.dobject.start.val == 0):
                        return f'(({to_string(ir.dobject.step)})*({to_string(ir.idx)}))'
                    else:
                        return f'(({to_string(ir.dobject.start)})+({to_string(ir.dobject.step)})*({to_string(ir.idx)}))'
            else:
                return f'{to_string(ir.dobject)}[{to_string(ir.idx)}]'
        case 'Decl':
            # variables are passed in as pytorch arguments
            if type(ir.dobject) == Scalar:
                if not ir.dobject.is_arg:
                    return f"{ir.dobject.dtype} {ir.dobject.name()};\n"
                else:
                    return ''
            elif type(ir.dobject) == Ndarray:
                code = ''
                if not ir.dobject.is_arg:
                    code = f'torch::Tensor obj_{ir.dobject.name()} = torch::empty({{{",".join([to_string(s) for s in ir.dobject.size])}}}, at::k{"Int" if ir.dobject.dtype=="int" else "Float"});\n'
                code += f'auto {ir.dobject.name()} = obj_{ir.dobject.name()}.accessor<{ir.dobject.dtype}, {len(ir.dobject.size)}>();\n'
                return code
        case 'Math':
            return f"{ir.type}({to_string(ir.val)})"
        case 'Code':
            code = ir.code
            for kw in ir.keywords:
                code = code.replace(kw, to_string(ir.keywords[kw]))
            return code + '\n'
        
        #cset extension
        case 'FilterLoop':
            code = f"for (int {to_string(ir.iterate)} = {to_string(ir.start)}; {to_string(ir.iterate)} < {to_string(ir.end)}; {to_string(ir.iterate)} += {to_string(ir.step)}) {{\n"
            for e in ir.body:
                if e:
                    code += to_string(e)
            if ir.cond:
                code += f"if({to_string(ir.cond)}){{\n"
                for e in ir.cond_body:
                    if e:
                        code += to_string(e)
                code += "} \n"
            code += "} \n"
            return code
        case 'Not':
            return f"!{to_string(ir.dobject)}"
        case 'BinarySearch':
            code = f"BinarySearch({to_string(ir.dobject)}, {to_string(ir.start)}, {to_string(ir.end)}, {to_string(ir.item)})"
            return code
        case _:
            return str(ir)


def print_cpp(asg):
    ir = []
    lower_bound_padding(asg)

    collect_ir(asg, ir)
    args = get_input_nodes(asg)
    args = ', '.join([f'torch::Tensor obj_{a}' if type(args[a]) == Tensor else f'{args[a].dtype} {a}' for a in args])

    code = ''
    for d in ir:
        if d:
            code += to_string(d)


    if type(asg.eval) == Scalar:
        rtype = asg.dtype
        code += f'return {asg.eval.name()};\n'
    elif type(asg.eval) == Ndarray:
        rtype = 'torch::Tensor'
        code += f'return obj_{asg.eval.name()};\n'
    else:
        raise TypeError('wrong output type', asg.eval)

    with open(f'{os.path.dirname(__file__)}/cpp_template.txt', 'r') as f:
        c_code = f.read()
        c_code = c_code.replace('RTYPE', rtype).replace('FNAME', asg.name[:24] + ''.join(random.choices(string.ascii_lowercase, k=8))).replace('ARGS', args).replace('CODE', code)
    return c_code
