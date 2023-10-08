from core.ast2ir import *
import helpers
import batch
from batch.opt.ir import *

def to_string(ir):
    match ir.__class__.__name__:
        case 'Expr':
            return f"({to_string(ir.left)}" + f" {ir.op} " + f"{to_string(ir.right)})"
        case 'Assignment':
            if ir.op is None:
                return f"{to_string(ir.lhs)} = {to_string(ir.rhs)};\n"
            else:
                return f"{to_string(ir.lhs)} {ir.op}= {to_string(ir.rhs)};\n"
        case 'Loop':
            code = f"for (int {to_string(ir.iterate)} = {to_string(ir.start)}; {to_string(ir.iterate)} < {to_string(ir.end)}; {to_string(ir.iterate)} += {to_string(ir.step)}) {{\n"
            for e in ir.body:
                if e:
                    code += to_string(e)
            code += "} \n"
            return code
        case 'CUDAThread':
            pass
        case 'Scalar' | 'Ndarray' | 'Ref':
            return ir.name()
        case 'Index':
            # print(ir.ind_arr, ir.index, ir.dobject, ir.index.addr())
            if ir.ind_arr != None:
                if type(ir.ind_arr) == Slice:
                    return f'{to_string(ir.dobject)}[(({to_string(ir.ind_arr.start)})+({to_string(ir.ind_arr.step)})*({to_string(ir.index)}))]'
                else: # idx is a Tensor
                    if ir.index == None:
                        return f'{to_string(ir.dobject)}[{to_string(ir.ind_arr)}]'
                    else:
                        return f'{to_string(ir.dobject)}[{to_string(ir.ind_arr)}[{to_string(ir.index)}]]'
            else:
                return f'{to_string(ir.dobject)}[{to_string(ir.index)}]'
        case 'Decl':
            # return ''
            # variables are passed in as pytorch arguments
            if type(ir.dobject) == Scalar:
                if not ir.dobject.is_arg:
                    # it is a zero or one
                    if ir.dobject.val != None:
                        return f"{ir.dobject.dtype} {ir.dobject.name()} = {to_string(ir.dobject.val)};\n"
                    else:
                        return f"{ir.dobject.dtype} {ir.dobject.name()};\n"
                else:
                    return ''
            elif type(ir.dobject) == Ndarray:
                code = ''
                if not ir.dobject.is_arg:
                    if ir.dobject.val != None:
                        code = f'torch::Tensor obj_{ir.dobject.name()} = torch::{"ones" if ir.dobject.val == 1 else "zeros"}({{{",".join([to_string(s) for s in ir.dobject.size])}}}, at::k{"Int" if ir.dobject.dtype=="int" else "Float"});\n'
                    else:
                        code = f'torch::Tensor obj_{ir.dobject.name()} = torch::empty({{{",".join([to_string(s) for s in ir.dobject.size])}}}, at::k{"Int" if ir.dobject.dtype=="int" else "Float"});\n'

                # code += f'auto {ir.dobject.name()} = obj_{ir.dobject.name()}.accessor<{ir.dobject.dtype}, {len(ir.dobject.size)}>();\n'
                return code
            elif type(ir.dobject) == Ref:
                code = f'{ir.dobject.dobject.dtype}* {ir.dobject.name()} = ({ir.dobject.dobject.dtype}*)&{ir.dobject.dobject.addr()}'
                return code
        case _:
            return str(ir)

def gen_cuda(ast, cpu_ir, gpu_ir):
    # 2 ir list for cpu and gpu
    def action_cpu(node, res):
        if type(node) == Var or type(node) == One or type(node) == Zero or type(node) == Ones or type(node) == Zeros or type(node) == Tensor:
            res.extend(node.decl)
        elif type(node) == TensorOp:
            res.extend(node.decl)
            # res.extend(node.compute)
        elif type(node) == batch.ast.BatchOp:
            res.extend(node.decl)
            # res.extend(node.compute)

    def action_cuda(node, res):
        if type(node) == TensorOp:
            # res.extend(node.decl)
            res.extend(node.compute)
        elif type(node) == batch.ast.BatchOp:
            res.extend(node.decl)
            res.extend(node.compute)
    t = helpers.Traversal(action_cpu)
    cpu_ir.extend(t(ast))

    t = helpers.Traversal(action_cuda)
    gpu_ir.extend(t(ast))

def print_cpp(ast):
    cpu_ir = []
    gpu_ir = []
    gen_cuda(ast, cpu_ir, gpu_ir)
    print(cpu_ir, "CUDA IR:::" , gpu_ir)

    args = helpers.get_input_nodes(ast)
    argscpu = ', '.join([f'torch::Tensor obj_{a}' if type(args[a]) == Tensor else f'{args[a].dtype} {a}' for a in args])
    # argsptr = ', '.join([f'obj_{a}.data_ptr<{args[a].dtype}>()' if type(args[a]) == Tensor else f'{a}' for a in args])
    # ptrs = ', '.join([f'{args[a].dtype}* {a}' if type(args[a]) == Tensor else f'{args[a].dtype} {a}' for a in args])

    print(args)
    
    argsptr = ', '.join([f'obj_{a}.packed_accessor32<{args[a].dtype}, {len(args[a].ref_size)}, torch::RestrictPtrTraits>()' if type(args[a]) == Tensor else f'{a}' for a in args])
    ptrs = ', '.join([f'torch::PackedTensorAccessor32<{args[a].dtype}, {len(args[a].ref_size)}, torch::RestrictPtrTraits> {a}' if type(args[a]) == Tensor else f'{args[a].dtype} {a}' for a in args])
    # in cuda kernel:     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
    # host call cuda:     .packed_accessor32<float, 2, torch::RestrictPtrTraits>()

    code = ''
    declare = ''
    for d in gpu_ir:
        if d:
            if type(d) == Decl and type(d.dobject) == Ndarray:
                declare += to_string(d)
                argsptr += f', obj_{d.dobject.name()}.packed_accessor32<{d.dobject.dtype}, {len(d.dobject.size)}, torch::RestrictPtrTraits>()'
                ptrs += f', torch::PackedTensorAccessor32<{d.dobject.dtype}, {len(d.dobject.size)}, torch::RestrictPtrTraits> {d.dobject.name()}'
            else:
                code += to_string(d)
            
    Return = ''
    if type(ast.eval) == Scalar:
        rtype = ast.dtype
        Return = f'return {ast.eval.name()};\n'
    elif type(ast.eval) == Ndarray:
        rtype = 'torch::Tensor'
        Return = f'return obj_{ast.eval.name()};\n'
    else:
        raise TypeError('wrong output type', ast.eval)

    with open('codegen/gpu_template.cu', 'r') as f:
        c_code = f.read()
        c_code = c_code.replace('RTYPE', rtype).replace('FNAME', ast.name).replace('ARGS', argscpu).replace('CODE', code).replace('PTR_VARS', argsptr).replace('PTRS', ptrs).replace('DECL', declare).replace('RETURN', Return)
    return c_code