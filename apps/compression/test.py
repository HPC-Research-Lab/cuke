from asg import *
import codegen
import torch
import run


def compression():
    nblocks = 50
    input = Tensor('input', (nblocks, 32), dtype='float')
    quant_res = (input * 1000).round()
    lorenzo_res = quant_res[:][0:32] - quant_res[:][-1:31]
    encode_nbits = lorenzo_res.abs().max(axis=1).nbits()
    ofs = encode_nbits.prefix_sum(inclusive=False)
    compressed_res = apply(lambda x, y: truncate(x, y), (lorenzo_res, encode_nbits), out_ofs=ofs)
    res = compressed_res
    code = codegen.cpu.print_cpp(res._gen_ir())

    print(code)

    code = '#include "../../compression/util.h"\n' + code

    data = torch.rand(nblocks, 32)

    d = run.cpu.compile_and_run(code, data)
    # d = run.cpu.run(data)
    print(d)


def invert_lorenzo():
    nblocks = 50
    input = Tensor('input', (nblocks, 32), dtype='float')
    quant_res = (input * 1000).round()
    lorenzo_res = quant_res[:][0:32] - quant_res[:][-1:31]

    recover_res = Tensor('recovered', input._size(), dtype='float')
    recover_res = setval(recover_res, recover_res[:][-1:31] + lorenzo_res[:][0:32])
    # recover_res = lorenzo_res.prefix_sum(axis=1)
    res = recover_res

    code = codegen.cpu.print_cpp(res._gen_ir())
    code = '#include "../../compression/util.h"\n' + code

    data = torch.rand(nblocks, 32)

    print((data * 1000).round())

    d = run.cpu.compile_and_run(code, data)
    print(d)

def regression():
    nblocks = 50
    input = Tensor('input', (nblocks, 32), dtype='float')
    quant_res = (input * 1000).round()
    res = apply(lambda x: einsum('i,i->', x, Const(slice(1, 33, 1), 'slice')), (quant_res, ))

    code = codegen.cpu.print_cpp(res._gen_ir())
    print(code)


def regression2():
    nblocks = 50
    input = Tensor('input', (nblocks, 6, 6, 6), dtype='float')
    quant_res = (input * 1000).round()
    tmp1 = apply(lambda x: einsum('ijk,i->', x, Const(slice(1, 33, 1), 'slice')), (quant_res, ))
    tmp2 = apply(lambda x: einsum('ijk,j->', x, Const(slice(1, 33, 1), 'slice')), (quant_res, ))
    tmp3 = apply(lambda x: einsum('ijk,k->', x, Const(slice(1, 33, 1), 'slice')), (quant_res, ))
    tmp4 = apply(lambda x: einsum('ijk,->', x, None), (quant_res, ))

    coefficient = (tmp1 + tmp2 + tmp3) / tmp4

    code = codegen.cpu.print_cpp(coefficient._gen_ir())
    print(code)



if __name__ == "__main__":
    # compression()
    # invert_lorenzo()
    # regression()
    regression2()
