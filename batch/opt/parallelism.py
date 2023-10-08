from core.ir import *
from batch.ast import *
from batch.ast2ir import *
import codegen
from batch.opt.ir import *
# for better optimization on GPU

def parallel(ast):
    if type(ast) == BatchOp:
        if type(ast.operators[1]) == BatchOp:
            parallel(ast.operators[1])
        if type(ast.operators[0]) == BatchOp:
            parallel(ast.operators[0])
    else:
        return
    print(ast.op_type)
    if ast.compute:
        stmt = ast.compute[0]
        if isinstance(stmt, Loop):
            print(stmt.end.name(), stmt.iterate.name())
            if stmt.end.name() == 'batch_size':
                ast.compute = stmt.body
                for i in ast.compute:
                    if isinstance(i, Loop):
                        i.start = 'threadIdx.x'
                        i.step = 'blockDim.x'
                # a = ThreadMapping(stmt.start, stmt.end, stmt.step, stmt.body)
                # 16 ty * 32 tx each thread block
                assign = Assignment(stmt.iterate, 'threadIdx.y + blockIdx.x * 16')
                # a.iterate = stmt.iterate
                ast.compute.insert(0, assign)
                ast.decl.append(Decl(stmt.iterate))
        
            