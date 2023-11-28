import copy
from cset.ast2ir import *
from codegen.cpu import *

num_thread = 16
thread_id = Scalar('int', 'thread_id')

def IRString(ir):
    code = ''
    for d in ir:
        if d:
            code += to_string(d)
    return code

def PrintCCode(ir):
    code = ''
    for d in ir:
        if d:
            code += to_string(d)
    print(code)
    return ir

def bind_thread(ir, scalar_set, ndarray_set):
    visited = set()

    def _bind_thread(ir):
        if type(ir) == list or type(ir) == tuple:
            for i in range(0, len(ir)):
                ir[i] = _bind_thread(ir[i])
            return ir
        elif type(ir) == Loop:
            ir.start =  _bind_thread(ir.start)
            ir.end =  _bind_thread(ir.end)
            ir.step =  _bind_thread(ir.step)
            ir.body = _bind_thread(ir.body)
            return ir
        elif type(ir) == FilterLoop:
            ir.start =  _bind_thread(ir.start)
            ir.end =  _bind_thread(ir.end)
            ir.step =  _bind_thread(ir.step)
            ir.body =  _bind_thread(ir.body)
            ir.cond = _bind_thread(ir.cond)
            ir.cond_body = _bind_thread(ir.cond_body)
            return ir
        elif type(ir) == Code:
            for key in ir.keywords:
                ir.keywords[key] = _bind_thread(ir.keywords[key])
            return ir
        elif type(ir) == Expr:
            ir.left = _bind_thread(ir.left)
            ir.right = _bind_thread(ir.right)
            return ir
        elif type(ir) == Assignment:
            ir.lhs = _bind_thread(ir.lhs)
            ir.rhs = _bind_thread(ir.rhs)
            return ir
        elif type(ir) == Indexing:
            if ir.idx !=thread_id:
                ir.dobject = _bind_thread(ir.dobject)
            ir.idx = _bind_thread(ir.idx)
            return ir
        elif type(ir) == Slice:
            ir.start = _bind_thread(ir.start)
            ir.stop = _bind_thread(ir.stop)
            ir.step = _bind_thread(ir.step)
            return ir
        elif type(ir) == Math:
            ir.val = _bind_thread(ir.val)
            return ir
        elif type(ir) == Scalar:
            if ir.name() in scalar_set:
                return Indexing(Ndarray(ir.dtype, [num_thread], ir.name()), thread_id)
            else:
                return ir
        elif type(ir) == Ndarray:
            if ir.name() in ndarray_set:
                return Indexing(ir, thread_id)
            else:
                return ir
        else:
            return ir
    _bind_thread(ir)

def replace_decl(ast, scalar_set, ndarray_set):
    def _replace_decl(decls):
        remove_list = []
        add_list = []
        for ir in decls:
            assert(type(ir)==Decl)
            if type(ir.dobject)== Scalar and ir.dobject.name() in scalar_set:
                remove_list.append(ir)
                add_list.append(Decl(Ndarray(ir.dobject.dtype, [num_thread], ir.dobject.name())))
            elif type(ir.dobject)== Ndarray and ir.dobject.name() in ndarray_set:
                remove_list.append(ir)
                add_list.append(Decl(Ndarray(ir.dobject.dtype, [num_thread] + ir.dobject.size, ir.dobject.name())))
            else:
                pass
        for ir in remove_list:
            decls.remove(ir)
        
        for ir in add_list:
            decls.append(ir)

    def action(node, res):
        if len(node.decl) != 0:
            _replace_decl(node.decl)

    t = helpers.ASGTraversal(action)
    t(ast)

def get_dobject_name(ir):
    if type(ir.dobject)!=Ndarray:
        return get_dobject_name(ir.dobject)
    else:
        return ir.dobject.name()

def get_all_lhs(ir):
    scalar_set = set()
    ndarray_set = set()
    
    def _get_all_lhs(ir):  
        if type(ir) == list or type(ir) == tuple:
            for l in ir:
                _get_all_lhs(l)
        elif type(ir) == Loop:
            _get_all_lhs(ir.body)
        elif type(ir) == FilterLoop:
            _get_all_lhs(ir.body)
            _get_all_lhs(ir.cond_body)
        elif type(ir) == Code:
            if 'res_arr' in ir.keywords:
                ndarray_set.add(get_dobject_name(ir.keywords['res_arr']))
            if 'res_size' in ir.keywords:
                scalar_set.add(ir.keywords['res_size'].name())
            for item in ir.keywords.values():
                _get_all_lhs(item)
        elif type(ir) == Expr:
            _get_all_lhs(ir.left)
            _get_all_lhs(ir.right)
        elif type(ir) == Assignment:
            if type(ir.lhs)==Scalar:
                scalar_set.add(ir.lhs.name())
            elif type(ir.lhs)==Indexing:
                ndarray_set.add(get_dobject_name(ir.lhs))
        elif type(ir) == Slice:
            return
        elif type(ir) == Math:
            return
        else:
            return
    
    _get_all_lhs(ir)
    return scalar_set, ndarray_set

def set_loop_levels(irs):
    for ir in irs:
        if type(ir)==Loop:
            ir.attr['ptype']= 'naive'
            ir.attr['plevel'] = 0
            ir.attr['nprocs'] = [(num_thread, ir)]
            ir.body.insert(0, Assignment(thread_id, Code('omp_get_thread_num();', {})))
            ir.body.insert(0, Decl(thread_id))



def reset_count(ast):
    def _reset_count(irs):
        count_ir = None
        remove_list = []
        add_list = []
        for ir in irs:
            if type(ir)== Assignment and type(ir.lhs)==Scalar and ir.lhs.name()=='count':
                count_ir = ir
        if count_ir !=None:
            irs.remove(count_ir)
        return count_ir

    def action(node, res):
        if len(node.compute) != 0:
            count_ir = _reset_count(node.compute)

    t = helpers.ASGTraversal(action)
    t(ast)
    
    sum_ir = Scalar('int', 'sum')
    sum_loop = Loop(0, num_thread, 1, [])
    sum_loop.body.extend([Assignment(sum_ir, bind(Ndarray('int', [num_thread], 'count'), sum_loop.iterate), '+')])
    ast.decl.extend([Decl(sum_ir)])
    ast.compute.append(sum_loop)
    ast.compute.append(Code('printf("count:%d\\n", sum);', {}))


def parallelize(ast):
    ast = ast.operators[0]
    scalar_set, ndarray_set = get_all_lhs(ast.compute)
    bind_thread(ast.compute, scalar_set, ndarray_set)
    replace_decl(ast, scalar_set, ndarray_set)
    set_loop_levels(ast.compute)
    reset_count(ast)




    # print(len(scalar_set))
    # print(len(ndarray_set))
    # for k in scalar_set:
    #     print(k)

    # for k in ndarray_set:
    #     print(k)