from cset.ast2ir import *
from codegen.cpu import *

fused_intersection_template = \
    "if(first_smaller_second) continue; \n\
    else if (first_larger_second) { \n\
        while(pj_smaller_secondsize && first_larger_second){    \n\
            pj_increment;                                         \n\
        } \n\
    } \n\
    if(pj_equal_secondsize) break;  \n\
    if(first_equal_second) { \n\
        assignment\n\
        pos_increment; \n\
        continue; \n\
    }"


fused_difference_template = \
    "if(first_smaller_second) { } \n\
    else if (first_larger_second) { \n\
        while(pj_smaller_secondsize && first_larger_second){    \n\
            pj_increment;                                           \n\
        } \n\
    } \n\
    if(pj_equal_secondsize) {  \n\
        setop_assignment\n\
        filter_cond \n\
        if(filter_meet){  \n\
            filter_assignment \n\
            filter_increment \n\
            continue; \n\
        } \n\
        else{\n\
            break;    \n\
        }\n\
    } \n\
    if(first_equal_second) { \n\
        pj_increment; \n\
    }  \n\
    else{ \n\
        setop_assignment\n\
        filter_cond \n\
        if(filter_meet){  \n\
            filter_assignment \n\
            filter_increment \n\
        } \n\
        else{\n\
            break;    \n\
        }\n\
    }"

def PrintCCode2(ir):
    code = ''
    for d in ir:
        if d:
            code += to_string(d)
    print(code)
    return ir

def fuse_irs(filter_op, setop, pre_ast, top_ast):
    filter_compute = filter_op.compute
    filter_decl = filter_op.decl

    setop_compute = setop.compute
    setop_decl = setop.decl

    setop_loop_ir = None
    for ir in setop_compute:
        if type(ir)==Loop:
            setop_loop_ir = ir
    assert(type(setop_loop_ir.body[0]==Code))
    setop_code = setop_loop_ir.body[0]
    setop_keywords = setop_code.keywords

    filter_loop_ir = None
    for ir in filter_compute:
        if type(ir)==FilterLoop:
            filter_loop_ir = ir

    filter_loop_body = filter_loop_ir.body
    filter_loop_cond = filter_loop_ir.cond
    filter_loop_cond_body = filter_loop_ir.cond_body
    filter_cond = filter_loop_body[0]
    filter_meet = filter_loop_cond
    filter_assignment = filter_loop_cond_body[0]
    filter_increment = filter_loop_cond_body[1]

    setop_assignment = setop_keywords['assignment']
    del setop_keywords['assignment']

    def _replace_with_tmp(setop_assignment, filter_cond, filter_assignment, tmp_var):
        setop_assignment.lhs = tmp_var
        filter_cond.rhs.left.right = tmp_var
        filter_assignment.rhs = tmp_var

    tmp_var = Scalar('int')
    _replace_with_tmp(setop_assignment, filter_cond, filter_assignment, tmp_var)

    
    setop_keywords['setop_assignment'] = setop_assignment
    setop_keywords['filter_cond'] = filter_cond
    setop_keywords['filter_meet'] = filter_meet
    setop_keywords['filter_assignment'] = filter_assignment
    setop_keywords['filter_increment'] = filter_increment

    setop_code.code = fused_difference_template
    setop_code.keywords = setop_keywords

    # PrintCCode2([setop_assignment])
    # PrintCCode2([setop_code])

    # filter_op.compute.clear()
    # setop.decl.extend(filter_decl)
    # setop.decl.append(tmp_var)
    # filter_op.decl.clear()

    #remove ir
    remove_list = []

    for ir in pre_ast.compute:
        if type(ir)==Loop:
            pre_ast_loop_ir = ir
    # print(level)
    # PrintCCode2(pre_ast_loop_ir.body)
    for ir in pre_ast_loop_ir.body:
        if ir in filter_op.compute:
            remove_list.append(ir)

    pre_ast_loop_ir.body.insert(0, Assignment(filter_increment.lhs, 0))
    for ir in remove_list:
        pre_ast_loop_ir.body.remove(ir)

    top_ast.decl.append(Decl(tmp_var))



def _op_fusion(ast, pre_ast, level, top_ast):

    if type(ast) != SetOp or ast.op_type!='apply':
        return
    
    if level>=1 and ast.operators[0].op_type=='filter':
        filter_op = ast.operators[0]
        setop = ast.operators[0].operators[0]
        fuse_irs(filter_op, setop, pre_ast, top_ast)

    
    _op_fusion(ast.operators[6], ast, level+1, top_ast)


def op_fusion(ast):
    ast = ast.operators[0]
    _op_fusion(ast, None, 0, ast)
    # input_set = _traverse_ast(ast, 0)

