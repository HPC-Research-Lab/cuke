import sys

from cset.ast import *
from cset.ir import *
from core.asg2ir import *
import helpers
import codegen

def get_first_unbind(index: (Indexing, Ndarray, Slice)):
    if type(index) == Indexing:
        x = get_first_unbind(index.dobject)
        if x != None:
            return x
        else:
            if type(index.idx) == Literal and index.idx.val == -1:
                return index
            else:
                y = get_first_unbind(index.idx)
                return y
    return None


def bind(index: (Indexing, Ndarray, Slice), idx, attr = {}):
    x = get_first_unbind(index)
    if x == None:
        res = Indexing(index, idx)
        res.attr.update(attr)
        return res
    else:
        old = copy.copy(x.idx)
        old_attr = copy.copy(x.attr)
        x.idx = idx
        x.attr.update(attr)
        new_index = copy.deepcopy(index)
        x.idx = old
        x.attr = old_attr
        return new_index

def _gen_and_detach_ir(ret):
    ret._gen_ir()

    def action(node, res):
        if isinstance(node, Tensor):
            res.extend(node.compute)
            node.compute.clear()
        elif isinstance(node, Set):
            res.extend(node.compute)
            node.compute.clear()

    t = helpers.ASGTraversal(action)
    ret_compute = t(ret)
    return ret_compute

def gen_ir(node):
    if type(node) == Set:
        node.nelem._gen_ir()
        node.storage._gen_ir()
        helpers.get_ir_of_size(node.storage._size())
        node.eval = node.storage.eval
        if len(node._tensor_size())>0:
            node.compute = [Assignment(node.nelem.eval, node._tensor_size()[0].eval)]

    elif type(node) == SetOp:        
        if node.op_type == 'apply':
            input_set = node.operators[0]
            input_func_ast = node.operators[1]
            input_init_ast = node.operators[2]
            item = node.operators[3]

            input_set._gen_ir()
            if input_init_ast!=None:
                input_init_ast._gen_ir()
            node.storage._gen_ir()
            node.nelem._gen_ir()
            node.eval = node.storage.eval
            
            outer_loop = Loop(0, input_set.nelem.eval, 1, [])
            item.eval = bind(input_set.eval, outer_loop.iterate)

            func_compute = _gen_and_detach_ir(input_func_ast)
            outer_loop.body.extend(func_compute)
            
            node.compute.extend([outer_loop])

            #Clear declaration of the result Set&Tensor
            node.storage.decl.clear()
        
        elif node.op_type== 'filter':
            input_set = node.operators[0]
            input_cond_ast = node.operators[1]
            item = node.operators[2]
            input_set._gen_ir()
            node.storage._gen_ir()
            node.nelem._gen_ir()
            node.eval = node.storage.eval
            res_size_ir = node.nelem.eval

            outer_loop = FilterLoop(0, input_set.nelem.eval, 1, [], None, [])
            item.eval = bind(input_set.eval,  outer_loop.iterate)

            cond_compute = _gen_and_detach_ir(input_cond_ast)
            outer_loop.body.extend(cond_compute)
            outer_loop.cond = input_cond_ast.eval

            lhs = bind(node.eval, res_size_ir) 
            rhs = item.eval
            res_init = Assignment(res_size_ir, 0)
            res_plus = Assignment(res_size_ir, 1, '+')
            if len(input_set._tensor_size())>1:
                pre_loop = None
                for i in range(1, len(input_set._tensor_size())):
                    loop = Loop(0, input_set._tensor_size()[i].eval, 1, [])
                    if pre_loop!=None:
                        pre_loop.body.append(loop)
                    pre_loop=loop
                    lhs = bind(lhs, pre_loop.iterate)
                    rhs = bind(rhs, pre_loop.iterate)
                pre_loop.body.append(Assignment(lhs, rhs))
                outer_loop.cond_body.extend([pre_loop, res_plus])
            else:
                outer_loop.cond_body.extend([Assignment(lhs, rhs), res_plus])

            node.compute = [res_init, outer_loop]
        
        elif node.op_type == 'binary_search':
            item = node.operators[0]
            item._gen_ir()
            input_set = node.operators[1]
            input_set._gen_ir()
            negative = node.operators[2]
            
            node.eval = Scalar(node.dtype)
            node.decl = [Decl(node.eval)]
            if negative.val:
                node.compute = [Assignment(node.eval,  Not(BinarySearch(bind(input_set.eval, 0), 0, input_set.nelem.eval, item.eval)))]
            else:
                node.compute = [Assignment(node.eval,  BinarySearch(bind(input_set.eval, 0), 0, input_set.nelem.eval, item.eval))]
       
        elif node.op_type == 'merge_search':
            input_set = node.operators[0]
            input_set._gen_ir()
            item = node.operators[1]
            item._gen_ir()
        
        elif node.op_type == 'smaller':
            val = node.operators[1]
            item = node.operators[0]
            val._gen_ir()
            item._gen_ir()

            node.eval = Scalar(node.dtype)
            node.decl = [Decl(node.eval)]
            node.compute = [Assignment(node.eval ,Expr(Expr(val.eval, item.eval, '-'), 0, 'bigger'))]
            # node.compute = [Assignment(node.eval ,0)]
        
        elif node.op_type == 'increment':
            input_set = node.operators[0]
            val =  node.operators[1]
            assert(type(input_set.storage)==Var)
            input_set._gen_ir()
            val._gen_ir()

            node.eval = input_set.storage.eval
            node.compute = [Assignment(node.eval, val.eval, '+')]
        
        elif node.op_type == 'retval':
            input_set =  node.operators[0]
            input_set._gen_ir()
            ret_val =  node.operators[1]
            ret_val._gen_ir()
            node.eval = ret_val.eval
            # node.compute=[]
        
        elif node.op_type == 'intersection' or node.op_type == 'difference':
            input_set1 = node.operators[0]
            input_func_ast = node.operators[1]
            item = node.operators[2]
            input_set1._gen_ir()
            item._gen_ir()

            input_set2 = input_func_ast.operators[1]
            input_set2._gen_ir()

            node.storage._gen_ir()
            node.eval = node.storage.eval
            node.nelem._gen_ir()

            if input_func_ast.op_type == 'function_call':
                if node.op_type == 'intersection':
                    func_template = "res_size = SetIntersection(first_arr, first_size, second_arr, second_size, res_arr);"
                else:
                    func_template = "res_size = SetDifference(first_arr, first_size, second_arr, second_size, res_arr);"

                keywords = {'first_arr' : bind(input_set1.eval, Literal(0, 'int')),
                            'first_size': input_set1.nelem.eval,
                            'second_arr' : bind(input_set2.eval, Literal(0, 'int')),
                            'second_size': input_set2.nelem.eval,
                            'res_arr':  bind(node.eval,Literal(0, 'int')),
                            'res_size': node.nelem.eval}
                node.compute.extend([Code(func_template, keywords)])
            
            elif input_func_ast.op_type == 'inline_code':
                outer_loop = Loop(0, input_set1.nelem.eval, 1, [])
                pi_ir = outer_loop.iterate
                pj_ir =  Scalar('int')
                pos_ir =  node.nelem.eval

                if node.op_type == 'intersection':
                    merge_template = \
                        "if(first_smaller_second) continue; \n\
                        else if (first_larger_second) { \n\
                            while(pj_smaller_secondsize && first_larger_second){    \n\
                                pj_increment;                                       \n\
                            } \n\
                        } \n\
                        if(pj_equal_secondsize) break;  \n\
                        if(first_equal_second) { \n\
                            assignment\n\
                            pos_increment; \n\
                            continue; \n\
                        }"
                else:
                    merge_template = \
                        "if(first_smaller_second) { } \n\
                        else if (first_larger_second) { \n\
                            while(pj_smaller_secondsize && first_larger_second){    \n\
                                pj_increment;                                           \n\
                            } \n\
                        } \n\
                        if(pj_equal_secondsize) {  \n\
                            assignment\n\
                            pos_increment; \n\
                            continue; \n\
                        } \n\
                        if(first_equal_second) { \n\
                            pj_increment; \n\
                        }  \n\
                        else{ \n\
                            assignment\n\
                            pos_increment; \n\
                        }"

                keywords = {'first_smaller_second' : Expr(bind(input_set1.eval, pi_ir), bind(input_set2.eval, pj_ir), '<'),
                            'first_larger_second': Expr(bind(input_set1.eval, pi_ir), bind(input_set2.eval, pj_ir), '>'),
                            'first_equal_second' : Expr(bind(input_set1.eval, pi_ir), bind(input_set2.eval, pj_ir), '=='),
                            'pj_smaller_secondsize': Expr(pj_ir,  input_set2.nelem.eval, '<'),
                            'pj_equal_secondsize':  Expr(pj_ir,  input_set2.nelem.eval, '=='),
                            'pj_increment': Expr(pj_ir,  1, '+='),
                            'pos_increment': Expr(pos_ir,  1, '+='),
                            'assignment': Assignment(bind(node.eval, pos_ir), bind(input_set1.eval, pi_ir))
                            }
                
                node.decl.extend([Decl(pj_ir)])  
                outer_loop.body.append(Code(merge_template, keywords))
                node.compute.extend([Assignment(pj_ir, 0), Assignment(pos_ir, 0), outer_loop])

    for d in node.decl:
        d.astnode = node

    if type(node) == SetOp:
        for s in node.compute:
            s.astnode = node

    return node
