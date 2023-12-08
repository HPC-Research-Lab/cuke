from cset.ast import *
from cset.ir import *
import helpers

# if input_fun:
#     # outer_loop.body.extend([Assignment(ret.eval, 0)])
#     # 

# else:
#     func_ret = input_set
#     # node.operators.append(func_ret)
#     node.eval = node.storage.eval
#     node.storage.ref_size = [node.storage._size()[0]] 
#     node.nelem[0].eval = input_set.nelem[0].eval
#     node.nelem = node.storage.ref_size 
#     node.storage._gen_ir()
#     node.eval = node.storage.eval
#     func_ret.eval = input_set.eval


# if input_cond:
#     cond_ret, cond_ret_decl, cond_ret_compute = _extend_ast(input_cond, item)
#     outer_loop.cond = cond_ret.eval

#     res_size = Scalar('int', 'nelem'+node.name, val=0)
    
#     assignment = Assignment(IndexOffset(node.eval, res_size), func_ret.eval)
#     res_add_one = Assignment(res_size, 1, '+')
#     outer_loop.cond_body.extend([assignment, res_add_one])

#     outer_loop.condition = cond_ret.eval
#     outer_loop.body.extend(cond_ret_compute)
#     node.decl.extend(cond_ret_decl)
# else:

def ExtractTensorIR(ret):
    def action(node, res):
        if type(node) == Var or type(node) == One or type(node) == Zero or type(node) == Ones or type(node) == Zeros or type(node) == Tensor:
            res.extend(node.decl)
            # node.decl.clear()
        elif type(node) == TensorOp:
            res.extend(node.decl)
            res.extend(node.compute)
            # node.decl.clear()
            # node.compute.clear()

    t = helpers.ASGTraversal(action)
    ret_ir = t(ret)
    return ret_ir

def bind2(arr: (Ndarray, Indexing), index, fix_ref=False):
    # fix_ref == True means index at current position instead of the first unbind
    if type(arr) == Ndarray or index == None or fix_ref:
        return Indexing(arr, idx=index)
    else:
        ref_chain = [arr]
        while (type(ref_chain[-1].dobject) != Ndarray):
            ref_chain.append(ref_chain[-1].dobject)
        for ref in ref_chain[::-1]:
            if ref.index == None:
                ref.index = index
                return arr
        return Indexing(arr, idx=index)

def RefBind(arr: Ref, index):
    if type(arr) == Ref:
        return Index(arr, idx=index)

def IndexOffset(arr, index):
    if type(arr) == Ndarray:
        return Indexing(arr, idx=index)
    elif type(arr) == Indexing:
        if type(arr.idx) == Indexing:
            if type(arr.idx.dobject) == Slice:
               arr.idx.idx = index
        else:
             arr.idx = Expr(arr.idx, idx, '+')
        return arr

def _extend_ast(callback_func, item):
    ret = callback_func(item)
    ret._gen_ir()

    def action(node, res):
        if node.valid == True:
            if type(node) == Var or type(node) == One or type(node) == Zero or type(node) == Ones or type(node) == Zeros or type(node) == Tensor:
                res.extend(node.decl)
                node.valid = False
            elif type(node) == TensorOp:
                res.extend(node.decl)
                res.extend(node.compute)
                node.valid = False
            elif type(node) == Set:
                res.extend(node.decl)
                node.valid = False
            elif type(node) == SetOp:
                res.extend(node.decl)
                res.extend(node.compute)
                node.valid = False

    t = helpers.ASGTraversal(action)
    ret_ir = t(ret)
    ret_decl = []
    ret_compute = []

    for ir in ret_ir:
        if type(ir) == Decl:                       
            ret_decl.extend([ir])
        else:
            ret_compute.extend([ir])
    return ret, ret_decl, ret_compute

def gen_ir(node):
    if type(node) == Set:
        helpers.get_ir_of_size(node.nelem)
        node.storage._gen_ir()
        node.eval = node.storage.eval

    elif type(node) == SetOp:        
        if node.op_type == 'apply':
            input_set = node.operators[0]
            input_func = node.operators[1]
            input_cond = node.operators[2]
            input_axis = node.operators[3]
            input_decl_ret = node.operators[4]
            item = node.operators[5]
            
            input_set._gen_ir()
            input_axis._gen_ir()

            
            if input_func and input_cond:
                pass

            elif not input_func and input_cond:

                outer_loop = FilterLoop(0, input_set.nelem[input_axis.eval.val].eval, 1, [], None, [])

                item.eval = input_set.eval
                # for i in range(input_axis.eval.val):
                #     item.eval = IndexOffset(item.eval, 0)
                item.eval = IndexOffset(item.eval,  outer_loop.iterate)
                
                node.storage._gen_ir()
                node.eval = node.storage.eval

                cond_ret, cond_ret_decl, cond_ret_compute = _extend_ast(input_cond, item)

                node.operators.append(cond_ret)

                outer_loop.body.extend(cond_ret_compute)
                outer_loop.cond = cond_ret.eval

                res_size = Scalar('int', 'nelem_'+node.name, val=0)
                assignment = Assignment(IndexOffset(node.eval, res_size), item.eval)
                res_add_one = Assignment(res_size, 1, '+')
                outer_loop.cond_body.extend([assignment, res_add_one])

                node.decl.extend(cond_ret_decl)
                node.decl.extend([Decl(res_size)])  
                node.compute = [Assignment(res_size, 0), outer_loop]

                node.storage.ref_size = [node.storage._size()[0]] + cond_ret._size()
                node.storage.fix_size = []
                node.nelem = [input_set.nelem[0]] + cond_ret._size()
                input_set.nelem[0].eval = res_size

            elif input_func and not input_cond:
                outer_loop = Loop(0, input_set.nelem[input_axis.eval.val].eval, 1, [])

                item.eval = input_set.eval
                # for i in range(input_axis.eval.val):
                #     item.eval = IndexOffset(item.eval, 0)
                item.eval = IndexOffset(item.eval,  outer_loop.iterate)

                func_ret, func_decl, func_compute = _extend_ast(input_func, item)

                node.dtype = func_ret.dtype
                node.operators.append(func_ret)
                
                node.storage.ref_size = [node.storage._size()[0]] + func_ret._size()
                node.storage.fix_size = []
                node.nelem = [input_set.nelem[0]] + func_ret._size()
                
                node.storage._gen_ir()

                outer_loop.body.extend(func_compute)

                node.eval = node.storage.eval

                if input_decl_ret:
                    node.decl.extend([Decl(node.eval)])
                    # TODO: instead of copying output to node.eval, we can replace the lhs of assignment in ret_compute
                    outer_loop.body.extend([Assignment(bind2(node.eval , outer_loop.iterate), func_ret.eval)])
                else:
                    node.storage.decl.clear()
                
                node.decl.extend(func_decl)
                node.compute = [outer_loop]
        
        elif node.op_type == 'search':
            input_set = node.operators[0]
            input_set._gen_ir()
            item = node.operators[1]
            item._gen_ir()
            search_start = 0
            search_end = input_set.nelem[0].eval

            node.eval = Scalar(node.dtype, node.name)
            node.decl = [Decl(node.eval)]
            node.compute = [Assignment(node.eval,  Search(IndexOffset(input_set.eval, 0), search_start, search_end, item.eval))]
        
        elif node.op_type == 'difference':
            input_set = node.operators[0]
            input_set._gen_ir()
            item = node.operators[1]
            item._gen_ir()
            search_start = 0
            search_end = input_set.nelem[0].eval

            node.eval = Scalar(node.dtype, node.name)
            node.decl = [Decl(node.eval)]
            node.compute = [Assignment(node.eval,  Not(Search(IndexOffset(input_set.eval, 0), search_start, search_end, item.eval)))]
        
        elif node.op_type == 'smaller':
            val = node.operators[0]
            item = node.operators[1]
            val._gen_ir()
            item._gen_ir()

            node.eval = Scalar(node.dtype, node.name)
            node.decl = [Decl(node.eval)]
            node.compute = [Assignment(node.eval ,Expr(Expr(val.eval, item.eval, '-'), 0, '>'))]
        
        elif node.op_type == 'addone':
            node.operators[0]._gen_ir()
            node.eval = node.operators[0].storage.eval
            node.compute = [Assignment(node.eval, 1, '+')]

        elif node.op_type == 'nelem':
            input_set = node.operators[0]
            input_set._gen_ir()
            node.eval = input_set.nelem[0].eval 

        elif node.op_type == 'sum':
            input_set = node.operators[0]
            input_storage = input_set.storage
            input_set._gen_ir()
            
            node.eval = Scalar(node.dtype, node.name, val=0)
            node.decl = [Decl(node.eval)]

            sum_loop = Loop(0, input_set.nelem[0].eval, 1, [])
            sum_loop.body.append(Assignment(node.eval, bind2(input_storage.eval, sum_loop.iterate), '+'))

            node.compute.extend([sum_loop])
        
        elif node.op_type == 'ret_val':
            program_body =  node.operators[0]
            program_body._gen_ir()
            ret_val =  node.operators[1]
            ret_val._gen_ir()
            node.eval = ret_val.eval

        # elif node.op_type == 'filter':
        #     input_set = node.operators[0]
        #     input_set._gen_ir()

        #     node.storage._gen_ir()
        #     node.eval = node.storage.eval
            
        #     cond_obj = node.operators[1]
        #     item = node.operators[2]

        #     ret = cond_obj(item)
            
        #     #把filter_loop的类型改成FilterLoop
        #     filter_loop = Loop(0, input_set.nelem.eval, 1, [])
        #     item.eval = IndexOffset(input_set.eval, filter_loop.iterate)

        #     ret._gen_ir()

        #     def action(node, res):
        #         if type(node) == Var or type(node) == One or type(node) == Zero or type(node) == Ones or type(node) == Zeros or type(node) == Tensor:
        #             res.extend(node.decl)
        #             node.decl.clear()
        #         elif type(node) == TensorOp:
        #             res.extend(node.decl)
        #             res.extend(node.compute)
        #             node.decl.clear()
        #             node.compute.clear()
        #         elif type(node) == Set:
        #             res.extend(node.decl)
        #             node.decl.clear()
        #         elif type(node) == SetOp:
        #             res.extend(node.decl)
        #             res.extend(node.compute)
        #             node.decl.clear()
        #             node.compute.clear()

        #     t = helpers.Traversal(action)
        #     ret_ir = t(ret)
        #     ret_decl = []
        #     ret_compute = []

        #     for ir in ret_ir:
        #         if type(ir) == Decl:                       
        #             ret_decl.extend([ir])
        #         else:
        #             ret_compute.extend([ir])
        #     node.operators.append(ret)

        #     filter_loop.body.extend(ret_compute)

        #     cond_ir = Condition(ret.eval, [])
        #     res_size = Scalar('int', 'nelem'+node.name, val=0)
        #     assignment = Assignment(IndexOffset(node.eval, res_size), item.eval)
        #     res_add_one = Assignment(res_size, 1, '+')
        #     cond_ir.body.extend([assignment, res_add_one])
            
        #     filter_loop.body.extend([cond_ir])

        #     node.nelem.eval = res_size
           
        #     node.decl.extend([Decl(node.nelem.eval)])
        #     node.decl.extend(ret_decl)

        #     filter_loop.body.append([assignment])
        #     node.compute = [filter_loop]

        #     # node.compute.append(for_loop_assignment)
            
            # node.compute = Search(input_set, search_start, search_end, item)
            # node.eval =  Var(f'search_res_of_{input_set.storage.name}', 'int', False)
            # node.decl = Decl(node.eval)
           # node.compute = [Condition(Search(input_set.eval, search_start, search_end, item.eval), [])]
            # node.eval = Var
            # node.decl.extend([Decl(node.eval)])


        # if node.op_type == 'intersect':
        #     node.operators[0]._gen_ir()
        #     node.operators[1]._gen_ir()
        #     node.storage._gen_ir()
        #     node.nelem._gen_ir()
        #     node.eval = Ref(node.storage.eval)
        #     node.decl = [Decl(node.eval)]

        #     first = node.operators[0].eval
        #     second = node.operators[1].eval

        #     first_size = node.operators[0].nelem.eval
        #     second_size = node.operators[1].nelem.eval


            # for(i=0; i<input.size; i++){
            #     if(condition(xxx)){
            #         output[i] = input[i]
            #     }
            # }

            # A.filter(is_in)
            # Loop()

            # filter_ir = Filter(first, None,  node.eval)
            # filter_ir.condition = Search(second, 0, second_size,  Index(first, filter_ir.iterate))

            # for
            # if(condition){
            #     out
            # }

            # intersect = Intersect(first, first_size, second, second_size, node.eval)
            # for_loop = Loop(0, first_size, 1, [])
            # res = Index(node.eval, for_loop.iterate)
            # for_loop.body.extend([Assignment(Index(node.eval, for_loop.iterate), Search(second, 0, second_size, Index(first, for_loop.iterate)), '+')])

            # node.compute = [Assignment(node.nelem.eval, intersect)]
            # node.compute = [filter_ir]
        
        # elif node.op_type == 'apply':
        #     pass

        # elif node.op_type == 'sum':
        #     node.operators[0]._gen_ir()
        #     node.storage._gen_ir()
        #     node.nelem._gen_ir()
        #     node.eval = Scalar(node.dtype)
        #     node.decl = [Decl(node.eval)]

        #     for_loop = Loop(0, node.operators[0].nelem.eval, 1, [])
        #     for_loop.body.extend([Assignment(node.eval, Index(node.operators[0].storage.eval, for_loop.iterate), '+')])

        #     node.compute = [for_loop]
        
        # elif node.op_type == 'index':
        #     node.operators[0]._gen_ir()
        #     node.operators[1]._gen_ir()

    return node
