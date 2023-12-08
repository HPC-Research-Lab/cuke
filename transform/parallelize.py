from helpers import ASGTraversal, IRTraversal, get_obj, replace_all_ref
from asg import *
from ir import *
from transform.fuse import fuser, basic_rule
import codegen




def _get_ref_idx(stmt, v):
    def _get_scalar_idx(idx):
        assert type(idx) == Indexing
        def action(s, res):
            if type(s) == Scalar:
                res.append(s)
            elif type(s) == Ndarray:
                return [False]

            return [True, True, True, True, True]

        t = IRTraversal(action)
        res = t(idx)
        return res

    def action(s, res):
        if type(s) == Indexing:
            if get_obj(s).dobject_id == v.dobject_id:
                if len(res) == 0:
                    res.append(_get_scalar_idx(s))
                return [False, False]
            else:
                return [False, True]
        elif type(s) in (Scalar, Ndarray):
            if s.dobject_id == v.dobject_id:
                if len(res) == 0:
                    res.append([])

        return [True, True, True, True, True]


    t = IRTraversal(action)
    res = t(stmt)
    if len(res) > 0:
        return res[0]
    else:
        return None

def _set_loop_levels(nodeir, num_procs):
    def action(ir, res):
        if type(ir) in (list, tuple):
            return [True]
        if type(ir) == Loop:
            if not 'plevel' in ir.attr and not 'nprocs' in ir.attr:
                ir.attr['plevel'] = 0
                ir.attr['nprocs'] = [(num_procs[0], ir)]
                ir.attr['tid'] = Scalar('int', 'tid0')
            for stmt in ir.body:
                if type(stmt) == Loop:
                    if 'plevel' in ir.attr and ir.attr['plevel'] + 1 < len(num_procs):
                        stmt.attr['plevel'] = ir.attr['plevel'] + 1
                        stmt.attr['nprocs'] = ir.attr['nprocs'] + [(num_procs[stmt.attr['plevel']], stmt)]
                        stmt.attr['tid'] = Scalar('int', f"tid{stmt.attr['plevel']}")
                    else:
                        stmt.attr['nprocs'] = ir.attr['nprocs']
            return [False, False, False, False, True]

        return [False, False, False, False, False]

    t = IRTraversal(action)
    t(nodeir)

def _isolate_vars(node):

    def find_all_vars(node, res):
        res.extend([d.dobject for d in node.decl])

    t1 = ASGTraversal(find_all_vars)
    vars = t1(node)

    def find_data_races(ir, new_vars):
        if len(new_vars) == 0:
            new_vars.append({})
        if type(ir) == Loop:
            if 'nprocs' in ir.attr:
                for stmt in ir.body:
                    if type(stmt) == Assignment:
                        for v in vars:
                            if v not in new_vars[0]:
                                idx = _get_ref_idx(stmt.lhs, v)
                                if idx != None:
                                    ext_size = []
                                    loop_info = []
                                    for l in ir.attr['nprocs']:
                                        indexed = False
                                        for ii in idx:
                                            if ('output_axis' in l[1].attr and l[1].attr['output_axis'] == ii.attr['loop'].attr['output_axis']) or ii.dobject_id == l[1].iterate.dobject_id:
                                                indexed = True
                                                break
                                        if not indexed:
                                            ext_size.append(l[0])
                                            loop_info.append(l[1])
                                    if len(ext_size) > 0:
                                        new_vars[0][v] = (Ndarray(v.dtype, ext_size + v.size), loop_info)

        return [True, True, True, True, True]

    t2 = IRTraversal(find_data_races)
    new_vars = t2(node.compute)[0]
    return new_vars

def parallelize(node, num_procs=[16]):

    def action(n, res):
        if len(res) == 0:
            res.append({})
        if len(n.compute) != 0:
            _set_loop_levels(n.compute, num_procs)
            res[0].update(_isolate_vars(n))

    t = ASGTraversal(action)
    to_replace = t(node)[0]

    def replace_decls(n, res):
        decl = []
        for d in n.decl:
            v = d.dobject.dobject_id
            replace_with = None
            for n in to_replace:
                if n.dobject_id == v:
                    replace_with = to_replace[n][0]
                    break
            if replace_with != None:
                decl.append(Decl(replace_with))
            else:
                decl.append(d)
        n.decl = decl

    t3 = ASGTraversal(replace_decls)
    t3(node)

    def replace_refs(n, res):
        if len(n.compute) > 0:
            for s in to_replace:
                new_var = to_replace[s][0]
                for l in to_replace[s][1]:
                    new_var = Indexing(new_var, l.attr['tid'])
                replace_all_ref(n.compute, s, new_var)

    t4 = ASGTraversal(replace_refs)
    t4(node)

    return node


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
    parallelize(f.fuse(ir1))
    code = codegen.cpu.print_cpp(ir1)
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
    ir1 = parallelize(f.fuse(ir1))
    code = codegen.cpu.print_cpp(ir1)
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
    ir1 = parallelize(f.fuse(ir1))
    print(codegen.cpu.print_cpp(ir1))



if __name__ == "__main__":
    # test1()
    # test2()
    test3()