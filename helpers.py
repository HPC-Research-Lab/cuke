import ir
import asg
import asg2ir

def same_object(a, b):
    if isinstance(a, ir.DObject) and isinstance(b, ir.DObject):
        return a.dobject_id == b.dobject_id
    return False

def is_int_var(v):
    return isinstance(v, asg.Tensor) and v.dtype == 'int' and len(v.ref_size) == 0


def is_scalar(v):
    return isinstance(v, int | float) or (isinstance(v, asg.Tensor) and len(v.ref_size) == 0)


def is_1d_tensor(v):
    return isinstance(v, asg.Tensor) and len(v.ref_size) == 1


def is_1dint_tensor(v):
    return is_1d_tensor(v) and v.dtype == 'int'


def eval_const_expr(expr):
    def _eval_expr(e):
        if type(e) == asg.TensorOp and (e.op_type in asg.arith_op):
            lhs = eval_const_expr(e.operators[0])
            if lhs != None:
                rhs = eval_const_expr(e.operators[1])
                if rhs != None:
                    match e.op_type:
                        case 'add':
                            return lhs + rhs
                        case 'sub':
                            return lhs - rhs
                        case 'mul':
                            return lhs * rhs
                        case 'floordiv':
                            return lhs // rhs
                        case 'truediv':
                            return lhs / rhs
                        case _:
                            return None
                else:
                    return None
            else:
                return None
        elif type(e) == asg.Const:
            return e.val
        else:
            return None

    v = _eval_expr(expr)
    if v != None:
        return v
    else:
        return expr


def has_same_iteration_space(l1, l2):
    return has_same_value(l1.start, l2.start) and has_same_value(l1.end, l2.end) and has_same_value(l1.step, l2.step)

def has_same_value(e1, e2):
    if type(e1) != type(e2):
        return False
    elif type(e1) == asg.Var or type(e1) == asg.Tensor:
        return e1.id == e2.id
    elif type(e1) == asg.Const:
        if e1.dtype == 'int' and e2.dtype == 'int':
            return e1.val == e2.val
        elif e1.dtype == 'slice' and e2.dtype == 'slice':
            return has_same_value(e1.val.start, e2.val.start) and has_same_value(e1.val.stop,
                                                                                 e2.val.stop) and has_same_value(
                e1.val.step, e2.val.step)
        else:
            return False
    elif type(e1) == asg.TensorOp:
        if e1.op_type != e2.op_type:
            return False
        elif e1.op_type in asg.arith_op:
            return has_same_value(e1.operators[0], e2.operators[0]) and has_same_value(e2.operators[1], e2.operators[1])
        else:
            if len(e1.operators) != len(e2.operators):
                return False
            else:
                for i in range(len(e1.operators)):
                    if not has_same_value(e1.operators[i], e2.operators[i]):
                        return False
    return True


def is_same_size(s1, s2):
    if (len(s1) != len(s2)):
        return False
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            if type(s1[i]) == type(s2[i]):
                return has_same_value(s1[i], s2[i])
            else:
                v1 = eval_const_expr(s1[i])
                v2 = eval_const_expr(s2[i])
                return v1 == v2
    return True


def prefix_match_size(s1, s2):
    length = min(len(s1), len(s2))
    for i in range(length):
        if s1[i] != s2[i]:
            if type(s1[i]) == type(s2[i]):
                return has_same_value(s1[i], s2[i])
            else:
                return False
    return True

class ASGTraversal:

    def __init__(self, action):
        self.action = action

    def _post_traverse(self, node, visited, res):
        if not isinstance(node, asg.ASTNode):
            return
        if node in visited:
            return
        else:
            visited.add(node)

        if type(node) == asg.Var:
            self.action(node, res)
        elif type(node) == asg.Const:
            if node.dtype == 'slice':
                self._post_traverse(node.val.start, visited, res)
                self._post_traverse(node.val.stop, visited, res)
                self._post_traverse(node.val.step, visited, res)
        elif type(node) == asg.Tensor:
            for s in node.ref_size:
                self._post_traverse(s, visited, res)
            self.action(node, res)
        elif type(node) == asg.TensorOp:
            for s in node.ref_size:
                self._post_traverse(s, visited, res)
            for c in node.operators:
                self._post_traverse(c, visited, res)
            self.action(node, res)


    def __call__(self, ast):
        visited = set()
        res = []
        self._post_traverse(ast, visited, res)
        return res

def get_input_nodes(node):
    def action(n, res):
        if type(n) == asg.Var or type(n) == asg.Tensor:
            if n.attr['is_arg']:
                res.append([n.eval.name(), n])

    t = ASGTraversal(action)
    return dict(t(node))

def get_ir_of_size(size):
    ir_size = []
    for s in size:
        assert isinstance(s, asg.ASTNode)
        asg2ir.gen_ir(s)
        if 'dynamic_size' in s.attr:
            s.eval.attr['dynamic_size'] = s.attr['dynamic_size']
        ir_size.append(s.eval)
    return ir_size

def collect_ir(ast, stmt):
    def action(node, res):
        if isinstance(node, asg.Tensor):
            res.extend(node.decl)
            res.extend(node.compute)

    t = ASGTraversal(action)
    stmt.extend(t(ast))


def new_op(func):
    def wrapper_func(*args, **kwargs):
        _res = func(*args, **kwargs)
        _res.attr[func.__name__] = True
        return _res
    return wrapper_func


def get_obj(stmt: (ir.Indexing, ir.Scalar)):
    obj = stmt
    while hasattr(obj, 'dobject'):
        obj = obj.dobject
    return obj

def get_val(stmt):
    if type(stmt) == ir.Literal:
        return stmt.val
    elif type(stmt) in (int, float):
        return stmt
    else:
        return stmt


class IRTraversal:

    def __init__(self, action):
        self.action = action

    def _preorder_traverse(self, stmt, res):
        if type(stmt) == list or type(stmt) == tuple:
            cond = self.action(stmt, res)
            if cond[0]:
                for l in stmt:
                    self._preorder_traverse(l, res)
        elif isinstance(stmt, ir.Loop):
            cond = self.action(stmt, res)
            if cond[0]:
                self._preorder_traverse(stmt.start, res)
            if cond[1]:
                self._preorder_traverse(stmt.end, res)
            if cond[2]:
                self._preorder_traverse(stmt.step, res)
            if cond[3]:
                self._preorder_traverse(stmt.body, res)
            if type(stmt) == ir.FilterLoop and cond[4]:
                self._preorder_traverse(stmt.cond_body, res)
        elif type(stmt) == ir.Expr:
            cond = self.action(stmt, res)
            if cond[0]:
                self._preorder_traverse(stmt.left, res)
            if cond[1]:
                self._preorder_traverse(stmt.right, res)
        elif type(stmt) == ir.Assignment:
            cond = self.action(stmt, res)
            if cond[0]:
                self._preorder_traverse(stmt.lhs, res)
            if cond[1]:
                self._preorder_traverse(stmt.rhs, res)
        elif type(stmt) == ir.Ndarray:
            cond = self.action(stmt, res)
            if cond[0]:
                self._preorder_traverse(stmt.size, res)
        elif type(stmt) == ir.Scalar:
            self.action(stmt, res)
        elif type(stmt) == ir.Indexing:
            cond = self.action(stmt, res)
            if cond[0]:
                self._preorder_traverse(stmt.dobject, res)
            if cond[1]:
                self._preorder_traverse(stmt.idx, res)
        elif type(stmt) == ir.Slice:
            cond = self.action(stmt, res)
            if cond[0]:
                self._preorder_traverse(stmt.start, res)
            if cond[1]:
                self._preorder_traverse(stmt.stop, res)
            if cond[2]:
                self._preorder_traverse(stmt.step, res)
        elif type(stmt) == ir.Math:
            cond = self.action(stmt, res)
            if cond[0]:
                self._preorder_traverse(stmt.val, res)
        elif type(stmt) == ir.Code:
            cond = self.action(stmt, res)
            if cond[0]:
                self._preorder_traverse(stmt.output[1], res)
            if cond[1]:
                for k in stmt.inputs:
                    self._preorder_traverse(stmt.inputs[k], res)

    def __call__(self, stmt):
        res = []
        self._preorder_traverse(stmt, res)
        return res


def rebind_iterate(stmt, old, new):
    def action(s, res):
        if type(s) == ir.Indexing and type(s.idx) in (ir.Scalar, ir.Literal):
            if s.idx.dobject_id == old.dobject_id:
                s.idx = new
        return [True, True, True, True, True]

    t = IRTraversal(action)
    t(stmt)

def replace_all_ref(stmt, old, new):
    def action(s, res):
        match s.__class__.__name__:
            case 'Loop':
                if same_object(s.start, old):
                    s.start = new
                if same_object(s.end, old):
                    s.end = new
                if same_object(s.step, old):
                    s.step = new
            case 'FilterLoop':
                if same_object(s.cond, old):
                    s.cond = new
                if same_object(s.start, old):
                    s.start = new
                if same_object(s.end, old):
                    s.end = new
                if same_object(s.step, old):
                    s.step = new
            case 'Expr':
                if same_object(s.left, old):
                    s.left = new
                if same_object(s.right, old):
                    s.right = new
            case 'Assignment':
                if same_object(s.lhs, old):
                    s.lhs = new
                if same_object(s.rhs, old):
                    s.rhs = new
            case 'Indexing':
                if same_object(s.dobject, old):
                    s.dobject = new
                if same_object(s.idx, old):
                    s.idx = new
            case 'Slice':
                if same_object(s.start, old):
                    s.start = new
                if same_object(s.stop, old):
                    s.stop = new
                if same_object(s.step, old):
                    s.step = new
            case 'Math':
                if same_object(s.val, old):
                    s.val = new
            case 'Code':
                if same_object(s.output[1], old):
                    s.output = (s.output[0], new)
                for k in s.inputs:
                    if s.inputs[k] == old:
                        s.inputs[k] = new
        return [True, True, True, True, True]

    t = IRTraversal(action)
    t(stmt)

def ir_uses(stmt, data, avoid = []):
    def action(s, res):
        if s in avoid:
            return [False, False, False, False, False]
        if s == data or same_object(s, data):
            if len(res) == 0:
                res.append(True)
            else:
                res[0] = True
        if type(s) == ir.Assignment:
            if s.op != None:
                return [True, True]
            else:
                return [False, True]
        else:
            return [True, True, True, True, True]

    t = IRTraversal(action)
    r = t(stmt)
    if len(r) > 0 and r[0] == True:
        return True
    else:
        return False


def ir_defs(stmt, data):
    def action(s, res):
        if s == data or (isinstance(s, ir.DObject) and s.dobject_id == data.dobject_id):
            if len(res) == 0:
                res.append(True)
            else:
                res[0] = True
        if type(s) == ir.Assignment:
            return [True, False]
        else:
            return [True, True, True, True, True]

    t = IRTraversal(action)
    r = t(stmt)
    if len(r) > 0 and r[0] == True:
        return True
    else:
        return False


def remove_decl(node, item):
    def action(n, res):
        n.decl = [d for d in n.decl if d.dobject != item]
    t = ASGTraversal(action)
    t(node)


def _remove_compute_of_node(loop, node):
    body = []
    for stmt in loop.body:
        if not ('asgnode' in stmt.attr and stmt.attr['asgnode'] == node):
            if type(stmt) == ir.Loop:
                _remove_compute_of_node(stmt, node)
            body.append(stmt)
    loop.body = body



def clear_compute(node):
    node.compute.clear()
    if 'scope' in node.attr:
        scope = node.attr['scope']
        compute = []
        for stmt in scope.compute:
            if not ('asgnode' in stmt.attr and stmt.attr['asgnode'] == node):
                if type(stmt) == ir.Loop:
                    _remove_compute_of_node(stmt, node)
                compute.append(stmt)
        scope.compute = compute


def ir_find_defs(stmt, data):
    def action(s, res):
        if type(s) == ir.Assignment and get_obj(s.lhs).dobject_id == data.dobject_id:
            res.append(s)
        elif type(s) == ir.Code and get_obj(s.output[1]).dobject_id == data.dobject_id:
            res.append(s)

        return [True, True, True, True, True]

    t = IRTraversal(action)
    return t(stmt)


def asg_find_defs(node, data):
    def action(n, res):
        if type(n) == asg.TensorOp:
            res.extend(ir_find_defs(n.compute, data))

    t = ASGTraversal(action)
    return t(node)


def get_vars(stmt):
    def action(s, res):
        if type(s) in (ir.Scalar, ir.Ndarray):
            res.append(s)

        return [True, True, True, True, True]

    t = IRTraversal(action)
    return t(stmt)


def ir_find_uses(stmt, data):
    def action(s, res):
        if type(s) == ir.Assignment and data in get_vars(s.rhs):
            res.append(s)
        elif type(s) == ir.Code:
            uses = False
            for x in s.inputs:
                if s.inputs[x] == data:
                    uses = True
                    break
            if uses:
                res.append(s)

        return [True, True, True, True, True]

    t = IRTraversal(action)
    return t(stmt)


def remove_defchain(stmt, stmts):

    all_removed = []
    def _remove_assigns(st, assigns):
        to_remove = []

        for s in assigns:
            if type(s) == ir.Assignment:
                uses = ir_uses(st, get_obj(s.lhs), all_removed)
                if not uses:
                    s.attr['invalid'] = True
                    all_removed.append(s)
                    use_vars = get_vars(s.rhs)
                    for v in use_vars:
                        to_remove.extend(ir_find_defs(st, v))
        if len(to_remove) > 0:
            _remove_assigns(st, to_remove)

    _remove_assigns(stmt, stmts)




