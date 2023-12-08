from asg import *
from apps import compression


def truncate(data, nbits):
    return Encoder('truncate', data, nbits)

class Encoder(Tensor):
    def __init__(self, op_type, *operators):
        self.op_type = op_type
        self.operators = []
        for opr in operators:
            self.operators.append(opr)
            if isinstance(opr, ASTNode):
                opr.ref_by.append(self)

        if type(self.operators[1]) == int:
            self.operators[1] = Const(self.operators[1], 'int')
        assert len(self.operators[0].ref_size) == 1
        ref_size = [self.operators[1]]
        fix_size = []

        name = f'{op_type}_' + '_'.join([op.name if hasattr(op, 'name') else '' for op in self.operators])
        super().__init__(name, ref_size, 'int', fix_size)

    def _gen_ir(self):
        return compression.asg2ir.gen_ir(self)


