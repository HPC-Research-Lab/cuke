from core.ir import *

class ThreadMapping:
    loop_id = 0
    def __init__(self, start, end, step, body: list):
        self.lid = ThreadMapping.loop_id
        ThreadMapping.loop_id += 1
        self.start = start
        self.end = end
        self.step = step
        self.body = body
        self.iterate = Scalar('int', f'_l{self.lid}')

class ThreadIdy:
    def __init__(self):
        pass

class ThreadIdx:
    def __init__(self):
        pass

class Sync:
    def __init__(self):
        pass