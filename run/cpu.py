from torch.utils.cpp_extension import load

def compile_and_run(code, *args):
    f = open('run/.tmp/cpu_code.cpp', 'w')
    f.write(code)
    f.close()
    module = load(name='module', sources=['run/.tmp/cpu_code.cpp'], extra_cflags=['-O9', '-fopenmp'])
    return module.run(*args)

def run(*args):
    module = load(name='module', sources=['run/.tmp/cpu_code.cpp'])
    return module.run(*args)