import torch_mlir
import argparse
from models import LeNet, RNN, LSTM, GRU
from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend
from torch_mlir_e2e_test.configs.utils import (
    recursively_convert_to_numpy,
    recursively_convert_from_numpy,
)
from obfuscation_sequence import LeNet_obfuscation, RNN_obfuscation, LSTM_obfuscation, GRU_obfuscation
import time
import sys

tu = TestUtils()

def get_module_size(module):
    size = sys.getsizeof(module)
    return size

i = 0
jit_module = ""

start_time = time.time()

def runTest(filter, order):
    global i
    global jit_module
    if (i == 0):
        i += 1
        if(filter == "LeNet"):
            model = LeNet()
        elif(filter == "RNN"):
            model = RNN(10, 20, 18)
        elif(filter == "LSTM"):
            model = LSTM(10, 20, 18)
        elif(filter == "GRU"):
            model = GRU(10, 20, 18)
        
        if (filter == "LeNet"):
            module = torch_mlir.compile(
                    model, tu.rand(1, 1, 28, 28), output_type="torch", use_tracing=True, ignore_traced_shapes=True
                )
        else:
            module = torch_mlir.compile(
                    model, tu.rand(3, 1, 10), output_type="torch", use_tracing=True, ignore_traced_shapes=True
                )    
        name = order
        passes = LeNet_obfuscation.get(name)
        torch_mlir.compiler_utils.run_pipeline_with_repro_report(
            module,
            f"builtin.module(func.func({passes}))",
            name,
        )
            
        torch_mlir.compiler_utils.run_pipeline_with_repro_report(
            module,
            "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
            "Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
        )
        backend = refbackend.RefBackendLinalgOnTensorsBackend()
        compiled = backend.compile(module)
        jit_module = backend.load(compiled)
        # global i
        # if (i == 0):
        #     i += 1
        #     size = get_module_size(module)
        #     print("Compiled module size: ", size, "bytes")
    for i in range(0, 100):
        if(filter == "LeNet"):
            inputs=tu.rand(1, 1, 28, 28)
        else:
            inputs = tu.rand(3, 1, 10)
        numpy_inputs = recursively_convert_to_numpy(inputs)
        outputs = jit_module.forward(*numpy_inputs)
        outputs = recursively_convert_from_numpy(outputs)

def _get_argparse():
    parser = argparse.ArgumentParser(description="Run obfuscation tests.")
    parser.add_argument(
        "-f",
        "--filter",
        default=".*",
        help="""
Regular expression specifying which tests to include in this run.
""",
    )
    parser.add_argument(
        "-s",
        "--sequential",
        default=False,
        action="store_true",
        help="""Run tests sequentially rather than in parallel.
This can be useful for debugging, since it runs the tests in the same process,
which make it easier to attach a debugger or get a stack trace.""",
    )
    parser.add_argument(
        "-o",
        "--order",
        default="1",
        help="""Run the specified pass.
This can be useful for debugging, since it runs the tests in the same process,
which make it easier to attach a debugger or get a stack trace.""",
    )
    return parser

if __name__ == "__main__":
    args = _get_argparse().parse_args()
    for i in range(0, 1000):
        runTest(args.filter, args.order)

end_time = time.time()
print(f"run time is {((end_time - start_time) / 1000):.6f}\n")
