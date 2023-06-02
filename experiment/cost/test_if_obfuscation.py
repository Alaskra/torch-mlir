import torch_mlir
import argparse
from models import LeNet, RNN, LSTM, GRU
from torch_mlir_e2e_test.framework import TestUtils
from obfuscation_sequence import LeNet_obfuscation, RNN_obfuscation, LSTM_obfuscation, GRU_obfuscation


tu = TestUtils()

def runTest(filter, order):
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
    str1 = module.operation.get_asm(large_elements_limit=10)
    print(str1)

    name = order
    if (filter == "LeNet"):
        passes = LeNet_obfuscation.get(name)
    elif (filter == "RNN"):
        passes = RNN_obfuscation.get(name)
    elif (filter == "LSTM"):
        passes = LSTM_obfuscation.get(name)
    elif (filter == "GRU"):
        passes = GRU_obfuscation.get(name)

    torch_mlir.compiler_utils.run_pipeline_with_repro_report(
        module,
        f"builtin.module(func.func({passes}))",
        name,
    )
    str2 = module.operation.get_asm(large_elements_limit=10)
    print(str2)
    if (str1 == str2):
        print("obfuscation false!")
    else:
        print("obfuscation success!")

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
    runTest(args.filter, args.order)
