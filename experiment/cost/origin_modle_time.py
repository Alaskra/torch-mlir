import argparse
from models import LeNet, RNN, LSTM, GRU
from torch_mlir_e2e_test.framework import TestUtils
import time
from functools import reduce
from operator import mul
import sys

tu = TestUtils()

start_time = time.time()
i = 0
model = ""

def print_model_size(model):
    params = list(model.parameters())
    num_params = sum([reduce(mul, p.size(), 1) for p in params])
    size_bytes = sum([p.numel() * p.element_size() for p in params])
    size_mb = size_bytes / (1024 * 1024)
    print(f"Number of parameters: {num_params}")
    print(f"Model size: {size_mb:.2f} MB")

def get_model_size(model):
    size = sys.getsizeof(model)
    return size

def runTest(filter):
    global i
    global model
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
    
    # global i
    # if (i == 0):
    #     i += 1
    #     size = get_model_size(model)
    #     print("modlel size: ", size, "bytes")

    for i in range(0, 100):
        if(filter == "LeNet"):
            inputs=tu.rand(1, 1, 28, 28)
        else:
            inputs = tu.rand(3, 1, 10)
        outputs = model(inputs)

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
    return parser

if __name__ == "__main__":
    args = _get_argparse().parse_args()
    for i in range(0, 1000):
        runTest(args.filter)

end_time = time.time()
print(f"time is {((end_time - start_time) / 1000):.6f}\n")
