import torch_mlir, torch
import argparse
import re
from itertools import repeat
from test_case import addTests, GLOBAL_TEST_REGISTRY, Test, TEST_NOT_RUN
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend
from torch_mlir_e2e_test.reporting import ValueReport, ErrorContext
from torch_mlir_e2e_test.configs.utils import (
    recursively_convert_to_numpy,
    recursively_convert_from_numpy,
)
import multiprocess as mp


def runTest(test=Test(), verbose=False):
    if test.name in TEST_NOT_RUN:
        return ["NOT RUN", test.name]
    model = test.model
    inputs = test.inputs
    if verbose:
        print(model)

    module = torch_mlir.compile(
        model, inputs, output_type="torch", use_tracing=True, ignore_traced_shapes=True
    )
    if verbose:
        print("================")
        print("origin torch mlir")
        print("================")
        print(module.operation.get_asm(large_elements_limit=10))

    torch_mlir.compiler_utils.run_pipeline_with_repro_report(
        module,
        test.passesName,
        "Apply obfuscation passes",
    )
    if verbose:
        print("================")
        print("after obfuscation")
        print("================")
        print(module.operation.get_asm(large_elements_limit=10))

    torch_mlir.compiler_utils.run_pipeline_with_repro_report(
        module,
        "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
        "Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
    )

    numpy_inputs = recursively_convert_to_numpy(inputs)
    backend = refbackend.RefBackendLinalgOnTensorsBackend()
    compiled = backend.compile(module)
    jit_module = backend.load(compiled)
    outputs = jit_module.forward(*numpy_inputs)
    outputs = recursively_convert_from_numpy(outputs)

    out_pytorch = model(*inputs)
    if verbose:
        print("================")
        print("origin outputs")
        print("================")
        print(out_pytorch)
        print("================")
        print("outputs after obfuscation")
        print("================")
        print(outputs)

    if torch.allclose(outputs, out_pytorch, rtol=1e-2, atol=1e-6):
        # value_report = ValueReport(outputs, out_pytorch, ErrorContext([""]))
        # if not value_report.failed:
        return ["SUCCESS", test.name]
    else:
        return ["FAILED", test.name]


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
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="print IR and output value before and after obfuscation",
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


def runTests(tests, filter, verbose=False, sequential=True):
    filtered_tests = [test for test in tests if re.match(filter, test.name)]

    num_processes = min(int(mp.cpu_count() * 1.1), len(tests))
    # TODO: We've noticed that on certain 2 core machine parallelizing the tests
    # makes the llvm backend legacy pass manager 20x slower than using a
    # single process. Need to investigate the root cause eventually. This is a
    # hack to work around this issue.
    # Also our multiprocessing implementation is not the most efficient, so
    # the benefit at core count 2 is probably not worth it anyway.
    if mp.cpu_count() == 2:
        num_processes = 1
    # TODO: If num_processes == 1, then run without any of the multiprocessing
    # machinery. In theory it should work, but any crash in the testing process
    # seems to cause a cascade of failures resulting in undecipherable error
    # messages.
    results = []
    if num_processes == 1 or sequential:
        for test in filtered_tests:
            print(f"************************\nTest: {test.name}")
            results.append(runTest(test, verbose))
            print(results[-1][0])
    else:
        # This is needed because autograd does not support crossing process
        # boundaries.
        torch.autograd.set_grad_enabled(False)

        pool = mp.Pool(num_processes)
        arglist = zip(filtered_tests, repeat(verbose))
        handles = pool.starmap_async(runTest, arglist)
        results = handles.get()
        for result in results:
            print(f"************************\nTest: {result[1]}")
            print(result[0])
    print()
    print(f"Total {len(filtered_tests)} tests")
    failed_num = 0
    not_run_num = 0
    for result in results:
        if result[0] == "FAILED":
            failed_num += 1
        elif result[0] == "NOT RUN":
            not_run_num += 1
    print(f"Failed {failed_num} tests")
    print(f"Not run {not_run_num} tests")


if __name__ == "__main__":
    args = _get_argparse().parse_args()
    addTests()
    runTests(GLOBAL_TEST_REGISTRY, args.filter, args.verbose, args.sequential)
