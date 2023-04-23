from models import LeNet, RNN, LSTM, GRU
from torch_mlir_e2e_test.framework import TestUtils

tu = TestUtils()
# The global registry of tests.
GLOBAL_TEST_REGISTRY = []
# the tests that failed currently, if bug is fixed, remove it
TEST_NOT_RUN = [
    "RNNInsertSkip",
    "RNNInsertSepraConv",
    "RNNInsertInception",
    "RNNInsertRNN",
    "RNNInsertRNNWithZeros",
    "LSTMInsertSkip",
    "LSTMInsertSepraConv",
    "LSTMInsertInception",
    "LSTMInsertRNN",
    "LSTMInsertRNNWithZeros",
    "GRUInsertSkip",
    "GRUInsertSepraConv",
    "GRUInsertInception",
    "GRUInsertRNN",
    "GRUInsertRNNWithZeros",
]
# Ensure that there are no duplicate names in the global test registry.
_SEEN_UNIQUE_NAME = set()


class Test:
    def __init__(
        self,
        name="LeNet",
        model=LeNet(),
        passes=[],
        inputs=tu.rand(1, 1, 28, 28),
    ):
        self.name = name
        self.model = model
        self.passes = passes
        self.passesName = self.getPasses(passes)
        self.inputs = inputs

    def getPasses(self, passes):
        t = ""
        for p in passes:
            t += f"func.func({p}),"
        return f"builtin.module({t[:-1]})"


def addGlobalTest(name, model, inputs, passes):
    global GLOBAL_TEST_REGISTRY
    assert name not in GLOBAL_TEST_REGISTRY, f"test name: {name} dubpicated"
    _SEEN_UNIQUE_NAME.add(name)
    GLOBAL_TEST_REGISTRY.append(Test(name, model, passes, inputs))


# These obfuscations can apply to all models, include LeNet, RNN, LSTM, GRU
general_obfuscation = {
    "InsertSkip": ["torch-insert-skip{layer=2}"],
    "InsertConv": ["torch-insert-conv"],
    "InsertSepraConv": ["torch-insert-sepra-conv-layer{layer=2}"],
    "InsertLinear": ["torch-insert-linear"],
    "ValueSplit": ["torch-value-split"],
    "MaskSplit": ["torch-mask-split"],
    "InsertInception": ["torch-insert-Inception{number=5}"],
    "InsertRNN": ["torch-insert-RNN{number=5}"],
    "InsertRNNWithZeros": ["torch-insert-RNNWithZeros{activationFunc=tanh number=5}"],
}


def addLeNetTests():
    def addLeNetTest(name, passes):
        net = LeNet()
        inputs = [tu.rand(1, 1, 28, 28)]
        addGlobalTest(name, net, inputs, passes)

    for name, passes in general_obfuscation.items():
        addLeNetTest(f"LeNet{name}", passes)
    addLeNetTest(
        "LeNetBranchLayer",
        ["torch-branch-layer{layer=2 branch=4}"],
    )
    addLeNetTest(
        "LeNetWidenConv",
        ["torch-widen-conv-layer{layer=1 number=4}"],
    )
    addLeNetTest(
        "LeNetWidenInsertConv",
        ["torch-widen-conv-layer", "torch-insert-conv"],
    )
    addLeNetTest(
        "LeNetInsertMaxpool",
        ["torch-insert-Maxpool"],
    )


def addRNNTests():
    def addRNNTest(name, passes):
        net = RNN(10, 20, 18)
        inputs = [tu.rand(3, 1, 10)]
        addGlobalTest(name, net, inputs, passes)

    for name, passes in general_obfuscation.items():
        addRNNTest(f"RNN{name}", passes)
    addRNNTest(
        "RNNValueSplitRNN",
        ["torch-value-split{net=RNN number=5}"],
    )
    addRNNTest(
        "RNNMaskSplitRNN",
        ["torch-mask-split{net=RNN number=5}"],
    )
    addRNNTest(
        "RNNInsertConvRNN",
        ["torch-insert-conv{net=RNN}"],
    )
    addRNNTest(
        "RNNInsertLinearRNN",
        ["torch-insert-linear{net=RNN}"],
    )


def addLSTMTests():
    def addLSTMTest(name, passes):
        net = LSTM(10, 20, 18)
        inputs = [tu.rand(3, 1, 10)]
        addGlobalTest(name, net, inputs, passes)

    for name, passes in general_obfuscation.items():
        addLSTMTest(f"LSTM{name}", passes)


def addGRUTests():
    def addGRUTest(name, passes):
        net = GRU(10, 20, 18)
        inputs = [tu.rand(3, 1, 10)]
        addGlobalTest(name, net, inputs, passes)

    for name, passes in general_obfuscation.items():
        addGRUTest(f"GRU{name}", passes)


def addTests():
    addLeNetTests()
    addRNNTests()
    addLSTMTests()
    addGRUTests()
