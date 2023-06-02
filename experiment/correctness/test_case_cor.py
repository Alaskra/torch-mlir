from model_cor import InceptionV1, ResNet50, VGG16, DenseNet121
from torch_mlir_e2e_test.framework import TestUtils

tu = TestUtils()
# The global registry of tests.
GLOBAL_TEST_REGISTRY = []
# Ensure that there are no duplicate names in the global test registry.
_SEEN_UNIQUE_NAME = set()


class Test:
    def __init__(
        self,
        name="Inception",
        model=InceptionV1(),
        passes=[],
        inputs=tu.rand(1, 3, 224, 224),
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
    "InsertSkip": ["torch-insert-skip{layer=1}"],
    "InsertConv": ["torch-insert-conv{layer=1}"],
    "InsertSepraConv": ["torch-insert-sepra-conv-layer{layer=1}"],
    "InsertLinear": ["torch-insert-linear{layer=1}"],
    "ValueSplit": ["torch-value-split{layer=1}"],
    "MaskSplit": ["torch-mask-split{layer=1}"],
    "InsertInception": ["torch-insert-Inception{number=5 layer=1}"],
    "InsertRNN": ["torch-insert-RNN{number=5 layer=1}"],
    "InsertRNNWithZeros": ["torch-insert-RNNWithZeros{activationFunc=tanh number=5 layer=1}"],
}

def addInceptionTests():
    def addInceptionTest(name, passes):
        net = InceptionV1()
        inputs = [tu.rand(1, 3, 224, 224)]
        addGlobalTest(name, net, inputs, passes)

    for name, passes in general_obfuscation.items():
        addInceptionTest(f"Inception{name}", passes)

def addResNetTests():
    def addResNetTest(name, passes):
        net = ResNet50()
        inputs = [tu.rand(1, 3, 224, 224)]
        addGlobalTest(name, net, inputs, passes)

    for name, passes in general_obfuscation.items():
        addResNetTest(f"ResNet{name}", passes)

def addVGGTests():
    def addVGGTest(name, passes):
        net = VGG16()
        inputs = [tu.rand(1, 3, 224, 224)]
        addGlobalTest(name, net, inputs, passes)

    for name, passes in general_obfuscation.items():
        addVGGTest(f"VGG{name}", passes)

def addDenseNetTests():
    def addDenseNetTest(name, passes):
        net = DenseNet121()
        inputs = [tu.rand(1, 3, 224, 224)]
        addGlobalTest(name, net, inputs, passes)

    for name, passes in general_obfuscation.items():
        addDenseNetTest(f"DenseNet{name}", passes)


def addTests():
    addInceptionTests()
    addResNetTests()
    addVGGTests()
    addDenseNetTests()