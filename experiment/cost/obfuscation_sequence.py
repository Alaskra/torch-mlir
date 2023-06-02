# LeNet_obfuscation = {
#     "LeNetInsertSkip": "torch-insert-skip{layer=2}",
#     # "LeNetInsertConv": "torch-insert-conv{layer=2}",
#     # "LeNetInsertSepraConv": "torch-insert-sepra-conv-layer{layer=2}",
#     # "LeNetInsertLinear": "torch-insert-linear{layer=2}",
#     # "LeNetValueSplit": "torch-value-split{layer=1}",
#     # "LeNetMaskSplit": "torch-mask-split{layer=1}",
#     # "LeNetInsertInception": "torch-insert-Inception{number=5 layer=1}",
#     # "LeNetInsertRNN": "torch-insert-RNN{number=5 layer=1}",
#     # "LeNetInsertRNNWithZeros": "torch-insert-RNNWithZeros{activationFunc=tanh number=5 layer=1}",
#     # "LeNetBranchLayer": "torch-branch-layer{layer=2 branch=4}",
#     # "LeNetWidenConv": "torch-widen-conv-layer{layer=1 number=4}",
#     # "LeNetInsertMaxpool": "torch-insert-Maxpool",
# }

LeNet_obfuscation = {
    "1": "torch-insert-skip{layer=2}",
    "2": "torch-insert-conv{layer=2}",
    "3": "torch-insert-sepra-conv-layer{layer=2}",
    "4": "torch-insert-linear{layer=2}",
    "5": "torch-value-split{layer=1}",
    "6": "torch-mask-split{layer=1}",
    "7": "torch-insert-Inception{number=5 layer=1}",
    "8": "torch-insert-RNN{number=5 layer=1}",
    "9": "torch-insert-RNNWithZeros{activationFunc=tanh number=5 layer=1}",
    "10": "torch-branch-layer{layer=2 branch=4}",
    "11": "torch-widen-conv-layer{layer=1 number=4}",
    "12": "torch-insert-Maxpool",
}

# RNN_obfuscation = {
#     "RNNInsertSkip": "torch-insert-skip{layer=2}",
#     "RNNInsertConv": "torch-insert-conv{layer=2}",
#     "RNNInsertSepraConv": "torch-insert-sepra-conv-layer{layer=2}",
#     "RNNInsertLinear": "torch-insert-linear{layer=2}",
#     "RNNValueSplit": "torch-value-split{layer=1}",
#     "RNNMaskSplit": "torch-mask-split{layer=1}",
#     "RNNInsertInception": "torch-insert-Inception{number=5 layer=1}",
#     "RNNInsertRNN": "torch-insert-RNN{number=5 layer=1}",
#     "RNNInsertRNNWithZeros": "torch-insert-RNNWithZeros{activationFunc=tanh number=5 layer=1}",
# }

RNN_obfuscation = {
    "1": "torch-insert-skip{layer=2}",
    "2": "torch-insert-conv{layer=2}",
    "3": "torch-insert-sepra-conv-layer{layer=2}",
    "4": "torch-insert-linear{layer=2}",
    "5": "torch-value-split{layer=1}",
    "6": "torch-mask-split{layer=1}",
    "7": "torch-insert-Inception{number=5 layer=1}",
    "8": "torch-insert-RNN{number=5 layer=1}",
    "9": "torch-insert-RNNWithZeros{activationFunc=tanh number=5 layer=1}",
}

# LSTM_obfuscation = {
#     "LSTMInsertSkip": "torch-insert-skip{layer=2}",
#     "LSTMInsertConv": "torch-insert-conv{layer=2}",
#     "LSTMInsertSepraConv": "torch-insert-sepra-conv-layer{layer=2}",
#     "LSTMInsertLinear": "torch-insert-linear{layer=2}",
#     "LSTMValueSplit": "torch-value-split{layer=1}",
#     "LSTMMaskSplit": "torch-mask-split{layer=1}",
#     "LSTMInsertInception": "torch-insert-Inception{number=5 layer=1}",
#     "LSTMInsertRNN": "torch-insert-RNN{number=5 layer=1}",
#     "LSTMInsertRNNWithZeros": "torch-insert-RNNWithZeros{activationFunc=tanh number=5 layer=1}",
# }

LSTM_obfuscation = {
    "1": "torch-insert-skip{layer=2}",
    "2": "torch-insert-conv{layer=2}",
    "3": "torch-insert-sepra-conv-layer{layer=2}",
    "4": "torch-insert-linear{layer=2}",
    "5": "torch-value-split{layer=1}",
    "6": "torch-mask-split{layer=1}",
    "7": "torch-insert-Inception{number=5 layer=1}",
    "8": "torch-insert-RNN{number=5 layer=1}",
    "9": "torch-insert-RNNWithZeros{activationFunc=tanh number=5 layer=1}",
}

GRU_obfuscation = {
    "GRUInsertSkip": "torch-insert-skip{layer=2}",
    "GRUInsertConv": "torch-insert-conv{layer=2}",
    "GRUInsertSepraConv": "torch-insert-sepra-conv-layer{layer=2}",
    "GRUInsertLinear": "torch-insert-linear{layer=2}",
    "GRUValueSplit": "torch-value-split{layer=1}",
    "GRUMaskSplit": "torch-mask-split{layer=1}",
    "GRUInsertInception": "torch-insert-Inception{number=5 layer=1}",
    "GRUInsertRNN": "torch-insert-RNN{number=5 layer=1}",
    "GRUInsertRNNWithZeros": "torch-insert-RNNWithZeros{activationFunc=tanh number=5 layer=1}",
}

GRU_obfuscation = {
    "1": "torch-insert-skip{layer=2}",
    "2": "torch-insert-conv{layer=2}",
    "3": "torch-insert-sepra-conv-layer{layer=2}",
    "4": "torch-insert-linear{layer=2}",
    "5": "torch-value-split{layer=1}",
    "6": "torch-mask-split{layer=1}",
    "7": "torch-insert-Inception{number=5 layer=1}",
    "8": "torch-insert-RNN{number=5 layer=1}",
    "9": "torch-insert-RNNWithZeros{activationFunc=tanh number=5 layer=1}",
}