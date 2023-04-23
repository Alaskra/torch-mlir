#!/bin/bash
project_root=$(dirname $0)"/.."
cd ${project_root}
echo "======code format====="
find lib/Dialect/Torch/Transforms/Obfuscations -regex '.*\.\(cpp\|h\)' -exec clang-format -style=file -i {} \;
black obfuscation/*.py
echo "====run test===="
python obfuscation/test.py -s
