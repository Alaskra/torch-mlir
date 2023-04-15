#!/bin/bash
project_root=`dirname $0`"/.."
cd ${project_root}
echo "======code format====="
find lib/Dialect/Torch/Transforms/Obfuscations -regex '.*\.\(cpp\|h\)' -exec clang-format -style=file -i {} \;
black python/torch_mlir_e2e_test/test_suite/obfuscate.py
echo "====run test===="
python -m e2e_testing.main --filter="Obfuscate_*" --verbose