#!/bin/bash
find lib/Dialect/Torch/Transforms/Obfuscations -regex '.*\.\(cpp\|h\)' -exec clang-format -style=file -i {} \;
python -m e2e_testing.main --filter="Obfuscate_*" --verbose