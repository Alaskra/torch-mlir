
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

// insert one convolution
static Value InertOneConv(RewriteOp &rewrite, vector<int64_t> shape,
                          Value inputOp) {
  // get one kernel
  shape[0] = shape[1];
  shape[2] = shape[3] = 1;
  int kernelSize = getKernelSize(shape);
  std::vector<float> oneKernelVec(kernelSize, 0);
  creatOneTensor(oneKernelVec, shape[0]);
  Value oneKernel = rewrite.createTensorOp(shape, oneKernelVec);
  // get zero bias
  toBiasShape(shape);
  std::vector<float> zeroBiasVec(shape[0], 0);
  Value zeroBias = rewrite.createTensorOp(shape, zeroBiasVec);
  // get one conv
  return rewrite.createConvOp({inputOp, oneKernel, zeroBias});
}

// branch the layer and insert a
// convolution into the branchs randomly.
static void BranchLayer(MLIRContext *context, Operation *f, int layer,
                        int branch) {
  // input test
  input_assert(branch < 2, "branch > 1 \n");
  input_assert(layer < 1, "layer > 0 \n");
  // get operations that you need
  Operation *op = getReluOp(f, layer);
  if (op == nullptr)
    return;
  int type = getReluType(op);
  const int dim = 1;
  // get input information
  auto inputShape = getShape(op->getResult(0));
  int inputChannels = inputShape[dim];
  // branch test: channels
  llvm_assert(inputChannels < 2, "error: input_channels(%d) <= 1 \n",
              inputChannels);
  llvm_assert(inputChannels <= branch,
              "error: input_channels(%d) <= branch(%d) \n", inputChannels,
              branch);
  // init rewrite
  RewriteOp rewrite(context, op);
  // get output tensor
  auto newOp = rewrite.cloneOp();
  Value oldResult = newOp->getResult(0);

  // slice randomly
  std::vector<int> branchChannel(branch);
  // tempvar: current channels, current branch, min channels, spare channels
  int tempVar[4] = {inputChannels, branch, 0, 0};
  srand(time(0));
  for (int i = 0; i < branch; i++) {
    tempVar[2] = tempVar[0] / tempVar[1];
    tempVar[3] = tempVar[0] % tempVar[1];
    branchChannel[i] = tempVar[2] + rand() % (tempVar[3] + 1);
    tempVar[0] -= branchChannel[i];
    tempVar[1] -= 1;
  }

  // slice tensors
  std::vector<std::vector<int64_t>> branchShape(branch);
  std::vector<Value> branchTensorOp(branch);
  int curChannel = 0; // current channel
  Value startOp;
  Value endOp = rewrite.createIntOp(curChannel);
  Value dimOp = rewrite.createIntOp(dim);
  for (int i = 0; i < branch; i++) {
    // get shape
    branchShape[i] = inputShape;
    branchShape[i][dim] = branchChannel[i];
    // get slice tensor
    startOp = endOp;
    curChannel += branchChannel[i];
    endOp = rewrite.createIntOp(curChannel);
    branchTensorOp[i] = rewrite.createSliceTensorOp(branchShape[i], oldResult,
                                                    dimOp, startOp, endOp);
  }

  // handle every branch tensor randomly
  int handleWay;
  srand(time(0));
  for (int i = 0; i < branch; i++) {
    handleWay = rand() % 2;
    // 0: nop
    if (handleWay == 0)
      continue;
    // 1: insert one convolution
    branchTensorOp[i] =
        InertOneConv(rewrite, branchShape[i], branchTensorOp[i]);
  }

  // cat branch tensors
  Value catOp = rewrite.createCatTensorOp(inputShape, dimOp, branchTensorOp);
  Value relu = rewrite.createReluOp(type, catOp);
  rewrite.replaceOp(relu);
}

use_pass(BranchLayer, 2, int, layer, int, branch);
