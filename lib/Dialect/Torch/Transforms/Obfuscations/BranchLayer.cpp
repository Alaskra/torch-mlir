
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

// number the layer and insert a
// convolution into the numbers randomly.
static void BranchLayer(MLIRContext *context, Operation *f, int layer,
                        int number) {
  // input test
  input_assert(number < 2, "number > 1 \n");
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
  // number test: channels
  llvm_assert(inputChannels < 2, "error: input_channels(%d) <= 1 \n",
              inputChannels);
  llvm_assert(inputChannels <= number,
              "error: input_channels(%d) <= number(%d) \n", inputChannels,
              number);
  // init rewrite
  RewriteOp rewrite(context, op);
  // get output tensor
  auto newOp = rewrite.cloneOp();
  Value oldResult = newOp->getResult(0);

  // slice randomly
  std::vector<int> numberChannel(number);
  // tempvar: current channels, current number, min channels, spare channels
  int tempVar[4] = {inputChannels, number, 0, 0};
  srand(time(0));
  for (int i = 0; i < number; i++) {
    tempVar[2] = tempVar[0] / tempVar[1];
    tempVar[3] = tempVar[0] % tempVar[1];
    numberChannel[i] = tempVar[2] + rand() % (tempVar[3] + 1);
    tempVar[0] -= numberChannel[i];
    tempVar[1] -= 1;
  }

  // slice tensors
  std::vector<std::vector<int64_t>> numberShape(number);
  std::vector<Value> numberTensorOp(number);
  int curChannel = 0; // current channel
  Value startOp;
  Value endOp = rewrite.createIntOp(curChannel);
  Value dimOp = rewrite.createIntOp(dim);
  for (int i = 0; i < number; i++) {
    // get shape
    numberShape[i] = inputShape;
    numberShape[i][dim] = numberChannel[i];
    // get slice tensor
    startOp = endOp;
    curChannel += numberChannel[i];
    endOp = rewrite.createIntOp(curChannel);
    numberTensorOp[i] = rewrite.createSliceTensorOp(numberShape[i], oldResult,
                                                    dimOp, startOp, endOp);
  }

  // handle every number tensor randomly
  int handleWay;
  srand(time(0));
  for (int i = 0; i < number; i++) {
    handleWay = rand() % 2;
    // 0: nop
    if (handleWay == 0)
      continue;
    // 1: insert one convolution
    numberTensorOp[i] =
        InertOneConv(rewrite, numberShape[i], numberTensorOp[i]);
  }

  // cat number tensors
  Value catOp = rewrite.createCatTensorOp(inputShape, dimOp, numberTensorOp);
  Value relu = rewrite.createReluOp(type, catOp);
  rewrite.replaceOp(relu);
}

use_pass(BranchLayer, 2, int, layer, int, number);
