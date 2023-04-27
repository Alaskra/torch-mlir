//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

// insert a skip for convolution
static void InsertSkip(MLIRContext *context, Operation *f, int layer) {
  // input test
  input_assert(layer < 1, "layer > 0 \n");
  // get operations that you need
  Operation *op = getReluOp(f, layer);
  if (op == nullptr)
    return;
  int type = getReluType(op);
  // init rewrite
  RewriteOp rewrite(context, op);
  // get output tensor
  auto newOp = rewrite.cloneOp();
  Value oldResult = newOp->getResult(0);

  // get std shape
  auto oldShape = getShape(oldResult);
  std::vector<int64_t> shape = oldShape;
  bool needReshape = false;
  if (oldShape.size() < 4) {
    needReshape = true;
    shape = toStdShape(oldShape);
    oldResult = rewrite.createReshape(shape, oldResult);
  }
  // get zero kernel
  shape[0] = shape[1];
  shape[2] = shape[3] = 1;
  int kernelSize = getKernelSize(shape);
  std::vector<float> zeroKernelVec(kernelSize, 0);
  Value zeroKernel = rewrite.createTensorOp(shape, zeroKernelVec);
  // get zero bias
  toBiasShape(shape);
  std::vector<float> zeroBiasVec(shape[0], 0);
  auto zeroBias = rewrite.createTensorOp(shape, zeroBiasVec);
  // zero conv
  Value zeroConv = rewrite.createConvOp({oldResult, zeroKernel, zeroBias});
  Value relu = rewrite.createReluOp(type, zeroConv);
  // add zero conv
  Value int1 = rewrite.createIntOp(1);
  Value skip = rewrite.createAddTensorOp(oldResult, relu, int1);
  relu = rewrite.createReluOp(type, skip);
  // reshape back to origin shape
  if (needReshape)
    relu = rewrite.createReshape(oldShape, relu);
  rewrite.replaceOp(relu);
}

use_pass(InsertSkip, 1, int, layer);
