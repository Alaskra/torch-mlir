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
  OpList oplist;
  int type = getReluOp(oplist, f, layer);
  if (!type) return;
  // get convolution operations
  auto op = *oplist.begin();
  // init rewrite
  RewriteOp rewrite(context, op);
  // get output tensor
  auto newOp = rewrite.cloneOp();
  Value oldResult = newOp->getResult(0);

  // get zero kernel
  auto shape = getShape(oldResult);
  toStdShape(shape);
  int kernelSize = getKernelSize(shape);
  std::vector<float> zeroKernelVec(kernelSize, 0);
  Value zeroKernel = rewrite.createTensorOp(shape, zeroKernelVec);
  // get zero bias
  toBiasShape(shape);
  std::vector<float> zeroBiasVec(shape[0], 0);
  auto zeroBias = rewrite.createTensorOp(shape, zeroBiasVec);
  // zero conv
  Value zeroConv = rewrite.createConvOp({oldResult, zeroKernel, zeroBias});
  // add zero conv
  Value int1 = rewrite.createIntOp(1); // Value float0 = rewrite.createFloatOp(0);
  Value skip = rewrite.createAddTensorOp(oldResult, zeroConv, int1);
  // add new relu
  Value relu = rewrite.createReluOp(type, skip);
  // replace old relu
  rewrite.replaceOp(relu);
}

use_pass(InsertSkip, 1, int, layer);
