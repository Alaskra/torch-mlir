//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

// insert a separable convolution
static void InsertSepraConv(MLIRContext *context, Operation *f, int layer) { 
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

  // deep-wise convolution, (out, in, 1, 1), groups = in_channel
  // get deep-wise kernel
  auto shape = getShape(oldResult);
  int group = shape[1];
  shape[0] = shape[1];
  shape[1] = shape[2] = shape[3] = 1;
  int kernelSize = getKernelSize(shape);
  std::vector<float> deepKernelVec(kernelSize, 1);
  Value deepKernel = rewrite.createTensorOp(shape, deepKernelVec);
  // get zero bias
  toBiasShape(shape);
  std::vector<float> zeroBiasVec(shape[0], 0);
  Value zeroBias = rewrite.createTensorOp(shape, zeroBiasVec);
  // insert deep-wise conv
  Value groupsOp = rewrite.createIntOp(group);
  Value deepConv = rewrite.createConvOp(oldResult.getType(), {oldResult, deepKernel, zeroBias}, groupsOp);
  // add new relu
  Value relu = rewrite.createReluOp(type, deepConv);
  
  // point-wise convolution, (in, in, 1, 1), groups = 1
  // get point-wise kernel
  shape = getShape(oldResult);
  toStdShape(shape);
  kernelSize = getKernelSize(shape);
  std::vector<float> oneKernelVec(kernelSize);
  creatOneTensor(oneKernelVec, shape[0]);
  Value oneKernel = rewrite.createTensorOp(shape, oneKernelVec);
  // insert point-wise conv
  Value oneConv = rewrite.createConvOp(oldResult.getType(), {relu, oneKernel, zeroBias});
  // add new relu
  relu = rewrite.createReluOp(type, oneConv);
  // replace old relu
  rewrite.replaceOp(relu);
}

use_pass(InsertSepraConv, 1, int, layer);
