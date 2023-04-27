//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

// insert two separable convolution: deep-wise, point-wise
static void InsertSepraConv(MLIRContext *context, Operation *f, int layer) {
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

  // deep-wise convolution, (out, in, 1, 1), group = in_channel

  // get std shape
  auto oldShape = getShape(oldResult);
  std::vector<int64_t> shape = oldShape;
  bool needReshape = false;
  if (oldShape.size() < 4) {
    needReshape = true;
    shape = toStdShape(oldShape);
    oldResult = rewrite.createReshape(shape, oldResult);
  }
  // get deep-wise kernel
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
  Value deepConv =
      rewrite.createConvOp({oldResult, deepKernel, zeroBias}, {1, 0, 1, group});
  Value relu = rewrite.createReluOp(type, deepConv);

  // point-wise convolution, (in, in, 1, 1), group = 1
  // get point-wise kernel
  shape = getShape(oldResult);
  shape[0] = shape[1];
  shape[2] = shape[3] = 1;
  kernelSize = getKernelSize(shape);
  std::vector<float> oneKernelVec(kernelSize);
  creatOneTensor(oneKernelVec, shape[0]);
  Value oneKernel = rewrite.createTensorOp(shape, oneKernelVec);
  // insert point-wise conv
  Value oneConv = rewrite.createConvOp({relu, oneKernel, zeroBias});
  relu = rewrite.createReluOp(type, oneConv);
  // reshape back to origin shape
  if (needReshape)
    relu = rewrite.createReshape(oldShape, relu);
  rewrite.replaceOp(relu);
}

use_pass(InsertSepraConv, 1, int, layer);
