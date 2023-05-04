//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include <cstdlib>
#include <ctime>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

static std::vector<Value> createABCD(RewriteOp &rewrite, long channelSz,
                                     long kernelSz) {
  // generate A, B, C, D, satisfy C(Ax+B)+D == x
  float A = (rand() % 100 + 1) / 100.0;
  float C = 1 / A;
  float B = (rand() % 100);
  float D = -B * C;
  // create convolution kernel and bias
  // kernel size is: (channelSz, channelSz, kernelSz, kernelSz)
  // bias size is: (channelSz)
  std::vector<long> shapeKernel{channelSz, channelSz, kernelSz, kernelSz};
  std::vector<float> kernelVec(channelSz * channelSz * kernelSz * kernelSz, 0);
  // first conv kernel
  int index = (kernelSz / 2) * kernelSz + (kernelSz / 2);
  for (int i = 0; i < channelSz; i++) {
    // kernelVec[i][i][kernelSz/2][kernelSz/2] = A
    int base = (i * channelSz + i) * kernelSz * kernelSz;
    kernelVec[base + index] = A;
  }
  Value kernelA = rewrite.createTensorOp(shapeKernel, kernelVec);
  // first conv bias
  std::vector<long> shapeBias{channelSz};
  std::vector<float> biasVec(shapeBias[0], B);
  Value biasB = rewrite.createTensorOp(shapeBias, biasVec);
  // second conv kernel and bias
  for (int i = 0; i < channelSz; i++) {
    int base = (i * channelSz + i) * kernelSz * kernelSz;
    kernelVec[base + index] = C;
  }
  Value kernelC = rewrite.createTensorOp(shapeKernel, kernelVec);
  biasVec.assign(biasVec.size(), D);
  Value biasD = rewrite.createTensorOp(shapeBias, biasVec);
  return std::vector<Value>{kernelA, biasB, kernelC, biasD};
}

// insert 2 convolutions after relu on the layer
void InsertConv(MLIRContext *context, Operation *f, int layer) {
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

  // get conv params
  srand(time(0));
  int channelSz = shape[1];
  int kernelSz = (1 + rand() % 5) * 2 + 1;
  std::vector<Value> values = createABCD(rewrite, channelSz, kernelSz);
  int stride = 1;
  int dil = 1 + std::rand() % 5;
  int pad = (kernelSz - 1) * dil / 2;
  int group = 1;
  std::vector<int64_t> intParam{stride, pad, dil, group};

  // create first conv
  Value conv =
      rewrite.createConvOp({oldResult, values[0], values[1]}, intParam);
  Value relu = rewrite.createReluOp(type, conv);
  // create second conv
  conv = rewrite.createConvOp({relu, values[2], values[3]}, intParam);
  relu = rewrite.createReluOp(type, conv);

  // reshape back to origin shape
  if (needReshape)
    relu = rewrite.createReshape(oldShape, relu);
  rewrite.replaceOp(relu);
}

use_pass(InsertConv, 1, int, layer);