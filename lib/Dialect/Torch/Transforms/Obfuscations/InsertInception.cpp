//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include <iostream>
#include <random>

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

static void insertInception(MLIRContext *context, Operation *f, int number,
                            int layer) {
  // this demo insert a Inception into the network

  input_assert(layer < 1, "layer > 0 \n");
  input_assert(number < 1, "number > 0 \n");

  // get operations that you need
  Operation *op = getReluOp(f, layer);
  if (op == nullptr)
    return;

  IRRewriter rewriter(context);
  rewriter.setInsertionPointAfter(op);
  Operation *newOp = rewriter.clone(*op);
  Location loc = newOp->getLoc();
  Value rst = newOp->getResult(0);

  // change
  std::vector<long> shapeOrigin =
      rst.getType().cast<ValueTensorType>().getSizes().vec();
  bool needReshape = false;
  std::vector<long> shapeNew(4 - shapeOrigin.size(), 1);
  shapeNew.insert(shapeNew.end(), shapeOrigin.begin(), shapeOrigin.end());
  if (shapeOrigin.size() != 4) {
    needReshape = true;
  }
  if (needReshape)
    rst = createReshape(rewriter, loc, context, shapeNew, rst);

  auto shape = rst.getType().cast<ValueTensorType>().getSizes().vec();
  Value int0 =
      rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
  Value int1 =
      rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
  int number_branch = number;
  std::random_device rd;  // 用于获取真随机数种子
  std::mt19937 gen(rd()); // 以真随机数种子生成随机数生成器
  std::uniform_int_distribution<> dis(1, 3); // 定义随机数分布，范围为 1 到 3
  std::uniform_int_distribution<> dis_2(1,
                                        20); // 定义随机数分布，范围为 1 到 20

  // common parameters
  Value list_stride = rewriter.create<PrimListConstructOp>(
      loc, ListType::get(IntType::get(context)), ValueRange({int1, int1}));
  Value list_dilation = rewriter.create<PrimListConstructOp>(
      loc, ListType::get(IntType::get(context)), ValueRange({int1, int1}));
  Value constFalse = rewriter.create<ConstantBoolOp>(loc, false);
  Value list = rewriter.create<PrimListConstructOp>(
      loc, ListType::get(IntType::get(context)), ValueRange());

  std::vector<Value> values(number_branch + 1);
  std::vector<int64_t> shape_dim_1(number_branch);
  values[0] = rst;
  for (int i = 0; i < number_branch; i++) {
    int branch_structure = dis(gen);
    if (branch_structure == 1) {
      // 添加一个1 * 1的卷积
      auto shape_weight = shape;
      shape_weight[0] = shape_weight[1];
      shape_dim_1[i] = shape_weight[0];
      shape_weight[2] = shape_weight[3] = 1;
      int weightSize =
          shape_weight[0] * shape_weight[1] * shape_weight[2] * shape_weight[3];
      std::vector<float> WeightVec(weightSize);
      // 随机数引擎对象
      std::mt19937 generator(std::random_device{}());

      // 创建随机数分布器，范围为[-1, 1]
      std::uniform_real_distribution<float> distribution(-1.f, 1.f);

      // 循环对 WeightVec 中的每个元素进行随机化
      for (int i = 0; i < weightSize; ++i) {
        WeightVec[i] = distribution(generator);
      }
      Value Weight =
          createTensor(rewriter, loc, context, shape_weight, WeightVec);
      // bias
      auto shape_bias = shape;
      shape_bias.erase(shape_bias.begin() + 1, shape_bias.end());
      shape_bias[0] = shape_weight[0];
      std::vector<float> BiasVec(shape_bias[0]);
      for (int i = 0; i < shape_bias[0]; ++i) {
        BiasVec[i] = distribution(generator);
      }
      Value Bias = createTensor(rewriter, loc, context, shape_bias, BiasVec);
      Value list_padding = rewriter.create<PrimListConstructOp>(
          loc, ListType::get(IntType::get(context)), ValueRange({int0, int0}));
      values[i + 1] = rewriter.create<AtenConvolutionOp>(
          loc, rst.getType(), rst, Weight, Bias, list_stride, list_padding,
          list_dilation, constFalse, list, int1);
    } else if (branch_structure == 2) {
      // 添加一个1 * 1的卷积
      auto shape_weight = shape;
      shape_weight[0] = shape_weight[1];
      shape_weight[2] = shape_weight[3] = 1;
      int weightSize =
          shape_weight[0] * shape_weight[1] * shape_weight[2] * shape_weight[3];
      std::vector<float> WeightVec(weightSize);
      // 随机数引擎对象
      std::mt19937 generator(std::random_device{}());

      // 创建随机数分布器，范围为[-1, 1]
      std::uniform_real_distribution<float> distribution(-1.f, 1.f);

      // 循环对 WeightVec 中的每个元素进行随机化
      for (int i = 0; i < weightSize; ++i) {
        WeightVec[i] = distribution(generator);
      }
      Value Weight =
          createTensor(rewriter, loc, context, shape_weight, WeightVec);
      // bias
      auto shape_bias = shape;
      shape_bias.erase(shape_bias.begin() + 1, shape_bias.end());
      shape_bias[0] = shape_weight[0];
      std::vector<float> BiasVec(shape_bias[0]);
      for (int i = 0; i < shape_bias[0]; ++i) {
        BiasVec[i] = distribution(generator);
      }
      Value Bias = createTensor(rewriter, loc, context, shape_bias, BiasVec);
      Value list_padding = rewriter.create<PrimListConstructOp>(
          loc, ListType::get(IntType::get(context)), ValueRange({int0, int0}));
      Value randomConv_1 = rewriter.create<AtenConvolutionOp>(
          loc, rst.getType(), rst, Weight, Bias, list_stride, list_padding,
          list_dilation, constFalse, list, int1);
      Value relu_1 =
          rewriter.create<AtenReluOp>(loc, rst.getType(), randomConv_1);

      // 改变kernel_size的大小,进行第二次卷积
      int padNum = dis(gen);
      Value intPad = rewriter.create<ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(padNum));
      list_padding = rewriter.create<PrimListConstructOp>(
          loc, ListType::get(IntType::get(context)),
          ValueRange({intPad, intPad}));
      int kernel = padNum * 2 + 1;
      auto shape_weight_2 = shape;
      shape_weight_2[0] = dis_2(gen);
      shape_weight_2[1] = shape_weight[0];
      shape_dim_1[i] = shape_weight_2[0];
      shape_weight_2[2] = shape_weight_2[3] = kernel;
      int weightSize_2 = shape_weight_2[0] * shape_weight_2[1] *
                         shape_weight_2[2] * shape_weight_2[3];
      std::vector<float> WeightVec_2(weightSize_2);
      for (int i = 0; i < weightSize_2; ++i) {
        WeightVec_2[i] = distribution(generator);
      }
      Weight =
          createTensor(rewriter, loc, context, shape_weight_2, WeightVec_2);

      // bias
      shape_bias[0] = shape_weight_2[0];
      std::vector<float> BiasVec_2(shape_bias[0]);
      for (int i = 0; i < shape_bias[0]; ++i) {
        BiasVec_2[i] = distribution(generator);
      }
      Bias = createTensor(rewriter, loc, context, shape_bias, BiasVec_2);
      auto shape_conv = shape;
      shape_conv[1] = shape_weight_2[0];
      int conv_size =
          shape_conv[0] * shape_conv[1] * shape_conv[2] * shape_conv[3];
      std::vector<float> zeroConvVec(conv_size);
      Value zeroConv =
          createTensor(rewriter, loc, context, shape_conv, zeroConvVec);
      Value randomConv_2 = rewriter.create<AtenConvolutionOp>(
          loc, zeroConv.getType(), relu_1, Weight, Bias, list_stride,
          list_padding, list_dilation, constFalse, list, int1);
      values[i + 1] =
          rewriter.create<AtenReluOp>(loc, zeroConv.getType(), randomConv_2);
    } else if (branch_structure == 3) {
      // maxpool2dOp
      int padNum = dis(gen);
      Value intPad = rewriter.create<ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(padNum));
      Value list_padding = rewriter.create<PrimListConstructOp>(
          loc, ListType::get(IntType::get(context)),
          ValueRange({intPad, intPad}));
      int kernel = padNum * 2 + 1;
      Value intKernel = rewriter.create<ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(kernel));
      Value list_kernel = rewriter.create<PrimListConstructOp>(
          loc, ListType::get(IntType::get(context)),
          ValueRange({intKernel, intKernel}));
      Value maxPool2dOp = rewriter.create<AtenMaxPool2dOp>(
          loc, rst.getType(), rst, list_kernel, list_stride, list_padding,
          list_dilation, constFalse);

      // convOp
      padNum = dis(gen);
      intPad = rewriter.create<ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(padNum));
      list_padding = rewriter.create<PrimListConstructOp>(
          loc, ListType::get(IntType::get(context)),
          ValueRange({intPad, intPad}));
      kernel = padNum * 2 + 1;
      auto shape_weight = shape;
      shape_weight[0] = dis_2(gen);
      shape_weight[1] = shape[1];
      shape_dim_1[i] = shape_weight[0];
      shape_weight[2] = shape_weight[3] = kernel;
      int weightSize =
          shape_weight[0] * shape_weight[1] * shape_weight[2] * shape_weight[3];
      std::vector<float> WeightVec(weightSize);
      // 随机数引擎对象
      std::mt19937 generator(std::random_device{}());

      // 创建随机数分布器，范围为[-1, 1]
      std::uniform_real_distribution<float> distribution(-1.f, 1.f);

      // 循环对 WeightVec 中的每个元素进行随机化
      for (int i = 0; i < weightSize; ++i) {
        WeightVec[i] = distribution(generator);
      }
      Value Weight =
          createTensor(rewriter, loc, context, shape_weight, WeightVec);
      // bias
      auto shape_bias = shape;
      shape_bias.erase(shape_bias.begin() + 1, shape_bias.end());
      shape_bias[0] = shape_weight[0];
      std::vector<float> BiasVec(shape_bias[0]);
      for (int i = 0; i < shape_bias[0]; ++i) {
        BiasVec[i] = distribution(generator);
      }
      Value Bias = createTensor(rewriter, loc, context, shape_bias, BiasVec);
      auto shape_conv = shape;
      shape_conv[1] = shape_weight[0];
      int conv_size =
          shape_conv[0] * shape_conv[1] * shape_conv[2] * shape_conv[3];
      std::vector<float> zeroConvVec(conv_size);
      Value zeroConv =
          createTensor(rewriter, loc, context, shape_conv, zeroConvVec);
      Value randomConv = rewriter.create<AtenConvolutionOp>(
          loc, zeroConv.getType(), maxPool2dOp, Weight, Bias, list_stride,
          list_padding, list_dilation, constFalse, list, int1);
      values[i + 1] =
          rewriter.create<AtenReluOp>(loc, zeroConv.getType(), randomConv);
    }
  }

  // list
  auto vtensorType = ValueTensorType::getWithLeastStaticInformation(context);
  Value list_cat = rewriter.create<PrimListConstructOp>(
      loc, ListType::get(vtensorType), ValueRange(values));

  // cat
  auto shape_cat = shape;
  for (int i = 0; i < number_branch; i++) {
    shape_cat[1] += shape_dim_1[i];
  }
  int size_cat = shape_cat[0] * shape_cat[1] * shape_cat[2] * shape[3];
  std::vector<float> catVec(size_cat);
  auto resultTensorType = ValueTensorType::get(
      context, llvm::ArrayRef(shape_cat), rewriter.getF32Type());
  auto dense = DenseElementsAttr::get(
      RankedTensorType::get(llvm::ArrayRef(shape_cat), rewriter.getF32Type()),
      llvm::ArrayRef(catVec));
  Value zeroCat =
      rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
  Value cat =
      rewriter.create<AtenCatOp>(loc, zeroCat.getType(), list_cat, int1);

  // conv
  auto shape_weight = shape;
  shape_weight[0] = shape[1];
  shape_weight[1] = shape_cat[1];
  shape_weight[2] = shape_weight[3] = 1;
  int weightSize =
      shape_weight[0] * shape_weight[1] * shape_weight[2] * shape_weight[3];
  std::vector<float> WeightVec(weightSize, 0);
  int index;
  for (int i = 0; i < shape_weight[0]; i++) {
    index = i * shape_weight[1]; // 计算当前i对应的起始下标
    WeightVec[index + i] = 1;    // 将WeightVec[i][i][0][0]设置为1
  }
  resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape_weight),
                                          rewriter.getF32Type());
  dense =
      DenseElementsAttr::get(RankedTensorType::get(llvm::ArrayRef(shape_weight),
                                                   rewriter.getF32Type()),
                             llvm::ArrayRef(WeightVec));
  Value Weight =
      rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
  // bias
  auto shape_bias = shape;
  shape_bias.erase(shape_bias.begin() + 1, shape_bias.end());
  shape_bias[0] = shape_weight[0];
  std::vector<float> BiasVec(shape_bias[0], 0);
  resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape_bias),
                                          rewriter.getF32Type());
  dense = DenseElementsAttr::get(
      RankedTensorType::get(llvm::ArrayRef(shape_bias), rewriter.getF32Type()),
      llvm::ArrayRef(BiasVec));
  Value Bias =
      rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
  Value list_padding = rewriter.create<PrimListConstructOp>(
      loc, ListType::get(IntType::get(context)), ValueRange({int0, int0}));
  Value conv = rewriter.create<AtenConvolutionOp>(
      loc, rst.getType(), cat, Weight, Bias, list_stride, list_padding,
      list_dilation, constFalse, list, int1);

  if (needReshape)
    conv = createReshape(rewriter, loc, context, shapeOrigin, rst);

  rewriter.replaceOp(op, conv);
}

namespace {
class InsertInceptionPass : public InsertInceptionBase<InsertInceptionPass> {
public:
  InsertInceptionPass() = default;
  InsertInceptionPass(int number, int layer) {
    this->number = number, this->layer = layer;
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    insertInception(context, f, number, layer);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createInsertInceptionPass(int number, int layer) {
  return std::make_unique<InsertInceptionPass>(number, layer);
}
