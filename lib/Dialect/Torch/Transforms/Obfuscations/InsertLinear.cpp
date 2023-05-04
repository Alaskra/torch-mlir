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

namespace {
// LUP分解
void LUP_Descomposition(float A[], float L[], float U[], int P[], int N) {
  int row = 0;
  for (int i = 0; i < N; i++) {
    P[i] = i;
  }
  for (int i = 0; i < N - 1; i++) {
    float p = 0.0;
    for (int j = i; j < N; j++) {
      if (std::abs(A[j * N + i]) > p) {
        p = std::abs(A[j * N + i]);
        row = j;
      }
    }
    if (0 == p) {
      llvm::errs() << "矩阵奇异，无法计算逆\n";
      return;
    }

    // 交换P[i]和P[row]
    int tmp = P[i];
    P[i] = P[row];
    P[row] = tmp;

    float tmp2 = 0.0;
    for (int j = 0; j < N; j++) {
      // 交换A[i][j]和 A[row][j]
      tmp2 = A[i * N + j];
      A[i * N + j] = A[row * N + j];
      A[row * N + j] = tmp2;
    }

    // 以下同LU分解
    float u = A[i * N + i], l = 0.0;
    for (int j = i + 1; j < N; j++) {
      l = A[j * N + i] / u;
      A[j * N + i] = l;
      for (int k = i + 1; k < N; k++) {
        A[j * N + k] = A[j * N + k] - A[i * N + k] * l;
      }
    }
  }

  // 构造L和U
  for (int i = 0; i < N; i++) {
    for (int j = 0; j <= i; j++) {
      if (i != j) {
        L[i * N + j] = A[i * N + j];
      } else {
        L[i * N + j] = 1;
      }
    }
    for (int k = i; k < N; k++) {
      U[i * N + k] = A[i * N + k];
    }
  }
}

// LUP求解方程
float *LUP_Solve(float L[], float U[], int P[], float b[], int N) {
  float *x = new float[N]();
  float *y = new float[N]();

  // 正向替换
  for (int i = 0; i < N; i++) {
    y[i] = b[P[i]];
    for (int j = 0; j < i; j++) {
      y[i] = y[i] - L[i * N + j] * y[j];
    }
  }
  // 反向替换
  for (int i = N - 1; i >= 0; i--) {
    x[i] = y[i];
    for (int j = N - 1; j > i; j--) {
      x[i] = x[i] - U[i * N + j] * x[j];
    }
    x[i] /= U[i * N + i];
  }
  return x;
}

/*****************矩阵原地转置BEGIN********************/

/* 后继 */
int getNext(int i, int m, int n) { return (i % n) * m + i / n; }

/* 前驱 */
int getPre(int i, int m, int n) { return (i % m) * n + i / m; }

/* 处理以下标i为起点的环 */
void movedata(float *mtx, int i, int m, int n) {
  float temp = mtx[i]; // 暂存
  int cur = i;         // 当前下标
  int pre = getPre(cur, m, n);
  while (pre != i) {
    mtx[cur] = mtx[pre];
    cur = pre;
    pre = getPre(cur, m, n);
  }
  mtx[cur] = temp;
}

/* 转置，即循环处理所有环 */
void transpose(float *mtx, int m, int n) {
  for (int i = 0; i < m * n; ++i) {
    int next = getNext(i, m, n);
    while (
        next >
        i) // 若存在后继小于i说明重复,就不进行下去了（只有不重复时进入while循环）
      next = getNext(next, m, n);
    if (next == i) // 处理当前环
      movedata(mtx, i, m, n);
  }
}

// LUP求逆(将每列b求出的各列x进行组装)
float *LUP_solve_inverse(float A[], int N) {
  // todo: 内存泄漏，先不管
  // 创建矩阵A的副本，注意不能直接用A计算，因为LUP分解算法已将其改变
  float *A_mirror = new float[N * N]();
  float *inv_A = new float[N * N]();  // 最终的逆矩阵（还需要转置）
  float *inv_A_each = new float[N](); // 矩阵逆的各列
  // float *B    =new float[N*N]();
  float *b = new float[N](); // b阵为B阵的列矩阵分量

  for (int i = 0; i < N; i++) {
    float *L = new float[N * N]();
    float *U = new float[N * N]();
    int *P = new int[N]();

    // 构造单位阵的每一列
    for (int i = 0; i < N; i++) {
      b[i] = 0;
    }
    b[i] = 1;

    // 每次都需要重新将A复制一份
    for (int i = 0; i < N * N; i++) {
      A_mirror[i] = A[i];
    }

    LUP_Descomposition(A_mirror, L, U, P, N);

    inv_A_each = LUP_Solve(L, U, P, b, N);
    memcpy(inv_A + i * N, inv_A_each, N * sizeof(float)); // 将各列拼接起来
  }
  transpose(inv_A, N, N); // 由于现在根据每列b算出的x按行存储，因此需转置

  return inv_A;
}
} // namespace

static std::vector<long> createNewShape(std::vector<long> shapeOrigin) {
  // if dimansion more than 2, need to reshape to 2
  // for example: (1,2,3,4) -> (6,4)
  std::vector<long> shapeNew;
  int mul = 1;
  for (unsigned long i = 0; i < shapeOrigin.size() - 1; ++i) {
    mul *= shapeOrigin[i];
  }
  shapeNew.push_back(mul);
  shapeNew.push_back(shapeOrigin[shapeOrigin.size() - 1]);
  return shapeNew;
}

static std::vector<Value> createABCD(IRRewriter &rewriter, Location loc,
                                     MLIRContext *context, long N) {
  std::vector<long> shapeWeight{N, N};
  std::vector<long> shapeBias = {N};

  // generate A, B, C, D, satisfy (xA+B)C+D == x
  float *A = new float[N * N]();
  float *B = new float[N]();
  float *C;
  float *D = new float[N]();
  // srand((unsigned)time(0));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = rand() % 100 * 0.01;
    }
    B[i] = rand() % 100 * 0.01;
  }
  C = LUP_solve_inverse(A, N);
  float sum;
  for (int i = 0; i < N; i++) {
    sum = 0;
    for (int j = 0; j < N; j++) {
      sum += B[j] * C[j * N + i];
    }
    D[i] = -sum;
  }

  // weight A
  Value weightA = createTensor(rewriter, loc, context, shapeWeight,
                               std::vector<float>(A, A + N * N));
  Value biasB = createTensor(rewriter, loc, context, shapeBias,
                             std::vector<float>(B, B + N));
  Value weightC = createTensor(rewriter, loc, context, shapeWeight,
                               std::vector<float>(C, C + N * N));
  Value biasD = createTensor(rewriter, loc, context, shapeBias,
                             std::vector<float>(D, D + N));
  return std::vector<Value>{weightA, biasB, weightC, biasD};
}

static void insertLinearRNN(MLIRContext *context,
                            SmallPtrSet<Operation *, 16> opWorklist) {
  // insert 2 linear layer for every op in opWorklist
  // special for RNN: hidden layer in loop share the same weight
  // prerequest: all ops in opWorklist is same op in unrolling RNN loop

  IRRewriter rewriter(context);
  Operation *op = *opWorklist.begin();
  rewriter.setInsertionPoint(op);
  Location loc = op->getLoc();

  // create reusable ops
  Value int1 = rewriter.create<ConstantIntOp>(op->getLoc(),
                                              rewriter.getI64IntegerAttr(1));
  std::vector<long> shapeOrigin =
      op->getResult(0).getType().cast<ValueTensorType>().getSizes().vec();
  std::vector<long> shapeNew;
  bool needReshape = true;
  if (shapeOrigin.size() == 2) {
    needReshape = false;
    shapeNew = shapeOrigin;
  } else {
    shapeNew = createNewShape(shapeOrigin);
  }
  std::vector<Value> values = createABCD(rewriter, loc, context, shapeNew[1]);

  for (auto op : opWorklist) {
    rewriter.setInsertionPointAfter(op);
    // copy op, for convinience of replace use of op
    Operation *newOp = rewriter.clone(*op);
    Location loc = newOp->getLoc();
    Value rst = newOp->getResult(0);

    if (needReshape)
      rst = createReshape(rewriter, loc, context, shapeNew, rst);
    // create 2 linear layer
    rst = rewriter.create<AtenMmOp>(loc, rst.getType(), rst, values[0]);
    rst = rewriter.create<AtenAddTensorOp>(loc, rst.getType(), rst, values[1],
                                           int1);
    rst = rewriter.create<AtenReluOp>(loc, rst.getType(), rst);
    rst = rewriter.create<AtenMmOp>(loc, rst.getType(), rst, values[2]);
    rst = rewriter.create<AtenAddTensorOp>(loc, rst.getType(), rst, values[3],
                                           int1);
    rst = rewriter.create<AtenReluOp>(loc, rst.getType(), rst);
    // reshape back
    if (needReshape)
      rst = createReshape(rewriter, loc, context, shapeOrigin, rst);

    rewriter.replaceOp(op, rst);
  }
}

static void insertLinear(MLIRContext *context,
                         llvm::SmallPtrSet<Operation *, 16> opWorklist) {
  // insert 2 linear layer for every op in opWorklist

  IRRewriter rewriter(context);

  for (auto op : opWorklist) {
    rewriter.setInsertionPointAfter(op);
    // copy op, for convinience of replace use of op
    Operation *newOp = rewriter.clone(*op);
    Location loc = newOp->getLoc();
    Value rst = newOp->getResult(0);

    Value int1 = rewriter.create<ConstantIntOp>(op->getLoc(),
                                                rewriter.getI64IntegerAttr(1));
    std::vector<long> shapeOrigin =
        op->getResult(0).getType().cast<ValueTensorType>().getSizes().vec();
    std::vector<long> shapeNew;
    bool needReshape = true;
    if (shapeOrigin.size() == 2) {
      needReshape = false;
      shapeNew = shapeOrigin;
    } else {
      shapeNew = createNewShape(shapeOrigin);
    }
    std::vector<Value> values = createABCD(rewriter, loc, context, shapeNew[1]);

    if (needReshape)
      rst = createReshape(rewriter, loc, context, shapeNew, rst);
    // create 2 linear layer
    rst = rewriter.create<AtenMmOp>(loc, rst.getType(), rst, values[0]);
    rst = rewriter.create<AtenAddTensorOp>(loc, rst.getType(), rst, values[1],
                                           int1);
    rst = rewriter.create<AtenReluOp>(loc, rst.getType(), rst);
    rst = rewriter.create<AtenMmOp>(loc, rst.getType(), rst, values[2]);
    rst = rewriter.create<AtenAddTensorOp>(loc, rst.getType(), rst, values[3],
                                           int1);
    rst = rewriter.create<AtenReluOp>(loc, rst.getType(), rst);
    // reshape back
    if (needReshape)
      rst = createReshape(rewriter, loc, context, shapeOrigin, rst);

    rewriter.replaceOp(op, rst);
  }
}

namespace {
class InsertLinearPass : public InsertLinearBase<InsertLinearPass> {
public:
  InsertLinearPass() = default;
  InsertLinearPass(std::string net) { this->net = net; }
  void runOnOperation() override {
    auto f = getOperation();
    llvm::SmallPtrSet<Operation *, 16> opWorklist = getPositiveLayers(f);
    MLIRContext *context = &getContext();

    if (opWorklist.empty()) {
      llvm::errs() << "Not run InsertLinear\n";
      return;
    }

    if (net == "") {
      // todo: opWorklist too large will cause precision error
      while (opWorklist.size() >= 3)
        opWorklist.erase(*opWorklist.begin());
      insertLinear(context, opWorklist);
    } else if (net == "RNN") {
      insertLinearRNN(context, opWorklist);
    } else {
      llvm::errs() << "unsupported net: " << net << "\n";
      return;
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createInsertLinearPass(std::string net) {
  return std::make_unique<InsertLinearPass>(net);
}
