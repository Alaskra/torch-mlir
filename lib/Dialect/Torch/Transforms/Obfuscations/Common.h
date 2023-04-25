#pragma once

#include "../PassDetail.h"

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
using std::vector;

// **************frequently-used macro**************
// assert macro
#include <cstdio>
#define llvm_assert(exp, ...)                                                  \
  if (exp) {                                                                   \
    printf(__VA_ARGS__);                                                       \
    return;                                                                    \
  }
#define input_assert(exp, ...)                                                 \
  llvm_assert(exp, "input error, require: " __VA_ARGS__)
#define llvm_assert_ret(exp, ret, ...)                                         \
  if (exp) {                                                                   \
    printf(__VA_ARGS__);                                                       \
    return ret;                                                                \
  }
#define input_assert_ret(exp, ret, ...)                                        \
  llvm_assert_ret(exp, ret, "input error, require: " __VA_ARGS__)
// debug macro
#define print_line() printf("line = %d\n", __LINE__)
#define print_value(value) llvm::outs() << value << '\n'

// **************frequently-used function***************

// frequently-used function about getting ops
typedef llvm::SmallPtrSet<Operation *, 16> OpList;
bool getConvMiddleOps(OpList &oplist, Operation *f, int layer);
bool getConvOp(OpList &oplist, Operation *f, int layer);
int getReluOp(OpList &oplist, Operation *f, int layer);

// frequently-used function about tensor
inline vector<int64_t> getShape(Value tensorOp) {
  // kernel shape: out_channels, in_channels, height, width
  // bias shape: out_channels
  return tensorOp.getType().cast<ValueTensorType>().getSizes().vec();
}
inline ValueTensorLiteralOp getTensor(Value tensorOp) {
  return tensorOp.getDefiningOp<ValueTensorLiteralOp>();
}
inline void toStdShape(vector<int64_t> &shape) {
  shape[0] = shape[1];
  shape[2] = shape[3] = 1;
}
inline void toBiasShape(vector<int64_t> &kernelShape) {
  kernelShape.erase(kernelShape.begin() + 1, kernelShape.end());
}
inline int getChannelSize(vector<int64_t> kernelShape) {
  return kernelShape[1] * kernelShape[2] * kernelShape[3];
}
inline int getKernelSize(vector<int64_t> kernelShape) {
  return kernelShape[0] * kernelShape[1] * kernelShape[2] * kernelShape[3];
}

inline void pushBackVec(vector<float> &ktensor, vector<float> source, int start, int size) {
  ktensor.insert(ktensor.end(), source.begin() + start, source.begin() + start + size);
}
inline void pushBackVec(vector<float> &ktensor, int start, int size) {
  pushBackVec(ktensor, ktensor, start, size);
}
void copyTensor(vector<float> &ktensor, ValueTensorLiteralOp tensor);
void creatOneTensor(vector<float> &ktensor, int64_t len);


Value createTensor(IRRewriter &rewriter, Location loc, MLIRContext *context,
                   std::vector<long> shape, std::vector<float> weight);
Value createReshape(IRRewriter &rewriter, Location loc, MLIRContext *context,
                    std::vector<long> shape, Value originVal);
SmallPtrSet<Operation *, 16> getPositiveLayers(Operation *f);



// **************class about rewriter***************
class RewriteOp {
private:
  MLIRContext *context;
  IRRewriter rewriter;
  Location loc;
  Operation *op;
public:
  RewriteOp(MLIRContext *context, Operation *op);
  Operation *cloneOp();
  // create operations
  Value createBoolOp(bool value);
  Value createIntOp(int64_t value);
  Value createFloatOp(double value);
  Value createTensorOp(vector<int64_t> shape, vector<float> tensor);
  Value createAddTensorOp(Value tensor1, Value tensor2, Value alpha);
  Value createSliceTensorOp(vector<int64_t> branchShape, Value input, Value dim, Value start, Value end);
  Value createListOp(Type elemType, vector<Value> elemVec);
  Value createCatTensorOp(vector<int64_t> resultShape, Value dim, vector<Value> tensorVec);
  Value createConvOp(Type result, ValueRange convParam, Value groupOp);
  Value createConvOp(Type result, ValueRange convParam);
  Value createConvOp(ValueRange convParam);
  Value createReluOp(Value inputOp);
  Value createLeakyReluOp(Value inputOp, Value nslope);
  Value createLeakyReluOp(Value inputOp);
  Value createReluOp(int type, Value inputOp);
  //replace operations
  void replaceTensorOp(ValueTensorLiteralOp &oldTensor, vector<int64_t> shape, vector<float> tensor);
  void replaceOp(Value newOp);
  // about tensor 
  ValueTensorType getValueTensorType(vector<int64_t> shape);
  ValueTensorType getLeastValueTensorType();
  DenseElementsAttr getTensorDense(vector<int64_t> shape, vector<float> tensor);
};

// *********************macro for pass********************************
// handle param
#define type_param(type, param) type param
#define notype_param(type, param) param
#define this_param(type, param) this->param
#define init_param(type, param) this->param = param
// handle param list
#define handle_param(n, micro, ...) handle_param##n(micro, __VA_ARGS__)
#define handle_param1(micro, type1, param1) micro(type1, param1)
#define handle_param2(micro, type1, param1, type2, param2)                     \
  micro(type1, param1), micro(type2, param2)
// namespace, class, function
#define use_pass(name, n, ...)                                                 \
  namespace {                                                                  \
  class name##Pass : public name##Base<name##Pass> {                           \
  public:                                                                      \
    name##Pass() = default;                                                    \
    name##Pass(handle_param(n, type_param, __VA_ARGS__)) {                     \
      handle_param(n, init_param, __VA_ARGS__);                                \
    }                                                                          \
    void runOnOperation() override {                                           \
      MLIRContext *context = &getContext();                                    \
      auto f = getOperation();                                                 \
      name(context, f, handle_param(n, this_param, __VA_ARGS__));              \
    }                                                                          \
  };                                                                           \
  }                                                                            \
  std::unique_ptr<OperationPass<func::FuncOp>>                                 \
      mlir::torch::Torch::create##name##Pass(                                  \
          handle_param(n, type_param, __VA_ARGS__)) {                          \
    return std::make_unique<name##Pass>(                                       \
        handle_param(n, notype_param, __VA_ARGS__));                           \
  }
