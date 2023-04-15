#include "Common.h"

bool getConvMiddleOps(OpList &oplist, Operation *f, int layer) {
  int convLayer = layer;
  f->walk([&](Operation *op) {
    if (isa<AtenConvolutionOp>(op)) {
      convLayer--;
      if (convLayer == -1)
        oplist.insert(op);
    }
    if (convLayer == 0)
      oplist.insert(op);
  });
  // input test
  input_assert_ret(convLayer > -1, false, "layer < max_layer(%d) \n",
                   (layer - convLayer)) return true;
}
bool getConvOp(OpList &oplist, Operation *f, int layer) {
  int convLayer = layer;
  f->walk([&](Operation *op) {
    if (isa<AtenConvolutionOp>(op)) {
      convLayer--;
      if (convLayer == 0)
        oplist.insert(op);
    }
  });
  // input test
  input_assert_ret(convLayer > 0, false, "layer <= max_layer(%d) \n",
                   (layer - convLayer)) return true;
}

void creatOneTensor(vector<float> &ktensor, int64_t len) {
  for (int i = 0; i < len; i++) {
    ktensor[i * len + i] = 1.0;
  }
}
void copyTensor(std::vector<float> &ktensor, ValueTensorLiteralOp tensor) {
  for (auto i : tensor.getValue().getValues<float>()) {
    ktensor.push_back(i);
  }
}

Value createTensor(IRRewriter &rewriter, Location loc, MLIRContext *context,
                   std::vector<long> shape, std::vector<float> weight) {
  auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                               rewriter.getF32Type());
  auto dense = DenseElementsAttr::get(
      RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()),
      llvm::ArrayRef(weight));
  return rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
}

Value createReshape(IRRewriter &rewriter, Location loc, MLIRContext *context,
                    std::vector<long> shape, Value originVal) {
  // reshape originVal to according shape
  std::vector<Value> values;
  for (auto i : shape) {
    values.push_back(
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(i)));
  }
  Value listShape = rewriter.create<PrimListConstructOp>(
      loc, ListType::get(IntType::get(context)), ValueRange(values));
  Type resultType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                         rewriter.getF32Type());
  return rewriter.create<AtenViewOp>(loc, resultType, originVal, listShape);
}

llvm::SmallPtrSet<Operation *, 16> getPositiveLayers(Operation *f) {
  // get ops which output is positive
  llvm::SmallPtrSet<Operation *, 16> opWorklist;
  f->walk([&](Operation *op) {
    if (isa<AtenReluOp, AtenSigmoidOp>(op)) {
      if (op->getResult(0).getType().isa<ValueTensorType>()) {
        opWorklist.insert(op);
      }
    }
  });
  return opWorklist;
}