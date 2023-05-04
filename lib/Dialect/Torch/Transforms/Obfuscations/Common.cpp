#include "Common.h"

// frequently-used function about getting ops
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
                   (layer - convLayer));
  return true;
}

#define ReluList AtenReluOp, AtenSigmoidOp
#define LeakyReluList AtenTanhOp
#define AllReluList ReluList, LeakyReluList
Operation *getReluOp(Operation *f, int layer) {
  int rlayer = layer;
  Operation *reluOp = nullptr;
  f->walk([&](Operation *op) {
    if (isa<AllReluList>(op)) {
      rlayer--;
      auto resType = op->getResult(0).getType();
      if (rlayer == 0 && resType.isa<ValueTensorType>()) {
        reluOp = op;
      }
    }
  });
  // input test
  if (rlayer > 0) {
    printf("layer <= max_layer(%d) \n", (layer - rlayer));
  }
  return reluOp;
}
int getReluType(Operation *reluOp) {
  int type = 1;
  if (isa<LeakyReluList>(reluOp))
    type = 2;
  return type;
}

// frequently-used function about tensor and shape
vector<int64_t> getShape(Value tensorOp) {
  // kernel shape: out_channels, in_channels, height, width
  // bias shape: out_channels
  return tensorOp.getType().cast<ValueTensorType>().getSizes().vec();
}
ValueTensorLiteralOp getTensor(Value tensorOp) {
  return tensorOp.getDefiningOp<ValueTensorLiteralOp>();
}
vector<int64_t> toStdShape(vector<int64_t> shape) {
  // if dimansion less than 4, need to reshape to 4
  vector<int64_t> newShape(4, 1);
  for (int i = 4 - shape.size(); i < 4; i++) {
    newShape[i] = shape[i - shape.size()];
  }
  return newShape;
}
void toBiasShape(vector<int64_t> &kernelShape) {
  kernelShape.erase(kernelShape.begin() + 1, kernelShape.end());
}
int getChannelSize(vector<int64_t> kernelShape) {
  return kernelShape[1] * kernelShape[2] * kernelShape[3];
}
int getKernelSize(vector<int64_t> kernelShape) {
  return kernelShape[0] * kernelShape[1] * kernelShape[2] * kernelShape[3];
}
// frequently-used function about tensor
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

// zwj: frequently-used function
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

llvm::SmallPtrSet<Operation *, 16> getReluLayers(Operation *f) {
  // get ops which output is positive
  llvm::SmallPtrSet<Operation *, 16> opWorklist;
  f->walk([&](Operation *op) {
    if (isa<AtenReluOp>(op)) {
      if (op->getResult(0).getType().isa<ValueTensorType>()) {
        opWorklist.insert(op);
      }
    }
  });
  return opWorklist;
}
llvm::SmallPtrSet<Operation *, 16> getReluLayers(Operation *f) {
  // get ops which output is positive
  llvm::SmallPtrSet<Operation *, 16> opWorklist;
  f->walk([&](Operation *op) {
    if (isa<AtenReluOp>(op)) {
      if (op->getResult(0).getType().isa<ValueTensorType>()) {
        opWorklist.insert(op);
      }
    }
  });
  return opWorklist;
}

llvm::SmallPtrSet<Operation *, 16> getConvLayers(Operation *f) {
  // get ops which output is positive
  llvm::SmallPtrSet<Operation *, 16> opWorklist;
  f->walk([&](Operation *op) {
    if (isa<AtenConvolutionOp>(op)) {
      if (op->getResult(0).getType().isa<ValueTensorType>()) {
        opWorklist.insert(op);
      }
    }
  });
  return opWorklist;
}