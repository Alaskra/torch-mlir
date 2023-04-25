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
  input_assert_ret(convLayer > -1, false, 
        "layer < max_layer(%d) \n", (layer - convLayer));
  return true;
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
  input_assert_ret(convLayer > 0, false, 
      "layer <= max_layer(%d) \n", (layer - convLayer));
  return true;
}

#define ReluList    AtenReluOp, AtenSigmoidOp
#define LeakyReluList   AtenTanhOp
#define AllReluList   ReluList, LeakyReluList
int getReluOp(OpList &oplist, Operation *f, int layer) {
  int rlayer = layer;
  int type = 0;
  f->walk([&](Operation *op) {
    if (isa<AllReluList>(op)) {
      rlayer--;
      if (rlayer == 0) {
        oplist.insert(op);
        if (isa<AllReluList>(op)) {
          type = 1;
        } else {
          type = 2;
        }
      }
    }
  });
  // input test
  input_assert_ret(rlayer > 0, type, 
      "layer <= max_layer(%d) \n", (layer - rlayer));
  return type;
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



// frequently-used class about rewriter
RewriteOp::RewriteOp(MLIRContext *context, Operation *op)  :
      context(context), rewriter(context), loc(op->getLoc()), op(op) {
  rewriter.setInsertionPoint(op);
}
Operation *RewriteOp::cloneOp() {
  return rewriter.clone(*(this->op));
}
// create operations
Value RewriteOp::createBoolOp(bool value) {
  return rewriter.create<ConstantBoolOp>(this->loc, value);
}
Value RewriteOp::createIntOp(int64_t value) {
  return rewriter.create<ConstantIntOp>(this->loc, rewriter.getI64IntegerAttr(value));
}
Value RewriteOp::createFloatOp(double value) {
  return rewriter.create<ConstantFloatOp>(this->loc, rewriter.getF64FloatAttr(value));
}
Value RewriteOp::createTensorOp(vector<int64_t> shape, vector<float> tensor) {
  auto tensorType = getValueTensorType(shape);
  auto tensorDense = getTensorDense(shape, tensor);
  return rewriter.create<ValueTensorLiteralOp>(this->loc, tensorType, tensorDense);
}
Value RewriteOp::createAddTensorOp(Value tensor1, Value tensor2, Value alpha) {
  return rewriter.create<AtenAddTensorOp>(this->loc, tensor1.getType(), tensor1, tensor2, alpha);
}
Value RewriteOp::createSliceTensorOp(vector<int64_t> branchShape, Value input, Value dim, Value start, Value end) {
  auto branchTensorType = getValueTensorType(branchShape);
  auto step = createIntOp(1);
  return rewriter.create<AtenSliceTensorOp>(
      this->loc, branchTensorType, input, dim, start, end, step);
}
Value RewriteOp::createListOp(Type elemType, vector<Value> elemVec) {
  return rewriter.create<PrimListConstructOp>(
      this->loc, ListType::get(elemType), ValueRange(elemVec));
}
Value RewriteOp::createCatTensorOp(vector<int64_t> resultShape, Value dim, vector<Value> tensorVec) {
  auto vtensorType = getLeastValueTensorType();
  auto tensorList = createListOp(vtensorType, tensorVec);
  auto resultType = getValueTensorType(resultShape);
  return rewriter.create<AtenCatOp>(this->loc, resultType, tensorList, dim);
}
Value RewriteOp::createConvOp(Type result, ValueRange convParam, Value groupOp) {
  Type intType = IntType::get(this->context);
  Value strideOp = createIntOp(1);
  Value padOp = createIntOp(0);
  Value dilOp = createIntOp(1);
  Value liststrideOp = createListOp(intType, {strideOp, strideOp});
  Value listPadOp = createListOp(intType, {padOp, padOp});
  Value listDilOp = createListOp(intType, {dilOp, dilOp});
  Value transOp = createBoolOp(false);
  Value outPadOp = createListOp(intType, {padOp, padOp});
  // convParam[0]:input, convParam[1]:weight, convParam[2]: bias
  return rewriter.create<AtenConvolutionOp>(
      loc, result, convParam[0], convParam[1], convParam[2], liststrideOp,
      listPadOp, listDilOp, transOp, outPadOp, groupOp);
}
Value RewriteOp::createConvOp(Type result, ValueRange convParam) {
  Value groupOp = createIntOp(1);
  return createConvOp(result, convParam, groupOp);
}
Value RewriteOp::createConvOp(ValueRange convParam) {
  return createConvOp(convParam[0].getType(), convParam);
}
Value RewriteOp::createReluOp(Value inputOp) {
  return rewriter.create<AtenReluOp>(this->loc, inputOp.getType(), inputOp);
}
Value RewriteOp::createLeakyReluOp(Value inputOp, Value nslope) {
  return rewriter.create<AtenLeakyReluOp>(this->loc, inputOp.getType(), inputOp, nslope);
}
Value RewriteOp::createLeakyReluOp(Value inputOp) {
  Value nslope = createFloatOp(1e-02);
  return createLeakyReluOp(inputOp, nslope);
}
Value RewriteOp::createReluOp(int type, Value inputOp) {
  Value relu;
  if (type == 1) {
    relu = createReluOp(inputOp);
  } else {
    relu = createLeakyReluOp(inputOp);
  }
  return relu;
}
//replace operations
void RewriteOp::replaceTensorOp(ValueTensorLiteralOp &oldTensor, vector<int64_t> shape, vector<float> tensor) {
  auto tensorType = getValueTensorType(shape);
  auto tensorDense = getTensorDense(shape, tensor);
  rewriter.replaceOpWithNewOp<ValueTensorLiteralOp>(oldTensor, tensorType, tensorDense);
}
void RewriteOp::replaceOp(Value newOp) {
  rewriter.replaceOp(this->op, newOp);
}
// about tensor 
ValueTensorType RewriteOp::getValueTensorType(vector<int64_t> shape) {
  return ValueTensorType::get(this->context, llvm::ArrayRef(shape), rewriter.getF32Type());
}
ValueTensorType RewriteOp::getLeastValueTensorType() {
  return ValueTensorType::getWithLeastStaticInformation(this->context);
}
DenseElementsAttr RewriteOp::getTensorDense(vector<int64_t> shape, vector<float> tensor) {
  return DenseElementsAttr::get(
      RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()),
      llvm::ArrayRef(tensor));
}
