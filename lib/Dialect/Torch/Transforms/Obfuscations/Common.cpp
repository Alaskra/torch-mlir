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

#define ReluList    AtenReluOp, AtenSigmoidOp
#define LeakyReluList   AtenTanhOp
#define AllReluList   ReluList, LeakyReluList
int getReluOp(OpList &oplist, Operation *f, int layer) {
  int rlayer = layer;
  int type = 0;
  f->walk([&](Operation *op) {
    if (isa<AllReluList>(op)) {
      rlayer--;
      auto resType = op->getResult(0).getType();
      if (rlayer == 0 && resType.isa<ValueTensorType>()) {
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
Value RewriteOp::createConvOp(Type result, ValueRange tensorParam, vector<int64_t> intParam) {
  Type intType = IntType::get(this->context);
  // intParam: stride, pad, dil, group
  Value strideOp = createIntOp(intParam[0]);
  Value padOp = createIntOp(intParam[1]);
  Value dilOp = createIntOp(intParam[2]);
  Value liststrideOp = createListOp(intType, {strideOp, strideOp});
  Value listPadOp = createListOp(intType, {padOp, padOp});
  Value listDilOp = createListOp(intType, {dilOp, dilOp});
  Value transOp = createBoolOp(false);
  Value outPadOp = createListOp(intType, {});
  Value groupOp = createIntOp(intParam[3]);
  // tensorParam: input, weight, bias
  return rewriter.create<AtenConvolutionOp>(
      this->loc, result, tensorParam[0], tensorParam[1], tensorParam[2], 
      liststrideOp, listPadOp, listDilOp, transOp, outPadOp, groupOp);
}
Value RewriteOp::createConvOp(ValueRange tensorParam, vector<int64_t> intParam) {
  return createConvOp(tensorParam[0].getType(), tensorParam, intParam);
}
Value RewriteOp::createConvOp(Type result, ValueRange tensorParam) {
  return createConvOp(result, tensorParam, {1, 0, 1, 1});
}
Value RewriteOp::createConvOp(ValueRange tensorParam) {
  return createConvOp(tensorParam, {1, 0, 1, 1});
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
Value RewriteOp::createReshape(vector<long> shape, Value originOp) {
  // reshape originOp to according shape
  std::vector<Value> values;
  for (auto i : shape) {
    values.push_back(this->createIntOp(i));
  }
  auto intType = IntType::get(this->context);
  Value listShapeOp = createListOp(intType, values);
  auto resType = getValueTensorType(shape);
  return rewriter.create<AtenViewOp>(this->loc, resType, originOp, listShapeOp);  
}
Value RewriteOp::createMmOp(Type result, Value inputOp, Value weightOp) {
  return rewriter.create<AtenMmOp>(this->loc, result, inputOp, weightOp);
}
Value RewriteOp::createMmOp(Value inputOp, Value weightOp) {
  return this->createMmOp(inputOp.getType(), inputOp, weightOp);
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
