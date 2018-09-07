#include <tvm/tvm.h>
#include <tvm/operation.h>
#include <tvm/runtime/packed_func.h>
/* #include <tvm/tensor.h> */
/* #include <tvm/build_module.h> */
/* #include <topi/reduction.h> */

#include <ir/IROperator.h>
#include <ir/IR.h>

#include <tvm/relay/base.h>
#include <tvm/relay/type.h>
#include <tvm/relay/expr.h>

using namespace std;
using namespace tvm;
using namespace tvm::runtime;


extern "C" {
int test_dispatch();
int test_call_node(void *c);
};

int test_dispatch()
{
  using namespace HalideIR::Internal;
  using namespace HalideIR;


  IRFunctor<std::string (const NodeRef& n, std::string prefix)> tostr;

  tostr.set_dispatch<Add>([](const Add* op, std::string prefix) {
    return prefix + ":" + "Add";
  });
  tostr.set_dispatch<IntImm>([](const IntImm* op, std::string prefix) {
    return prefix + ":" + "IntImm";
  });


  Expr x = make_const(Int(32),1);
  Expr y = x + x;
  // dispatch to IntImm, outputs "MyIntImm"
  LOG(INFO) << tostr(x, "My");
  // dispatch to IntImm, outputs "MyAdd"
  LOG(INFO) << tostr(y, "My");

  return 0;
}

int test_call_node(void *c_) {
  tvm::relay::CallNode *c = (tvm::relay::CallNode*)c_;
  LOG(INFO) << c->_type_key << " " << c->args.size();
}


TVM_REGISTER_GLOBAL("test1")
.set_body([](TVMArgs args, TVMRetValue* rv) {

  LOG(INFO) << "test1: "
    << args[0].node_sptr()->derived_from<Node>() << " "
    << args[0].node_sptr()->derived_from<relay::RelayNode>() << " "
    << args[0].node_sptr()->derived_from<relay::ExprNode>() << " "
    << args[0].node_sptr()->derived_from<relay::SourceNameNode>();
});

template<typename T>
T* mycast(TVMArgValue v) {
  CHECK(v.node_sptr()->derived_from<T>()) << "cast: Wrong type";
  return static_cast<T*>(v.node_sptr().get());
}


TVM_REGISTER_GLOBAL("test2")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  using namespace relay;

  relay::ExprNode *e = mycast<relay::ExprNode>(args[0]);

  LOG(INFO) << e << " " << e->_type_key;

});



/*
int test_relay() {
}

int main() {
  test_dispatch();
}
*/
