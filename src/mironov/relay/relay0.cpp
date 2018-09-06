#include <tvm/tvm.h>
#include <tvm/operation.h>
/* #include <tvm/tensor.h> */
/* #include <tvm/build_module.h> */
/* #include <topi/reduction.h> */

#include <ir/IROperator.h>
#include <ir/IR.h>

#include <tvm/relay/type.h>
#include <tvm/relay/expr.h>

using namespace std;
using namespace tvm;

using namespace HalideIR::Internal;
using namespace HalideIR;


extern "C" {
int test_dispatch();
int test_call_node(void *c);
};

int test_dispatch()
{

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

/*
int test_relay() {
}

int main() {
  test_dispatch();
}
*/
