/*
Build and execute like this:

g++ -std=c++14  tvm1.cpp -ltvm -o tvm1.gen
tvm1.gen > tvm1.s
*/

#include <random>
#include <iomanip>
#include <array>
#include <exception>

#include <tvm/tvm.h>
#include <tvm/operation.h>
#include <tvm/tensor.h>
#include <tvm/build_module.h>
#include <topi/broadcast.h>

#include <ir/IR.h>

using namespace std;
using namespace tvm;

int main()
{
  BuildConfig config = build_config();

  auto n = var("n");
  Array<Expr> shape = {n};
  Tensor A = placeholder(shape, Float(32), "A");
  Tensor B = placeholder(shape, Float(32), "B");
  Tensor X = compute(shape, FCompute([=](auto i){ return A(i) + B(i); } )) ;

  auto vecadd_lowered = ({
    Schedule s = create_schedule({X->op});
    std::unordered_map<Tensor, Buffer> binds;
    auto args = Array<Tensor>({A, B, X});
    auto lowered = lower(s, args, "vecadd", binds, config);
    lowered;
    });

  cerr << "VECADD_LOWERED" << endl
       << "==============" << endl
       << vecadd_lowered[0]->body << endl;

  Tensor C = placeholder(shape, Float(32), "C");
  Tensor Y = compute(shape, FCompute([=](auto i){
      return HalideIR::Internal::Call::make(
        Float(32),"vecadd",
        Array<Expr>({C(i),C(i)}),
        HalideIR::Internal::Call::PureExtern,
        vecadd_lowered[0], 0);
      } )) ;


  auto double_lowered = ({
    Schedule s = create_schedule({Y->op});
    std::unordered_map<Tensor, Buffer> binds;
    auto args = Array<Tensor>({C, Y});
    auto lowered = lower(s, args, "double", binds, config);
    lowered;
    });

  cerr << "DOUBLE_LOWERED" << endl
       << "==============" << endl
       << double_lowered[0]->body << endl;

  auto target = Target::create("llvm");
  auto target_host = Target::create("llvm");
  runtime::Module mod = build(double_lowered, target, target_host, config);

  /* Output LLVM assembly to stdout */
  cout << mod->GetSource("asm") << endl;
  return 0;
}

