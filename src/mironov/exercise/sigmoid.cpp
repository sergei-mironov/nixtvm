/*
Inestigate how to perform scheduling in C++

g++ -std=c++14  scheduling0.cpp -ltvm -o scheduling0.gen
scheduling0.gen > scheduling0.s
*/

#include <random>
#include <iomanip>
#include <array>
#include <exception>

#include <tvm/tvm.h>
#include <tvm/operation.h>
#include <tvm/schedule.h>
#include <tvm/tensor.h>
#include <tvm/build_module.h>
#include <tvm/ir.h>
#include <tvm/ir_operator.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>

#include <topi/broadcast.h>
#include <topi/reduction.h>

#include <ir/IR.h>
#include <ir/IRPrinter.h>

using namespace std;
using namespace tvm;

int main()
{
  BuildConfig config = build_config();

  auto n = var("n");
  Array<Expr> shape = {n,n};
  Tensor A = placeholder(shape, Float(32), "A");
  Tensor X = compute(shape, FCompute([=](auto i){ return tvm::sigmoid(A(i)); } )) ;


  auto vecadd_lowered = ({
    Schedule s = create_schedule({X->op});
    std::unordered_map<Tensor, Buffer> binds;
    auto args = Array<Tensor>({A, X});
    auto lowered = lower(s, args, "vecadd", binds, config);
    lowered;
  });

  auto target = Target::create("llvm");
  auto target_host = Target::create("llvm");
  runtime::Module mod = build(vecadd_lowered, target, target_host, config);

  /* Output LLVM assembly to stdout */
  cout << mod->GetSource("asm") << endl;
  return 0;
}


