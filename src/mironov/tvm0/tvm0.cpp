#include <random>
#include <iomanip>
#include <array>
#include <exception>

// #define NDEBUG
#include <tvm/tvm.h>
#include <tvm/operation.h>
#include <tvm/tensor.h>
#include <tvm/build_module.h>
#include <topi/broadcast.h>
// #undef NDEBUG

using namespace std;

int main()
{
	// Defining the computatoin of a vector addition
	// TOPI (Tensor Operator Inventory) id very powerful that it per-defines a lot
	// of common operations for us,
	auto n = tvm::var("n");
	tvm::Array<tvm::Expr> shape = {n};
	tvm::Tensor A = tvm::placeholder(shape, tvm::Float(32), "A");
	tvm::Tensor B = tvm::placeholder(shape, tvm::Float(32), "B");
	//tvm::Tensor C = topi::broadcast_add(A, B, "C");
  //tvm::Expr E = A[0] + B[0];

	tvm::Tensor C = tvm::compute(shape, tvm::FCompute([=](auto i){ return A(i) + B(i); } )) ;

	tvm::Schedule s = tvm::create_schedule({C->op});
	tvm::BuildConfig config = tvm::build_config();
	std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
	auto args = tvm::Array<tvm::Tensor>({A, B, C});
	auto lowered = tvm::lower(s, args, "vecadd", binds, config);

  cerr << lowered << endl;

	auto target = tvm::Target::create("llvm");
	auto target_host = tvm::Target::create("llvm");
	tvm::runtime::Module mod = tvm::build(lowered, target, target_host, config);

  cout << mod->GetSource("asm") << endl;

  /*
	// Define a Schedule
	tvm::Schedule s = tvm::create_schedule({C->op});
  auto axis = C->op.as<tvm::ComputeOpNode>()->axis;

  tvm::IterVar bx, tx;
  s[C].split(axis[0], 1, &bx, &tx);
  s[C].bind(bx, tvm::thread_axis(tvm::Range(), "blockIdx.x"));
  s[C].bind(tx, tvm::thread_axis(tvm::Range(), "threadIdx.x"));

	tvm::BuildConfig config = tvm::build_config();
	std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
	auto args = tvm::Array<tvm::Tensor>({A, B, C});
	tvm::Array<tvm::LoweredFunc> lowered = tvm::lower(s, args, "vecadd", binds, config);

  cout << tvm::NodeRef(lowered) << endl;

	auto target = tvm::Target::create("llvm");
	auto target_host = tvm::Target::create("llvm");
	tvm::runtime::Module mod = tvm::build(lowered, target, target_host, config);

  cout << mod->GetSource() << endl;
  */

  return 0;
}

