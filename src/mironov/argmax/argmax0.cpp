#include <tvm/tvm.h>
#include <tvm/operation.h>
#include <tvm/tensor.h>
#include <tvm/build_module.h>
#include <topi/reduction.h>

using namespace std;
using namespace tvm;

int main()
{
  Var row("row");
  Tensor A = placeholder({row}, Float(32), "A");
  Tensor C = topi::argmax(A, {}, false);

	tvm::Schedule s = tvm::create_schedule({C->op});
	tvm::BuildConfig config = tvm::build_config();
	std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
	auto args = tvm::Array<tvm::Tensor>({A, C});
	auto lowered = tvm::lower(s, args, "vecargmax", binds, config);

  cerr << lowered[0]->body << endl;

	auto target = tvm::Target::create("llvm");
	auto target_host = tvm::Target::create("llvm");
	tvm::runtime::Module mod = tvm::build(lowered, target, target_host, config);

  cout << mod->GetSource("asm") << endl;

  return 0;
}
