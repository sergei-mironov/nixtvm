#include <random>
#include <iomanip>
#include <array>
#include <exception>

#define NDEBUG
#include <tvm/tvm.h>
#include <tvm/build_module.h>
#include <topi/broadcast.h>
#undef NDEBUG

// Returns a pointer to the data if aviable
template <typename dtype>
inline dtype* TVMARRAY_ADDR(TVMArrayHandle handle)
{
	::DLTensor* dl_tensor = handle;
	uint64_t offset = dl_tensor->byte_offset; // normally this is 0
	dtype* addr = reinterpret_cast<dtype*> (static_cast<char*> (dl_tensor->data) + offset);

	return addr;
}

// returns the dimernsions of a TVMArray
inline int64_t* TVMARRAY_SHAPE(TVMArrayHandle handle)
{
	DLTensor* dl_tensor = handle;
	//Other common attributes
	//int dimensions = dl_tensor->ndim;
  //
	if(handle->shape == nullptr) // nullptr means the tensor is conpact
		throw std::runtime_error("Cannot handle comact tensor for now");

	return handle->shape;
}

int main()
{
	// Defining the computatoin of a vector addition
	// TOPI (Tensor Operator Inventory) id very powerful that it per-defines a lot
	// of common operations for us,
	auto n = tvm::var("n");
	tvm::Array<tvm::Expr> shape = {n};
	tvm::Tensor A = tvm::placeholder(shape, tvm::Float(32), "A");
	tvm::Tensor B = tvm::placeholder(shape, tvm::Float(32), "B");
	tvm::Tensor C = topi::broadcast_add(A, B, "C");

	// Define a Schedule
	tvm::Schedule s = tvm::create_schedule({C->op});

	std::string tgt = "opencl";
	bool onGPU = tgt == "opencl" || tgt == "cuda";
	//If ruunning on a GPU, bind GPU specific values
	if(onGPU == true)
	{
		auto cAxis = C->op.as<tvm::ComputeOpNode>()->axis;
		tvm::IterVar bx, tx;
		s[C].split(cAxis[0], 1, &bx, &tx);
		s[C].bind(bx, tvm::thread_axis(tvm::Range(), "blockIdx.x"));
		s[C].bind(tx, tvm::thread_axis(tvm::Range(), "threadIdx.x"));
	}

	// Build the defined computation
	tvm::BuildConfig config = tvm::build_config();
	auto target = tvm::Target::create(tgt); // Guesst device: tgt (OpenCL)
	auto target_host = tvm::Target::create("stackvm");  // Host Device: LLVM
	auto args = tvm::Array<tvm::Tensor>({A, B, C});
	std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
	tvm::Array<tvm::LoweredFunc> lowered = tvm::lower(s, args, "vecadd", binds, config);
	tvm::runtime::Module mod = tvm::build(lowered, target, target_host, config);
	auto vecadd = mod.GetFunction("vecadd");
	std::cout << std::endl << std::endl;

	std::cout << "=== Target info ===" << std::endl;
	std::cout << "Type: " << target->target_name << std::endl;
	std::cout << "Name: " << target->device_name << std::endl;
	std::cout << "Max num threads: " << target->max_num_threads << std::endl;
	std::cout << "Max wrap size: " << target->thread_warp_size << std::endl;
	std::cout << std::endl;

	//std::cout << "=== Halide Statement ===" << std::endl;
	//std::cout << lowered[0]->body << std::endl;

	// Show the generated source code. This is not avliable on all targets.
	// But it works on OpenCL
	if(onGPU == true)
	{
		std::vector<tvm::runtime::Module> imported_modules = mod->imports();
		std::cout << "=== Generated Kernel ===" << std::endl;
		std::cout << imported_modules[0]->GetSource() << std::endl;
	}

	// Allocate memory on both host and guest device
	std::array<TVMArrayHandle, 3> hHostArr;
	std::array<TVMArrayHandle, 3> hTgtArr;
	std::array<tvm_index_t, 1> arr_shape = {16};
	int device_type = kDLCPU;
	if(tgt == "opencl")
		device_type = kDLOpenCL;
	else if(tgt == "cuda")
		device_type = kDLGPU; // GPU == CUDA :(

	for (int i=0;i<3;i++)
	{
		TVMArrayAlloc(&arr_shape[0], arr_shape.size(), kDLFloat, 32, 1, kDLCPU, 0, &hHostArr[i]);
		TVMArrayAlloc(&arr_shape[0], arr_shape.size(), kDLFloat, 32, 1, device_type, 0, &hTgtArr[i]);
	}
	assert(arr_shape[0] == TVMARRAY_SHAPE(hHostArr[0])[0]);

	// Initalize the array on host
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> urd(0.0, 1.0);
	for (int i=0;i<2;i++)
	{
		for (int j=0;j<arr_shape[0];j++)
			TVMARRAY_ADDR<float>(hHostArr[i])[j] = urd(mt);
	}

	// Copy data from host to guest
	for (int i=0;i<2;i++)
		TVMArrayCopyFromTo(hHostArr[i], hTgtArr[i], nullptr);

	// Execute the define computation
	vecadd(hTgtArr[0], hTgtArr[1], hTgtArr[2]);

	// Retreve the result
	TVMArrayCopyFromTo(hTgtArr[2], hHostArr[2], nullptr);
	TVMArrayCopyFromTo(hTgtArr[2], hHostArr[2], nullptr);

	// Print the results
	std::cout << "=== Result ===" << std::endl;
	std::cout << std::fixed << std::setprecision(4);
	for (int i=0;i<arr_shape[0];i++)
	{
		std::cout << "[" << i << "] " << TVMARRAY_ADDR<float>(hHostArr[0])[i] << " + ";
		std::cout << TVMARRAY_ADDR<float>(hHostArr[1])[i] << " = ";
		std::cout << TVMARRAY_ADDR<float>(hHostArr[2])[i] << std::endl;
	}

	// Free the memory
	for (int i=0;i<3;i++)
	{
		TVMArrayFree(hHostArr[i]);
		TVMArrayFree(hTgtArr[i]);
	}

	return 0;
}


