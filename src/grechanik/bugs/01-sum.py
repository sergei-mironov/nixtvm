import nnvm.compiler
import nnvm.symbol as sym
import tvm

x = sym.Variable("x", shape=(100,), dtype=1)
z = sym.sum(x, axis=(0,))

graph = nnvm.graph.create(z)
print(graph.ir())

with nnvm.compiler.build_config(opt_level=2):
    deploy_graph, lib, params = nnvm.compiler.build(
        graph, target="llvm")

