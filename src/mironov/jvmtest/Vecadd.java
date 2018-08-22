import ml.dmlc.tvm.Module;
import ml.dmlc.tvm.NDArray;
import ml.dmlc.tvm.TVMContext;

import java.io.File;
import java.util.Arrays;

public class Vecadd {
  public static void main(String[] args) {
    String loadingDir = args[0];
    Module fadd = Module.load(loadingDir + File.separator + "vecadd.so");

    TVMContext ctx = TVMContext.cpu();

    long[] shape = new long[]{2};
    NDArray arr = NDArray.empty(shape, ctx);
    arr.copyFrom(new float[]{3f, 4f});
    NDArray res = NDArray.empty(shape, ctx);

    fadd.entryFunc().pushArg(arr).pushArg(arr).pushArg(res).invoke();
    System.out.println(Arrays.toString(res.asFloatArray()));

    arr.release();
    res.release();
    fadd.release();
  }
}

