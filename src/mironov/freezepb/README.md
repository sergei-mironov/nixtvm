
TODO
----

* Measure all possible times (Wall time, CPU time, etc)
* Figure out parallelism. Which module does schedule parallel execution in TVM?
* Run `run` several times?
* Try with batch size ~100?
* Try with default NNVM optimisations disabled

Problems
--------

* Simple TF runners using `session.run` may be suboptimal
* TV vs TVM error rises with increasing absolute value of input
