import os

import numpy as np
import tvm
from tvm import te, auto_scheduler
from time import time

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul_add(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)
    C = te.placeholder((N, M), name="C", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    matmul = te.compute(
        (N, M),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="matmul",
        attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
    )
    out = te.compute((N, M), lambda i, j: matmul[i, j] + C[i, j], name="out")

    return [A, B, C, out]

target = tvm.target.cuda(arch="sm_70")
N = M = 1024
L = 1024
task = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(N, L, M, "float32"), target=target)

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

log_file = "matmul.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=256,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)
start_time = time()
# Run auto-tuning (search)
task.tune(tune_option)
total_time = time() - start_time
print("total time: ", total_time, "one round:", total_time / 4)
# Apply the best schedule
sch, args = task.apply_best(log_file)

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))

print("Equivalent python schedule:")
print(task.print_best(log_file))

func = tvm.build(sch, args, target)
a_np = np.random.uniform(size=(N, L)).astype(np.float32)
b_np = np.random.uniform(size=(L, M)).astype(np.float32)
c_np = np.random.uniform(size=(N, M)).astype(np.float32)
out_np = a_np.dot(b_np) + c_np

dev = tvm.gpu()
a_tvm = tvm.nd.array(a_np, device=dev)
b_tvm = tvm.nd.array(b_np, device=dev)
c_tvm = tvm.nd.array(c_np, device=dev)
out_tvm = tvm.nd.empty(out_np.shape, device=dev)
#func(a_tvm, b_tvm, c_tvm, out_tvm)

# Check results
#np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
dev_module = func.imported_modules[0]
print("-----GPU code-----")
print(dev_module.get_source())

# Evaluate execution time.
evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
print(
    "Execution time of this operator: %.3f ms"
    % (np.median(evaluator(a_tvm, b_tvm, c_tvm, out_tvm).results) * 1000)
)

