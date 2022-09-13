import sys
import time
import math
import numpy as np
import freetensor as ft
from freetensor.libop import *
from freetensor import debug

sys.path.append('../..')
from common.gpu import profile_start, profile_stop


def compile_all(n, c_in, c_out, h, w, k_h, k_w, device):
    # yapf: disable

    @ft.transform
    def inference(X, W1, W2, Y):
        X: ft.Var[(n, c_in, h, w), "float32", "input"]
        W1: ft.Var[(k_h, k_w, 2, c_in, k_h, k_w), "float32", "input"]
        W2: ft.Var[(c_out, c_in, k_h, k_w), "float32", "input"]
        Y: ft.Var[(n, c_out, h, w), "float32", "output"]

        #! label: Li
        for i in range(n):
            #! label: Lp
            for p in range(h):
                #! label: Lq
                for q in range(w):
                    pos = ft.empty((k_h, k_w, 2), "float32")
                    pos_int = ft.empty((k_h, k_w, 2), "int32")
                    #! label: Lro0
                    for ro in range(k_h):
                        for so in range(k_w):
                            for t in range(2):
                                pos[ro, so, t] = 0
                            for ki in range(c_in):
                                for ri in range(k_h):
                                    for si in range(k_w):
                                        if p + ri >= 0 and p + ri < h and q + si >= 0 and q + si < w:
                                            for t in range(2):
                                                pos[ro, so, t] += X[i, ki, p + ri, q + si] * W1[ro, so, t, ki, ri, si]
                            for t in range(2):
                                pos[ro, so, t] /= c_in
                                pos_int[ro, so, t] = ft.cast(ft.floor(pos[ro, so, t]), "int32")

                    #! label: pixel
                    pixel = ft.empty((c_in, k_h, k_w), "float32")
                    for ki in range(c_in):
                        for ro in range(k_h):
                            for so in range(k_w):
                                x = ft.empty((2, 2), "int32")
                                y = ft.empty((2, 2), "int32")
                                x[0, 0] = p + ro + pos_int[ro, so, 0]
                                y[0, 0] = q + so + pos_int[ro, so, 1]
                                x[0, 1] = p + ro + pos_int[ro, so, 0]
                                y[0, 1] = q + so + pos_int[ro, so, 1] + 1
                                x[1, 0] = p + ro + pos_int[ro, so, 0] + 1
                                y[1, 0] = q + so + pos_int[ro, so, 1]
                                x[1, 1] = p + ro + pos_int[ro, so, 0] + 1
                                y[1, 1] = q + so + pos_int[ro, so, 1] + 1
                                dist = ft.empty((2, 2), "float32")
                                dist[0, 0] = (pos[ro, so, 0] - pos_int[ro, so, 0]) * (pos[ro, so, 1] - pos_int[ro, so, 1])
                                dist[0, 1] = (pos[ro, so, 0] - pos_int[ro, so, 0]) * (pos_int[ro, so, 1] + 1 - pos[ro, so, 1])
                                dist[1, 0] = (pos_int[ro, so, 0] + 1 - pos[ro, so, 0]) * (pos[ro, so, 1] - pos_int[ro, so, 1])
                                dist[1, 1] = (pos_int[ro, so, 0] + 1 - pos[ro, so, 0]) * (pos_int[ro, so, 1] + 1 - pos[ro, so, 1])
                                pixel[ki, ro, so] = 0
                                for t in range(2):
                                    for u in range(2):
                                        if x[t, u] >= 0 and x[t, u] < h and y[t, u] >= 0 and y[t, u] < w:
                                            pixel[ki, ro, so] += X[i, ki, x[t, u], y[t, u]] * dist[t, u]

                    #! label: Lko
                    einsum_("krs,lkrs->l", pixel, W2, Y[i, :, p, q])

    # yapf: enable

    forward, backward, requires, privdes = ft.grad_(inference,
                                                    set(["X", "W1", "W2"]),
                                                    set(["Y"]))

    print("# Inference:")
    print(inference)
    s = ft.Schedule(inference)
    if device.type() == ft.TargetType.CPU:
        Lko = s.fission("Li", ft.FissionSide.Before, "<<~Lko")[1]["Li"]
        _, _, _, Y_t_def = s.cache(Lko, "Y", "cpu")
        s.var_reorder(Y_t_def, [0, 2, 3, 1])
    else:
        Lko = s.fission("Li", ft.FissionSide.Before, "<<~Lko")[1]["Li"]
        _, _, _, Y_t_def = s.cache(Lko, "Y", "gpu/global")
        s.var_reorder(Y_t_def, [0, 2, 3, 1])
        s.var_reorder("pixel", [3, 4, 5, 0, 1, 2])
    s.auto_schedule(device.target())
    f = ft.lower(s.func(), device.target())
    print(f)
    code = ft.codegen(f, device.target())
    print(ft.debug.with_line_no(code))
    inference_exe = ft.Driver(inference, code, device)

    print("# Forward:")
    print(forward)
    s = ft.Schedule(forward)
    if device.type() == ft.TargetType.CPU:
        Lko = s.fission("Li", ft.FissionSide.Before, "<<~Lko")[1]["Li"]
        _, _, _, Y_t_def = s.cache(Lko, "Y", "cpu")
        s.var_reorder(Y_t_def, [0, 2, 3, 1])
    else:
        Lko = s.fission("Li", ft.FissionSide.Before, "<<~Lko")[1]["Li"]
        _, _, _, Y_t_def = s.cache(Lko, "Y", "gpu/global")
        s.var_reorder(Y_t_def, [0, 2, 3, 1])
        s.var_reorder("pixel", [3, 4, 5, 0, 1, 2])
    s.auto_schedule(device.target())
    f = ft.lower(s.func(), device.target())
    print(f)
    code = ft.codegen(f, device.target())
    print(ft.debug.with_line_no(code))
    forward_exe = ft.Driver(forward, code, device)

    print("# Backward:")
    print(backward)
    s = ft.Schedule(backward)
    if device.type() == ft.TargetType.CPU:
        s.cache_reduction("$grad{Lro0}", "X.grad", "cpu")
        s.fission("$grad{Li}", ft.FissionSide.Before, "$grad{<<~Lko}")
        Lko = s.fission("$fission.1{$grad{Li}}", ft.FissionSide.After,
                        "$fission.1{$grad{<<~Lko}}")[0]["$fission.1{$grad{Li}}"]
        _, _, _, Y_t_def = s.cache(Lko, "Y.grad", "cpu")
        s.var_reorder(Y_t_def, [0, 2, 3, 1])
    else:
        #s.cache_reduction("$grad{Lro0}", "X.grad", "gpu/global")
        s.fission("$grad{Li}", ft.FissionSide.Before, "$grad{<<~Lko}")
        Lko = s.fission("$fission.1{$grad{Li}}", ft.FissionSide.After,
                        "$fission.1{$grad{<<~Lko}}")[0]["$fission.1{$grad{Li}}"]
        _, _, _, Y_t_def = s.cache(Lko, "Y.grad", "gpu/global")
        s.var_reorder(Y_t_def, [0, 2, 3, 1])
        s.var_reorder("pixel", [3, 4, 5, 0, 1, 2])
        s.var_reorder("pixel.grad", [3, 4, 5, 0, 1, 2])
    s.auto_schedule(device.target())
    f = ft.lower(s.func(), device.target())
    print(f)
    code = ft.codegen(f, device.target())
    print(ft.debug.with_line_no(code))
    backward_exe = ft.Driver(backward, code, device)

    def run_backward(x, w1, w2, y, d_y, d_x, d_w1, d_w2):
        kvs = {}
        kvs[privdes['Y']] = d_y
        kvs[requires['X']] = d_x
        kvs[requires['W1']] = d_w1
        kvs[requires['W2']] = d_w2
        backward_exe(x, w1, w2, y, **kvs)

    return inference_exe, forward_exe, run_backward


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cpu/gpu>")
        exit(-1)
    device = sys.argv[1]

    n = 8
    c_in = 256
    c_out = 256
    h = 56
    w = 56
    k_h = 3
    k_w = 3
    x = np.random.uniform(size=(n, c_in, h, w)).astype("float32") * 2 - 1
    w1 = np.random.uniform(size=(k_h, k_w, 2, c_in, k_h,
                                 k_w)).astype("float32") * 2 - 1
    w2 = np.random.uniform(size=(c_out, c_in, k_h,
                                 k_w)).astype("float32") * 2 - 1
    y = np.zeros((n, c_out, h, w), dtype="float32")
    d_x = np.zeros(x.shape, dtype='float32')
    d_w1 = np.zeros(w1.shape, dtype='float32')
    d_w2 = np.zeros(w2.shape, dtype='float32')
    d_y = np.random.uniform(size=y.shape).astype('float32')

    if device == 'gpu':
        ir_dev = ft.GPU()
    else:
        assert device == 'cpu'
        ir_dev = ft.CPU()

    x = ft.Array(x)
    w1 = ft.Array(w1)
    w2 = ft.Array(w2)
    y = ft.Array(y)
    d_x = ft.Array(d_x)
    d_w1 = ft.Array(d_w1)
    d_w2 = ft.Array(d_w2)
    d_y = ft.Array(d_y)

    with ir_dev:
        inference, forward, backward = compile_all(n, c_in, c_out, h, w, k_h,
                                                   k_w, ir_dev)

    warmup_num = 10
    test_num = 100

    for i in range(warmup_num):
        inference(x, w1, w2, y)
    ir_dev.sync()
    t0 = time.time()
    for i in range(test_num):
        inference(x, w1, w2, y)
    ir_dev.sync()
    t1 = time.time()

    print(f"Inference Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        forward(x, w1, w2, y)
    ir_dev.sync()
    t0 = time.time()
    for i in range(test_num):
        forward(x, w1, w2, y)
    ir_dev.sync()
    t1 = time.time()

    print(f"Forward Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        backward(x, w1, w2, y, d_y, d_x, d_w1, d_w2)
    ir_dev.sync()
    #profile_start()
    t0 = time.time()
    for i in range(test_num):
        backward(x, w1, w2, y, d_y, d_x, d_w1, d_w2)
    ir_dev.sync()
    t1 = time.time()
    #profile_stop()

    print(f"Backward Time = {(t1 - t0) / test_num * 1000} ms")
