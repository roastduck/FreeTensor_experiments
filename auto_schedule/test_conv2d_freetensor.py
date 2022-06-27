import freetensor as ft
import numpy as np
from freetensor import libop

target = ft.GPU()
device = ft.Device(target)


def test_c2d():
    N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 3, 3, (
        1, 1), (0, 0)

    @ft.transform
    def f(x, w, b, o):
        x: ft.Var[(N, CI, H, W), "float32", "input", "gpu/global"]
        w: ft.Var[(CO, CI, KH, KW), "float32", "input", "gpu/global"]
        b: ft.Var[(CO,), "float32", "input", "gpu/global"]
        o: ft.Var[(N, CO, H - KH + 1, W - KW + 1), "float32", "output",
                  "gpu/global"]
        y = ft.empty((N, CO, H - KH + 1, W - KW + 1), "float32", "gpu/local")
        #! nid: conv
        libop.conv_(x, w, None, y, auto_pad='VALID')
        for i in range(N):
            for j in range(CO):
                for k in range(H - KH + 1):
                    for l in range(W - KW + 1):
                        o[i, j, k, l] = ft.max(y[i, j, k, l] + b[j], 0.)

    s = ft.Schedule(f)
    print(s.ast())
    # w_np = np.zeros((a, b, c), dtype="float32")
    # w_np = np.zeros((a, b), dtype="float32")
    # x_np = np.zeros((b, a), dtype="float32")
    # y_np = np.zeros((a, a), dtype="float32")
    x_np = np.zeros((N, CI, H, W), dtype="float32")
    w_np = np.zeros((CO, CI, KH, KW), dtype="float32")
    b_np = np.zeros((CO,), dtype="float32")
    o_np = np.zeros((N, CO, H - KH + 1, W - KW + 1), dtype="float32")
    # u_np = np.zeros((m, m), dtype="float32")
    # y_np = np.zeros((a, b), dtype="float32")
    w_arr = ft.Array(w_np)
    x_arr = ft.Array(x_np)
    b_arr = ft.Array(b_np)
    o_arr = ft.Array(o_np)
    # u_arr = ft.Array(u_np, device)
    print("Start constructing...")
    s = ft.AutoSchedule(s, target, device, 512, tag="c2d", min_block_size=64)
    s.set_params(w=w_arr, x=x_arr, b=b_arr, o=o_arr)
    # s.set_params(w=w_arr, x=x_arr, y=y_arr)
    print("Start running...")
    s = s.run(10)
    print("Start lowering...")
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)


if __name__ == '__main__':
    test_c2d()
