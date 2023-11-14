import sys
import numpy as np

sys.path.append('../../../')
from common.numpy.io import load_txt

angle = sys.argv[1]
image_size = int(sys.argv[2])

def cmp_faces():
    #data1 = load_txt(f"./result/d_faces{angle}_{image_size}.txt", "float32")
    data1 = load_txt(f"./result/tapenade_grad_faces{angle}_{image_size}.txt", "float32")
    data2 = load_txt(f"./result/enzyme_grad_faces{angle}_{image_size}.txt", "float32")
    #data2 = load_txt(f"/home/trrbivial/FreeTensor_experiments/softrast/ours/ours_render/result/dd_faces000.txt", "float32")
    x = np.isclose(data2, data1, 5e-2, 5e-3)
    sum = 0.
    t = [0, 0, 0]
    for i in range(5856):
        for k in range(3):
            for l in range(3):
                if x[0][i][k][l] == 0:
                    print("faces: ", i, k, l, data1[0][i][k][l],
                          data2[0][i][k][l])
                    t[l] += 1
                    sum += 1.
    return (sum, t)


def cmp_textures():
    #data1 = load_txt(f"./result/d_textures{angle}_{image_size}.txt", "float32")
    data1 = load_txt(f"./result/tapenade_grad_textures{angle}_{image_size}.txt", "float32")
    data2 = load_txt(f"./result/enzyme_grad_textures{angle}_{image_size}.txt", "float32")
    #data2 = load_txt(f"/home/trrbivial/FreeTensor_experiments/softrast/ours/ours_render/result/dd_textures000.txt", "float32")
    x = np.isclose(data2, data1, 5e-2, 5e-3)
    sum = 0.
    for i in range(5856):
        for j in range(25):
            for k in range(3):
                if x[0][i][j][k] == 0:
                    print("textures: ", i, j, k, data1[0][i][j][k],
                          data2[0][i][j][k])
                    sum += 1.
    return sum


if __name__ == '__main__':
    sum1, t = cmp_faces()
    sum2 = cmp_textures()
    print(sum1 / (5856 * 3 * 3))
    print(t)
    print(sum2 / (5856 * 25 * 3))
