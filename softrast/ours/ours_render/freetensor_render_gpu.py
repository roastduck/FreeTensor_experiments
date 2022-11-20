import sys
import time
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio
import argparse

import torch
import torch.nn as nn

import freetensor as ft


def load_txt(filename: str, dtype: str):
    with open(filename) as f:
        shape = list(map(int, f.readline().split()))
        if dtype.startswith('int'):
            data = list(map(int, f.readline().split()))
        elif dtype.startswith('float'):
            data = list(map(float, f.readline().split()))
        else:
            assert False
        return np.array(data, dtype=dtype).reshape(shape)


def store_txt(filename: str, tensor: np.array):
    with open(filename, "w") as f:
        f.write(" ".join(map(str, tensor.shape)) + "\n")
        f.write(" ".join(map(str, tensor.flatten())) + "\n")


device = ft.GPU(0)

with device:

    @ft.inline
    def face_inv(v):
        """
        compute inv of
        [[x0, y0, 1],
         [x1, y1, 1],
         [x2, y2, 1]]
        """

        inv = ft.empty((3, 3), "float32")
        det = ft.empty((), "float32")
        det[()] = 0.
        for p in range(3):
            det[()] += v[p][0] * (v[(p + 1) % 3][1] - v[(p + 2) % 3][1])

        det[()] = ft.if_then_else(det > 0, ft.max(det, 1e-10),
                                  ft.min(det, -1e-10))

        for p in range(3):
            inv[p][0] = (v[(p + 1) % 3][1] - v[(p + 2) % 3][1]) / det
            inv[p][1] = (v[(p + 2) % 3][0] - v[(p + 1) % 3][0]) / det
            inv[p][2] = (v[(p + 1) % 3][0] * v[(p + 2) % 3][1] -
                         v[(p + 2) % 3][0] * v[(p + 1) % 3][1]) / det

        return inv

    @ft.inline
    def face_sym(v):
        """
        compute matmul of
        [[x0, y0, 1],
         [x1, y1, 1],
         [x2, y2, 1]]
        *
        [[x0, y0, 1],
         [x1, y1, 1],
         [x2, y2, 1]]^T
        """
        sym = ft.empty((3, 3), "float32")

        for j in range(3):
            for k in range(3):
                sym[j][k] = v[j][0] * v[k][0] + v[j][1] * v[k][1] + 1
        return sym

    @ft.inline
    def dot_xyz(v1, v2):
        y = ft.empty((), "float32")
        y[()] = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        return y

    @ft.inline
    def dot_xy(v1, v2):
        y = ft.empty((), "float32")
        y[()] = v1[0] * v2[0] + v1[1] * v2[1]
        return y

    @ft.inline
    def sub(v1, v2):
        y = ft.empty((3,), "float32")
        for k in range(3):
            y[k] = v1[k] - v2[k]
        return y

    @ft.inline
    def face_obt(v):
        obt = ft.empty((3, 3), "bool")

        for k in range(3):
            if dot_xy(sub(v[(k + 1) % 3], v[k]), sub(v[(k + 2) % 3], v[k])) < 0:
                obt[k][0] = 1
        return obt

    @ft.inline
    def barycentric_coordinate(x, y, inv):
        w = ft.empty((3,), "float32")
        w[0] = inv[0, 0] * x + inv[0, 1] * y + inv[0, 2]
        w[1] = inv[1, 0] * x + inv[1, 1] * y + inv[1, 2]
        w[2] = inv[2, 0] * x + inv[2, 1] * y + inv[2, 2]
        return w

    @ft.inline
    def check_border(x, y, f, threshold):
        t = ft.sqrt(threshold)
        ret = ft.empty((), "bool")
        ret[()] = (x > ft.max(ft.max(f[0, 0], f[1, 0]), f[2, 0]) + t) or (
            x < ft.min(ft.min(f[0, 0], f[1, 0]), f[2, 0]) - t) or (
                y > ft.max(ft.max(f[0, 1], f[1, 1]), f[2, 1]) + t) or (
                    y < ft.min(ft.min(f[0, 1], f[1, 1]), f[2, 1]) - t)
        return ret[()]

    @ft.inline
    def check_face_frontside(face):
        ret = ft.empty((), "bool")
        ret[()] = (face[2, 1] - face[0, 1]) * (face[1, 0] - face[0, 0]) < (
            face[1, 1] - face[0, 1]) * (face[2, 0] - face[0, 0])
        return ret[()]

    @ft.inline
    def barycentric_clip(w):
        w_clip = ft.empty((3,), "float32")
        for k in range(3):
            w_clip[k] = ft.max(ft.min(w[k], 1.), 0.)
        w_sum = ft.empty((), "float32")
        w_sum[()] = ft.max(w_clip[0] + w_clip[1] + w_clip[2], 1e-5)
        for k in range(3):
            w_clip[k] /= w_sum[()]
        return w_clip

    @ft.inline
    def euclidean_p2f_distance(w, face, face_info, xp, yp, sign):
        sym = face_info[1]
        obt = face_info[2]

        t = ft.empty((3,), "float32")
        dis_x = ft.empty((), "float32")
        dis_y = ft.empty((), "float32")

        if sign > 0:
            dis_min = ft.empty((), "float32")
            dis_x_min = ft.empty((), "float32")
            dis_y_min = ft.empty((), "float32")
            dis_min[()] = 100000000
            dis_x_min[()] = 0
            dis_y_min[()] = 0
            a0 = ft.empty((3,), "float32")
            t0 = ft.empty((3,), "float32")
            for k in range(3):
                v0, v1, v2 = k, (k + 1) % 3, (k + 2) % 3
                a0 = sub(sym[v0], sym[v1])

                t0[v0] = (w[0] * a0[0] + w[1] * a0[1] + w[2] * a0[2] -
                          a0[v1]) / (a0[v0] - a0[v1])
                t0[v1] = 1 - t0[v0]
                t0[v2] = 0

                ft.sub_to(t0, w)

                dis_x[(
                )] = t0[0] * face[0, 0] + t0[1] * face[1, 0] + t0[2] * face[2,
                                                                            0]
                dis_y[(
                )] = t0[0] * face[0, 1] + t0[1] * face[1, 1] + t0[2] * face[2,
                                                                            1]

                dis = ft.empty((), "float32")
                dis[()] = dis_x * dis_x + dis_y * dis_y

                if dis < dis_min:
                    dis_min[()] = dis
                    dis_x_min[()] = dis_x
                    dis_y_min[()] = dis_y
                    ft.assign(t, t0)
            dis_x[()] = dis_x_min
            dis_y[()] = dis_y_min
        else:
            v0 = ft.empty((), "int32")
            v0[()] = -1
            for k in range(3):
                if v0 == -1 and w[(k + 1) % 3] <= 0 and w[(k + 2) % 3] <= 0:
                    v0[()] = k
                    if obt[k, 0] == 1 and (
                        (xp - face[k, 0]) *
                        (face[(k + 2) % 3, 0] - face[k, 0]) +
                        (yp - face[k, 1]) *
                        (face[(k + 2) % 3, 1] - face[k, 1])) > 0:
                        v0[()] = (k + 2) % 3
            for k in range(3):
                if v0 == -1 and w[k] <= 0:
                    v0[()] = (k + 1) % 3

            v1 = (v0 + 1) % 3
            v2 = (v0 + 2) % 3

            a0 = ft.empty((3,), "float32")
            a0 = sub(sym[v0], sym[v1])

            t[v0] = (w[0] * a0[0] + w[1] * a0[1] + w[2] * a0[2] -
                     a0[v1]) / (a0[v0] - a0[v1])
            t[v1] = 1 - t[v0]
            t[v2] = 0

            for k in range(3):
                t[k] = ft.min(ft.max(t[k], 0.), 1.)
                t[k] -= w[k]

            dis_x[(
            )] = t[0] * face[0, 0] + t[1] * face[1, 0] + t[2] * face[2, 0]
            dis_y[(
            )] = t[0] * face[0, 1] + t[1] * face[1, 1] + t[2] * face[2, 1]
        return (dis_x, dis_y, t)

    @ft.inline
    def forward_sample_texture(texture, w, r, k, texture_type):
        texture_k = ft.empty((), "float32")
        texture_k[()] = 0.
        if texture_type == 0:
            w_x = ft.empty((), "int32")
            w_y = ft.empty((), "int32")
            w_x[()] = w[0] * r
            w_y[()] = w[1] * r
            if (w[0] + w[1]) * r - w_x - w_y <= 1:
                if w_y * r + w_x == 25:
                    texture_k[()] = texture[w_y * r + w_x - 1, k]
                else:
                    texture_k[()] = texture[w_y * r + w_x, k]
            else:
                texture_k[()] = texture[(r - 1 - w_y) * r + (r - 1 - w_x), k]
        elif texture_type == 1:
            texture_k[(
            )] = w[0] * texture[0, k] + w[1] * texture[1, k] + w[2] * texture[2,
                                                                              k]
        return texture_k

    @ft.transform(verbose=2)
    def our_render_main(faces, textures, soft_colors):
        batch_size = 1
        num_faces = 5856
        texture_size = 25
        texture_res = 5
        image_size = 256
        near = 1.
        far = 100.
        eps = 1e-3
        sigma_val = 1e-5
        gamma_val = 1e-4
        dist_eps = 9.21024036697585
        threshold = dist_eps * sigma_val
        double_side = False
        texture_type = 0

        faces: ft.Var[(batch_size, num_faces, 3, 3), "float32"]
        textures: ft.Var[(batch_size, num_faces, texture_size, 3), "float32"]
        soft_colors: ft.Var[(batch_size, 4, image_size, image_size), "float32",
                            "inout"]

        faces_info = ft.empty((batch_size, num_faces, 3, 3, 3), "float32")

        for bn in range(batch_size):
            for fn in range(num_faces):
                ft.assign(faces_info[bn, fn, 0], face_inv(faces[bn, fn]))
                ft.assign(faces_info[bn, fn, 1], face_sym(faces[bn, fn]))
                ft.assign(faces_info[bn, fn, 2], face_obt(faces[bn, fn]))

        for bn in range(batch_size):
            for pn in range(image_size * image_size):
                yi = image_size - 1 - (pn // image_size)
                xi = pn % image_size
                xp = ft.empty((), "float32")
                yp = ft.empty((), "float32")
                xp[()] = (2. * xi + 1. - image_size) / image_size
                yp[()] = (2. * yi + 1. - image_size) / image_size

                softmax_sum = ft.empty((), "float32")
                softmax_sum[()] = ft.exp(eps / gamma_val)

                softmax_max = ft.empty((), "float32")
                softmax_max[()] = eps

                soft_color = ft.empty((4,), "float32")
                soft_color[3] = 1.

                for k in range(3):
                    soft_color[k] = soft_colors[bn, k, pn // image_size,
                                                pn % image_size] * softmax_sum

                sign = ft.empty((num_faces,), "float32")

                for fn in range(num_faces):
                    face = faces[bn, fn]
                    texture = textures[bn, fn]
                    face_info = faces_info[bn, fn]
                    if not check_border(xp, yp, face, threshold):

                        w = barycentric_coordinate(xp, yp, face_info[0])

                        if w[0] > 0 and w[1] > 0 and w[2] > 0 and w[
                                0] < 1 and w[1] < 1 and w[2] < 1:
                            sign[fn] = 1
                        else:
                            sign[fn] = -1

                        dis_x, dis_y, t = euclidean_p2f_distance(
                            w, face, face_info, xp, yp, sign[fn])

                        dis = ft.empty((), "float32")
                        dis[()] = dis_x * dis_x + dis_y * dis_y

                        if not (sign[fn] < 0 and dis >= threshold):

                            soft_fragment = ft.empty((), "float32")
                            soft_fragment[(
                            )] = 1. / (1. + ft.exp(-sign[fn] * dis / sigma_val))

                            soft_color[3] *= 1. - soft_fragment

                            w_clip = barycentric_clip(w)

                            zp = ft.empty((), "float32")

                            zp[()] = 1. / (w_clip[0] / face[0, 2] + w_clip[1] /
                                           face[1, 2] + w_clip[2] / face[2, 2])

                            if not (zp < near or zp > far):
                                if check_face_frontside(face) or double_side:
                                    zp_norm = ft.empty((), "float32")
                                    exp_delta_zp = ft.empty((), "float32")

                                    zp_norm[()] = (far - zp) / (far - near)
                                    exp_delta_zp[()] = 1.

                                    if zp_norm > softmax_max:
                                        exp_delta_zp[()] = ft.exp(
                                            (softmax_max - zp_norm) / gamma_val)
                                        softmax_max[()] = zp_norm

                                    exp_z = ft.exp(
                                        (zp_norm - softmax_max) / gamma_val)
                                    softmax_sum[(
                                    )] = exp_delta_zp * softmax_sum + exp_z * soft_fragment
                                    for k in range(3):
                                        color_k = forward_sample_texture(
                                            texture, w_clip, texture_res, k,
                                            texture_type)
                                        soft_color[k] = exp_delta_zp * soft_color[
                                            k] + exp_z * soft_fragment * color_k

                soft_colors[bn, 3, pn // image_size,
                            pn % image_size] = 1. - soft_color[3]
                for k in range(3):
                    soft_colors[bn, k, pn // image_size,
                                pn % image_size] = soft_color[k] / softmax_sum

        return soft_colors

    render_exe = ft.optimize(
        our_render_main,
        schedule_callback=lambda s: s.auto_schedule(device),
        verbose=2)


def our_render(faces, textures):
    """
        faces = faces[batch_size][num_faces][3(vertices)][3(xyz)]
        textures = textures[batch_size][num_faces][texture_size][3]
        face_info = face_info[batch_size][num_faces][3(inv, sym, obt)][3][3] // matrix
    """
    batch_size = 1
    image_size = 256
    soft_colors = torch.ones((batch_size, 4, image_size, image_size),
                             dtype=torch.float32)
    background_color = [0, 0, 0]
    soft_colors[:, 0, :, :] *= background_color[0]
    soft_colors[:, 1, :, :] *= background_color[1]
    soft_colors[:, 2, :, :] *= background_color[2]

    return render_exe(faces, textures, soft_colors)


def main():

    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = current_dir
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--output-dir',
                        type=str,
                        default=os.path.join(data_dir, 'newnew'))

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # draw object from different view
    loop = tqdm.tqdm(list(range(0, 360, 4)))
    writer = imageio.get_writer(os.path.join(args.output_dir, 'rotation.gif'),
                                mode='I')

    warmup_num = 1
    repeat_num = 0

    face_vertices = []
    face_textures = []

    for i in range(warmup_num):
        loop = tqdm.tqdm(list(range(0, 360, 4)))
        for num, azimuth in enumerate(loop):
            loop.set_description('Drawing rotation')

            if i == 0:
                face_vertices.append(
                    load_txt(f"./data/face_vertices{str(num*4).zfill(3)}",
                             "float32"))
                face_textures.append(
                    load_txt(f"./data/face_textures{str(num*4).zfill(3)}",
                             "float32"))

            images = our_render(face_vertices[num], face_textures[num])

            if i == 0:
                image = images.torch().detach().cpu().numpy()[0].transpose(
                    (1, 2, 0))
                writer.append_data((255 * image).astype(np.uint8))
        if i == 0:
            writer.close()

    t0 = time.time()

    for i in range(repeat_num):
        loop = tqdm.tqdm(list(range(0, 360, 4)))
        for num, azimuth in enumerate(loop):
            loop.set_description('Drawing rotation')
            images = our_render(face_vertices[num], face_textures[num])

    t1 = time.time()

    if repeat_num > 0:
        print(f"Drawing Rotation Time = {(t1 - t0) / repeat_num * 1000} ms")


if __name__ == '__main__':
    main()
