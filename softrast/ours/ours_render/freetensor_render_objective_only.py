import sys
import time
#import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio
#import argparse

#import torch
#import torch.nn as nn

import freetensor as ft

angle = sys.argv[1]
image_size = int(sys.argv[2])


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
        f.write(" ".join(map(lambda x: f"{x:.8f}", tensor.flatten())) + "\n")


device = ft.CPU(0)
#device = ft.GPU(0)

batch_size = 1
num_faces = 5856
texture_size = 25
texture_res = 5
#image_size = 256
near = 1.
far = 100.
eps = 1e-3
sigma_val = 1e-5
gamma_val = 1e-4
dist_eps = 9.21024036697585
threshold = dist_eps * sigma_val
double_side = False
texture_type = 0

with device:

    @ft.inline
    def face_inv(v):
        """
        compute inv of
        [[x0, y0, 1],
         [x1, y1, 1],
         [x2, y2, 1]]
        """

        f_inv = ft.empty((3, 3), "float32")
        det = ft.empty((), "float32")
        det[()] = 0.
        for p in range(3):
            det[()] += v[p][0] * (v[(p + 1) % 3][1] - v[(p + 2) % 3][1])

        det[()] = ft.if_then_else(det > 0, ft.max(det, 1e-10),
                                  ft.min(det, -1e-10))

        for p in range(3):
            f_inv[p][0] = (v[(p + 1) % 3][1] - v[(p + 2) % 3][1]) / det
            f_inv[p][1] = (v[(p + 2) % 3][0] - v[(p + 1) % 3][0]) / det
            f_inv[p][2] = (v[(p + 1) % 3][0] * v[(p + 2) % 3][1] -
                           v[(p + 2) % 3][0] * v[(p + 1) % 3][1]) / det

        return f_inv

    @ft.inline
    def dot_xy(v1, v2):
        y = ft.empty((), "float32")
        y[()] = v1[0] * v2[0] + v1[1] * v2[1]
        return y

    @ft.inline
    def cross_xy(v1, v2):
        y = ft.empty((), "float32")
        y[()] = v1[0] * v2[1] - v1[1] * v2[0]
        return y

    @ft.inline
    def sub_xy(v1, v2):
        y = ft.empty((2,), "float32")
        for k in range(2):
            y[k] = v1[k] - v2[k]
        return y

    @ft.inline
    def norm2(v):
        y = ft.empty((), "float32")
        y[()] = v[0] * v[0] + v[1] * v[1]
        return y

    @ft.inline
    def barycentric_coordinate(p, inv):
        w = ft.empty((3,), "float32")
        w[0] = inv[0, 0] * p[0] + inv[0, 1] * p[1] + inv[0, 2]
        w[1] = inv[1, 0] * p[0] + inv[1, 1] * p[1] + inv[1, 2]
        w[2] = inv[2, 0] * p[0] + inv[2, 1] * p[1] + inv[2, 2]
        return w

    @ft.inline
    def check_border(p, f, threshold):
        t = ft.sqrt(threshold)
        ret = ft.empty((), "bool")
        ret[()] = (p[0] > ft.max(ft.max(f[0, 0], f[1, 0]), f[2, 0]) + t) or (
            p[0] < ft.min(ft.min(f[0, 0], f[1, 0]), f[2, 0]) -
            t) or (p[1] > ft.max(ft.max(f[0, 1], f[1, 1]), f[2, 1]) +
                   t) or (p[1] < ft.min(ft.min(f[0, 1], f[1, 1]), f[2, 1]) - t)
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
    def euclidean_p2f_distance(f, p):
        '''
            Using barycentric coordinate will be convenient for human to deduce gradient function
            However, we use autograd. There is no need to use barycentric coordinate to calculate distance
            TODO: reduce calculations when sign < 0
        '''

        dist = ft.empty((3,), "float32")
        for k in range(3):
            area = cross_xy(sub_xy(p, f[k]), sub_xy(f[(k + 1) % 3], f[k]))

            d1 = dot_xy(sub_xy(p, f[k]), sub_xy(f[(k + 1) % 3], f[k]))
            if d1[()] >= 0:
                d2 = dot_xy(sub_xy(p, f[(k + 1) % 3]),
                            sub_xy(f[k], f[(k + 1) % 3]))
                if d2[()] >= 0:
                    len = norm2(sub_xy(f[(k + 1) % 3], f[k]))
                    dist[k] = area / ft.max(len[()], 1e-10) * area
                else:
                    p2_dist = norm2(sub_xy(p, f[(k + 1) % 3]))
                    dist[k] = p2_dist[()]
            else:
                p1_dist = norm2(sub_xy(p, f[k]))
                dist[k] = p1_dist[()]

        dis = ft.empty((), "float32")
        dis[()] = ft.min(ft.min(dist[0], dist[1]), dist[2])
        return dis

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

    @ft.transform
    def our_render_main_original(faces, textures, soft_colors):
        faces: ft.Var[(batch_size, num_faces, 3, 3), "float32"]
        textures: ft.Var[(batch_size, num_faces, texture_size, 3), "float32"]
        soft_colors: ft.Var[(batch_size, 4, image_size, image_size), "float32",
                            "output"]

        faces_inv = ft.empty((batch_size, num_faces, 3, 3), "float32")

        for bn in range(batch_size):
            for fn in range(num_faces):
                ft.assign(faces_inv[bn, fn], face_inv(faces[bn, fn]))

        for bn in range(batch_size):
            for pn in range(image_size * image_size):
                yi = image_size - 1 - (pn // image_size)
                xi = pn % image_size
                pixel = ft.empty((2,), "float32")

                pixel[0] = (2. * xi + 1. - image_size) / image_size
                pixel[1] = (2. * yi + 1. - image_size) / image_size

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
                    inv = faces_inv[bn, fn]
                    if not check_border(pixel, face, threshold):

                        w = barycentric_coordinate(pixel, inv)

                        if w[0] > 0 and w[1] > 0 and w[2] > 0 and w[
                                0] < 1 and w[1] < 1 and w[2] < 1:
                            sign[fn] = 1
                        else:
                            sign[fn] = -1

                        dis = ft.empty((), "float32")
                        dis[()] = euclidean_p2f_distance(face, pixel)

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
                                        """
                                        color_k = 0.5
                                        """
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

    @ft.transform
    def our_render_main_no_iterative_softmax(faces, textures, soft_colors):
        faces: ft.Var[(batch_size, num_faces, 3, 3), "float32"]
        textures: ft.Var[(batch_size, num_faces, texture_size, 3), "float32"]
        soft_colors: ft.Var[(batch_size, 4, image_size, image_size), "float32",
                            "inout"]

        faces_inv = ft.empty((batch_size, num_faces, 3, 3), "float32")

        for bn in range(batch_size):
            for fn in range(num_faces):
                ft.assign(faces_inv[bn, fn], face_inv(faces[bn, fn]))

        for bn in range(batch_size):
            for pn in range(image_size * image_size):
                yi = image_size - 1 - (pn // image_size)
                xi = pn % image_size
                pixel = ft.empty((2,), "float32")

                pixel[0] = (2. * xi + 1. - image_size) / image_size
                pixel[1] = (2. * yi + 1. - image_size) / image_size

                softmax_max = ft.empty((), "float32")
                softmax_max[()] = eps

                soft_color = ft.empty((4,), "float32")
                soft_color[3] = 1.

                for fn in range(num_faces):
                    face = faces[bn, fn]
                    inv = faces_inv[bn, fn]
                    if not check_border(pixel, face, threshold):

                        w = barycentric_coordinate(pixel, inv)

                        w_clip = barycentric_clip(w)

                        zp = ft.empty((), "float32")

                        zp[()] = 1. / (w_clip[0] / face[0, 2] + w_clip[1] /
                                       face[1, 2] + w_clip[2] / face[2, 2])

                        if not (zp < near or zp > far):
                            if check_face_frontside(face) or double_side:
                                zp_norm = ft.empty((), "float32")
                                zp_norm[()] = (far - zp) / (far - near)

                                softmax_max[()] = ft.max(softmax_max, zp_norm)

                softmax_sum = ft.empty((), "float32")
                softmax_sum[()] = ft.exp((2 * eps - softmax_max) / gamma_val)

                for k in range(3):
                    soft_color[k] = soft_colors[bn, k, pn // image_size,
                                                pn % image_size] * softmax_sum

                for fn in range(num_faces):
                    face = faces[bn, fn]
                    texture = textures[bn, fn]
                    inv = faces_inv[bn, fn]
                    if not check_border(pixel, face, threshold):

                        w = barycentric_coordinate(pixel, inv)

                        sign = ft.empty((), "float32")

                        if w[0] > 0 and w[1] > 0 and w[2] > 0 and w[
                                0] < 1 and w[1] < 1 and w[2] < 1:
                            sign[()] = 1
                        else:
                            sign[()] = -1

                        dis = ft.empty((), "float32")
                        dis[()] = euclidean_p2f_distance(face, pixel)

                        if not (sign < 0 and dis >= threshold):

                            soft_fragment = ft.empty((), "float32")
                            soft_fragment[(
                            )] = 1. / (1. + ft.exp(-sign * dis / sigma_val))

                            soft_color[3] *= ft.cast(1. - soft_fragment,
                                                     "float32>0")

                            w_clip = barycentric_clip(w)

                            zp = ft.empty((), "float32")

                            zp[()] = 1. / (w_clip[0] / face[0, 2] + w_clip[1] /
                                           face[1, 2] + w_clip[2] / face[2, 2])

                            if not (zp < near or zp > far):
                                if check_face_frontside(face) or double_side:
                                    zp_norm = ft.empty((), "float32")
                                    zp_norm[()] = (far - zp) / (far - near)

                                    coef = ft.exp((zp_norm - softmax_max) /
                                                  gamma_val) * soft_fragment
                                    softmax_sum[()] += coef
                                    for k in range(3):
                                        color_k = forward_sample_texture(
                                            texture, w_clip, texture_res, k,
                                            texture_type)

                                        soft_color[k] += coef * color_k

                soft_colors[bn, 3, pn // image_size,
                            pn % image_size] = 1. - soft_color[3]
                for k in range(3):
                    soft_colors[bn, k, pn // image_size,
                                pn % image_size] = soft_color[k] / softmax_sum

        return soft_colors

    @ft.transform
    def our_render_main(faces, textures, soft_colors):
        faces: ft.Var[(batch_size, num_faces, 3, 3), "float32"]
        textures: ft.Var[(batch_size, num_faces, texture_size, 3), "float32"]
        soft_colors: ft.Var[(batch_size, 4, image_size, image_size), "float32",
                            "inout"]

        faces_inv = ft.empty((batch_size, num_faces, 3, 3), "float32")

        for bn in range(batch_size):
            for fn in range(num_faces):
                ft.assign(faces_inv[bn, fn], face_inv(faces[bn, fn]))

        for bn in range(batch_size):
            #! label: L_pn
            for pn in range(image_size * image_size):
                yi = image_size - 1 - (pn // image_size)
                xi = pn % image_size
                pixel = ft.empty((2,), "float32")

                pixel[0] = (2. * xi + 1. - image_size) / image_size
                pixel[1] = (2. * yi + 1. - image_size) / image_size

                #! label: softmax_max
                softmax_max = ft.empty((), "float32")
                with ft.StmtRange() as rng:
                    softmax_max[()] = eps
                    for fn in range(num_faces):
                        face = faces[bn, fn]
                        inv = faces_inv[bn, fn]
                        if not check_border(pixel, face, threshold):

                            w = barycentric_coordinate(pixel, inv)

                            w_clip = barycentric_clip(w)

                            zp = ft.empty((), "float32")

                            zp[()] = 1. / (w_clip[0] / face[0, 2] + w_clip[1] /
                                           face[1, 2] + w_clip[2] / face[2, 2])

                            if not (zp < near or zp > far):
                                if check_face_frontside(face) or double_side:
                                    zp_norm = ft.empty((), "float32")
                                    zp_norm[()] = (far - zp) / (far - near)

                                    softmax_max[()] = ft.max(
                                        softmax_max, zp_norm)
                with ft.UserGrad(stmt_range=rng):
                    pass

                background_weight = ft.empty((), "float32")
                background_weight[...] = 2 * eps
                coef = ft.exp((background_weight - softmax_max) / gamma_val)

                soft_color = ft.empty((3,), "float32")
                soft_color_alpha_log = ft.empty((), "float32")
                soft_color_alpha_log[...] = 0.
                for k in ft.static_range(3):
                    soft_color[k] = soft_colors[bn, k, pn // image_size,
                                                pn % image_size] * coef

                #! label: softmax_sum
                softmax_sum = ft.empty((), "float32")
                softmax_sum[()] = coef

                #! label: L_fn
                for fn in range(num_faces):
                    face = faces[bn, fn]
                    texture = textures[bn, fn]
                    inv = faces_inv[bn, fn]
                    if not check_border(pixel, face, threshold):

                        w = barycentric_coordinate(pixel, inv)

                        sign = ft.empty((), "float32")

                        if w[0] > 0 and w[1] > 0 and w[2] > 0 and w[
                                0] < 1 and w[1] < 1 and w[2] < 1:
                            sign[()] = 1
                        else:
                            sign[()] = -1

                        dis = ft.empty((), "float32")
                        dis[()] = euclidean_p2f_distance(face, pixel)

                        if not (sign < 0 and dis >= threshold):

                            soft_fragment = ft.empty((), "float32")
                            soft_fragment[(
                            )] = 1. / (1. + ft.exp(-sign * dis / sigma_val))

                            soft_color_alpha_log[...] += ft.ln(1. -
                                                               soft_fragment)

                            w_clip = barycentric_clip(w)

                            zp = ft.empty((), "float32")

                            zp[()] = 1. / (w_clip[0] / face[0, 2] + w_clip[1] /
                                           face[1, 2] + w_clip[2] / face[2, 2])

                            if not (zp < near or zp > far):
                                if check_face_frontside(face) or double_side:
                                    zp_norm = ft.empty((), "float32")
                                    zp_norm[()] = (far - zp) / (far - near)

                                    coef = ft.exp((zp_norm - softmax_max) /
                                                  gamma_val) * soft_fragment
                                    softmax_sum[()] += coef
                                    for k in range(3):
                                        color_k = forward_sample_texture(
                                            texture, w_clip, texture_res, k,
                                            texture_type)

                                        soft_color[k] += coef * color_k

                soft_colors[bn, 3, pn // image_size,
                            pn % image_size] = 1. - ft.exp(soft_color_alpha_log)
                for k in ft.static_range(3):
                    soft_colors[bn, k, pn // image_size,
                                pn % image_size] = soft_color[k] / softmax_sum

        return soft_colors

    def schedule_fwd(s):
        if 'PAPER_SERIAL' in os.environ:
            return
        s.auto_use_lib(device.target())
        s.auto_reorder(device.target())
        s.auto_parallelize(device.target())
        s.auto_set_mem_type(device.target())
        s.auto_unroll(device.target())

    def schedule_bwd(s):
        if 'PAPER_SERIAL' in os.environ:
            return
        s.auto_use_lib(device.target())
        s.auto_reorder(device.target())
        #s.reorder(['$grad{L_fn}', '$grad{L_pn}'],
        #          ft.ReorderMode.MoveOutImperfect)
        s.auto_parallelize(device.target())
        s.auto_set_mem_type(device.target())
        s.auto_unroll(device.target())

    our_render_main = ft.optimize(
        our_render_main,
        #schedule_callback=lambda s: s.auto_schedule(device.target()),
        schedule_callback=schedule_fwd,
        verbose=1)

    @ft.optimize(schedule_callback=schedule_fwd)
    def init_background_color():
        background_r = 0
        background_g = 0
        background_b = 0
        background_a = 1
        color = ft.empty((batch_size, 4, image_size, image_size), "float32")
        color[:, 0, :, :] = background_r
        color[:, 1, :, :] = background_g
        color[:, 2, :, :] = background_b
        color[:, 3, :, :] = background_a
        return color


d_soft_colors = ft.array(
    np.ones((batch_size, 4, image_size, image_size), dtype="float32"))


def our_render(faces, textures):
    """
        faces[batch_size][num_faces][3(vertices)][3(xyz)]
        textures[batch_size][num_faces][texture_size][3]
        inv[batch_size][num_faces][3][3]
    """
    soft_colors = init_background_color()
    our_render_main(faces, textures, soft_colors)
    return soft_colors.numpy()


def main():

    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = current_dir
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-o',
    #                    '--output-dir',
    #                    type=str,
    #                    default=os.path.join(data_dir, 'newnew'))
    #
    #args = parser.parse_args()
    #os.makedirs(args.output_dir, exist_ok=True)

    # draw object from different view
    #loop = tqdm.tqdm(list(range(0, 360, 4)))
    #writer = imageio.get_writer(os.path.join(args.output_dir, 'rotation.gif'),
    #                            mode='I')

    warmup_num = 3
    repeat_num = 1000

    #face_vertices = []
    #face_textures = []

    for i in range(warmup_num):
        #loop = tqdm.tqdm(list(range(0, 360, 4)))
        #for num, azimuth in enumerate(loop):
            #loop.set_description('Drawing rotation')

        if i == 0:
            face_vertices = load_txt(f"./data/face_vertices{angle}", "float32")
            face_textures = load_txt(f"./data/face_textures{angle}", "float32")

        our_render(face_vertices, face_textures)

    tot_time = 0.
    timed_rounds = 0
    for i in range(repeat_num):
        t0 = time.time()
        our_render(face_vertices, face_textures)
        t1 = time.time()
        tot_time += t1 - t0
        timed_rounds += 1
        if tot_time > 60:
            break

    if repeat_num > 0:
        print(f"Drawing Rotation Time = {tot_time / timed_rounds} s")


if __name__ == '__main__':
    main()
