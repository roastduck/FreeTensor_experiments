#include <assert.h>

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>

#define scalar_t double
#define batch_size (1)
#define num_faces (5856)
#define texture_size (25)
#define texture_res (5)
#define image_size (256)
#define near (1.)
#define far (100.)
#define eps (1e-3)
#define sigma_val (1e-5)
#define gamma_val (1e-4)
#define dist_eps (9.21024036697585)
#define threshold (dist_eps * sigma_val)
#define double_side (false)
#define texture_type (0)

#define MAX(a, b) (max((a), (b)))
#define MIN(a, b) (min((a), (b)))
extern void __device__ __enzyme_autodiff_render(void *, int, scalar_t *,
                                                const scalar_t *, int,
                                                const scalar_t *, scalar_t *,
                                                int, const scalar_t *,
                                                scalar_t *);

int __device__ enzyme_dup;

inline void __device__ face_inv(scalar_t *inv_t, const scalar_t *face_t) {
#define v(k, l) (face_t[(k) * (3) + (l)])
#define inv(k, l) (inv_t[(k) * (3) + (l)])
    scalar_t det = 0.;
    for (int p = 0; p < 3; p++) {
        det += v(p, 0) * (v((p + 1) % 3, 1) - v((p + 2) % 3, 1));
    }

    det = det > 0 ? MAX(det, 1e-10) : MIN(det, -1e-10);

    for (int p = 0; p < 3; p++) {
        inv(p, 0) = (v((p + 1) % 3, 1) - v((p + 2) % 3, 1)) / det;
        inv(p, 1) = (v((p + 2) % 3, 0) - v((p + 1) % 3, 0)) / det;
        inv(p, 2) = (v((p + 1) % 3, 0) * v((p + 2) % 3, 1) -
                     v((p + 2) % 3, 0) * v((p + 1) % 3, 1)) /
                    det;
    }
#undef v
#undef inv
}

inline scalar_t __device__ dot_xy(const scalar_t *v1, const scalar_t *v2) {
    return v1[0] * v2[0] + v1[1] * v2[1];
}
inline scalar_t __device__ cross_xy(const scalar_t *v1, const scalar_t *v2) {
    return v1[0] * v2[1] - v1[1] * v2[0];
}
inline void __device__ sub_xy(scalar_t *v, const scalar_t *v1,
                              const scalar_t *v2) {
    for (int k = 0; k < 2; k++) v[k] = v1[k] - v2[k];
}
inline void __device__ barycentric_coordinate(scalar_t *w, const scalar_t *p,
                                              const scalar_t *inv_t) {
#define inv(k, l) (inv_t[(k) * (3) + (l)])
    w[0] = inv(0, 0) * p[0] + inv(0, 1) * p[1] + inv(0, 2);
    w[1] = inv(1, 0) * p[0] + inv(1, 1) * p[1] + inv(1, 2);
    w[2] = inv(2, 0) * p[0] + inv(2, 1) * p[1] + inv(2, 2);
#undef inv
}
inline bool __device__ check_border(const scalar_t *p, const scalar_t *face_t,
                                    const scalar_t threshold_t) {
#define f(k, l) (face_t[(k) * (3) + (l)])
    scalar_t t = sqrt(threshold_t);
    return (p[0] > MAX(MAX(f(0, 0), f(1, 0)), f(2, 0)) + t) ||
           (p[0] < MIN(MIN(f(0, 0), f(1, 0)), f(2, 0)) - t) ||
           (p[1] > MAX(MAX(f(0, 1), f(1, 1)), f(2, 1)) + t) ||
           (p[1] < MIN(MIN(f(0, 1), f(1, 1)), f(2, 1)) - t);
#undef f
}

inline bool __device__ check_face_frontside(const scalar_t *face_t) {
#define f(k, l) (face_t[(k) * (3) + (l)])
    return (f(2, 1) - f(0, 1)) * (f(1, 0) - f(0, 0)) <
           (f(1, 1) - f(0, 1)) * (f(2, 0) - f(0, 0));
#undef f
}
inline void __device__ barycentric_clip(scalar_t *w_clip, const scalar_t *w) {
    for (int k = 0; k < 3; k++) w_clip[k] = MAX(MIN(w[k], 1.), 0.);
    scalar_t w_sum = MAX(w_clip[0] + w_clip[1] + w_clip[2], 1e-5);
    for (int k = 0; k < 3; k++) w_clip[k] /= w_sum;
}

inline scalar_t __device__ euclidean_p2f_distance(const scalar_t *face_t,
                                                  const scalar_t *p) {
#define f(k) (face_t + ((k) * (3)))
    scalar_t dis[3];
    for (int k = 0; k < 3; k++) {
        scalar_t t1[3], t2[3], t3[3];
        sub_xy(t1, p, f(k));
        sub_xy(t2, f((k + 1) % 3), f(k));
        scalar_t area = cross_xy(t1, t2);

        scalar_t d1 = dot_xy(t1, t2);
        if (d1 >= 0) {
            sub_xy(t3, f((k + 1) % 3), p);
            scalar_t d2 = dot_xy(t2, t3);
            if (d2 >= 0) {
                scalar_t len = t2[0] * t2[0] + t2[1] * t2[1];
                dis[k] = area / MAX(len, 1e-10) * area;
            } else {
                dis[k] = t3[0] * t3[0] * t3[1] * t3[1];
            }
        } else {
            dis[k] = t1[0] * t1[0] + t1[1] * t1[1];
        }
    }
    return MIN(MIN(dis[0], dis[1]), dis[2]);
#undef f
}
/*
inline scalar_t __device__ euclidean_p2f_distance(const scalar_t *face_t,
                                                  const scalar_t *p) {
#define f(k, l) (face_t[(k) * (3) + (l)])
    scalar_t dis[3];
    for (int k = 0; k < 3; k++) {
        scalar_t t1[3], t2[3], t3[3];

        for (int i = 0; i < 2; i++) {
            t1[i] = p[i] - f(k, i);
            t2[i] = f((k + 1) % 3, i) - f(k, i);
        }
        scalar_t area = cross_xy(t1, t2);

        scalar_t d1 = dot_xy(t1, t2);
        if (d1 >= 0) {
            for (int i = 0; i < 2; i++) {
                t3[i] = f((k + 1) % 3, i) - p[i];
            }
            scalar_t d2 = dot_xy(t2, t3);
            if (d2 >= 0) {
                scalar_t len = t2[0] * t2[0] + t2[1] * t2[1];
                dis[k] = area / max(len, 1e-10) * area;
            } else {
                dis[k] = t3[0] * t3[0] + t3[1] * t3[1];
            }
        } else {
            dis[k] = t1[0] * t1[0] + t1[1] * t1[1];
        }
    }
    return min(min(dis[0], dis[1]), dis[2]);
#undef f
}
*/
inline scalar_t __device__ forward_sample_texture(const scalar_t *texture_t,
                                                  const scalar_t *w,
                                                  const int r, const int k,
                                                  const int ty) {
#define texture(k, l) (texture_t[(k) * (3) + (l)])
    scalar_t texture_k = 0.;
    if (ty == 0) {
        int w_x = w[0] * r;
        int w_y = w[1] * r;
        if ((w[0] + w[1]) * r - w_x - w_y <= 1) {
            if (w_y * r + w_x == texture_size) {
                texture_k = texture(w_y * r + w_x - 1, k);
            } else {
                texture_k = texture(w_y * r + w_x, k);
            }
        } else {
            texture_k = texture((r - 1 - w_y) * r + (r - 1 - w_x), k);
        }
    } else if (ty == 1) {
        texture_k =
            w[0] * texture(0, k) + w[1] * texture(1, k) + w[2] * texture(2, k);
    }
    return texture_k;
#undef texture
}

void __device__ enzyme_render_main(scalar_t *soft_colors_t,
                                   const scalar_t *faces_t,
                                   const scalar_t *textures_t) {
#define faces(bn, fn) \
    (faces_t + ((bn) * (num_faces) * (3) * (3) + (fn) * (3) * (3)))
#define face(a, b) (face_t[(a) * (3) + (b)])
#define soft_colors(bn, k, pn)                                \
    (soft_colors_t[(bn) * (4) * (image_size) * (image_size) + \
                   (k) * (image_size) * (image_size) + (pn)])
#define textures(bn, fn)                                       \
    (textures_t + ((bn) * (num_faces) * (texture_size) * (3) + \
                   (fn) * (texture_size) * (3)))

    for (int bn = 0; bn < batch_size; bn++) {
        for (int pn = 0; pn < image_size * image_size; pn++) {
            int yi = image_size - 1 - (pn / image_size);
            int xi = pn % image_size;
            scalar_t pixel[2];

            pixel[0] = (2. * xi + 1. - image_size) / image_size;
            pixel[1] = (2. * yi + 1. - image_size) / image_size;

            scalar_t softmax_max = eps;

            scalar_t soft_color[4];
            soft_color[3] = 1.;
            for (int fn = 0; fn < num_faces; fn++) {
                scalar_t inv[3 * 3];
                const scalar_t *face_t = faces(bn, fn);
                face_inv(inv, face_t);
                if (!check_border(pixel, face_t, threshold)) {
                    scalar_t w[3], w_clip[3];
                    barycentric_coordinate(w, pixel, inv);
                    barycentric_clip(w_clip, w);

                    scalar_t zp =
                        1. / (w_clip[0] / face(0, 2) + w_clip[1] / face(1, 2) +
                              w_clip[2] / face(2, 2));

                    if (!(zp < near || zp > far)) {
                        if (check_face_frontside(face_t) || double_side) {
                            scalar_t zp_norm = (far - zp) / (far - near);

                            if (zp_norm > softmax_max) {
                                softmax_max = zp_norm;
                            }
                        }
                    }
                }
            }

            scalar_t softmax_sum = exp((2 * eps - softmax_max) / gamma_val);

            for (int k = 0; k < 3; k++) {
                soft_color[k] = 0;
            }

            for (int fn = 0; fn < num_faces; fn++) {
                const scalar_t *face_t = faces(bn, fn);
                const scalar_t *texture_t = textures(bn, fn);
                scalar_t inv[3 * 3];
                face_inv(inv, face_t);
                if (!check_border(pixel, face_t, threshold)) {
                    scalar_t w[3];
                    barycentric_coordinate(w, pixel, inv);

                    scalar_t sign;

                    if (w[0] > 0 && w[1] > 0 && w[2] > 0 && w[0] < 1 &&
                        w[1] < 1 && w[2] < 1)
                        sign = 1;
                    else
                        sign = -1;

                    scalar_t dis = euclidean_p2f_distance(face_t, pixel);

                    if (!(sign < 0 and dis >= threshold)) {
                        scalar_t soft_fragment =
                            1. / (1. + exp(-sign * dis / sigma_val));

                        soft_color[3] *= 1. - soft_fragment;

                        scalar_t w_clip[3];

                        barycentric_clip(w_clip, w);

                        scalar_t zp = 1. / (w_clip[0] / face(0, 2) +
                                            w_clip[1] / face(1, 2) +
                                            w_clip[2] / face(2, 2));

                        if (!(zp < near || zp > far)) {
                            if (check_face_frontside(face_t) || double_side) {
                                scalar_t zp_norm = (far - zp) / (far - near);

                                scalar_t coef =
                                    exp((zp_norm - softmax_max) / gamma_val) *
                                    soft_fragment;
                                softmax_sum += coef;
                                for (int k = 0; k < 3; k++) {
                                    scalar_t color_k = forward_sample_texture(
                                        texture_t, w_clip, texture_res, k,
                                        texture_type);
                                    soft_color[k] += coef * color_k;
                                }
                            }
                        }
                    }
                }
            }

            soft_colors(bn, 3, pn) = 1. - soft_color[3];
            for (int k = 0; k < 3; k++) {
                soft_colors(bn, k, pn) = soft_color[k] / softmax_sum;
            }
        }
    }
#undef faces
#undef face
#undef soft_colors
#undef textures
}

scalar_t soft_colors[batch_size * 4 * image_size * image_size];
scalar_t d_soft_colors[batch_size * 4 * image_size * image_size];
scalar_t faces[batch_size * num_faces * 3 * 3];
scalar_t d_faces[batch_size * num_faces * 3 * 3];
scalar_t textures[batch_size * num_faces * texture_size * 3];
scalar_t d_textures[batch_size * num_faces * texture_size * 3];

void read(scalar_t *a, std::string file) {
    freopen(file.c_str(), "r", stdin);
    size_t batchSize, numFaces, x, y;
    std::cin >> batchSize >> numFaces >> x >> y;
    assert(batchSize == batch_size);
    assert(numFaces == num_faces);
    size_t size = batchSize * numFaces * x * y;
    for (size_t i = 0; i < size; i++) {
        std::cin >> a[i];
    }
    fclose(stdin);
}
void print(const scalar_t *a, std::string file, std::vector<size_t> shape) {
    freopen(file.c_str(), "w", stdout);
    size_t size = 1;
    for (auto &&mul : shape) {
        size *= mul;
        std::cout << mul << ' ';
    }
    std::cout << '\n';
    for (size_t i = 0; i < size; i++) {
        std::cout << std::fixed << std::setprecision(8) << a[i] << ' ';
    }
    fclose(stdout);
}

void __global__ enzyme_grad(scalar_t *soft_colors,
                            const scalar_t *d_soft_colors,
                            const scalar_t *faces, scalar_t *d_faces,
                            const scalar_t *textures, scalar_t *d_textures) {
    __enzyme_autodiff_render((void *)enzyme_render_main, enzyme_dup,
                             soft_colors, d_soft_colors, enzyme_dup, faces,
                             d_faces, enzyme_dup, textures, d_textures);
}

int main() {
    read(faces, "./data/face_vertices000");
    read(textures, "./data/face_textures000");

    size_t size = sizeof(d_soft_colors) / sizeof(scalar_t);
    for (size_t i = 0; i < size; i++) {
        d_soft_colors[i] = 1.;
    }

    scalar_t *soft_colors_t, *d_soft_colors_t, *faces_t, *d_faces_t,
        *textures_t, *d_textures_t;

    cudaMalloc(&soft_colors_t, sizeof(soft_colors));
    cudaMalloc(&d_soft_colors_t, sizeof(d_soft_colors));
    cudaMalloc(&faces_t, sizeof(faces));
    cudaMalloc(&d_faces_t, sizeof(d_faces));
    cudaMalloc(&textures_t, sizeof(textures));
    cudaMalloc(&d_textures_t, sizeof(d_textures));

    cudaMemcpy(soft_colors_t, soft_colors, sizeof(soft_colors),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_soft_colors_t, d_soft_colors, sizeof(d_soft_colors),
               cudaMemcpyHostToDevice);
    cudaMemcpy(faces_t, faces, sizeof(faces), cudaMemcpyHostToDevice);
    cudaMemcpy(d_faces_t, d_faces, sizeof(d_faces), cudaMemcpyHostToDevice);
    cudaMemcpy(textures_t, textures, sizeof(textures), cudaMemcpyHostToDevice);
    cudaMemcpy(d_textures_t, d_textures, sizeof(d_textures),
               cudaMemcpyHostToDevice);

    /*
    memset(faces, 0, sizeof(faces));
    cudaMemcpy(faces, faces_t, sizeof(faces), cudaMemcpyDeviceToHost);
    size_t tmp0[] = {batch_size, num_faces, 3, 3};
    print(faces, "./newnew/faces.txt", std::vector<size_t>(tmp0, tmp0 + 4));
    */

    enzyme_grad<<<1, 1>>>(soft_colors_t, d_soft_colors_t, faces_t, d_faces_t,
                          textures_t, d_textures_t);

    cudaDeviceSynchronize();

    cudaMemcpy(soft_colors, soft_colors_t, sizeof(soft_colors),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(d_faces, d_faces_t, sizeof(d_faces), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_textures, d_textures_t, sizeof(d_textures),
               cudaMemcpyDeviceToHost);

    size_t tmp1[] = {batch_size, 4, image_size, image_size};
    print(soft_colors, "./newnew/soft_colors.txt",
          std::vector<size_t>(tmp1, tmp1 + 4));

    size_t tmp2[] = {batch_size, num_faces, 3, 3};
    print(d_faces, "./result/dd_faces000.txt",
          std::vector<size_t>(tmp2, tmp2 + 4));

    size_t tmp3[] = {batch_size, num_faces, texture_size, 3};
    print(d_textures, "./result/dd_textures000.txt",
          std::vector<size_t>(tmp3, tmp3 + 4));

    return 0;
}
