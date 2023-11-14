#include <assert.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#define scalar_t double
#define batch_size 1
#define image_size 256
#define num_faces 5856
#define texture_size 25

extern "C" {
void tapenade_render_main(scalar_t *soft_colors_t, const scalar_t *faces_t,
                          const scalar_t *textures_t);
void tapenade_render_main_b(scalar_t *soft_colors_t,
                            const scalar_t *soft_colors_tb,
                            const scalar_t *faces_t, scalar_t *faces_tb,
                            const scalar_t *textures_t, scalar_t *textures_tb);
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

int main() {
    read(faces, "../data/face_vertices000");
    read(textures, "../data/face_textures000");

    size_t size = sizeof(d_soft_colors) / sizeof(scalar_t);
    for (size_t i = 0; i < size; i++) {
        d_soft_colors[i] = 1.;
    }
    tapenade_render_main_b(soft_colors, d_soft_colors, faces, d_faces, textures,
                           d_textures);
    memset(soft_colors, 0, sizeof(soft_colors));
    tapenade_render_main(soft_colors, faces, textures);

    print(soft_colors, "../result/tapenade_soft_colors_fortran.txt",
          {batch_size, 4, image_size, image_size});
    print(d_faces, "../result/tapenade_faces000.txt",
          {batch_size, num_faces, 3, 3});
    print(d_textures, "../result/tapenade_textures000.txt",
          {batch_size, num_faces, texture_size, 3});

    return 0;
}
