#include <assert.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#define scalar_t float
#define batch_size 1
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
    std::ofstream os(file);
    size_t size = 1;
    for (auto &&mul : shape) {
        size *= mul;
        os << mul << ' ';
    }
    os << std::endl;
    for (size_t i = 0; i < size; i++) {
        os << std::fixed << std::setprecision(8) << a[i] << ' ';
    }
}

int main(int argc, char **argv) {
    namespace ch = std::chrono;
    int nWarmup = 3;
    int nTest = 1000;

    assert(argc == 2);
    read(faces, std::string("../../data/face_vertices") + argv[1]);
    read(textures, std::string("../../data/face_textures") + argv[1]);

    size_t size = batch_size * 4 * image_size * image_size;
    for (size_t i = 0; i < size; i++) {
        d_soft_colors[i] = 1.;
    }
    tapenade_render_main(soft_colors, faces, textures);
    memset(soft_colors, 0, size * sizeof(scalar_t));
    tapenade_render_main(soft_colors, faces, textures);

    print(soft_colors,
          std::string("../../result/tapenade_soft_colors_objective") + argv[1] +
              "_" + std::to_string(image_size) + ".txt",
          {batch_size, 4, image_size, image_size});
    /*print(d_faces,
          std::string("../../result/tapenade_grad_faces") + argv[1] + "_" +
              std::to_string(image_size) + ".txt",
          {batch_size, num_faces, 3, 3});
    print(d_textures,
          std::string("../../result/tapenade_grad_textures") + argv[1] + "_" +
              std::to_string(image_size) + ".txt",
          {batch_size, num_faces, texture_size, 3});*/

    for (int i = 0; i < nWarmup - 1; i++) {
        tapenade_render_main(soft_colors, faces, textures);
    }
    int timedRounds = 0;
    double totTime = 0;
    for (int i = 0; i < nTest; i++) {
        auto t0 = ch::high_resolution_clock::now();
        tapenade_render_main(soft_colors, faces, textures);
        auto t1 = ch::high_resolution_clock::now();
        totTime += ch::duration_cast<ch::duration<double>>(t1 - t0).count();
        timedRounds++;
        if (totTime > 60) {
            break;
        }
    }
    std::cout << "time = " << (totTime / timedRounds) << "s" << std::endl;

    return 0;
}
