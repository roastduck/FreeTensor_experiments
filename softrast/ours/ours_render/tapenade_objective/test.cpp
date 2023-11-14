#include <assert.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#define scalar_t int
#define batch_size 1
#define image_size 256
#define num_faces 5856
#define texture_size 25

extern "C" {
void tapenade_render_main(scalar_t *soft_colors_t);
}
scalar_t soft_colors[batch_size * 4 * image_size * image_size];

void print(const scalar_t *a, std::string file, std::vector<size_t> shape) {
    freopen(file.c_str(), "w", stdout);
    size_t size = 1;
    for (auto &&mul : shape) {
        size *= mul;
        std::cout << mul << ' ';
    }
    std::cout << '\n';
    for (size_t i = 0; i < size; i++) {
        std::cout << a[i] << ' ';
    }
    fclose(stdout);
}

int main() {
    tapenade_render_main(soft_colors, faces, textures);

    print(soft_colors, "../result/test_soft_colors_fortran.txt",
          {batch_size, 4, image_size, image_size});

    return 0;
}
