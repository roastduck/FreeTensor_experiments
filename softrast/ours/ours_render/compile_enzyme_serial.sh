#/home/spack/spack/opt/spack/linux-debian12-zen2/gcc-11.3.0/llvm-16.0.2-me2hmj2l7jeouxexrsmp6om76p2uuqpn/bin/clang enzyme_render.cpp -Xclang -load -Xclang /home/trrbivial/Enzyme-0.0.69/enzyme/build/Enzyme/LLVMEnzyme-16.so -o a.out -lm -lstdc++
/home/rd/src/Enzyme_experiments/llvm-project-12.0.1.src/build/bin/clang++ enzyme_render_serial.cpp \
    -Xclang -load -Xclang /home/rd/src/parallel-ad-minibench/enzyme/Enzyme/enzyme/build/Enzyme/ClangEnzyme-12.so \
    -O2 -o enzyme_render_serial -std=c++20 -g
#/home/spack/spack/opt/spack/linux-debian12-zen2/gcc-11.3.0/llvm-16.0.2-me2hmj2l7jeouxexrsmp6om76p2uuqpn/bin/clang enzyme_render.cpp -S -emit-llvm -o input.ll -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops
#/home/spack/spack/opt/spack/linux-debian12-zen2/gcc-11.3.0/llvm-16.0.2-me2hmj2l7jeouxexrsmp6om76p2uuqpn/bin/opt input.ll -load=/home/trrbivial/Enzyme-0.0.69/enzyme/build/Enzyme/LLVMEnzyme-16.so -enzyme -o output.ll -S --enable-new-pm=0
#/home/spack/spack/opt/spack/linux-debian12-zen2/gcc-11.3.0/llvm-16.0.2-me2hmj2l7jeouxexrsmp6om76p2uuqpn/bin/opt output.ll -O2 -o output_opt.ll -S
#/home/spack/spack/opt/spack/linux-debian12-zen2/gcc-11.3.0/llvm-16.0.2-me2hmj2l7jeouxexrsmp6om76p2uuqpn/bin/clang output_opt.ll -o a.out -lm -lstdc++
