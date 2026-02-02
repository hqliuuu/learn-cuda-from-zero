#include <stdio.h>

__global__ void square_matrix_kernel(const float* matrix, float* result, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x + blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int idx = row * width + col;
        matrix[idx] = matrix[idx] * matrix[idx];
    }
}


torch::Tensor square_matrix(torch:Tensor matrix) {
    const auto height = matrix.size(0);
    const auto width = matrix.size(1);

    auto result = torch::empty_like(matrix);

    dim3 threads_per_block(16, 16);
    dim3 numbers_of_block((width + threads_per_block - 1) / threads_per_block.x, (height + threads_per_block - 1) / threads_per_block.y);

    square_matrix_kernel<<<numbers_of_block, threads_per_block>>>(
        matrix.data_ptr<float>(), result.data_ptr<float>(), width, height);

    return result;
}