import torch
from torch.utils.cpp_extension import load_inline


# 定义cuda算子并且用torch做绑定
#  把矩阵向量化，然后遍历每个元素进行自乘
#  同时还需要一个包装函数 wrapper function，将pytorch张量转化成内核需要的格式
cuda_source = '''
__global__ void square_matrix_kernel(const float* matrix, float* result, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x + blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int idx = row * width + col;
        result[idx] = matrix[idx] * matrix[idx];
    }
}


torch::Tensor square_matrix(torch::Tensor matrix) {
    const auto height = matrix.size(0);
    const auto width = matrix.size(1);

    auto result = torch::empty_like(matrix);

    dim3 threads_per_block(16, 16);
    dim3 numbers_of_block((width + threads_per_block.x - 1) / threads_per_block.x,
                          (height + threads_per_block.y - 1) / threads_per_block.y);

    square_matrix_kernel<<<numbers_of_block, threads_per_block>>>(
        matrix.data_ptr<float>(), result.data_ptr<float>(), width, height);
    
    return result;
}
'''

#  实际的c++函数在main.cpp中暴露，在这里进行调用
cpp_source = "torch::Tensor square_matrix(torch::Tensor matrix);"

# 加载cuda作为pytorch的一个延申
square_matrix_extension = load_inline(
    name='square_matrix_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['square_matrix'],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    build_directory='F:/c+code/cuda/load_inline_cuda'
)

#  调用pytorch的模块
a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device='cuda')
print(square_matrix_extension.square_matrix(a))