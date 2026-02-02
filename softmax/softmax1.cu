#include<stdio.h>
#include<stdlib.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32

// naive softmax kernel
// threads in block = 256

template <const int NUM_THREADS = 256>
__global__ void softmax_block_reduce_kernel(
    const float* x,
    float* y,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sdata[NUM_THREADS];

    float val = 0.0f;
    if (idx < N) {
        val = expf(x[idx]);
    }
    sdata[threadIdx.x] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float sum = sdata[0];

    if (idx < N) {
        y[idx] = val / sum;
    }
}


template <const int NUM_THREADS = 256>
__global__ void naive_softmax_kernel(const float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float sum;
  if (threadIdx.x == 0) {
    sum = 0.0f;
  }
  __syncthreads();


  if (idx < N) {
    float val = expf(x[idx]);
    atomicAdd(&sum, val);
  }
  __syncthreads();


  if (idx < N) {
    y[idx] = expf(x[idx]) / sum;
  }
}


template <const int NUM_THREADS = 256>
__global__ void naive_softmax_kernel_1(const float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float sum;
  __shared__ float exp_vals[NUM_THREADS];
  if (threadIdx.x == 0) {
    sum = 0.0f;
  }
  __syncthreads();


  if (idx < N) {
    float val = expf(x[idx]);
    exp_vals[threadIdx.x] = val;
    atomicAdd(&sum, val);
  }
  __syncthreads();


  if (idx < N) {
    y[idx] = expf(exp_vals[threadIdx.x] )/ sum;
  }
}


#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

// if dtype and shape is unmatchbale, return error
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                                       \
  assert((T1).dim() == (T2).dim());                                            \
  for (int i = 0; i < (T1).dim(); ++i) {                                       \
    if ((T2).size(i) != (T1).size(i)) {                                        \
      throw std::runtime_error("Tensor size mismatch!");                       \
    }                                                                          \
  }

// grid memory fence
// #define TORCH_BINDING_SOFTMAX(packed_type, th_type, element_type, n_elements)  \
//   void softmax_##packed_type(torch::Tensor x, torch::Tensor y) {               \
//     CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
//     CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
//     auto options =                                                             \
//         torch::TensorOptions().dtype((th_type)).device(torch::kCUDA, 0);       \
//     const int N = x.size(0);                                                   \
//     CHECK_TORCH_TENSOR_SHAPE(x, y)                                             \
//     auto total = torch::zeros({1}, options);                                   \
//     dim3 block(64);                                                           \
//     dim3 grid(((N + 64 - 1) / 64) / (n_elements));                           \
//     softmax_##packed_type##_kernel<64><<<grid, block>>>(                      \
//         reinterpret_cast<element_type *>(x.data_ptr()),                        \
//         reinterpret_cast<element_type *>(y.data_ptr()),                        \
//         reinterpret_cast<element_type *>(total.data_ptr()), N);                \
//   }

// caculate grid and block
#define LANUCH_NAIVE_SOFTMAX_F32_KERNEL() \
    naive_softmax_kernel<<<grid, block>>>(reinterpret_cast<float*>(x.data_ptr()), \
                                              reinterpret_cast<float*>(y.data_ptr()), N)

// 1D flatten
#define DISPATCH_NAIVE_SOFTMAX_F32(S, H) \
    dim3 block(256);                     \
    dim3 grid((S * H + 256 - 1) / 256);  \
    LANUCH_NAIVE_SOFTMAX_F32_KERNEL();


#define LANUCH_NAIVE_SOFTMAX_F32_KERNEL_1() \
    naive_softmax_kernel_1<<<grid, block>>>(reinterpret_cast<float*>(x.data_ptr()), \
                                              reinterpret_cast<float*>(y.data_ptr()), N)

// 1D flatten
#define DISPATCH_NAIVE_SOFTMAX_F32_1(S, H) \
    dim3 block(256);                     \
    dim3 grid((S * H + 256 - 1) / 256);  \
    LANUCH_NAIVE_SOFTMAX_F32_KERNEL_1();



#define LANUCH_SOFTMAX_BLOCK_REDUCE_KERNEL() \
    softmax_block_reduce_kernel<<<grid, block>>>(reinterpret_cast<float*>(x.data_ptr()), \
                                              reinterpret_cast<float*>(y.data_ptr()), N)

// 1D flatten
#define DISPATCH_SOFTMAX_BLOCK_REDUCE_F32(S, H) \
    dim3 block(256);                     \
    dim3 grid((S * H + 256 - 1) / 256);  \
    LANUCH_SOFTMAX_BLOCK_REDUCE_KERNEL();



// // 1 block caculate 1 token(1 row)
// #define DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)                            \
//   dim3 block((H));                                                             \
//   dim3 grid((S));                                                              \
//   switch ((H)) {                                                               \
//   case 32:                                                                     \
//     LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(32)                                    \
//     break;                                                                     \
//   case 64:                                                                     \
//     LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(64)                                    \
//     break;                                                                     \
//   case 128:                                                                    \
//     LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(128)                                   \
//     break;                                                                     \
//   case 256:                                                                    \
//     LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(256)                                   \
//     break;                                                                     \
//   case 512:                                                                    \
//     LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(512)                                   \
//     break;                                                                     \
//   case 1024:                                                                   \
//     LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)                                  \
//     break;                                                                     \
//   default:                                                                     \
//     throw std::runtime_error("only support H: 64/128/256/512/1024");           \
//     break;                                                                     \
//   }

// pytorch can call the kernel from here
void softmax_f32_naive(torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
    CHECK_TORCH_TENSOR_SHAPE(x, y)

    const int S = x.size(0);
    const int H = x.size(1);
    const int N = S * H;

    DISPATCH_NAIVE_SOFTMAX_F32(S, H)
}

void softmax_f32_naive_1(torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
    CHECK_TORCH_TENSOR_SHAPE(x, y)

    const int S = x.size(0);
    const int H = x.size(1);
    const int N = S * H;

    DISPATCH_NAIVE_SOFTMAX_F32_1(S, H)
}

void softmax_block_reduce(torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
    CHECK_TORCH_TENSOR_SHAPE(x, y)

    const int S = x.size(0);
    const int H = x.size(1);
    const int N = S * H;

    DISPATCH_SOFTMAX_BLOCK_REDUCE_F32(S, H)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(softmax_f32_naive)
    TORCH_BINDING_COMMON_EXTENSION(softmax_f32_naive_1)
    TORCH_BINDING_COMMON_EXTENSION(softmax_block_reduce)
}
