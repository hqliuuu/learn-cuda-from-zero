#include <cuda_runtime.h>
#include <iostream>

__global__ void helloworld() {
  printf("threadIdx: %d\n", threadIdx.x);
  if (threadIdx.x == 0) {
    printf("GPU: Hello world!");
  }
}

int main(int argc, char **argv) {
  printf("CPU: Hello World!");
  helloworld<<<1, 10>>>();
  cudaDeviceSynchronize();
  if (cudaGetLastError() != cudaSuccess) {
    std::cerr << "CUDA: Error :" << cudaGetErrorString(cudaGetLastError()) << std::endl;
    return 1;
  }
  else {
    std::cout << "GPU: Hello World Success!" << std::endl;
  }
  std::cout << "CPU: Hello World Success!" << std::endl;
  return 0;
}