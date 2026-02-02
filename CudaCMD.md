# knowledge
L2 cache 命中： 每次读取数据是从显存获取对应数据和相邻一块内存的地址，如果下一个SM正好计算用到了这块相邻地址的数据，那就节省了再去读取数据的时间。

SM（计算核心）  
  ↓
L1 / Shared Memory（片上缓存）
  ↓
L2 Cache（整个GPU共享）
  ↓
DRAM（显存）

## 内存管理
| | 标准C语言内存管理函数 | CUDA内存管理函数 |
| 内存分配 | malloc | cudaMalloc |
| 数据传输 | memcpy | cudaMemcpy |
| 内存初始化 | memset | cudaMemset |
| 内存释放 | free | cudaFree|

### 内存分配
```
主机内存分配： extern void *malloc(unsigned int num_bytes);
           -> float *fpHost_A;
              fpHost_A = (float*)malloc(nBytes);

设备内存分配： float *fpDevice_A;
              cudaMalloc((float**)&fpDevice_A,nBytes)
```

### 数据拷贝
```
主机数据拷贝 ： void *memcpy(void *dest,const void *src,size_t n);
            -> memcpy((void*)d,(void*)s,nBytes);

设备数据拷贝： cudaMemcpy(Device_A,Host_A,nBytes,cudaMemcpyHostToHost)
           种类:   cudaMemcpyHostToHost 主机 -> 主机
                   cudaMemcpyHostToDevice 主机 -> 设备
                   cudaMemcpyDeviceToHost 设备 -> 主机
                   cudaMemcpyDeviceToDevice 设备 -> 设备
```

### 内存初始化
```
主机内存初始化： void *memset(void *str,int c,size_t n);
             -> memset(fpHost_A,0,nBytes);

设备内存初始化： cudaMemset(fpDevice_A,0,nBytes);
```

### 内存释放
```
主机内存释放： free(pHost_A);

设备内存释放： cudaFree(pDevice_A);
```

# kernel 的一些
## mat_transpose 矩阵转置
一个将列优先转置成行优先，一个行优先转置成列优先，必须是要线程从内存拿数据的时候要连续内存。
```
__global__ void mat_transpose_f32_col2row_kernel(float *x, float *y,
                                                 const int row, const int col) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_row = global_idx / col;
  const int global_col = global_idx % col;
  if (global_idx < row * col) {
    y[global_col * row + global_row] = x[global_idx];
  }
}

__global__ void mat_transpose_f32_row2col_kernel(float *x, float *y,
                                                 const int row, const int col) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_col = global_idx / row;
  const int global_row = global_idx % row;
  if (global_idx < row * col) {
    y[global_idx] = x[global_row * col + global_col];
  }
}
```


# 算子解读

## reduce
```
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float *a, float *y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;      // 总共的warp数量
  __shared__ float reduce_smem[NUM_WARPS];  // 从shared_memory读取数据到reduce_smem(大小为warp数量，为了reduce所有warp内的和)
  // keep the data in register is enough for warp operaion.
  float sum = (idx < N) ? a[idx] : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum = warp_reduce_sum_f32<WARP_SIZE>(sum);   // warp内reduce
  // warp leaders store the data to shared memory.
  if (lane == 0)
    reduce_smem[warp] = sum;  // 把lane0的值搬到shared memory
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;  //把warp内的和搬运进warp内，只是后
  面有效搬运和计算发生在warp0内
  if (warp == 0)
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);   // 
  if (tid == 0)
    atomicAdd(y, sum);
}
```

torch 的 profile 会生成main.cpp，通过这个管道绑定矩阵
同时会生成.cu文件，里面包含真正的算子
生成build.ninja，编译.cpp文件，然后利用编译器标志（compiler flags）运行编译器，输出一个目标文件，这个目标文件会被调用