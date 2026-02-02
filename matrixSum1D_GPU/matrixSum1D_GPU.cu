#include <stdio.h>
#include "../device/setDevice.cuh"


__global__ void addFromGPU(float *A, float *B, float *C,const int N)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = tid + bid * blockDim.x;

    C[id] = A[id] + B[id];  
}

void initialData(float *addr, int elemCount)
{
    for (int i = 0; i < elemCount; i ++)
    {
        addr[i] = (float)(rand() &0xFF) / 10.f;
    }
    return;
}


int main(void)
{
    // 设置GPU设备
    setGPU();

    // 分配主机内存和设备内存，并初始化
    int iElemCount = 512;  //设置元素数量
    size_t stBytesCount = iElemCount * sizeof(float);  //计算字节数

    // 分配主机内存并初始化
    float *fpHost_A, *fpHost_B, *fpHost_C;
    fpHost_A = (float*)malloc(stBytesCount);
    fpHost_B = (float*)malloc(stBytesCount);
    fpHost_C = (float*)malloc(stBytesCount);
    if (fpHost_A != NULL && fpHost_B != NULL && fpHost_C != NULL)
    {
        memset(fpHost_A,0,stBytesCount);
        memset(fpHost_B,0,stBytesCount);
        memset(fpHost_C,0,stBytesCount);
    }
    else
    {
        printf("Fail to allocate host memory!\n");
        exit(-1);
    }

    // 分配设备内存并初始化
    float *fpDevice_A, *fpDevice_B, *fpDevice_C;
    cudaMalloc((float**)&fpDevice_A,stBytesCount);
    cudaMalloc((float**)&fpDevice_B,stBytesCount);
    cudaMalloc((float**)&fpDevice_C,stBytesCount);

    if (fpDevice_A != NULL && fpDevice_B != NULL && fpDevice_C != NULL)
    {
        cudaMemset(fpDevice_A,0,stBytesCount);
        cudaMemset(fpDevice_B,0,stBytesCount);
        cudaMemset(fpDevice_C,0,stBytesCount);
    }
    else
    {
        printf("Fail to allocate memory!\n");
        free(fpHost_A);
        free(fpHost_B);
        free(fpHost_C);
        exit(-1);
    }

    // 初始化主机数据
    srand(666);
    initialData(fpHost_A,iElemCount);
    initialData(fpHost_B,iElemCount);

    // 将数据从主机复制到设备
    cudaMemcpy(fpDevice_A,fpHost_A,stBytesCount,cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_B,fpHost_B,stBytesCount,cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_C,fpHost_C,stBytesCount,cudaMemcpyHostToDevice);

    // 调用核函数在设备中计算
    dim3 block(32);
    dim3 grid(iElemCount/32);

    addFromGPU<<<grid,block>>>(fpDevice_A, fpDevice_B, fpDevice_C, iElemCount);
    cudaDeviceSynchronize();

    // 将计算好的数据从设备传输给主机
    cudaMemcpy(fpHost_C, fpDevice_C,stBytesCount, cudaMemcpyDeviceToHost);

    for (int i = 0;i < 10; i++)
    {
        printf("idx=%2d\tmatrix_A:%.2f\tmatrix_B:%.2f\tresult=%.2f\n",i+1, fpHost_A[i], fpHost_B[i], fpHost_C[i]);
    }

    // 释放主机与设备内存
    free(fpHost_A);
    free(fpHost_B);
    free(fpHost_C);
    cudaFree(fpDevice_A);
    cudaFree(fpDevice_B);
    cudaFree(fpDevice_C);
}