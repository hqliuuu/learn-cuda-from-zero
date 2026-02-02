#include<stdio.h>

cudaError_t Errorcheck(cudaError_t error_code, const char* filename, int lineNumber)
{
    if (error_code != cudaSuccess)
    {
        printf("CUDA error:\r\ncode=%d, name=%s, description=%s\r\nfile=%s,line=%d\r\n",
        error_code,cudaGetErrorName(error_code), cudaGetErrorString(error_code), filename, lineNumber);
        return error_code;
    }
    return error_code;
}

void setGPU()
{
    //检测GPU数量
    int iDeviceCount = 0;
    cudaError_t error = Errorcheck(cudaGetDeviceCount(&iDeviceCount),__FILE__,__LINE__);

    if (error != cudaSuccess || iDeviceCount == 0)
    {
        printf("No CUDA campatable GPU found!\n");
        exit(-1);
    }
    else
    {
        printf("The count of GPUs is %d.\n",iDeviceCount);
    }

    //设置执行
    int iDev = 0;
    error = cudaSetDevice(iDev);
    if (error != cudaSuccess)
    {
        printf("fail to set GPU 0 for computing!\n");
        exit(-1);
    }
    else
    {
        printf("set GPU 0 for computing!");
    }

}