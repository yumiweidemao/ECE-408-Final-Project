#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

const int BLOCK_WIDTH = 24;

// This is the shared memory tiled version (+2 points) kernel. 
// This kernel uses Strategy 2 - read all inputs
__global__ void conv_forward_kernel_tiled(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {

    __shared__ float subTile[4][BLOCK_WIDTH][BLOCK_WIDTH]; // largest channel number = 4 (second conv layer)

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int tile_out_width = BLOCK_WIDTH - K + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int W_grid = (Width_out-1)/tile_out_width + 1;

    int b = blockIdx.z;
    int m = blockIdx.x;
    int by = (blockIdx.y / W_grid); // block y-index
    int bx = (blockIdx.y % W_grid); // block x-index
    int h = by*tile_out_width+threadIdx.y;
    int w = bx*tile_out_width+threadIdx.x;

    // load the whole input tensor into shared memory
    for (int c = 0; c < Channel; c++) {
        if (h < Height && w < Width)
            subTile[c][threadIdx.y][threadIdx.x] = in_4d(b, c, h, w);
        else
            subTile[c][threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    float acc = 0.0f;
    if (threadIdx.y < tile_out_width && threadIdx.x < tile_out_width) {
        for (int c = 0; c < Channel; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) { 
                    acc += subTile[c][threadIdx.y+p][threadIdx.x+q] * mask_4d(m, c, p, q);
                }
            }
        }
        if (h < Height_out && w < Width_out)
            out_4d(b, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // allocate memory on GPU
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    cudaMalloc(device_output_ptr, Batch*Map_out*Height_out*Width_out*sizeof(float));
    cudaMalloc(device_input_ptr, Batch*Channel*Height*Width*sizeof(float));
    cudaMalloc(device_mask_ptr, Map_out*Channel*K*K*sizeof(float));

    // pass input & mask to GPU
    cudaMemcpy(*device_input_ptr, host_input, Batch*Channel*Height*Width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out*Channel*K*K*sizeof(float), cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    dim3 DimGrid(Map_out, ((Height_out-1)/(BLOCK_WIDTH-K+1) + 1)*((Width_out-1)/(BLOCK_WIDTH-K+1) + 1), Batch);
    dim3 DimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    conv_forward_kernel_tiled<<<DimGrid, DimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    
    cudaMemcpy(host_output, device_output, Batch*Map_out*Height_out*Width_out*sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
