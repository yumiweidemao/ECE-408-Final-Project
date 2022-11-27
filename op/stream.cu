#include <cmath>
#include <iostream>
#include <cuda_fp16.h>
#include "gpu-new-forward.h"

/*
    List of optimizations:
    - Tiled shared memory convolution (2 points)
    - Kernel values in constant memory (1 point)
    - Multiple kernel implementations for different layer sizes (1 point)
    - FP16 arithmetic (4 points)
    - Using streams to overlap computation with data transfer (4 points)
    - Total: 12 points
*/

/* Constant memory allocation */
// First conv layer K = 7, Map_out = 4, Channel = 1, size=7*7*4*1=196
const int const_mask_size_1 = 196;
__constant__ half Mc1[const_mask_size_1];
// Second conv layer K = 7, Map_out = 16, Channel = 4, use half2 so divide by 2
// 7*7*16*4/2 = 1568
const int const_mask_size_2 = 1568;
__constant__ __half2 Mc2[const_mask_size_2];

/* Block dimensions */
const int BLOCK_WIDTH = 24;
const int BLOCK_WIDTH = 24;

/* Streams */
const int nStreams = 4;
static cudaStream_t stream[nStreams];

static const float * global_host_input; 
static const float * global_host_output;

/*
    This kernel is used for the first conv layer with K=7, Channel=1, Map_out=4.
*/
__global__ void conv_forward_kernel_tiled_1(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {

    __shared__ half subTile[BLOCK_WIDTH*BLOCK_WIDTH];

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int tile_out_width = BLOCK_WIDTH - K + 1;
    const int tile_out_height = BLOCK_WIDTH - K + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) Mc1[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define tile_3d(i2, i1, i0) subTile[(i2) * (BLOCK_WIDTH*BLOCK_WIDTH) + (i1) * (BLOCK_WIDTH) + i0]

    int W_grid = (Width_out-1)/tile_out_width + 1;
    int half_width = BLOCK_WIDTH/2;

    int b = blockIdx.z;
    int m = blockIdx.x;

    // Use this indexing method to increase L1 hit rate since each word contains two half types.
    // Conventional indexing method is slightly slower (2-3ms @ Batch=10000),
    // L1 hit rate is lower but there are fewer bank conflicts
    int tile_y = threadIdx.y - (threadIdx.y % 2) + (threadIdx.x / half_width);
    int tile_x = (threadIdx.x % half_width)*2 + (threadIdx.y % 2);
    int h = (blockIdx.y / W_grid)*tile_out_height+tile_y;
    int w = (blockIdx.y % W_grid)*tile_out_width+tile_x;

    // load the whole input tensor into shared memory
    if (h < Height && w < Width)
        tile_3d(0, tile_y, tile_x) = __float2half(in_4d(b, 0, h, w));
    else
        tile_3d(0, tile_y, tile_x) = 0;
    __syncthreads();

    half acc = 0.0f;
    if (tile_y < tile_out_height && tile_x < tile_out_width) {
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                acc += __hmul(tile_3d(0, tile_y+p, tile_x+q), mask_4d(m, 0, p, q));
            }
        }
        if (h < Height_out && w < Width_out)
            out_4d(b, m, h, w) = __half2float(acc);
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef tile_3d
}


__global__ void conv_forward_kernel_tiled_2(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {

    // largest channel = 4, one half2 stores two values, 4/2=2
    __shared__ __half2 subTile[2*BLOCK_WIDTH*BLOCK_WIDTH];

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int tile_out_width = BLOCK_WIDTH - K + 1;
    const int tile_out_height = BLOCK_WIDTH - K + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) Mc2[(i3) * (Channel/2 * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define tile_3d(i2, i1, i0) subTile[(i2) * (BLOCK_WIDTH*BLOCK_WIDTH) + (i1) * (BLOCK_WIDTH) + i0]

    int W_grid = (Width_out-1)/tile_out_width + 1;

    int b = blockIdx.z;
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid)*tile_out_height+threadIdx.y;
    int w = (blockIdx.y % W_grid)*tile_out_width+threadIdx.x;

    // load the whole input tensor into shared memory
    for (int c = 0; c < Channel/2; c++) {
        if (h < Height && w < Width)
            tile_3d(c, threadIdx.y, threadIdx.x) = __floats2half2_rn(in_4d(b, c*2, h, w), in_4d(b, c*2 + 1, h, w));
        else
            tile_3d(c, threadIdx.y, threadIdx.x) = __float2half2_rn(0.0f);
    }
    __syncthreads();

    // low half of acc stores channel 0 & 1, high half stores channel 2 & 3
    __half2 acc = __float2half2_rn(0.0f);
    if (threadIdx.y < tile_out_height && threadIdx.x < tile_out_width) {
        for (int c = 0; c < Channel/2; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    acc = __hadd2(acc, __hmul2(tile_3d(c, threadIdx.y+p, threadIdx.x+q), mask_4d(m, c, p, q)));
                }
            }
        }
        if (h < Height_out && w < Width_out)
            out_4d(b, m, h, w) = __half2float(__hadd(__high2half(acc), __low2half(acc)));
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef tile_3d
}
	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Prefer shared memory since it's faster
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    // allocate memory on GPU
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    cudaMalloc(device_output_ptr, Batch*Map_out*Height_out*Width_out*sizeof(float));
    cudaMalloc(device_input_ptr, Batch*Channel*Height*Width*sizeof(float));

    #define host_mask_4d(i3, i2, i1, i0) host_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define mask_half2_4d(i3, i2, i1, i0) mask_half2[(i3) * (Channel/2 * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // convert mask to half precision
    // the mask for conv layer 1 is just an array of half
    half mask_half[const_mask_size_1];
    if (Channel == 1) {
        for (int i = 0; i < const_mask_size_1; i++) {
            mask_half[i] = __float2half(host_mask[i]);
        }
    }

    // the mask for conv layer 2 is __half2, each __half2 stores data for two channels
    // For example, the first element stores the Map_out=0, (y, x) = (0, 0) kernel value for channel 0 and 1
    __half2 mask_half2[const_mask_size_2];
    if (Channel > 1) {
        for (int m = 0; m < Map_out; m++) {
            for (int c = 0; c < Channel/2; c++) {
                for (int p = 0; p < K; p++) {
                    for (int q = 0; q < K; q++) {
                        mask_half2_4d(m, c, p, q) = __floats2half2_rn(host_mask_4d(m, c*2, p, q), host_mask_4d(m, c*2+1, p, q));
                    }
                }
            }
        }
    }

    #undef host_mask_4d
    #undef mask_half2_4d

    // pass input & mask to GPU
    cudaMemcpyToSymbol(Mc1, mask_half, const_mask_size_1*sizeof(half));
    cudaMemcpyToSymbol(Mc2, mask_half2, const_mask_size_2*sizeof(__half2));
    //cudaMemcpy(*device_input_ptr, host_input, Batch*Channel*Height*Width*sizeof(float), cudaMemcpyHostToDevice);

    // create streams
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&stream[i]);
    }

    global_host_input = host_input;
    global_host_output = host_output;
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    dim3 DimGrid(Map_out, ((Height_out-1)/(BLOCK_WIDTH-K+1) + 1)*((Width_out-1)/(BLOCK_WIDTH-K+1) + 1), Batch/nStreams);
    dim3 DimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    
    int streamSize = (Batch*Channel*Height*Width) / nStreams;
    int streamSize_out = (Batch*Map_out*Height_out*Width_out) / nStreams;
    if (Channel == 1) {
        for (int i = 0; i < nStreams; i++) {
            int offset = i * streamSize;
            int offset_out = i * streamSize_out;
            cudaMemcpyAsync((float*)&device_input[offset], &global_host_input[offset], streamSize*sizeof(float), cudaMemcpyHostToDevice, stream[i]);
            conv_forward_kernel_tiled_1<<<DimGrid, DimBlock, 0, stream[i]>>>(device_output+offset_out, device_input+offset, device_mask, Batch, Map_out, Channel, Height, Width, K);
            cudaMemcpyAsync((float*)&global_host_output[offset_out], &device_output[offset_out], streamSize_out*sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
        }
    }
    else {
        for (int i = 0; i < nStreams; i++) {
            int offset = i * streamSize;
            int offset_out = i * streamSize_out;
            cudaMemcpyAsync((float*)&device_input[offset], &global_host_input[offset], streamSize*sizeof(float), cudaMemcpyHostToDevice, stream[i]);
            conv_forward_kernel_tiled_2<<<DimGrid, DimBlock, 0, stream[i]>>>(device_output+offset_out, device_input+offset, device_mask, Batch, Map_out, Channel, Height, Width, K);
            cudaMemcpyAsync((float*)&global_host_output[offset_out], &device_output[offset_out], streamSize_out*sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
        }
    }
    cudaDeviceSynchronize();
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);

    // destroy streams
    for (int i = 0; i < nStreams; i++) {
        cudaStreamDestroy(stream[i]);
    }
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
