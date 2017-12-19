#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include <cuda.h>
#include <unistd.h>

#define CEIL(a, b) ((a) + (b) -1)/(b)
#define INF 1000000000

int **Dist;
int *data;
int block_size;
int vert, edge;
int vert2;

inline void init(){
    vert2 = vert*vert;
    Dist = new int*[vert];
    data = new int[vert2];

    std::fill(data, data + vert2, INF);

    for(int i=0;i<vert;++i){
        Dist[i] = data + i*vert;
        Dist[i][i] = 0;
    }
}

inline void finalize(){
    delete[] Dist;
    delete[] data;
}

void dump_from_file_and_init(const char *file){
    std::ifstream fin(file);
    std::stringstream ss;

    ss << fin.rdbuf();
    ss >> vert >> edge;
    
    init();

    int i, j, w;
    while(--edge >= 0){
        ss >> i >> j >> w;
        Dist[i][j] = w;
    }
    fin.close();
}

void dump_to_file(const char *file){
    std::ofstream fout(file);
    fout.write((char*)data, sizeof(int)*vert2);
    fout.close();
}

__global__ void init_gpu(int reps){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= reps) return;
}

__global__ void phase_one(int32_t* const dist, int block_size, int round, int width, int vert){
    extern __shared__ int s[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int br = block_size * round;
    const int c = br + ty;
    const int r = br + tx;

    const int ty_block = ty * block_size;
    int k_block = 0;

    const int cell = c * width + r;
    const int s_cell = ty_block + tx;

    __syncthreads();

    s[s_cell] = dist[cell];

    if(c >= vert || r >= vert)
        s[s_cell] = INF;

    __syncthreads();

    int n, k;
    for(k=0;k<block_size;++k){
        n = s[ty_block + k] + s[k_block + tx];
        if(n < s[s_cell])
            s[s_cell] = n;
        k_block += block_size;
        __syncthreads();
    }

    dist[cell] = s[s_cell];
}

__global__ void phase_two(int32_t* const dist, int block_size, int round, int width, int vert){
    extern __shared__ int s[];
    int *const s_m = s;
    int *const s_c = s + block_size * block_size;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int mc, mr;
    int cc, cr;
    const int br = block_size * round;

    if(bx >= round)++bx;

    if(by == 0){
        mc = br + ty;
        mr = block_size * bx + tx;
        cc = mc;
        cr = br + tx;
    }else{
        mc = block_size * bx + ty;
        mr = br + tx;
        cc = br + ty;
        cr = mr;
    }

    const int ty_block = ty * block_size;
    int k_block = 0;
    
    int m_cell = mc * width + mr;
    int c_cell = cc * width + cr;
    int s_cell = ty_block + tx;

    __syncthreads();

    s_m[s_cell] = dist[m_cell];
    s_c[s_cell] = dist[c_cell];

    if(mc >= vert || mr >= vert)
        s_m[s_cell] = INF;
    if(cc >= vert || cr >= vert)
        s_c[s_cell] = INF;

    __syncthreads();

    int n, k;
    if(by == 0){
        for(k=0;k<block_size;++k){
            n = s_c[ty_block + k] + s_m[k_block + tx];
            if(n < s_m[s_cell])
                s_m[s_cell] = n;
            k_block += block_size;
            __syncthreads();
        }
    }else{
        for(k=0;k<block_size;++k){
            n = s_m[ty_block + k] + s_c[k_block + tx];
            if(n < s_m[s_cell])
                s_m[s_cell] = n;
            k_block += block_size;
            __syncthreads();
        }
    }

    dist[m_cell] = s_m[s_cell];
}

__global__ void phase_three(int32_t* const dist, int block_size, int round, int width, int vert){

    const int bs2 = block_size*block_size;

    extern __shared__ int s[];
    int *const s_m = s;
    int *const s_l = s + bs2;
    int *const s_r = s_l + bs2;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    if(bx >= round)++bx;
    if(by >= round)++by;

    const int br = block_size * round;
    const int ty_block = ty * block_size;
    int k_block = 0;

    const int mc = block_size * by + ty;
    const int mr = block_size * bx + tx;
    const int lc = mc;
    const int lr = br + tx;
    const int rc = br + ty;
    const int rr = mr;
    const int m_cell = mc * width + mr;
    const int l_cell = lc * width + lr;
    const int r_cell = rc * width + rr;
    const int s_cell = ty_block + tx;

    __syncthreads();

    s_m[s_cell] = dist[m_cell];
    s_l[s_cell] = dist[l_cell];
    s_r[s_cell] = dist[r_cell];

    if(mc >= vert || mr >= vert)
        s_m[s_cell] = INF;
    if(lc >= vert || lr >= vert)
        s_l[s_cell] = INF;
    if(rc >= vert || rr >= vert)
        s_r[s_cell] = INF;

    __syncthreads();

    int n, k;
    for(k=0;k<block_size;++k){
        n = s_l[ty_block + k] + s_r[k_block + tx];
        if(n < s_m[s_cell])
            s_m[s_cell] = n;
        k_block += block_size;
        __syncthreads();
    }

    dist[m_cell] = s_m[s_cell];
}

void block_FW(){
    int Round = CEIL(vert, block_size);
    size_t vert_bytes = vert * sizeof(int);

    int32_t *device_ptr;
    size_t padded_size = Round * block_size;
    size_t padded_bytes = padded_size * sizeof(int);
    size_t pitch_bytes;

    dim3 p2b(Round-1, 2, 1);
    dim3 p3b(Round-1, Round-1, 1);

    dim3 dimt(block_size, block_size, 1);

    cudaSetDevice(0);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMallocPitch(&device_ptr, &pitch_bytes, padded_bytes, padded_size);
    int pitch = pitch_bytes / sizeof(int);

    init_gpu<<< 1 , dimt >>>(32);

    cudaMemcpy2DAsync(device_ptr, pitch_bytes, data, vert_bytes,
                    vert_bytes, vert, cudaMemcpyHostToDevice, stream);

    size_t bs2b3 = block_size * block_size * sizeof(int) * 3;

    cudaDeviceSynchronize();

    for(int r=0;r<Round;++r){
        phase_one<<< 1 , dimt , bs2b3 >>>(device_ptr, block_size, r, pitch, vert);
        phase_two<<< p2b , dimt , bs2b3 >>>(device_ptr, block_size, r, pitch, vert);
        phase_three<<< p3b , dimt , bs2b3 >>>(device_ptr, block_size, r, pitch, vert);
    }

    cudaDeviceSynchronize();

    cudaMemcpy2D(data, vert_bytes, device_ptr, pitch_bytes, vert_bytes, vert, cudaMemcpyDeviceToHost);

    cudaFree(device_ptr);
    cudaStreamDestroy(stream);

}


int main(int argc, char **argv){
    dump_from_file_and_init(argv[1]);
    block_size = std::atoi(argv[3]);

    block_FW();

    dump_to_file(argv[2]);
    finalize();

    return 0;
}






