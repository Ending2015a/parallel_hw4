#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <chrono>


#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include <cuda.h>
#include <unistd.h>

//#define _DEBUG_
//#define _TIME_MEASURE_

#ifdef _DEBUG_
    #include <string>
    #include <sstream>

    int __print_step = 0;
    
    void __pt_log(const char *h_, const char *f_, ...){
        std::stringstream ss;
        ss << h_ << f_ << '\n';
        std::string format = ss.str();

        va_list va;
        va_start(va, f_);
            vprintf(format.c_str(), va);
        va_end(va);
        __print_step++;
    }

    #define VA_ARGS(...) , ##__VA_ARGS__
    #define LOG(f_, ...) __pt_log(\
                                    "[LOG] Step %3d: ", (f_), \
                                     __print_step VA_ARGS(__VA_ARGS__))

#else
    #define LOG(f_, ...)
#endif


#define INF 1000000000
#define CEIL(a, b) (( (a) - 1 ) / (b) + 1 )


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

    if(vert < block_size){
        block_size = vert;
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
    LOG("vert: %d, edge: %d", vert, edge);

    init();

    int i, j, w;
    while(--edge >=0){
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
    const int c = block_size * round + ty;
    const int r = block_size * round + tx;
    const int cell = c*width+r;
    const int s_cell = ty*block_size+tx;

    if(c >= vert || r >= vert){
        s[s_cell] = INF;
    }else{
        s[s_cell] = dist[cell];
    }

    __syncthreads();

    int n, k;
    for(k=0;k<block_size;++k){
        // min(dist[ty][tx], dist[ty][i] + dist[i][tx])
        n = s[ty*block_size+k] + s[k*block_size+tx];
        if(n < s[s_cell]){
            s[s_cell] = n;
        }
        __syncthreads();
    }

    dist[cell] = s[s_cell];
}

__global__ void phase_two(int32_t* const dist, int block_size, int round, int width, int vert){
    extern __shared__ int s2[];
    int* const s_m = s2;  //main(block)
    int* const s_c = s2 + block_size*block_size;  //center(pivot)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int mc, mr;  //main
    int cc, cr;  //center(pivot)

    if(bx >= round)++bx; //shift

    if(by == 0){  //horizontal
        mc = block_size * round + ty;
        mr = block_size * bx + tx;
        cc = mc;
        cr = block_size * round + tx;
    }else{ //vertical
        mc = block_size * bx + ty;
        mr = block_size * round + tx;
        cc = block_size * round + ty;
        cr = mr;
    }

    int m_cell = mc * width + mr;
    int c_cell = cc * width + cr;
    int s_cell = ty * block_size + tx;


    if(mc >= vert || mr >= vert) s_m[s_cell] = INF;
    else s_m[s_cell] = dist[m_cell];
    
    if(cc >= vert || cr >= vert) s_c[s_cell] = INF;
    else s_c[s_cell] = dist[c_cell];
    

    __syncthreads();
    
    int n, k;
    if(by == 0){
        for(k=0;k<block_size;++k){
            n = s_c[ty*block_size+k] + s_m[k*block_size+tx];
            if(n < s_m[s_cell]){
                s_m[s_cell] = n;
            }
            __syncthreads();
        }

    }else{
        for(k=0;k<block_size;++k){
            n = s_m[ty*block_size+k] + s_c[k*block_size+tx];
            if(n < s_m[s_cell]){
                s_m[s_cell] = n;
            }
            __syncthreads();
        }
    }

    dist[m_cell] = s_m[s_cell];
}

__global__ void phase_three(int32_t* const dist, int block_size, int round, int width, int vert){
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    if(bx >= round)++bx;  //shift x
    if(by >= round)++by;  //shift y

    const int c = block_size * by + ty;
    const int r = block_size * bx + tx;
    const int cell = c*width + r;

    if(c >= vert || r >= vert){  //out of bounds, filled in with INF for each element
        dist[cell] = INF;
    }

    __syncthreads();

    int low = round * block_size;
    int up = low + block_size;
    int n;
    for( ; low<up ; ++low){
        // min(dist[c][r], dist[c][low] + dist[low][r])
        n = dist[ c*width+low ] + dist[ low*width+r ];
        if(n < dist[cell]){
            dist[cell] = n;
        }
        __syncthreads();
    }
}

extern __shared__ int S[];
void block_FW(){

#ifdef _TIME_MEASURE_
    auto start = std::chrono::high_resolution_clock::now();
#endif

    int Round = CEIL(vert, block_size);
    int padded_size = Round * block_size;
    
    size_t vert_w_bytes = vert * sizeof(int);
    size_t padded_w_bytes = padded_size * sizeof(int);

    int32_t *device_ptr;
    //size_t pitch;

    dim3 p2b(Round-1, 2, 1);  //phase 2 block
    dim3 p3b(Round-1, Round-1, 1);  //phase 3 block

    dim3 dimt(block_size, block_size, 1);  //thread

    //cudaMallocPitch(&device_ptr, &pitch, vert_byte, vert_byte, vert);
    cudaMalloc(&device_ptr, padded_w_bytes * padded_size);

    //size_t pitch_int = pitch / sizeof(int);
    //LOG("pitch => %zu bytes (%zu words)", pitch, pitch_int);

    LOG("the number of blocks: %d", Round);

    //dst_ptr, dst_pitch, src, src_pitch, w, h, kind
    cudaMemcpy2D(device_ptr, padded_w_bytes, data, vert_w_bytes, 
                   vert_w_bytes, vert, cudaMemcpyHostToDevice);

    size_t bs2b = block_size * block_size * sizeof(int);
    
    for(int r=0; r < Round; ++r){
        LOG("Round %d/%d", r+1, Round);
        phase_one<<< 1 , dimt , bs2b >>>(device_ptr, block_size, r, padded_size, vert);
        phase_two<<< p2b , dimt , bs2b*2 >>>(device_ptr, block_size, r, padded_size, vert);
        phase_three<<< p3b , dimt >>>(device_ptr, block_size, r, padded_size, vert);
    }

    cudaMemcpy2D(data, vert_w_bytes, device_ptr, padded_w_bytes, 
                    vert_w_bytes, vert, cudaMemcpyDeviceToHost);
    
    cudaFree(device_ptr);

                    
#ifdef _TIME_MEASURE_
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    double elapsed_time = diff.count() * 1000;
    printf("Total time: %f ms (%f GFLOPS)\n", elapsed_time, 2*vert*vert*vert / (elapsed_time * 1e6));
#endif

    
}

int main(int argc, char **argv){

    dump_from_file_and_init(argv[1]);
    block_size = std::atoi(argv[3]);

    block_FW();

    dump_to_file(argv[2]);
    finalize();
    return 0;
}
