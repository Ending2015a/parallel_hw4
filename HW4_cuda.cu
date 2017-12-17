#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>


#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <unistd.h>

#define _DEBUG_
#define _TIME_MEASURE_


#ifdef _DEBUG_
    #include <string>
    #include <sstream>

    int __print_step = 0;
    
    void __pt_log(const char *h_, const char *f_, ...){
        std::stringstream ss;
        ss << d_ << f_ << '\n';
        std::string format = ss.str()

        va_list va;
        va_start(va, f_);
            vprintf(format.c_str(), va);
        va_end(va);
        __print_step++;
    }

    #define VA_ARGS(...) , ##__VA_ARGS__
    #define LOG_GPU(f_, b_, t_, ...) __pt_log( \
                                                "[block %d / thread %d] Step %08d: ", (f_), \
                                                b_, t_, __print_step VA_ARGS(__VA_ARGS__))
    #define LOG(f_, ...) __pt_log(\
                                    "[LOG] Step %08d: ", (f_), \
                                     __print_step VA_ARGS(__VA_ARGS__))

#else
    #define LOG_GPU(f_, b_, t_, ...)
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

    for(int i=0;i<vert;++i){
        Dist[i] = data + i*vert;
    }

    std::fill(Dist, Dist + vert2, INF);
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
    while(--edge >=0){
        ss >> i >> j >> w;
        Dist[i][j] = Dist[j][i] = w;
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

__global__ void cal(const int *dist, int block_size, int Round, 
        int block_start_i, int block_start_j, int block_height, int block_width){
    
    int block_end_i = block_start_i + block_height;
    int block_end_j = block_start_j + block_width;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    int row = 
    int col = 

}

extern __shared__ int S[];
int block_FW(){

#ifdef _TIME_MEASURE_
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif

    size_t vert_byte = vert * sizeof(int);

    int32_t *device_ptr;
    //size_t pitch;

    //cudaMallocPitch(&device_ptr, &pitch, vert_byte, vert_byte, vert);
    cudaMalloc(&device_ptr, vert_bytes);

    //size_t pitch_int = pitch / sizeof(int);
    //LOG("pitch => %zu bytes (%zu words)", pitch, pitch_int);

    int Round = CEIL(vert, block_size);

    LOG("the number of blocks: %d", num_block);

    //dst_ptr, dst_pitch, src, src_pitch, w, h, kind
    //cudaMemcpy2D(device_ptr, pitch, data, vert_byte, 
    //                vert_byte, vert, cudaMemcpyHostToDevice);

    cudaMamcpy(device_ptr, data, vert_bytes*vert, cudaMemcpyHostToDevice);
    
    for(int r=0; r < Round; ++r){
        // phase 1
        // block num, thread num, shared memory size,
        cal<<< 1 , block_size*block_size , block_size*block_size >>>(device_ptr, block_size, r, r, r, 1, 1);
    
        // phase 2
        cal<<< r , block_size*block_size , block_size*block_size >>>(device_ptr, block_size, r, r,   0,   1, r);
        cal<<< 1 , block_size*block_size , block_size*block_size >>>(device_ptr, block_size, r, r,   r+1, 1, Round-r-1);
        cal<<< 1 , block_size*block_size , block_size*block_size >>>(device_ptr, block_size, r, r+1, 0,   r, 1);
        cal<<< 1 , block_size*block_size , block_size*block_size >>>(device_ptr, block_size, r, r+1, r+1, Round-r-1, 1);

        // phase 3
        cal<<< 1 , block_size*block_size , block_size*block_size >>>(device_ptr, block_size, r, 0,   0,   r, r);
        cal<<< 1 , block_size*block_size , block_size*block_size >>>(device_ptr, block_size, r, 0,   r+1, r, Round-r-1);
        cal<<< 1 , block_size*block_size , block_size*block_size >>>(device_ptr, block_size, r, r+1, 0,   Round-r-1, r);
        cal<<< 1 , block_size*block_size , block_size*block_size >>>(device_ptr, block_size, r, r+1, r+1, Round-r-1, 1);

    }

    //cudaMemcpy2D(data, vert_byte, device_ptr, pitch, 
    //                vert_byte, vert, cudaMemcpyDeviceToHost);
    
    cudaMemcpy(data, device_ptr, vert_bytes*vert, cudaMemcpyDeviceToHost);


    cudaFree(device_ptr);

                    
#ifdef _TIME_MEASURE_
    cudaEventRecord(end, 0);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, end);
    elapsed_time /= 1000;
    LOG("Total time: %f sec (%f GFLOPS)", elapsed_time, 2*vert*vert*vert / (elapsed_time * 1e9));
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
