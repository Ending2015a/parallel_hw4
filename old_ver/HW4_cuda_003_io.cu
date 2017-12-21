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


#ifdef _TIME_MEASURE_

    #define PRECISION 1000

    #include <chrono>
    #include <map>

    using hr_clock = std::chrono::high_resolution_clock;

    struct __timer{
        bool state;
        double total;
        std::chrono::time_point<hr_clock> start;
        __timer() : state(false), total(0){}
    };

    std::map<std::string, struct __timer> __t_map;
    inline void __ms_tic(std::string tag, bool cover=true){
        try{
            __timer &t = __t_map[tag];
            if(!cover && t.state) 
                throw "the timer has already started";
            t.state = true;
            t.start = std::chrono::high_resolution_clock::now();
        }catch(std::string msg){
            msg += std::string(": %s");
            LOG(msg.c_str(), tag.c_str());
        }
    }

    inline void __ms_toc(std::string tag, bool restart=false){
        auto end = std::chrono::high_resolution_clock::now();
        try{
            __timer &t = __t_map[tag];
            if(!t.state)
                throw "the timer is inactive";
            t.state = restart;
            t.total = (end-t.start).count() * PRECISION;
            t.start = end;
        }catch(std::string msg){
            msg += std::string(": %s");
            LOG(msg.c_str(), tag.c_str());
        }
    }

    inline void __log_all(){
        LOG("%-15s %-15s", "Timers", "Elapsed time");
        for(auto it=__t_map.begin(); it!=__t_map.end(); ++it)
            LOG("%15s %.5f", it->first.c_str(), it->second.total);
    }

    #ifndef VA_ARGS
        #define VA_ARGS(...) , ##__VA_ARGS__
    #endif

    #define TIC(tag, ...) __ms_tic((tag) )
    #define TOC(tag, ...)
    #define GET(tag) __t_map[tag].total;
    #define _LOG_ALL() __log_all()
#else
    #define TIC(tag, ...)
    #define TOC(tag, ...)
    #define GET(tag) 0
    #define _LOG_ALL()
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

    TIC("io");
    ss << fin.rdbuf();
    ss >> vert >> edge;
    TOC("io");

    //LOG("vert: %d, edge: %d", vert, edge);

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

    int ty_block = ty * block_size;
    int k_block = 0;

    const int cell = c * width + r;
    const int s_cell = ty_block + tx;

    if(c < vert && r < vert){
        s[s_cell] = dist[cell];
    }else{
        s[s_cell] = INF;
    }

    int n, k;
    for(k=0;k<block_size;++k){

        __syncthreads();

        // min(dist[ty][tx], dist[ty][i] + dist[i][tx])
        n = s[ty_block + k] + s[k_block + tx];
        if(n < s[s_cell]){
            s[s_cell] = n;
        }
        k_block += block_size;
    }

    if(c < vert && r < vert){
        dist[cell] = s[s_cell];
    }
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
    int br = block_size*round;

    if(bx >= round)++bx; //shift

    if(by == 0){  //horizontal
        mc = br + ty;
        mr = block_size * bx + tx;
        cc = mc;
        cr = br + tx;
    }else{ //vertical
        mc = block_size * bx + ty;
        mr = br + tx;
        cc = br + ty;
        cr = mr;
    }

    int ty_block = ty*block_size;
    int k_block = 0;

    int m_cell = mc * width + mr;
    int c_cell = cc * width + cr;
    int s_cell = ty_block + tx;

    if(mc < vert && mr < vert)
        s_m[s_cell] = dist[m_cell];
    else
        s_m[s_cell] = INF;

    if(cc < vert && cr < vert)
        s_c[s_cell] = dist[c_cell];
    else
        s_c[s_cell] = INF;
    

    int n, k;
    if(by == 0){
        for(k=0;k<block_size;++k){
            __syncthreads();
            n = s_c[ty_block+ k] + s_m[k_block + tx];
            if(n < s_m[s_cell]){
                s_m[s_cell] = n;
            }
            k_block += block_size;
        }

    }else{
        for(k=0;k<block_size;++k){
            __syncthreads();
            n = s_m[ty_block+k] + s_c[k_block+tx];
            if(n < s_m[s_cell]){
                s_m[s_cell] = n;
            }
            k_block += block_size;
        }
    }

    if(mc < vert && mr < vert)
        dist[m_cell] = s_m[s_cell];
}

__global__ void phase_three(int32_t* const dist, int block_size, int round, int width, int vert){
    
    int bs2 = block_size*block_size;

    extern __shared__ int s3[];
    int* const s_m = s3;
    int* const s_l = s3 + bs2;
    int* const s_r = s_l + bs2;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    if(bx >= round)++bx;  //shift x
    if(by >= round)++by;  //shift y

    int br = block_size * round;
    int ty_block = ty*block_size;
    int k_block = 0;

    const int mc = block_size * by + ty;
    const int mr = block_size * bx + tx;
    const int lc = mc;
    const int lr = br + tx;
    const int rc = br + ty;
    const int rr = mr;
    const int m_cell = mc*width + mr; //dist
    const int l_cell = lc*width + lr; //dist
    const int r_cell = rc*width + rr; //dist
    const int s_cell = ty_block + tx; //shared

    if(mc >= vert || mr >= vert) s_m[s_cell] = INF;
    else s_m[s_cell] = dist[m_cell];

    if(lc >= vert || lr >= vert) s_l[s_cell] = INF;
    else s_l[s_cell] = dist[l_cell];

    if(rc >= vert || rr >= vert) s_r[s_cell] = INF;
    else s_r[s_cell] = dist[r_cell];


    int n, k;
    for(k=0;k<block_size;++k){
        __syncthreads();
        n = s_l[ty_block + k] + s_r[k_block + tx];
        if(n<s_m[s_cell]){
            s_m[s_cell] = n;
        }
        k_block += block_size;
    }

    if(mc < vert && mr < vert)
        dist[m_cell] = s_m[s_cell];
}


void block_FW(){

#ifdef _TIME_MEASURE_
    auto start = std::chrono::high_resolution_clock::now();
#endif

    TIC("total");

    cudaStream_t init_stream;
    cudaStreamCreate(&init_stream);

    int Round = CEIL(vert, block_size);
    
    size_t vert_w_bytes = vert * sizeof(int);

    int32_t *device_ptr;
    size_t pitch;

    dim3 p2b(Round-1, 2, 1);  //phase 2 block
    dim3 p3b(Round-1, Round-1, 1);  //phase 3 block

    dim3 dimt(block_size, block_size, 1);  //thread


    TIC("cuda_init");
    cudaSetDevice(0);

    cudaMallocPitch(&device_ptr, &pitch, vert_w_bytes, vert);
    int pitch_int = pitch >> 2;

    init_gpu<<< 1, dimt >>>(32);

    //dst_ptr, dst_pitch, src, src_pitch, w, h, kind
    cudaMemcpy2DAsync(device_ptr, pitch, data, vert_w_bytes, 
                   vert_w_bytes, vert, cudaMemcpyHostToDevice, init_stream);

    c

    size_t bs2b3 = block_size * block_size * sizeof(int) * 3;

    cudaDeviceSynchronize();

    TOC("cuda_init");

    TIC("cuda_calc");

    for(int r=0; r < Round; ++r){
        //LOG("Round %d/%d", r+1, Round);
        phase_one<<< 1 , dimt , bs2b3 >>>(device_ptr, block_size, r, pitch_int, vert);
        phase_two<<< p2b , dimt , bs2b3 >>>(device_ptr, block_size, r, pitch_int, vert);
        phase_three<<< p3b , dimt , bs2b3 >>>(device_ptr, block_size, r, pitch_int, vert);
    }

    TOC("cuda_calc");

    TIC("cuda_fin");
    cudaMemcpy2D(data, vert_w_bytes, device_ptr, pitch, 
                    vert_w_bytes, vert, cudaMemcpyDeviceToHost);
    
    TOC("cuda_fin");

    cudaFree(device_ptr);
    cudaStreamDestroy(init_stream);

#ifdef _TIME_MEASURE_
    TOC("total");
    double elapsed_time = GET("total");
    printf("Total time: %f ms (%f GFLOPS)\n", elapsed_time, 2*vert*vert*vert / (elapsed_time * 1e6));
#endif

}

int main(int argc, char **argv){

    dump_from_file_and_init(argv[1]);
    block_size = std::atoi(argv[3]);

    block_FW();

    dump_to_file(argv[2]);
    finalize();

    _LOG_ALL();
    return 0;
}
