#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <vector>

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
                throw std::string("the timer has already started");
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
                throw std::string("the timer is inactive");
            t.state = restart;
            std::chrono::duration<double> d = end-t.start;
            t.total += d.count() * PRECISION;
            t.start = end;
        }catch(std::string msg){
            msg += std::string(": %s");
            LOG(msg.c_str(), tag.c_str());
        }
    }

    inline void __log_all(){
        LOG("%-15s %-15s", "Timers", "Elapsed time");
        for(auto it=__t_map.begin(); it!=__t_map.end(); ++it)
            LOG("%-15s %.6lf ms", it->first.c_str(), it->second.total);
    }

    #define TIC(tag, ...) __ms_tic((tag))
    #define TOC(tag, ...) __ms_toc((tag))
    #define GET(tag) __t_map[tag].total;
    #define _LOG_ALL() __log_all()
#else
    #define TIC(tag, ...)
    #define TOC(tag, ...)
    #define GET(tag) 0
    #define _LOG_ALL()
#endif



#define CEIL(a, b) ((a) + (b) -1)/(b)
#define INF 1000000000
#define BLOCK_SIZE block_size


int **Dist;
int *data;
int block_size;
int vert, edge;
int vert2;

inline void init(){
    vert2 = vert*vert;
    Dist = new int*[vert];
    cudaHostAlloc((void**)&data, vert2*sizeof(int), cudaHostAllocDefault);

    std::fill(data, data + vert2, INF);

    for(int i=0;i<vert;++i){
        Dist[i] = data + i*vert;
        Dist[i][i] = 0;
    }
}

inline void finalize(){
    delete[] Dist;
    cudaFree(data);
}

void parse_string(std::stringstream &ss, std::vector<int> &int_list){

    std::string str = ss.str();
    const char *buf = str.c_str();
    size_t sz = str.size();

    int lc = 0;
    int item = 0;
    for (size_t i = 0; i < sz; ++i){
        switch (buf[i]){
            case '\n':
                int_list.push_back(item);
                item = 0; lc++;
                break;
            case ' ':
                int_list.push_back(item);
                item = 0;
                break;
            default:
                item = 10*item + buf[i] - '0';
                break;
        }    
    }
}

void dump_from_file_and_init(const char *file){
    TIC("read_file");
    std::ifstream fin(file);
    std::stringstream ss;

    ss << fin.rdbuf();
    ss >> vert >> edge;

    TOC("read_file");

    TIC("parse_int");

    std::vector<int> int_list;
    int_list.reserve(edge * 3+2);

    init();

    parse_string(ss, int_list);

    TOC("parse_int");
    TIC("init_mat");

    for(auto e = int_list.begin()+2; e != int_list.end(); e+=3){
        Dist[*e][*(e+1)] = *(e+2);
    }

    fin.close();

    TOC("init_mat");
}

void dump_to_file(const char *file){
    FILE *fout = fopen(file, "w");
    fwrite(data, sizeof(int), vert2, fout);
    fclose(fout);
}

__global__ void init_gpu(int reps){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= reps) return;
}


template<int block_size>
__global__ void phase_one(int32_t* const dist, int round, int width, int vert){
    __shared__ int s[block_size][block_size];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int br = block_size * round;
    const int c = br + ty;
    const int r = br + tx;
    const int cell = c*width + r;

    s[ty][tx] = dist[cell];
    if(c >= vert || r >= vert)
        s[ty][tx] = INF;
    
    __syncthreads();
    int mn = s[ty][tx];
    int n, k;
    #pragma unroll
    for(k=0;k<block_size;++k){
        n = s[ty][k] + s[k][tx];
        if(n < mn){
            s[ty][tx] = n;
            mn = n;
        }
        __syncthreads();
    }

    dist[cell] = mn;
}

template<int block_size>
__global__ void phase_two(int32_t* const dist, int round, int width, int vert){
    __shared__ int s_m[block_size][block_size];
    __shared__ int s_c[block_size][block_size];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int mc, mr;
    int cc, cr;
    const int br = block_size * round;

    if(bx >= round) ++bx;

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

    int m_cell = mc * width + mr;
    int c_cell = cc * width + cr;

    s_m[ty][tx] = dist[m_cell];
    s_c[ty][tx] = dist[c_cell];

    if(mc >= vert || mr >= vert)
        s_m[ty][tx] = INF;
    if(cc >= vert || cr >= vert)
        s_c[ty][tx] = INF;

    __syncthreads();

    int mn = s_m[ty][tx];
    int n, k;

    if(by == 0){
        #pragma unroll
        for(k=0;k<block_size;++k){
            n = s_c[ty][k] + s_m[k][tx];
            if(n < mn){
                s_m[ty][tx] = n;
                mn = n;
            }
            __syncthreads();
        }
    }else{
        #pragma unroll
        for(k=0;k<block_size;++k){
            n = s_m[ty][k] + s_c[k][tx];
            if(n < mn){
                s_m[ty][tx] = n;
                mn = n;
            }
            __syncthreads();
        }
    }

    dist[m_cell] = mn;
}

template<int block_size>
__global__ void phase_three(int32_t* const dist, int round, int width, int vert){
    __shared__ int s_m[block_size][block_size];
    __shared__ int s_l[block_size][block_size];
    __shared__ int s_r[block_size][block_size];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    if(bx >= round)++bx;
    if(by >= round)++by;

    const int br = block_size * round;

    const int mc = block_size * by + ty;
    const int mr = block_size * bx + tx;
    const int lc = mc;
    const int lr = br + tx;
    const int rc = br + ty;
    const int rr = mr;

    const int m_cell = mc * width + mr;
    const int l_cell = lc * width + lr;
    const int r_cell = rc * width + rr;
    
    s_m[ty][tx] = dist[m_cell];
    s_l[ty][tx] = dist[l_cell];
    s_r[ty][tx] = dist[r_cell];

    if(mc >= vert || mr >= vert)
        s_m[ty][tx] = INF;
    if(lc >= vert || lr >= vert)
        s_l[ty][tx] = INF;
    if(rc >= vert || rr >= vert)
        s_r[ty][tx] = INF;

    __syncthreads();

    int mn = s_m[ty][tx];
    int n, k;
    #pragma unroll
    for(k=0;k<block_size;++k){
        n = s_l[ty][k] + s_r[k][tx];
        if(n < mn){
            s_m[ty][tx] = n;
            mn = n;
        }
        __syncthreads();
    }

    dist[m_cell] = mn;
}

__global__ void phase_one(int32_t* const dist, int block_size, int round, int width, int vert){
    extern __shared__ int s[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int br = BLOCK_SIZE * round;
    const int c = br + ty;
    const int r = br + tx;

    const int ty_block = ty * BLOCK_SIZE;

    const int cell = c * width + r;
    const int s_cell = ty_block + tx;

    s[s_cell] = dist[cell];

    if(c >= vert || r >= vert)
        s[s_cell] = INF;

    __syncthreads();


    int n, k;
    for(k=0;k<BLOCK_SIZE;++k){
        n = s[ty_block + k] + s[k*BLOCK_SIZE + tx];
        if(n < s[s_cell])
            s[s_cell] = n;
        __syncthreads();
    }

    dist[cell] = s[s_cell];
}

__global__ void phase_two(int32_t* const dist, int block_size, int round, int width, int vert){
    extern __shared__ int s[];
    int *const s_m = s;
    int *const s_c = s + BLOCK_SIZE * BLOCK_SIZE;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int mc, mr;
    int cc, cr;
    const int br = BLOCK_SIZE * round;

    if(bx >= round)++bx;

    if(by == 0){
        mc = br + ty;
        mr = BLOCK_SIZE * bx + tx;
        cc = mc;
        cr = br + tx;
    }else{
        mc = BLOCK_SIZE * bx + ty;
        mr = br + tx;
        cc = br + ty;
        cr = mr;
    }

    const int ty_block = ty * BLOCK_SIZE;
    
    int m_cell = mc * width + mr;
    int c_cell = cc * width + cr;
    int s_cell = ty_block + tx;

    s_m[s_cell] = dist[m_cell];
    s_c[s_cell] = dist[c_cell];

    if(mc >= vert || mr >= vert)
        s_m[s_cell] = INF;
    if(cc >= vert || cr >= vert)
        s_c[s_cell] = INF;

    __syncthreads();
    

    int n, k;
    if(by == 0){
        for(k=0;k<BLOCK_SIZE;++k){
            n = s_c[ty_block + k] + s_m[k*BLOCK_SIZE + tx];
            if(n < s_m[s_cell])
                s_m[s_cell] = n;
            __syncthreads();
        }
    }else{
        for(k=0;k<BLOCK_SIZE;++k){
            n = s_m[ty_block + k] + s_c[k*BLOCK_SIZE + tx];
            if(n < s_m[s_cell])
                s_m[s_cell] = n;
            __syncthreads();
        }
    }

    dist[m_cell] = s_m[s_cell];
}

__global__ void phase_three(int32_t* const dist, int block_size, int round, int width, int vert){

    const int bs2 = BLOCK_SIZE*BLOCK_SIZE;

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

    const int br = BLOCK_SIZE * round;
    const int ty_block = ty * BLOCK_SIZE;

    const int mc = BLOCK_SIZE * by + ty;
    const int mr = BLOCK_SIZE * bx + tx;
    const int lc = mc;
    const int lr = br + tx;
    const int rc = br + ty;
    const int rr = mr;
    const int m_cell = mc * width + mr;
    const int l_cell = lc * width + lr;
    const int r_cell = rc * width + rr;
    const int s_cell = ty_block + tx;

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

    for(k=0;k<BLOCK_SIZE;++k){
        n = s_l[ty_block + k] + s_r[k*BLOCK_SIZE + tx];
        if(n < s_m[s_cell])
            s_m[s_cell] = n;
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

    switch(block_size){
        case 16:
            for(int r=0;r<Round;++r){
                phase_one<16><<< 1 , dimt >>>(device_ptr, r, pitch, vert);
                phase_two<16><<< p2b , dimt >>>(device_ptr, r, pitch, vert);
                phase_three<16><<< p3b , dimt >>>(device_ptr, r, pitch, vert);
            }break;
        case 32:
            for(int r=0;r<Round;++r){
                phase_one<32><<< 1 , dimt >>>(device_ptr, r, pitch, vert);
                phase_two<32><<< p2b , dimt >>>(device_ptr, r, pitch, vert);
                phase_three<32><<< p3b , dimt >>>(device_ptr, r, pitch, vert);
            }break;
        default:

            for(int r=0;r<Round;++r){
                phase_one<<< 1 , dimt , bs2b3 >>>(device_ptr, block_size, r, pitch, vert);
                phase_two<<< p2b , dimt , bs2b3 >>>(device_ptr, block_size, r, pitch, vert);
                phase_three<<< p3b , dimt , bs2b3 >>>(device_ptr, block_size, r, pitch, vert);
            }break;
    }

    cudaDeviceSynchronize();

    cudaMemcpy2D(data, vert_bytes, device_ptr, pitch_bytes, vert_bytes, vert, cudaMemcpyDeviceToHost);

    cudaFree(device_ptr);
    cudaStreamDestroy(stream);

}

int main(int argc, char **argv){

    TIC("init");
    dump_from_file_and_init(argv[1]);

    TOC("init");


    TIC("block");

    block_size = std::atoi(argv[3]);
    block_FW();

    TOC("block");



    TIC("write_file");

    dump_to_file(argv[2]);

    TOC("write_file");


    TIC("finalize");

    finalize();

    TOC("finalize");

    _LOG_ALL();
    return 0;
}






