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
        LOG("%-30s %-30s", "Timers", "Elapsed time");
        for(auto it=__t_map.begin(); it!=__t_map.end(); ++it)
            LOG("%-30s %.6lf ms", it->first.c_str(), it->second.total);
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

int **Dist;
int *data;
int block_size;
int vert, edge;
int vert2;

inline void init(){
    vert2 = vert*vert;
    Dist = new int*[vert];
    cudaMallocHost(&data, vert2*sizeof(int));

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

    int item = 0;
    for (size_t i = 0; i < sz; ++i){
        switch (buf[i]){
            case '\n':
                int_list.push_back(item);
                item = 0;
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
    TIC("init/read_file");
    std::ifstream fin(file);
    std::stringstream ss;

    ss << fin.rdbuf();
    ss >> vert >> edge;

    TOC("init/read_file");

    TIC("init/parse_int");

    std::vector<int> int_list;
    int_list.reserve(edge * 3+2);

    init();

    parse_string(ss, int_list);

    TOC("init/parse_int");
    TIC("init/init_mat");

    for(auto e = int_list.begin()+2; e != int_list.end(); e+=3){
        Dist[*e][*(e+1)] = *(e+2);
    }

    fin.close();

    TOC("init/init_mat");
}

void dump_to_file(const char *file){
    FILE *fout = fopen(file, "w");
    fwrite(data, sizeof(int) * vert2, 1, fout);
    fclose(fout);
}

template<int block_size>
__global__ void phase_one(int32_t* const dist, const int round, const int width, const int vert, const int br){

    __shared__ int s[block_size][block_size];

    const int c = br + threadIdx.y;
    const int r = br + threadIdx.x;
    const int cell = c * width + r;

    const bool mb = (c < vert && r < vert);
    s[threadIdx.y][threadIdx.x] = (mb) ? dist[cell] : INF;

    if( !mb ) return;
    
    __syncthreads();
    int mn = s[threadIdx.y][threadIdx.x];
    int n;
    for(int k=0;k<block_size;++k){
        n = s[threadIdx.y][k] + s[k][threadIdx.x];
        if(n < mn){
            s[threadIdx.y][threadIdx.x] = n;
            mn = n;
        }
        __syncthreads();
    }

    dist[cell] = mn;
}

template<int block_size>
__global__ void phase_two(int32_t* const dist, const int round, const int width, const int vert, const int br){

    if(blockIdx.x == round) return;

    __shared__ int s_m[block_size][block_size];
    __shared__ int s_c[block_size][block_size];

    int mc, mr;
    int cc, cr;

    if(blockIdx.y == 0){
        mc = br + threadIdx.y;
        mr = block_size * blockIdx.x + threadIdx.x;
        cc = mc;
        cr = br + threadIdx.x;
    }else{
        mc = block_size * blockIdx.x + threadIdx.y;
        mr = br + threadIdx.x;
        cc = br + threadIdx.y;
        cr = mr;
    }

    const int m_cell = mc * width + mr;
    //const int c_cell = cc * width + cr;

    const bool mb = (mc < vert && mr < vert);
    const bool cb = (cc < vert && cr < vert);

    s_m[threadIdx.y][threadIdx.x] = (mb) ? dist[m_cell] : INF;
    s_c[threadIdx.y][threadIdx.x] = (cb) ? dist[cc * width + cr] : INF;

    if( !mb ) return;

    __syncthreads();

    int mn = s_m[threadIdx.y][threadIdx.x];
    int n;

    if(blockIdx.y == 0){
        for(int k=0;k<block_size;++k){
            n = s_c[threadIdx.y][k] + s_m[k][threadIdx.x];
            if(n < mn){
                s_m[threadIdx.y][threadIdx.x] = n;
                mn = n;
            }
            __syncthreads();
        }
    }else{
        for(int k=0;k<block_size;++k){
            n = s_m[threadIdx.y][k] + s_c[k][threadIdx.x];
            if(n < mn){
                s_m[threadIdx.y][threadIdx.x] = n;
                mn = n;
            }
            __syncthreads();
        }
    }

    dist[m_cell] = mn;
}

template<int block_size>
__global__ void phase_three(int32_t* const dist, const int round, const int width, const int vert, const int br){
    
    if(blockIdx.x == round || blockIdx.y == round) return;

    __shared__ int s_l[block_size][block_size];
    __shared__ int s_r[block_size][block_size];

    const int mc = block_size * blockIdx.y + threadIdx.y;
    const int mr = block_size * blockIdx.x + threadIdx.x;
    const int lr = br + threadIdx.x;
    const int rc = br + threadIdx.y;

    s_l[threadIdx.y][threadIdx.x] = (mc < vert && lr < vert) ? dist[mc * width + lr] : INF;
    s_r[threadIdx.y][threadIdx.x] = (rc < vert && mr < vert) ? dist[rc * width + mr] : INF;

    if( !(mc < vert && mr < vert) ) return;

    const int m_cell = mc * width + mr;
    int mn = dist[m_cell];

    __syncthreads();

    int n;
    for(int k=0;k<block_size;++k){
        n = s_l[threadIdx.y][k] + s_r[k][threadIdx.x];
        if( n < mn) mn = n;
    }

    dist[m_cell] = mn;
}

template<int BLOCK_SIZE> void block_FW(){
    int Round = CEIL(vert, BLOCK_SIZE);
    size_t vert_bytes = vert * sizeof(int);

    int32_t *device_ptr;
    size_t pitch_bytes;

    dim3 p2b(Round, 2, 1);
    dim3 p3b(Round, Round, 1);

    dim3 dimt(BLOCK_SIZE, BLOCK_SIZE, 1);


    cudaMallocPitch(&device_ptr, &pitch_bytes, vert_bytes, vert);

    cudaMemcpy2DAsync(device_ptr, pitch_bytes, data, vert_bytes,
                    vert_bytes, vert, cudaMemcpyHostToDevice);

    int pitch = pitch_bytes / sizeof(int);
    cudaDeviceSynchronize();

    int br = 0;
    for(int r=0;r<Round;++r){
        phase_one< BLOCK_SIZE ><<< 1 , dimt >>>(device_ptr, r, pitch, vert, br);
        phase_two< BLOCK_SIZE ><<< p2b , dimt >>>(device_ptr, r, pitch, vert, br);
        phase_three< BLOCK_SIZE ><<< p3b , dimt >>>(device_ptr, r, pitch, vert, br);
        br += BLOCK_SIZE;
    }

    cudaMemcpy2D(data, vert_bytes, device_ptr, pitch_bytes, vert_bytes, vert, cudaMemcpyDeviceToHost);

    cudaFree(device_ptr);
}

int main(int argc, char **argv){

    TIC("init");
    dump_from_file_and_init(argv[1]);

    TOC("init");


    TIC("block");

    block_size = std::atoi(argv[3]);
    switch(block_size){
        case 8:
            block_FW<8>();
            break;
        case 16:
            block_FW<16>();
            break;
        case 24:
            block_FW<24>();
            break;
        case 32:
            block_FW<32>();
            break;
    }

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






