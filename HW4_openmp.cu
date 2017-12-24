#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include <cuda.h>
#include <omp.h>
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


#define MAX(a, b) ((a)>(b)?(a):(b))
#define MIN(a, b) ((a)<(b)?(a):(b))
#define CEIL(a, b) ((a) + (b) -1)/(b)
#define INF 1000000000
#define MAX_BLOCK_SIZE 32


int **Dist;
int *data;
int block_size;
int vert, edge;
int vert2;

inline void init(){
    vert2 = vert*vert;
    Dist = new int*[vert];
    //data = new int[vert2*sizeof(int)];
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
    //delete[] data;
}

//end_list = pointer to the last element in the int_list
void parse_string(std::stringstream &ss, int *int_list, int *end_list){

    std::string str = ss.str();
    char *buf = (char*)str.c_str();
    size_t sz = str.size();
    char *end = buf+sz;
    char *mid = buf + (sz/2);

    while( mid < end && *mid != ' ' && *mid != '\n' )
        ++mid;

    ++mid;
    int* mid_list = end_list;
#pragma omp parallel num_threads(2) shared(int_list, end_list, mid_list, buf, end, str, mid)
    {
        int num = omp_get_thread_num();
        int item = 0;
        if(num){ //num = 1
            for(char* m = mid; m < end; ++m){
                switch(*m){
                    case '\n':
                    case ' ':
                        *mid_list = item;
                        --mid_list;
                        item = 0;
                        break;
                    default:
                        item = 10*item + (*m - '0');
                        break;
                }
            }
        }else{
            for (; buf < mid; ++buf){
                switch (*buf){
                    case '\n':
                    case ' ':
                        *int_list=item;
                        ++int_list;
                        item = 0;
                        break;
                    default:
                        item = 10*item + (*buf - '0');
                        break;
                }
            }
        }
    }
//end parallel

    ++mid_list;
    while(mid_list < end_list){
        (*mid_list) ^= (*end_list) ^= (*mid_list) ^= (*end_list);
        ++mid_list;
        --end_list;
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
    int sz = edge*3+2;
    int *int_list = new int[sz];
    init();

    parse_string(ss, int_list, int_list+sz-1);

    TOC("init/parse_int");
    TIC("init/init_mat");

    int *end = int_list + sz;
    for(int* e = int_list+2; e < end ; e+=3){
        Dist[*e][*(e+1)] = *(e+2);
    }

    fin.close();

    delete[] int_list;
    TOC("init/init_mat");
}

void dump_to_file(const char *file){
    FILE *fout = fopen(file, "w");
    fwrite(data, sizeof(int) * vert2, 1, fout);
    fclose(fout);
}


void print_out(std::string msg){
    LOG("%s", msg.c_str());
    std::stringstream ss;
    ss << "\n";
    for(int i=0;i<vert;++i){
        for(int j=0;j<vert;++j){
            ss << " " << Dist[i][j];
        }
        ss << "\n";
    }

    LOG("%s", ss.str().c_str());
}


template<int block_size>
__global__ void phase_one(int32_t* const dist, const int width, const int pivot, const int bound){

    __shared__ int s[block_size][block_size];

    const int c = pivot + threadIdx.y;
    const int r = pivot + threadIdx.x;
    const int cell = c * width + r;

    const bool mb = (c < bound && r < bound);
    s[threadIdx.y][threadIdx.x] = (mb) ? dist[cell] : INF;

    if( !mb ) return;

    int o = s[threadIdx.y][threadIdx.x];
    int n;
    for(int k=0;k<block_size;++k){

        __syncthreads();

        n = s[threadIdx.y][k] + s[k][threadIdx.x];
        if(n < s[threadIdx.y][threadIdx.x]){
            s[threadIdx.y][threadIdx.x] = n;
        }
    }

    if( s[threadIdx.y][threadIdx.x] < o)
        dist[cell] = s[threadIdx.y][threadIdx.x];
}


template<int block_size>
__global__ void phase_two(int32_t* const dist, const int width, const int pivot,
                        const int start, const int bound, const int skip){

    if(blockIdx.x == skip)return;

    __shared__ int s_m[block_size][block_size];
    __shared__ int s_c[block_size][block_size];

    int mc, mr;
    int cc, cr;

    /*     +---+
     *     |   |
     * +---+---+---+---+
     * |   | P |   |   |  y=0
     * +---+---+---+---+
     *     |   |
     *     +---+
     *     |   |
     *     +---+  y=1
     * skip [p]
     */

    if(blockIdx.y == 0){  //horizontal
        mc = pivot + threadIdx.y;
        mr = start + block_size * blockIdx.x + threadIdx.x;
        cc = mc;
        cr = pivot + threadIdx.x;
    }else{
        mc = start + block_size * blockIdx.x + threadIdx.y;
        mr = pivot + threadIdx.x;
        cc = pivot + threadIdx.y;
        cr = mr;
    }

    const int m_cell = mc * width + mr;
    s_m[threadIdx.y][threadIdx.x] = (mc < bound && mr < bound) ? dist[m_cell] : INF;
    s_c[threadIdx.y][threadIdx.x] = (cc < bound && cr < bound) ? dist[cc * width + cr] : INF;
    
    if( mc >= bound || mr >= bound ) return;
    
    int o = s_m[threadIdx.y][threadIdx.x];
    int n;
    if(blockIdx.y == 0){
        for(int k=0;k<block_size;++k){
            __syncthreads();
            n = s_c[threadIdx.y][k] + s_m[k][threadIdx.x];
            if(n < s_m[threadIdx.y][threadIdx.x])
                s_m[threadIdx.y][threadIdx.x] = n;
        }
    }else{
        for(int k=0;k<block_size;++k){
            __syncthreads();
            n = s_m[threadIdx.y][k] + s_c[k][threadIdx.x];
            if(n < s_m[threadIdx.y][threadIdx.x])
                s_m[threadIdx.y][threadIdx.x] = n;
        }
    }

    if(s_m[threadIdx.y][threadIdx.x] < o)
        dist[m_cell] = s_m[threadIdx.y][threadIdx.x];
}

template<int block_size>
__global__ void phase_three(int32_t* const dist, const int width, const int pivot, 
                        const int start, const int bound, const int bound_p, const int skip){

    if(blockIdx.x == skip || blockIdx.y == skip) return;

    __shared__ int s_l[block_size][block_size];
    __shared__ int s_r[block_size][block_size];

    const int mc = start + block_size * blockIdx.y + threadIdx.y;
    const int mr = start + block_size * blockIdx.x + threadIdx.x;
    const int lr = pivot + threadIdx.x;
    const int rc = pivot + threadIdx.y;

    s_l[threadIdx.y][threadIdx.x] = (mc < bound && lr < bound_p) ? dist[mc * width + lr] : INF;
    s_r[threadIdx.y][threadIdx.x] = (rc < bound_p && mr < bound) ? dist[rc * width + mr] : INF;

    if( mc >= bound || mr >= bound ) return;

    const int m_cell = mc * width + mr;
    __syncthreads();

    int o = dist[m_cell];
    int n;
    int mn = s_l[threadIdx.y][0] + s_r[0][threadIdx.x];
    for(int k=1;k<block_size;++k){
        n = s_l[threadIdx.y][k] + s_r[k][threadIdx.x];
        if( n < mn ) mn = n;
    }

    if( mn < o )
        dist[m_cell] = mn;
}

template<int block_size>
__global__ void phase_two_v(int32_t* const dist, const int width, const int pivot,
        const int start, const int bound_y, const int bound_x, const int bound){
    
    __shared__ int s_m[block_size][block_size];
    __shared__ int s_c[block_size][block_size];

    const int mc = start + block_size * blockIdx.x + threadIdx.y;
    const int mr = pivot + threadIdx.x;
    const int cc = pivot + threadIdx.y;
    const int cr = mr;

    const int m_cell = mc * width + mr;

    const bool mb = (mc < bound_y && mr < bound_x);
    const bool cb = (cc < bound && cr < bound);

    s_m[threadIdx.y][threadIdx.x] = (mb) ? dist[m_cell] : INF;
    s_c[threadIdx.y][threadIdx.x] = (cb) ? dist[cc * width + cr] : INF;

    if( !mb ) return;

    int o = s_m[threadIdx.y][threadIdx.x];
    int n;

    for(int k=0;k<block_size; ++k){
        __syncthreads();
        n = s_m[threadIdx.y][k] + s_c[k][threadIdx.x];
        if(n < s_m[threadIdx.y][threadIdx.x]){
            s_m[threadIdx.y][threadIdx.x] = n;
        }
    }

    if(s_m[threadIdx.y][threadIdx.x] < o)
        dist[m_cell] = s_m[threadIdx.y][threadIdx.x];
}

template<int block_size>
__global__ void phase_two_h(int32_t* const dist, const int width, const int pivot,
        const int start, const int bound_y, const int bound_x, const int bound){

    __shared__ int s_m[block_size][block_size];
    __shared__ int s_c[block_size][block_size];

    const int mc = pivot + threadIdx.y;
    const int mr = start + block_size * blockIdx.x + threadIdx.x;
    const int cc = mc;
    const int cr = pivot + threadIdx.x;

    const int m_cell = mc * width + mr;

    const bool mb = (mc < bound_y && mr < bound_x);
    const bool cb = (cc < bound && cr < bound);

    s_m[threadIdx.y][threadIdx.x] = (mb) ? dist[m_cell] : INF;
    s_c[threadIdx.y][threadIdx.x] = (cb) ? dist[cc * width + cr] : INF;

    if ( !mb ) return;

    int o = s_m[threadIdx.y][threadIdx.x];
    int n;
    for(int k=0;k<block_size;++k){
        __syncthreads();
        n = s_c[threadIdx.y][k] + s_m[k][threadIdx.x];
        if(n < s_m[threadIdx.y][threadIdx.x]){
            s_m[threadIdx.y][threadIdx.x] = n;
        }
    }

    if(s_m[threadIdx.y][threadIdx.x] < o)
        dist[m_cell] = s_m[threadIdx.y][threadIdx.x];
}

template<int block_size>
__global__ void phase_three_h(int32_t* const dist, const int width, const int pivot, 
                        const int start_y, const int start_x, const int bound_y, const int bound_x,
                        const int bound, const int skip){

    if(blockIdx.y == skip) return;

    __shared__ int s_l[block_size][block_size];
    __shared__ int s_r[block_size][block_size];

    const int mc = start_y + block_size * blockIdx.y + threadIdx.y;
    const int mr = start_x + block_size * blockIdx.x + threadIdx.x;
    const int lr = pivot + threadIdx.x;
    const int rc = pivot + threadIdx.y;

    s_l[threadIdx.y][threadIdx.x] = (mc < bound_y && lr < bound) ? dist[mc * width + lr] : INF;
    s_r[threadIdx.y][threadIdx.x] = (rc < bound && mr < bound_x) ? dist[rc * width + mr] : INF;

    if( mc >= bound_y || mr >= bound_x ) return;

    const int m_cell = mc * width + mr;
    __syncthreads();

    int o = dist[m_cell];
    int n;
    int mn = s_l[threadIdx.y][0] + s_r[0][threadIdx.x];
    for(int k=1;k<block_size;++k){
        n = s_l[threadIdx.y][k] + s_r[k][threadIdx.x];
        if( n < mn ) mn = n;
    }

    if( mn < o )
        dist[m_cell] = mn;
}

template<int block_size>
__global__ void phase_three_v(int32_t* const dist, const int width, const int pivot, 
                        const int start_y, const int start_x, const int bound_y, const int bound_x,
                        const int bound, const int skip){

    if(blockIdx.x == skip) return;

    __shared__ int s_l[block_size][block_size];
    __shared__ int s_r[block_size][block_size];

    const int mc = start_y + block_size * blockIdx.y + threadIdx.y;
    const int mr = start_x + block_size * blockIdx.x + threadIdx.x;
    const int lr = pivot + threadIdx.x;
    const int rc = pivot + threadIdx.y;

    s_l[threadIdx.y][threadIdx.x] = (mc < bound_y && lr < bound) ? dist[mc * width + lr] : INF;
    s_r[threadIdx.y][threadIdx.x] = (rc < bound && mr < bound_x) ? dist[rc * width + mr] : INF;

    if( mc >= bound_y || mr >= bound_x ) return;

    const int m_cell = mc * width + mr;
    __syncthreads();

    int o = dist[m_cell];
    int n;
    int mn = s_l[threadIdx.y][0] + s_r[0][threadIdx.x];
    for(int k=1;k<block_size;++k){
        n = s_l[threadIdx.y][k] + s_r[k][threadIdx.x];
        if( n < mn ) mn = n;
    }

    if( mn < o )
        dist[m_cell] = mn;
}



/*
template<int BLOCK_SIZE> 
void block_FW(){
    int Round = CEIL(vert, BLOCK_SIZE);
    size_t vert_bytes = vert * sizeof(int);

    int32_t *device_ptr;
    size_t pitch_bytes;

    dim3 p2b(Round, 2, 1);
    dim3 p3b(Round, Round, 1);

    dim3 dimt(BLOCK_SIZE, BLOCK_SIZE, 1);


    cudaMallocPitch(&device_ptr, &pitch_bytes, vert_bytes, vert);

    cudaMemcpy2D(device_ptr, pitch_bytes, data, vert_bytes,
                    vert_bytes, vert, cudaMemcpyHostToDevice);

    int pitch = pitch_bytes / sizeof(int);
    //cudaDeviceSynchronize();

    int br = 0;
    for(int r=0;r<Round;++r){
        phase_one< BLOCK_SIZE ><<< 1 , dimt >>>(device_ptr, r, pitch, vert, br);
        phase_two< BLOCK_SIZE ><<< p2b , dimt >>>(device_ptr, r, pitch, vert, br);
        phase_three< BLOCK_SIZE ><<< p3b , dimt >>>(device_ptr, r, pitch, vert, br);
        br += BLOCK_SIZE;
    }

    cudaDeviceSynchronize();
    cudaMemcpy2D(data, vert_bytes, device_ptr, pitch_bytes, vert_bytes, vert, cudaMemcpyDeviceToHost);

    cudaFree(device_ptr);
}*/

template<int BLOCK_SIZE>
void block_FW(){
    int sp[2];
    sp[0] = CEIL(vert, 2);  //lower
    sp[1] = vert - sp[0];  //upper

    size_t vert_bytes = vert * sizeof(int);

    dim3 dimt(BLOCK_SIZE, BLOCK_SIZE, 1);

    int32_t *device_ptr[2];
    size_t pitch_bytes[2];
    int pitch[2];

    cudaStream_t stream[2];

    cudaEvent_t sync_1[2];
    cudaEvent_t sync_2[2];

#pragma omp parallel num_threads(2) shared(device_ptr, pitch_bytes, pitch, sp, stream)
    {
        int num = omp_get_thread_num();

#ifdef _DEBUG_
        cudaSetDevice(0);
#else
        cudaSetDevice(num);
#endif
        cudaStreamCreate(&stream[num]);
        cudaMallocPitch(device_ptr + num, pitch_bytes + num, vert_bytes, vert);
        cudaMemcpy2DAsync(device_ptr[num], pitch_bytes[num], data, vert_bytes, vert_bytes, vert, cudaMemcpyHostToDevice, stream[num]);
        pitch[num] = pitch_bytes[num] / sizeof(int);

        //
        //      sp0  sp1
        // *  +---+-----+
        // *  |G11| G12 |  sp0
        // *  +---+-----+
        // *  |   |     |  sp1
        // *  |G21| G22 |
        // *  +---+-----+
        // *
        // *   Event  |   Round   |    Boundary    |     grid     |   phase
        // *----------+-----------+----------------+--------------+----------
        // *    G11   |  Round_0  |   [0, sp[0]) s |  sp[0]xsp[0] | 1, 2, 3
        // *    G22   |  Round_1  | [sp[0], vert)s |  sp[1]xsp[1] | 1, 2, 3
        // * G11->G12 |  Round_0  | [sp[0], vert)h |  sp[0]xsp[1] | 2p, 3p
        // * G22->G21 |  Round_1  |   [0, sp[0]) h |  sp[1]xsp[0] | 2p, 3p
        // * G11->G21 |  Round_0  | [sp[0], vert)v |  sp[1]xsp[0] | 2p, 3p
        // * G22->G12 |  Round_1  |   [0, sp[0]) v |  sp[0]xsp[1] | 2p, 3p
        // *   ->G11  |  Round_0  |   [0, 
        // *

        int Round_0 = CEIL(sp[0], BLOCK_SIZE);
        int Round_1 = CEIL(sp[1], BLOCK_SIZE);

        int Round = (num==0) ? Round_0:Round_1;

        dim3 p2b(Round, 2, 1);
        dim3 p3b(Round, Round, 1);

        dim3 p2b_0(Round_0, 1, 1);
        dim3 p2b_1(Round_1, 1, 1);

        dim3 p3b_g11(Round_0, Round_0, 1);
        dim3 p3b_g12(Round_0, Round_1, 1);
        dim3 p3b_g21(Round_1, Round_0, 1);
        dim3 p3b_g22(Round_1, Round_1, 1);

        cudaEventCreate(&sync_1[num]);
        cudaEventCreate(&sync_2[num]);

        LOG("[thread %d] sp: %d, Round: %d", num, sp[num], Round);

        cudaDeviceSynchronize();
#pragma omp barrier

        //  Step 1: update G11, G22
        //
        //    stream 0      stream 1
        //
        //  +---+-----+    +---+-----+
        //  | U |     |    | U |     |   U: update (pivot)
        //  +---+-----+    +---+-----+
        //  |   |     |    |   |     |
        //  |   |     |    |   |     |
        //  +---+-----+    +---+-----+


        if(num==0){  //G11 update
            int pivot = 0;
            for(int r=0;r<Round_0;++r){
                phase_one<BLOCK_SIZE><<< 1 , dimt , 0 , stream[0] >>>(device_ptr[0], pitch[0], pivot, sp[0]);
                phase_two<BLOCK_SIZE><<< p2b , dimt , 0 , stream[0] >>>(device_ptr[0], pitch[0], pivot, 0, sp[0], r);   //0~sp[0]
                phase_three<BLOCK_SIZE><<< p3b , dimt , 0 , stream[0] >>>(device_ptr[0], pitch[0], pivot, 0, sp[0], sp[0], r); //0~sp[0]
                pivot += BLOCK_SIZE;
            }
        }else{  //G11 update
             int pivot = 0;
            for(int r=0;r<Round_0;++r){
                phase_one<BLOCK_SIZE><<< 1 , dimt , 0 , stream[1] >>>(device_ptr[1], pitch[1], pivot, sp[0]);
                phase_two<BLOCK_SIZE><<< p2b , dimt , 0 , stream[1] >>>(device_ptr[1], pitch[1], pivot, 0, sp[0], r);   //0~sp[0]
                phase_three<BLOCK_SIZE><<< p3b , dimt , 0 , stream[1] >>>(device_ptr[1], pitch[1], pivot, 0, sp[0], sp[0], r); //0~sp[0]
                pivot += BLOCK_SIZE;
            }
        }

        //G11 -> G12 G21
        if(num == 0){ //G11 -> G12
            int pivot = 0;
            for(int r=0;r<Round_0;++r){
                phase_two_h<BLOCK_SIZE><<< p2b_1 , dimt , 0 , stream[0] >>>(device_ptr[0], pitch[0], pivot, sp[0], sp[0], vert, sp[0]);
                phase_three_h<BLOCK_SIZE><<< p3b_g12, dimt , 0 , stream[0] >>>(device_ptr[0], pitch[0], pivot, 0, sp[0], sp[0], vert, sp[0], r);
                pivot += BLOCK_SIZE;
            }

            cudaMemcpy2DAsync(device_ptr[1], pitch_bytes[1],
                            device_ptr[0], pitch_bytes[0],
                            vert_bytes, sp[0], cudaMemcpyDeviceToDevice, stream[0]);
            cudaEventRecord(sync_1[0], stream[0]);

        }else{  //G11 -> G21
            int pivot = 0;
            for(int r=0;r<Round_0;++r){
                phase_two_v<BLOCK_SIZE><<< p2b_1 , dimt , 0 , stream[1] >>>(device_ptr[1], pitch[1], pivot, sp[0], vert, sp[0], sp[0]);
                phase_three_v<BLOCK_SIZE><<< p3b_g21 , dimt , 0 , stream[1]>>>(device_ptr[1], pitch[1], pivot, sp[0], 0, vert, sp[0], sp[0], r);
                pivot += BLOCK_SIZE;
            }

            cudaMemcpy2DAsync(device_ptr[0] + sp[0]*pitch[0], pitch_bytes[1],
                            device_ptr[1] + sp[0]*pitch[1], pitch_bytes[0],
                            vert_bytes, sp[1], cudaMemcpyDeviceToDevice, stream[1]);
            cudaEventRecord(sync_1[1], stream[1]);
        }

//#pragma omp barrier

        //relax G12 G21 -> G22
        if(num == 0){  //G22 relax
            cudaStreamWaitEvent(stream[0], sync_1[1], 0);

            int pivot = 0;
            for(int r=0;r<Round_0;++r){
                phase_three<BLOCK_SIZE><<< p3b_g22 , dimt , 0 , stream[0] >>>(device_ptr[0], pitch[0], pivot, sp[0], vert, sp[0], vert);
                pivot += BLOCK_SIZE;
            }  
        }else{ //G22 relax
            cudaStreamWaitEvent(stream[1], sync_1[0], 0);

            int pivot = 0;
            for(int r=0;r<Round_0;++r){
                phase_three<BLOCK_SIZE><<< p3b_g22 , dimt , 0 , stream[1] >>>(device_ptr[1], pitch[1], pivot, sp[0], vert, sp[0], vert);
                pivot += BLOCK_SIZE;
            } 
        }

        //G22 update
        if(num == 0){  //G22 update
            int pivot = sp[0];
            for(int r=0;r<Round_1;++r){
                phase_one<BLOCK_SIZE><<< 1 , dimt , 0 , stream[0] >>>(device_ptr[0], pitch[0], pivot, vert);
                phase_two<BLOCK_SIZE><<< p2b , dimt , 0 , stream[0] >>>(device_ptr[0], pitch[0], pivot, sp[0], vert, r);  //sp[0]~vert
                phase_three<BLOCK_SIZE><<< p3b , dimt , 0 , stream[0] >>>(device_ptr[0], pitch[0], pivot, sp[0], vert, vert, r);//sp[0]~vert
                pivot += BLOCK_SIZE;
            }
        }else{  //G22 update
            int pivot = sp[0];
            for(int r=0;r<Round_1;++r){
                phase_one<BLOCK_SIZE><<< 1 , dimt , 0 , stream[1] >>>(device_ptr[1], pitch[1], pivot, vert);
                phase_two<BLOCK_SIZE><<< p2b , dimt , 0 , stream[1] >>>(device_ptr[1], pitch[1], pivot, sp[0], vert, r);  //sp[0]~vert
                phase_three<BLOCK_SIZE><<< p3b , dimt , 0 , stream[1] >>>(device_ptr[1], pitch[1], pivot, sp[0], vert, vert, r);//sp[0]~vert
                pivot += BLOCK_SIZE;
            }
        }

        //G22 -> G12
        if(num == 0){  //G22 -> G12

            int pivot = sp[0];
            for(int r=0;r<Round_1;++r){
                phase_two_v<BLOCK_SIZE><<< p2b_0 , dimt , 0 , stream[0] >>>(device_ptr[0], pitch[0], pivot, 0, sp[0], vert, vert);
                phase_three_v<BLOCK_SIZE><<< p3b_g12 , dimt , 0 , stream[0] >>>(device_ptr[0], pitch[0], pivot, 0, sp[0], sp[0], vert, vert, r);
                pivot += BLOCK_SIZE;
            }

        }else{  //G22 -> G21
            int pivot = sp[0];
            for(int r=0;r<Round_1;++r){
                phase_two_h<BLOCK_SIZE><<< p2b_0 , dimt , 0 , stream[1] >>>(device_ptr[1], pitch[1], pivot, 0, vert, sp[0], vert);
                phase_three_h<BLOCK_SIZE><<< p3b_g21, dimt , 0 , stream[1] >>>(device_ptr[1], pitch[1], pivot, sp[0], 0, vert, sp[0], vert, r);
                pivot += BLOCK_SIZE;
            }

            cudaMemcpy2DAsync(device_ptr[0] + sp[0]*pitch[0], pitch_bytes[0], 
                            device_ptr[1] + sp[0]*pitch[1], pitch_bytes[1],
                            vert_bytes, sp[1], cudaMemcpyDeviceToDevice, stream[1]);
            cudaEventRecord(sync_2[1], stream[1]);
 
            cudaMemcpy2DAsync(data + sp[0]*vert, vert_bytes,
                            device_ptr[1] + sp[0]*pitch[1], pitch_bytes[1],
                            vert_bytes, sp[1], cudaMemcpyDeviceToHost, stream[1]); 
        }

//#pragma omp barrier
        if(num==0){  //G11 relax
            cudaStreamWaitEvent(stream[0], sync_2[1], 0);

            int pivot = sp[0];
            for(int r=0;r<Round_1;++r){
                phase_three<BLOCK_SIZE><<< p3b_g11 , dimt , 0 , stream[0] >>>(device_ptr[0], pitch[0], pivot, 0, sp[0], vert, vert);
                pivot += BLOCK_SIZE;
            }

            cudaMemcpy2DAsync(data, vert_bytes, 
                            device_ptr[0], pitch_bytes[0],
                            vert_bytes, sp[0],cudaMemcpyDeviceToHost, stream[0]); 
        }

        
/*
        if(num){ //G22
            int pivot = sp[0];
            for(int r=0;r<Round;++r){
                phase_one<BLOCK_SIZE><<< 1 , dimt , 0 , stream[1] >>>(device_ptr[1], pitch[1], pivot, vert);
                phase_two<BLOCK_SIZE><<< p2b , dimt , 0 , stream[1] >>>(device_ptr[1], pitch[1], pivot, sp[0], vert, r);  //sp[0]~vert
                phase_three<BLOCK_SIZE><<< p3b , dimt , 0 , stream[1] >>>(device_ptr[1], pitch[1], pivot, sp[0], vert, vert, r);//sp[0]~vert
                pivot += BLOCK_SIZE;
            }
        }else{ //G11
            int pivot = 0;
            for(int r=0;r<Round;++r){
                phase_one<BLOCK_SIZE><<< 1 , dimt , 0 , stream[0] >>>(device_ptr[0], pitch[0], pivot, sp[0]);
                phase_two<BLOCK_SIZE><<< p2b , dimt , 0 , stream[0] >>>(device_ptr[0], pitch[0], pivot, 0, sp[0], r);   //0~sp[0]
                phase_three<BLOCK_SIZE><<< p3b , dimt , 0 , stream[0] >>>(device_ptr[0], pitch[0], pivot, 0, sp[0], sp[0], r); //0~sp[0]
                pivot += BLOCK_SIZE;
            }
        }

#pragma omp barrier
        // *   Step 2: G11->G12, G22->G21
        // *
        // *    stream 0      stream 1
        // *
        // *  +---+-----+    +---+-----+
        // *  | NP|  U  |    |   |     |   P: pivot
        // *  +---+-----+    +---+-----+   U: update
        // *  |   |     |    |   |  P  |   N: new block (not sync yet)
        // *  |   |     |    | U |  N  |
        // *  +---+-----+    +---+-----+
        // *

        if(num){  //G22 relax to G21
            int pivot = sp[0];
            for(int r=0;r<Round;++r){
                phase_two_h<BLOCK_SIZE><<< p2b_0 , dimt , 0 , stream[1] >>>(device_ptr[1], pitch[1], pivot, 0, vert, sp[0], vert);
                phase_three_h<BLOCK_SIZE><<< p3b_g21, dimt , 0 , stream[1] >>>(device_ptr[1], pitch[1], pivot, sp[0], 0, vert, sp[0], vert, r);
                pivot += BLOCK_SIZE;
            }

            cudaMemcpy2DAsync(device_ptr[0] + sp[0]*pitch[0], pitch_bytes[0], 
                            device_ptr[1] + sp[0]*pitch[1], pitch_bytes[1],
                            vert_bytes, sp[1], cudaMemcpyDeviceToDevice, stream[1]);
            cudaEventRecord(sync_1[1], stream[1]);
        }else{  //G11 relax to G12
            int pivot = 0;
            for(int r=0;r<Round;++r){
                phase_two_h<BLOCK_SIZE><<< p2b_1 , dimt , 0 , stream[0] >>>(device_ptr[0], pitch[0], pivot, sp[0], sp[0], vert, sp[0]);
                phase_three_h<BLOCK_SIZE><<< p3b_g12, dimt , 0 , stream[0] >>>(device_ptr[0], pitch[0], pivot, 0, sp[0], sp[0], vert, sp[0], r);
                pivot += BLOCK_SIZE;
            }
            cudaMemcpy2DAsync(device_ptr[1], pitch_bytes[1],
                            device_ptr[0], pitch_bytes[0],
                            vert_bytes, sp[0], cudaMemcpyDeviceToDevice, stream[0]);
            cudaEventRecord(sync_1[0], stream[0]);
        }

        //         Synchronize
        // *
        // *    stream 0      stream 1
        // *
        // *  +---+-----+    +---+-----+
        // *  | N |  N  | -> |   |     |
        // *  +---+-----+    +---+-----+  N: new block
        // *  |   |     |    |   |     |
        // *  |   |     | <- | N |  N  |
        // *  +---+-----+    +---+-----+
        // *

#pragma omp barrier

        //   Step 3: G11->G21, G22->G12
        // *
        // *    stream 0      stream 1
        // *
        // *  +---+-----+    +---+-----+
        // *  | P |     |    |   |  U  |  P: pivot
        // *  +---+-----+    +---+-----+  U: update
        // *  |   |     |    |   |     |
        // *  | U |     |    |   |  P  |
        // *  +---+-----+    +---+-----+
        // *

        if(num){
            cudaStreamWaitEvent(stream[1], sync_1[0], 0);

            int pivot = sp[0];
            for(int r=0;r<Round;++r){
                phase_two_v<BLOCK_SIZE><<< p2b_0 , dimt , 0 , stream[1] >>>(device_ptr[1], pitch[1], pivot, 0, sp[0], vert, vert);
                phase_three_v<BLOCK_SIZE><<< p3b_g12 , dimt , 0 , stream[1] >>>(device_ptr[1], pitch[1], pivot, 0, sp[0], sp[0], vert, vert, r);
                pivot += BLOCK_SIZE;
            }

            cudaMemcpy2DAsync(device_ptr[0], pitch_bytes[0],
                            device_ptr[1], pitch_bytes[1],
                            vert_bytes, sp[0], cudaMemcpyDeviceToDevice, stream[1]);
            cudaEventRecord(sync_2[1], stream[1]);
        }else{
            cudaStreamWaitEvent(stream[0], sync_1[1], 0);

            int pivot = 0;
            for(int r=0;r<Round;++r){
                phase_two_v<BLOCK_SIZE><<< p2b_1 , dimt , 0 , stream[0] >>>(device_ptr[0], pitch[0], pivot, sp[0], vert, sp[0], sp[0]);
                phase_three_v<BLOCK_SIZE><<< p3b_g21 , dimt , 0 , stream[0]>>>(device_ptr[0], pitch[0], pivot, sp[0], 0, vert, sp[0], sp[0], r);
                pivot += BLOCK_SIZE;
            }

            cudaMemcpy2DAsync(device_ptr[1] + sp[0]*pitch[1], pitch_bytes[1],
                            device_ptr[0] + sp[0]*pitch[0], pitch_bytes[0],
                            vert_bytes, sp[1], cudaMemcpyDeviceToDevice, stream[0]);
            cudaEventRecord(sync_2[0], stream[0]);
        }

        //         Synchronize
        // *
        // *    stream 0      stream 1
        // *
        // *  +---+-----+    +---+-----+
        // *  |   |     | <- |   |  N  |
        // *  +---+-----+    +---+-----+  N: new block
        // *  |   |     |    |   |     |
        // *  | N |     | -> |   |     |
        // *  +---+-----+    +---+-----+
        // *

#pragma omp barrier

        //   Step 4: G12, G21 both relax to G11 and G22
        // *
        // *    stream 0      stream 1
        // *
        // *  +---+-----+    +---+-----+
        // *  | P |     |    | U |     |  P: pivot
        // *  +---+-----+    +---+-----+  U: update
        // *  |   |     |    |   |     |
        // *  |   |  U  |    |   |  P  |
        // *  +---+-----+    +---+-----+
        // *

        if(num){
            cudaStreamWaitEvent(stream[1], sync_2[0], 0);

            int pivot = sp[0];
            for(int r=0;r<Round;++r){
                phase_three<BLOCK_SIZE><<< p3b_g11 , dimt , 0 , stream[1] >>>(device_ptr[1], pitch[1], pivot, 0, sp[0], vert, vert);
                pivot += BLOCK_SIZE;
            }

            cudaMemcpy2DAsync(data, vert_bytes, 
                            device_ptr[1], pitch_bytes[1],
                            vert_bytes, sp[0],cudaMemcpyDeviceToHost, stream[1]); //to host
        }else{
            cudaStreamWaitEvent(stream[0], sync_2[1], 0);

            int pivot = 0;
            for(int r=0;r<Round;++r){
                phase_three<BLOCK_SIZE><<< p3b_g22 , dimt , 0 , stream[0] >>>(device_ptr[0], pitch[0], pivot, sp[0], vert, sp[0], vert);
                pivot += BLOCK_SIZE;
            }
            cudaMemcpy2DAsync(data + sp[0]*vert, vert_bytes,
                            device_ptr[0] + sp[0]*pitch[0], pitch_bytes[0],
                            vert_bytes, sp[1], cudaMemcpyDeviceToHost, stream[0]); //to host
        }


        //            Synchronize
        // *
        // *       stream 0      stream 1
        // *   
        // *     +---+-----+    +---+-----+
        // *     |   |     |    | N |     |  --+
        // *     +---+-----+    +---+-----+    |     N: new block
        // *     |   |     |    |   |     |    |
        // * +-- |   |  N  |    |   |     |    |
        // * |   +---+-----+    +---+-----+    |
        // * |                                 |
        // * |              Host               |
        // * |                                 |
        // * |          +---+-----+            |
        // * +-----+    |   |     | <----------+
        // *       |    +---+-----+
        // *       |    |   |     |
        // *       +--> |   |     |
        // *            +---+-----+
        // *
*/

        cudaDeviceSynchronize();
#pragma omp barrier
        cudaStreamDestroy(stream[num]);
    } //end parallel

    for(int i=0;i<2;++i){
        cudaFree(device_ptr[i]);
    }
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
        default:
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






