#include "hip/hip_runtime.h"
// --------------------------------------------------------------------//
//                                                                     //
// title                  :gridding.cu                                 //
// description            :Sort and Gridding process.                  //
// author                 :                                            //
//                                                                     //
// --------------------------------------------------------------------//

#include <boost/sort/sort.hpp>
#include "gridding.h"
#include "/usr/local/mpich-3.4.1/include/mpi.h"
#include <omp.h>
using boost::sort::block_indirect_sort;

#define stream_size 3
#define max_group_size 24
hipStream_t stream[stream_size];

double *d_lons;
double *d_lats;
double *d_data;
double *d_weights;
uint64_t *d_hpx_idx;
uint32_t *d_start_ring;
//texture<uint32_t> tex_start_ring;
__constant__ uint32_t d_const_zyx[3];
uint32_t *d_zyx;
double *d_xwcs;
double *d_ywcs;
double **d_datacube;
double **d_weightscube;
double **tempArray;
__constant__ double d_const_kernel_params[3];
__constant__ GMaps d_const_GMaps;
__constant__ Healpix d_const_Healpix;
size_t pitch_d,pitch_h, pitch_r;
double coordinate_order_time = 0;
double data_order_time = 0;
double pre_order_time = 0;

/*********************************Sort input points with CPU*******************************/
/**
 * @brief   Sort input points with CPU and Create two level lookup table.
 * @param   sort_param: set the sort parameters for the chosen type.
 * */
void init_input_with_cpu(const int &sort_param) {
    double iTime1 = cpuSecond();
    uint32_t data_shape = h_GMaps.data_shape;
    std::vector<HPX_IDX> V(data_shape);
    V.reserve(data_shape);

    // Get HEALPix index and input index of each input point.
    for(int i=0; i < data_shape; ++i) {
        double theta = HALFPI - DEG2RAD * h_lats[i];
        double phi = DEG2RAD * h_lons[i];
        uint64_t hpx = h_ang2pix(theta, phi);
        V[i] = HPX_IDX(hpx, i);             // (HEALPix_index, input_index)
    }

    // Sort input points by param (key-value sort). KEY: HEALPix index VALUE: array index
    double iTime2 = cpuSecond();
    if (sort_param == BLOCK_INDIRECT_SORT) {
        unsigned int con_threads;
        con_threads = thread::hardware_concurrency();
        // printf("con_threads=%d\n", con_threads);
        boost::sort::block_indirect_sort(V.begin(), V.end(),con_threads);
    } else if (sort_param == PARALLEL_STABLE_SORT) {
        boost::sort::parallel_stable_sort(V.begin(), V.end());
    } else if (sort_param == STL_SORT) {
        std::sort(V.begin(), V.end());
    }
    double iTime3 = cpuSecond();

    // Copy the HEALPix, lons, lats and data for sorted input points
    // Sort the input points according the sorted input index.
    h_hpx_idx = RALLOC(uint64_t, data_shape + 1);
    h_inx_idx = RALLOC(uint32_t, data_shape + 1);
    for(int i=0; i < data_shape; ++i){
        h_hpx_idx[i] = V[i].hpx;
        h_inx_idx[i] = V[i].inx;
    }
    h_hpx_idx[data_shape] = h_Healpix._npix;
    double *tempArray = RALLOC(double, data_shape);
    for(int i=0; i < data_shape; ++i){
        tempArray[i] = h_lons[V[i].inx];
    }
    swap(h_lons, tempArray);
    for(int i=0; i < data_shape; ++i){
        tempArray[i] = h_lats[V[i].inx];
    }
    swap(h_lats, tempArray);
    DEALLOC(tempArray);

    // Pre-process by h_hpx_idx
    double iTime4 = cpuSecond();
    uint64_t first_ring = h_pix2ring(h_hpx_idx[0]); 
    uint32_t temp_count = (uint32_t)(1 + h_pix2ring(h_hpx_idx[data_shape - 1]) - first_ring);   
    h_Healpix.firstring = first_ring;
    h_Healpix.usedrings = temp_count;
    h_start_ring = RALLOC(uint32_t, temp_count + 1); 
    h_start_ring[0] = 0;    
    uint64_t startpix, num_pix_in_ring;   
    uint32_t ring_idx = 0;
    bool shifted;
    for(uint64_t cnt_ring = 1; cnt_ring < temp_count; ++cnt_ring) { 
        h_get_ring_info_small(cnt_ring + first_ring, startpix, num_pix_in_ring, shifted);
        uint32_t cnt_ring_idx = searchLastPosLessThan(h_hpx_idx, ring_idx, data_shape, startpix);   
        if (cnt_ring_idx == data_shape) {
            cnt_ring_idx = ring_idx - 1;
        }
        ring_idx = cnt_ring_idx + 1;    // The start array index of the first HEALPix index in one ring.
        h_start_ring[cnt_ring] = ring_idx;  // Construct the R_start  
    }
    h_start_ring[temp_count] = data_shape;  
    double iTime5 = cpuSecond();

    // Release
    vector<HPX_IDX> vtTemp;
    vtTemp.swap(V);

    // Get runtime
    double iTime6 = cpuSecond();
    coordinate_order_time = iTime6 - iTime1;
}
void pre_order_data(const int dim){
    uint32_t data_shape = h_GMaps.data_shape;
    // double *tempArray = RALLOC(double, data_shape);
    // omp_set_num_threads(8);
    int i;
#pragma omp parallel for shared(tempArray, h_data) private(i)
    for(i=0; i < data_shape; ++i){
        tempArray[dim][i] = h_data[dim][h_inx_idx[i]];
    }
    swap(h_data[dim], tempArray[dim]);
    // DEALLOC(tempArray);
}

/*********************************Sort input points with GPU*******************************/
/**
 * @brief   Sort input points with GPU and Create two level lookup table.
 * @param   sort_param: set the sort parameters for the chosen type.
 * @note    Through our testing, the performance of sort input points with
 *          CPU is higher than its in GPU, here we provide the GPU sort
 *          interface for reference.
 * */

/* Initialize output spectrals and weights. */
void init_output(){
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    uint32_t channels = h_GMaps.spec_dim;
    HANDLE_ERROR(hipHostMalloc((void**)& h_datacube, sizeof(double*)*channels));
    HANDLE_ERROR(hipHostMalloc((void**)& h_weightscube, sizeof(double*)*channels));
    for(int i = 0; i < channels; i++){
        HANDLE_ERROR(hipHostMalloc((void**)& h_datacube[i], sizeof(double)*num)); 
        HANDLE_ERROR(hipHostMalloc((void**)& h_weightscube[i], sizeof(double)*num));
        for(int j = 0; j < num; j++){
            h_datacube[i][j] = 0;
            h_weightscube[i][j] = 0;
        }
    }
}

/* Sinc function with simple singularity check. */
__device__ double sinc(double x){
    if(fabs(x) < 1.e-10)
        return 1.;
    else
        return sin(x) / x;
}

/* Grid-kernel definitions. */
__device__ double kernel_func_ptr(double distance, double bearing){
    if(d_const_GMaps.kernel_type == GAUSS1D){   // GAUSS1D
        return exp(-distance * distance * d_const_kernel_params[0]);
    }
    else if(d_const_GMaps.kernel_type == GAUSS2D){  // GAUSS2D
        double ellarg = (\
                pow(d_const_kernel_params[0], 2.0)\
                    * pow(sin(bearing - d_const_kernel_params[2]), 2.0)\
                + pow(d_const_kernel_params[1], 2.0)\
                    * pow(cos(bearing - d_const_kernel_params[2]), 2.0));
        double Earg = pow(distance / d_const_kernel_params[0] /\
                       d_const_kernel_params[1], 2.0) / 2. * ellarg;
        return exp(-Earg);
    }
    else if(d_const_GMaps.kernel_type == TAPERED_SINC){ // TAPERED_SINC
        double arg = PI * distance / d_const_kernel_params[0];
        return sinc(arg / d_const_kernel_params[2])\
            * exp(pow(-(arg / d_const_kernel_params[1]), 2.0));
    }
}

/* Binary search key in hpx_idx array. */
__host__ __device__ uint32_t searchLastPosLessThan(uint64_t *values, uint32_t left, uint32_t right, uint64_t _key){
    if(right <= left)
        return right;
    uint32_t low = left, mid, high = right - 1;
    while (low < high){
        mid = low + ((high - low + 1) >> 1);
        if (values[mid] < _key)
            low = mid;
        else
            high = mid - 1;
    }
    if(values[low] < _key)
        return low;
    else
        return right;
}


/********************************************HEGrid****************************************/
/**
 * @brief   Execute gridding in GPU.
 * @param   d_lons: longitude
 * */
__global__ void hegrid (
        double *d_lons,
        double *d_lats,
        double *d_data,
        double *d_weights,
        double *d_xwcs,
        double *d_ywcs,
        double *d_datacube,
        double *d_weightscube,
        uint32_t *d_start_ring,
        uint64_t *d_hpx_idx) {
    uint32_t warp_id = blockIdx.x * (blockDim.x / 64) + threadIdx.x / 64;   // warp index of the whole 1Dim grid and 1Dim block. 32-->64
    uint32_t thread_id = ((warp_id % d_const_GMaps.block_warp_num) * 64 + threadIdx.x % 64) * d_const_GMaps.factor;   // thread index in one ring.
    int get_num = 0;
    int target_num = 0;
    if (thread_id < d_const_zyx[1]) {
        uint32_t left = thread_id;    //Initial left
        uint32_t right = left + d_const_GMaps.factor - 1;   // Initial right
        if (right > d_const_zyx[1]) {                      // 这块儿感觉有问题，怎么是等于每行的最大索引时-1了？(待定）
            right = d_const_zyx[1];
        }
        uint32_t step = (warp_id / d_const_GMaps.block_warp_num) * d_const_zyx[1];    // Thread step for change the ring
        left = left + step;
        right = right + step;
        double temp_weights[3], temp_data[3], l1[3], b1[3];  //这里预设为最大线程粗化因子为3，所以每次连续写入factor个值
        for (thread_id = left; thread_id <= right; ++thread_id) {
            temp_weights[thread_id - left] = d_weightscube[thread_id];
            temp_data[thread_id - left] = d_datacube[thread_id];
            l1[thread_id - left] = d_xwcs[thread_id] * DEG2RAD;
            b1[thread_id - left] = d_ywcs[thread_id] * DEG2RAD;
        }

        // get northeast ring and southeast ring
        double disc_theta = HALFPI - b1[0];     // disc中心点所在行的赤纬
        double disc_phi = l1[0];                // disc中心点所在行的赤经
        double utheta = disc_theta - d_const_GMaps.disc_size;   // 最上面一行中心点的赤纬
        double north_theta = utheta * RAD2DEG;
        if (utheta * RAD2DEG < 0){
            utheta = 0;
        }  // 这里修改了，影响极点位置
        uint64_t upix = d_ang2pix(utheta, disc_phi);            //最上面一行中心点所属的HEALPix的pixel索引
        uint64_t uring = d_pix2ring(upix);                      //最上面一行中心点所在行的HEALPix的行号
        if (uring < d_const_Healpix.firstring){
            uring = d_const_Healpix.firstring;
        }
        utheta = disc_theta + d_const_GMaps.disc_size;  // 最下面一行中心点的赤纬
        upix = d_ang2pix(utheta, disc_phi);             // 最下面一行中心点所属的HEALPix的pixel索引
        uint64_t dring = d_pix2ring(upix);              // 最下面一行中心点所在行的HEALPix行号
        if (dring >= d_const_Healpix.firstring + d_const_Healpix.usedrings){
            dring = d_const_Healpix.firstring + d_const_Healpix.usedrings - 1;
        }

        // Go from the Northeast ring to the Southeast one
        uint32_t start_int = d_start_ring[uring - d_const_Healpix.firstring];  // get the first HEALPix index
        // tex1Dfetch(tex_start_ring, uring - d_const_Healpix.firstring);
        while (uring <= dring) {                                                            // of one ring.
            // get ring info
            uint32_t end_int = d_start_ring[uring - d_const_Healpix.firstring+1];
                    // tex1Dfetch(tex_start_ring, uring - d_const_Healpix.firstring+1);
            uint64_t startpix, num_pix_in_ring, mid_pixel;
            bool shifted;
            d_get_ring_info_small(uring, startpix, num_pix_in_ring, shifted);
            double utheta, uphi, hpx_zero_theta, hpx_zero_phi, rc_theta, rc_phi;
            double d;
            d_pix2ang(startpix, utheta, uphi);

            disc_theta = HALFPI - b1[0];
            disc_phi = l1[0];
            uphi = disc_phi - d_const_GMaps.disc_size;
            d_pix2ang(d_hpx_idx[0], hpx_zero_theta, hpx_zero_phi);
            double zero_angle = uphi * RAD2DEG;
            uint64_t lpix = d_ang2pix(utheta, uphi); // disc size 范围首行的起始pixel
            if (disc_theta * RAD2DEG <= NORTH_B || disc_theta * RAD2DEG >= SOUTH_B){
                lpix = startpix;
            } else{
                lpix = lpix;
            }

            if (!(lpix >= startpix && lpix <= startpix + num_pix_in_ring)) {
                start_int = end_int;
                continue;
            }

            uphi = disc_phi + d_const_GMaps.disc_size;
            uint64_t rpix = d_ang2pix(utheta, uphi);
            if (disc_theta * RAD2DEG <= NORTH_B || disc_theta * RAD2DEG >= SOUTH_B){
                rpix = startpix + num_pix_in_ring - 1;
            } else{
                rpix = rpix;
            }
            if (!(rpix >= startpix && rpix < startpix + num_pix_in_ring)) {
                start_int = end_int;
                continue;
            }

            // find position of lpix
            uint32_t upix_idx = searchLastPosLessThan(d_hpx_idx, start_int - 1, end_int, lpix);
            ++upix_idx;
            if (upix_idx > end_int) {
                upix_idx = d_const_GMaps.data_shape;
            }

            // Gridding
            while(upix_idx < d_const_GMaps.data_shape){
                double l2 = d_lons[upix_idx] * DEG2RAD;
                double b2 = d_lats[upix_idx] * DEG2RAD;
                upix = d_ang2pix(HALFPI - b2, l2);
                if (upix > rpix) {
                    break;
                }

                double in_weights = d_weights[upix_idx];
                double in_data = d_data[upix_idx];

                for (thread_id = left; thread_id <= right; ++thread_id) {
                    double sdist = true_angular_distance(l1[thread_id - left], b1[thread_id - left], l2, b2) * RAD2DEG;
                    double sbear = 0.;
                    if (d_const_GMaps.bearing_needed) {
                        sbear = great_circle_bearing(l1[thread_id - left], b1[thread_id - left], l2, b2);
                    }
                    if (sdist < d_const_GMaps.sphere_radius) {
                        double sweight = kernel_func_ptr(sdist, sbear);
                        double tweight = in_weights * sweight;
                        temp_data[thread_id - left] += in_data * tweight;
                        temp_weights[thread_id - left] += tweight;
                    }
                    d_datacube[thread_id] = temp_data[thread_id - left];
                    d_weightscube[thread_id] = temp_weights[thread_id - left];
                }
                ++upix_idx;
            }
            start_int = end_int;
            ++uring;
        }
    }
    __syncthreads();
}

/* Alloc data for GPU. */
void data_alloc(){
    uint32_t data_shape = h_GMaps.data_shape;
    uint32_t channels = h_GMaps.spec_dim;
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    uint32_t usedrings = h_Healpix.usedrings;

    HANDLE_ERROR(hipMalloc((void**)& d_lons, sizeof(double)*data_shape));
    HANDLE_ERROR(hipMalloc((void**)& d_lats, sizeof(double)*data_shape));

    HANDLE_ERROR(hipHostMalloc((void**)& h_data, sizeof(double*)*channels));
    for(int i = 0; i < channels; i++){
        HANDLE_ERROR(hipHostMalloc((void**)& h_data[i], sizeof(double)*data_shape));
    }   
    
    HANDLE_ERROR(hipMallocPitch((void**)& d_data, &pitch_d, sizeof(double)*data_shape, channels));
    HANDLE_ERROR(hipMallocPitch((void**)& d_datacube, &pitch_r, sizeof(double)*num, channels));
    HANDLE_ERROR(hipMallocPitch((void**)& d_weightscube, &pitch_r, sizeof(double)*num, channels));

    HANDLE_ERROR(hipMalloc((void**)& d_weights, sizeof(double)*data_shape));
    HANDLE_ERROR(hipMalloc((void**)& d_xwcs, sizeof(double)*num));
    HANDLE_ERROR(hipMalloc((void**)& d_ywcs, sizeof(double)*num));
    HANDLE_ERROR(hipMalloc((void**)& d_hpx_idx, sizeof(uint64_t)*(data_shape+1)));
    HANDLE_ERROR(hipMalloc((void**)& d_start_ring, sizeof(uint32_t)*(usedrings+1)));
//    HANDLE_ERROR(hipBindTexture(NULL, tex_start_ring, d_start_ring, sizeof(uint32_t)*(usedrings+1)));
}

/* Send data from CPU to GPU. */
void data_h2d(){
    uint32_t data_shape = h_GMaps.data_shape;
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    uint32_t usedrings = h_Healpix.usedrings;

    // Copy constants memory
    HANDLE_ERROR(hipMemcpy(d_lons, h_lons, sizeof(double)*data_shape, hipMemcpyHostToDevice));
    HANDLE_ERROR(hipMemcpy(d_lats, h_lats, sizeof(double)*data_shape, hipMemcpyHostToDevice));
    //HANDLE_ERROR(hipMemcpy(d_data, h_data, sizeof(double)*data_shape, hipMemcpyHostToDevice));
    HANDLE_ERROR(hipMemcpy(d_weights, h_weights, sizeof(double)*data_shape, hipMemcpyHostToDevice));
    HANDLE_ERROR(hipMemcpy(d_xwcs, h_xwcs, sizeof(double)*num, hipMemcpyHostToDevice));
    HANDLE_ERROR(hipMemcpy(d_ywcs,h_ywcs, sizeof(double)*num, hipMemcpyHostToDevice));
    //HANDLE_ERROR(hipMemcpy(d_datacube, h_datacube, sizeof(double)*num, hipMemcpyHostToDevice));
    //HANDLE_ERROR(hipMemcpy(d_weightscube, h_weightscube, sizeof(double)*num, hipMemcpyHostToDevice));
    HANDLE_ERROR(hipMemcpy(d_hpx_idx, h_hpx_idx, sizeof(uint64_t)*(data_shape+1), hipMemcpyHostToDevice));
    HANDLE_ERROR(hipMemcpy(d_start_ring, h_start_ring, sizeof(uint32_t)*(usedrings+1), hipMemcpyHostToDevice));
    HANDLE_ERROR(hipMemcpyToSymbol(HIP_SYMBOL(d_const_kernel_params), h_kernel_params, sizeof(double)*3));
    HANDLE_ERROR(hipMemcpyToSymbol(HIP_SYMBOL(d_const_zyx), h_zyx, sizeof(uint32_t)*3));
    HANDLE_ERROR(hipMemcpyToSymbol(HIP_SYMBOL(d_const_Healpix), &h_Healpix, sizeof(Healpix)));
    HANDLE_ERROR(hipMemcpyToSymbol(HIP_SYMBOL(d_const_GMaps), &h_GMaps, sizeof(GMaps)));
}

/* Send data from GPU to CPU. */
void data_d2h(uint32_t s_index, uint32_t dim){
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    HANDLE_ERROR(hipMemcpyAsync(h_datacube[dim], (double*)((char*)d_datacube+dim*pitch_r), sizeof(double)*num, hipMemcpyDeviceToHost,stream[s_index]));
    HANDLE_ERROR(hipMemcpyAsync(h_weightscube[dim],(double*)((char*)d_weightscube+dim*pitch_r), sizeof(double)*num, hipMemcpyDeviceToHost,stream[s_index]));
}

/* Release data. */
void data_free(){
    DEALLOC(h_lons);
    HANDLE_ERROR( hipFree(d_lons) );
    DEALLOC(h_lats);
    HANDLE_ERROR( hipFree(d_lats) );
    //DEALLOC(h_data);
    for(int i = 0; i < h_GMaps.spec_dim; i++){
        hipHostFree(h_data[i]);
    }   
    HANDLE_ERROR(hipHostFree(h_data));

    HANDLE_ERROR( hipFree(d_data) );
    DEALLOC(h_weights);
    HANDLE_ERROR( hipFree(d_weights) );
    DEALLOC(h_xwcs);
    HANDLE_ERROR( hipFree(d_xwcs) );
    DEALLOC(h_ywcs);
    HANDLE_ERROR( hipFree(d_ywcs) );

    for(int i = 0; i < h_GMaps.spec_dim; i++){
        hipHostFree(h_datacube[i]);
    }
    hipHostFree(h_datacube);
    for(int i = 0; i < h_GMaps.spec_dim; i++){
        hipHostFree(h_weightscube[i]);
    }
    hipHostFree(h_weightscube);

    HANDLE_ERROR( hipFree(d_datacube) );
    HANDLE_ERROR( hipFree(d_weightscube) );
    DEALLOC(h_hpx_idx);
    HANDLE_ERROR( hipFree(d_hpx_idx) );
    DEALLOC(h_start_ring);
//    HANDLE_ERROR( hipUnbindTexture(tex_start_ring) );

    HANDLE_ERROR( hipFree(d_start_ring) );
    DEALLOC(h_header);
    DEALLOC(h_zyx);
    DEALLOC(h_kernel_params);
    DEALLOC(tempArray);
}

/*mpi read&pre-order input data*/
void MallocTempArray(){
    //read&pre-order input data 
    uint32_t channels = h_GMaps.spec_dim;
    uint32_t data_shape = h_GMaps.data_shape; 
    tempArray = RALLOC(double*, channels);
    for(int i = 0; i < channels; i++){
        tempArray[i] = RALLOC(double, data_shape);
    }
}

/* Gridding process. */
void solve_gridding(const char *infile, const char *tarfile, const char *outfile, const char *sortfile, const int& param, const int &bDim, int argc, char **argv) {
    // Read input points.
    read_input_coordinate(infile);

    // Read output map.
    read_output_map(tarfile);

    // Set wcs for output pixels.
    set_WCS();

    // Initialize output spectrals and weights.
    init_output();

    double iTime1 = cpuSecond();
    init_input_with_cpu(param);
    double iTime2 = cpuSecond();

/*******************sort_time*********************/
    double sort_time = (iTime2 - iTime1);
/********************************************/

    // get the cuda device count
    int count;
    hipGetDeviceCount(&count);
    // printf("设备数量为:%d ,",count);
    hipSetDevice(1);
    // Alloc data for GPU.
    data_alloc();
    double iTime3 = cpuSecond();
    // Send data from CPU to GPU.
    data_h2d();
    double iTime4 = cpuSecond();

/*****************load_time1**********************/
    double load_time1 = (iTime4 - iTime3);
/*********************************************/

    // read input data.
    read_input_data(infile);
    MallocTempArray();
    double iTime5 = cpuSecond();

/*****************read_value**********************/
double read_value = (iTime5 - iTime4);
/*********************************************/

    //create stream
    for(int i=0;i<stream_size;i++){
        hipStreamCreate(&stream[i]);
    }

    // printf("h_GMaps.spec_dim=%d\n",h_GMaps.spec_dim);
    uint32_t channels = h_GMaps.spec_dim;
    uint32_t data_shape = h_GMaps.data_shape;
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    // Set block and thread.
    dim3 block(bDim);
    dim3 grid((h_GMaps.block_warp_num * h_zyx[2] - 1) / (block.x / 64) + 1);  //32-->64
    // printf("grid.x=%d, block.x=%d, ", grid.x, block.x);

    // Get start time.
    hipEvent_t start, stop;
    HANDLE_ERROR(hipEventCreate(&start));
    HANDLE_ERROR(hipEventCreate(&stop));
    HANDLE_ERROR(hipEventRecord(start, 0));

    // for(int i = 0; i < channels; i++){
    //     int j = i % stream_size;
    //     printf("channel_id=%d\n", i);
    //     HANDLE_ERROR(hipMemcpyAsync((double*)((char*)d_data+i*pitch_d), h_data[i], sizeof(double)*data_shape, hipMemcpyHostToDevice, stream[j]));
    //     HANDLE_ERROR(hipMemcpyAsync((double*)((char*)d_datacube+i*pitch_r), h_datacube[i], sizeof(double)*num, hipMemcpyHostToDevice, stream[j]));
    //     HANDLE_ERROR(hipMemcpyAsync((double*)((char*)d_weightscube+i*pitch_r), h_weightscube[i], sizeof(double)*num, hipMemcpyHostToDevice,stream[j]));
    //     hipLaunchKernelGGL(hegrid, dim3(grid), dim3(block ), 0, stream[j], d_lons, d_lats, (double*)((char*)d_data+i*pitch_d), d_weights, d_xwcs, d_ywcs, (double*)((char*)d_datacube+i*pitch_r), (double*)((char*)d_weightscube+i*pitch_r), d_start_ring, d_hpx_idx);
    //     // data_d2h(i % stream_size, i);
    //     // hegrid<<< grid, block, 0, stream[j] >>>(d_lons, d_lats, (double*)((char*)d_data+i*pitch_d), d_weights, d_xwcs, d_ywcs, (double*)((char*)d_datacube+i*pitch_r), (double*)((char*)d_weightscube+i*pitch_r), d_hpx_idx);
    // }
    // for(int i = 0; i < channels; i++){
    //     data_d2h(i % stream_size, i);
    // }    

    omp_set_num_threads(stream_size);
    for(int j=0; j < channels/stream_size; j++){
        #pragma omp parallel
        {
            int i = omp_get_thread_num();
            // printf("thread_id=%d\n", threadsss);
            int channel_id = i + stream_size * j;
            // printf("channel_id=%d\n", channel_id);
            // printf("CPU start!\n");
            pre_order_data(channel_id);
            // printf("CPU finish!\n");
            HANDLE_ERROR(hipMemcpyAsync((double*)((char*)d_data+channel_id*pitch_d), h_data[channel_id], sizeof(double)*data_shape, hipMemcpyHostToDevice, stream[i]));
            HANDLE_ERROR(hipMemcpyAsync((double*)((char*)d_datacube+channel_id*pitch_r), h_datacube[channel_id], sizeof(double)*num, hipMemcpyHostToDevice, stream[i]));
            HANDLE_ERROR(hipMemcpyAsync((double*)((char*)d_weightscube+channel_id*pitch_r), h_weightscube[channel_id], sizeof(double)*num, hipMemcpyHostToDevice,stream[i]));
            hipLaunchKernelGGL(hegrid, dim3(grid), dim3(block ), 0, stream[i], d_lons, d_lats, (double*)((char*)d_data+channel_id*pitch_d), d_weights, d_xwcs, d_ywcs, (double*)((char*)d_datacube+channel_id*pitch_r), (double*)((char*)d_weightscube+channel_id*pitch_r), d_start_ring, d_hpx_idx);
        }

    }
    // for(int i = 0; i < channels; i++){
    //     data_d2h(i % stream_size, i);
    // }    

    // Get stop time.
    HANDLE_ERROR(hipEventRecord(stop, 0));
    HANDLE_ERROR(hipEventSynchronize(stop));
    float kernel_time;
/***************************kernel_time*********************************/
    HANDLE_ERROR(hipEventElapsedTime(&kernel_time, start, stop));
/********************************************************************/

    // printf("%f, ", kernel_time);

    hipDeviceSynchronize();

    //destroy stream
    for(int i=0;i<stream_size;i++){
        hipStreamDestroy(stream[i]);
    }

/*********************************************************************/
    double cost_time = sort_time + load_time1 + kernel_time/1000;
    printf("time cost %lf\n", cost_time);
/*********************************************************************/
    // // Write output FITS file
    // write_output_map(outfile);

    // Write sorted input FITS file
    if (sortfile) {
        write_ordered_map(infile, sortfile);
    }

    // Release data
    data_free();
    HANDLE_ERROR( hipEventDestroy(start) );
    HANDLE_ERROR( hipEventDestroy(stop) );
    HANDLE_ERROR( hipDeviceReset() );
}
