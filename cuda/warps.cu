
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math_constants.h>
#include <device_launch_parameters.h>

#define CHECK_CUDA_ERROR() do { \
    cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error)); \
        return -1; \
    } \
} while (0)

using f32 = float;
using u32 = std::uint32_t;

const u32 m = 128;
const u32 n_reps = 30;
#define integrate 0

struct vec2
{
    f32 x,y;
};

struct uvec2
{
    u32 x,y;
};

struct uvec4
{
    u32 x,y,z,w;
};

__device__ u32 pcg(u32 v)
{
	u32 state = v * 747796405u + 2891336453u;
	u32 word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}
// http://www.jcgt.org/published/0009/03/02/
__device__ uvec2 pcg2d(uvec2 v)
{
    v.x = v.x * 1664525u + 1013904223u;
    v.y = v.y * 1664525u + 1013904223u;

    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;

    v.x = v.x ^ (v.x >> 16u);
    v.y = v.y ^ (v.y >> 16u);

    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;

    v.x = v.x ^ (v.x >> 16u);
    v.y = v.y ^ (v.y >> 16u);

    return v; 
}

// http://www.jcgt.org/published/0009/03/02/
__device__ uvec4 pcg4d(uvec4 v)
{
    v.x = 1013904223u + 1664525u * v.x;
    v.y = 1013904223u + 1664525u * v.y;
    v.z = 1013904223u + 1664525u * v.z;
    v.w = 1013904223u + 1664525u * v.w;
    
    v.x += v.y*v.w;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    v.w += v.y*v.z;
    
    v.x = v.x ^ (v.x >> 16u);
    v.y = v.y ^ (v.y >> 16u);
    v.z = v.z ^ (v.z >> 16u);
    v.w = v.w ^ (v.w >> 16u);
    
    v.x += v.y*v.w;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    v.w += v.y*v.z;
    
    return v;
}

__global__ void random(vec2* a, u32 n) 
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(1234, idx, 0, &state);

    if (idx < n) {
        a[idx].x   = curand_uniform(&state);
        a[idx].y   = curand_uniform(&state);
    }
}

__global__ void random2(vec2* a, u32 n) 
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    u32 s = pcg(idx);
    if (idx < n) {
        a[idx].x   = s * 0x1.0p-32f;
        a[idx].y   = pcg(s) * 0x1.0p-32f;
    }
}

__device__ f32 spiral_func(f32 x, f32 y)
{
	f32 theta = std::atan2(y,x);
	if (theta < 0) theta += 2*CUDART_PI_F;
	f32 r = std::sqrt(x * x + y * y);
	return r * (theta / (2*CUDART_PI_F));
}

#define test_func spiral_func

__global__ void polar_sampling(vec2* out, u32 n)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) 
    {
        f32 x_final{}, y_final{};
        u32 s = idx;
    for(u32 u{}; u<m; u++)
    {
        f32 phi = 2*CUDART_PI_F*(s=pcg(s))*0x1.0p-32f;
        f32 r   = sqrtf((s=pcg(s))*0x1.0p-32f);
        #if integrate 
            x_final += test_func(r*__cosf(phi), r*__sinf(phi));
        #else
            x_final += r*__cosf(phi);
            y_final += r*__sinf(phi);
        #endif
    }
        out[idx].x = x_final/m;
        #if !integrate
            out[idx].y = y_final/m;
        #endif
    }
}

__global__ void polar_sampling2(vec2* out, u32 n)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) 
    {
        f32 x_final{}, y_final{};
        u32 s = idx;
    for(u32 u{}; u<m; u++)
    {
        f32 r   = sqrtf((s=pcg(s))*0x1.0p-32f);
        f32 si,co;
        sincospif((s=pcg(s))*0x1.0p-31f, &si,&co); 
        #if integrate 
            x_final += test_func(r*co, r*si);
        #else
            x_final += r*co;
            y_final += r*si;
        #endif
    }
        out[idx].x = x_final/m;
        #if !integrate
            out[idx].y = y_final/m;
        #endif
    }
}

__global__ void polar_sampling3(vec2* out, u32 n)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) 
    {
        f32 x_final{}, y_final{};
        uvec2 s = {idx, idx+n};
    for(u32 u{}; u<m; u++)
    {
        s = pcg2d(s);
        f32 r   = sqrtf(s.x*0x1.0p-32f);
        f32 si,co;
        sincospif(s.y*0x1.0p-31f, &si,&co); 
        #if integrate 
            x_final += test_func(r*co, r*si);
        #else
            x_final += r*co;
            y_final += r*si;
        #endif
    }
        out[idx].x = x_final/m;
        #if !integrate
            out[idx].y = y_final/m;
        #endif
    }
}

__global__ void concentric_sampling(vec2* out, u32 n)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) 
    {
        f32 x_final{}, y_final{};
        u32 s = idx;
    for(u32 u{}; u<m; u++)
    {
        f32 x,y,r,phi;
        x = (s=pcg(s)) * 0x1.0p-31f - 1.f;
        y = (s=pcg(s)) * 0x1.0p-31f - 1.f;
        if (x == 0 && y == 0)
        {
            r   = 0;
            phi = 0;
        }
        else if (abs(x) > abs(y)) 
        {
            r   = x;
            phi = CUDART_PI_F/4 * (y / x);
        } 
        else 
        {
            r  = y;
            phi= CUDART_PI_F/2 - CUDART_PI_F/4 * (x / y);
        }
        #if integrate 
            x_final += test_func(r*cosf(phi), r*sinf(phi));
        #else
            x_final += r*cosf(phi);
            y_final += r*sinf(phi);
        #endif
    }
        out[idx].x = x_final/m;
        #if !integrate
            out[idx].y = y_final/m;
        #endif
    }
}

__global__ void rejection_sampling(vec2* out, u32 n)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) 
    {
        f32 x_final{}, y_final{};
        u32 s = idx;
    for(u32 u{}; u<m; u++)
    {
        f32 x,y,r2;
        do
        {
            x = (s=pcg(s)) * 0x1.0p-31f - 1.f;
            y = (s=pcg(s)) * 0x1.0p-31f - 1.f;
            r2 = x * x + y * y;
        } while (r2 > 1.);
        #if integrate 
            x_final += test_func(x,y);
        #else
            x_final += x;
            y_final += y;
        #endif
    }
        out[idx].x = x_final/m;
        #if !integrate
            out[idx].y = y_final/m;
        #endif
    }
}

__global__ void rejection_sampling2(vec2* out, u32 n)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) 
    {
        f32 x_final{}, y_final{};
        uvec2 s = {idx, idx+n};
    for(u32 u{}; u<m; u++)
    {
        f32 x,y,r2;
        do
        {
            s = pcg2d(s);
            x = s.x * 0x1.0p-31f - 1.f;
            y = s.y * 0x1.0p-31f - 1.f;
            r2 = x * x + y * y;
        } while (r2 > 1.);
        #if integrate 
            x_final += test_func(x,y);
        #else
            x_final += x;
            y_final += y;
        #endif
    }
        out[idx].x = x_final/m;
        #if !integrate
            out[idx].y = y_final/m;
        #endif
    }
}

__global__ void adoption_sampling(vec2* out, u32 n)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) 
    {
        f32 x_final{}, y_final{};
        u32 s = idx;
    for(u32 u{}; u<m; u++)
    {
        f32 x,y;
        x = (s=pcg(s)) * 0x1.0p-31f - 1.f;
        y = (s=pcg(s)) * 0x1.0p-31f - 1.f;

		f32 r2 = x * x + y * y;
		f32 t = r2 + 2.f;
		f32 x_cache = x;
		f32 y_cache = y;


		if (t <= 4 * x)
		{
			x_cache -= 2;
		}
		else if (t <= -4 * x)
		{
			x_cache += 2;
		}
		else if (t <= 4 * y)
		{
			y_cache -= 2;
		}
		else if (t <= -4 * y)
		{
			y_cache += 2;
		}
        else
        {
            if((s=pcg(s)) * 0x1.0p-32f < 1-CUDART_2_OVER_PI_F)
            {
                x_cache = x;
                y_cache = y;
            }
            else
            {
                x = (s=pcg(s)) * 0x1.0p-31f - 1.f;
                y = (s=pcg(s)) * 0x1.0p-31f - 1.f;

                r2 = x * x + y * y;
                t = r2 + 2.f;
                x_cache = x;
                y_cache = y;

                if (t <= 4 * x)
                {
                    x_cache -= 2;
                }
                else if (t <= -4 * x)
                {
                    x_cache += 2;
                }
                else if (t <= 4 * y)
                {
                    y_cache -= 2;
                }
                else if (t <= -4 * y)
                {
                    y_cache += 2;
                }
            }   
        }

        if((s=pcg(s)) < 1u<<31)
        {
            x_cache = x;
            y_cache = y;
        }

        #if integrate 
            x_final += test_func(CUDART_SQRT_HALF_F * x_cache, CUDART_SQRT_HALF_F * y_cache);
        #else
            x_final += CUDART_SQRT_HALF_F * x_cache;
            y_final += CUDART_SQRT_HALF_F * y_cache;
        #endif
    }
        out[idx].x = x_final/m;
        #if !integrate
            out[idx].y = y_final/m;
        #endif
    }
}


__global__ void adoption_sampling2(vec2* out, u32 n)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) 
    {
        f32 x_final{}, y_final{};
        uvec4 s = {idx,idx+n,idx+2*n,idx+3*n};

    for(u32 u{}; u<m; u++)
    {
        f32 x,y;
        s = pcg4d(s);

        x = s.x * 0x1.0p-31f - 1.f;
        y = s.y * 0x1.0p-31f - 1.f;

        bool b  = s.x & 1;
        bool b2 = pcg(s.y) * 0x1.0p-32f < CUDART_2_OVER_PI_F;


		f32 r2 = x * x + y * y;
		f32 t = r2 + 2.f;
		f32 x_cache = x;
		f32 y_cache = y;

        f32 c = fmaxf(fabsf(x),fabsf(y));
        if(t > 4*c && b2)
        {
                x = s.z* 0x1.0p-31f-1.f;
                y = s.w* 0x1.0p-31f-1.f;
                x_cache = x;
                y_cache = y;
                r2 = x * x + y * y;
                t = r2 + 2.f;
                c = fmaxf(fabsf(x),fabsf(y));
        }

        if(t<=4*c && b)
        {
            if (t <= 4 * x)
            {
                x_cache -= 2;
            }
            else if (t <= -4 * x)
            {
                x_cache += 2;
            }
            else if (t <= 4 * y)
            {
                y_cache -= 2;
            }
            else if (t <= -4 * y)
            {
                y_cache += 2;
            }
        }

        #if integrate 
            x_final += test_func(CUDART_SQRT_HALF_F * (b ? x_cache : x), CUDART_SQRT_HALF_F *  (b ? y_cache : y));
        #else
            x_final += CUDART_SQRT_HALF_F * (b ? x_cache : x);
            y_final += CUDART_SQRT_HALF_F * (b ? y_cache : y);
        #endif
    }
        out[idx].x = x_final/m;
        #if !integrate
            out[idx].y = y_final/m;
        #endif
    }
}


__global__ void sum_vectors(vec2* vectors, vec2* result, u32 n) {
    extern __shared__ vec2 sdata[];
    u32 tid = threadIdx.x;
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? vectors[idx] : vec2{0.f,0.f};
    __syncthreads();

    for (u32 s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid].x += sdata[tid + s].x/s;
            sdata[tid].y += sdata[tid + s].y/s;
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}


using kernel_func = void(*)(vec2*, u32);

void kernel_wrapper(kernel_func f, u32 blocks, u32 threads, vec2* data, u32 n)
{
    f<<<blocks, threads>>>(data, n);
}

int main() {
    u32 n = 1<<26;
    vec2 *in,*out,*res, *res_host;

    u32 blockSize = 128;
    u32 numBlocks = (n + blockSize - 1) / blockSize;

    cudaMalloc((void **)&in,  n * sizeof(vec2));
    cudaMalloc((void **)&out, 2*n * sizeof(vec2));
    cudaMalloc((void **)&res, numBlocks * sizeof(vec2));

    res_host = (vec2*)malloc(numBlocks * sizeof(vec2));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    vec2 total_sum = {0, 0};
    f32 milliseconds = 0;

    kernel_func samplers[] = {polar_sampling, polar_sampling2, polar_sampling3, concentric_sampling, rejection_sampling, rejection_sampling2, adoption_sampling, adoption_sampling2 };
    const char* labels[] = {"polar_sampling", "polar_sampling2", "polar_sampling3", "concentric_sampling", "rejection_sampling", "rejection_sampling2", "adoption_sampling", "adoption_sampling2"};
    const u32 n_samplers = sizeof(samplers)/sizeof(samplers[0]);


    f32 avg_time[n_samplers];
    memset(avg_time,0, 4*n_samplers);


    for(u32 j{}; j<n_reps; j++)
    {
        for(u32 u{}; u<n_samplers; u++)
        {
            cudaEventRecord(start);
            kernel_wrapper(samplers[u], numBlocks, blockSize, out, n);
            CHECK_CUDA_ERROR();

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            sum_vectors<<<numBlocks, blockSize, blockSize * sizeof(vec2)>>>(out, res, n);
            CHECK_CUDA_ERROR();

            cudaMemcpy(res_host, res, numBlocks*sizeof(vec2), cudaMemcpyDeviceToHost);
            total_sum = {0, 0};
            for (u32 i = 0; i < numBlocks; i++) {
                total_sum.x += res_host[i].x/numBlocks;
                total_sum.y += res_host[i].y/numBlocks;
            }
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);

            if(j==0)
            {
            printf("%s\nTime taken: %f  ms.\n",labels[u], milliseconds);
            printf("\tPer sample: %f ns/sample. Res (%f,%f)\n", milliseconds * (1000.*1000./n/m) , total_sum.x, total_sum.y);
            }

            avg_time[u] += milliseconds * (1000.*1000./n/m);
        }
    }

    for(u32 u{}; u<n_samplers; u++)
    {
            printf("%s\nTime taken average per sample: %f ns.\n",labels[u], avg_time[u]/n_reps);
    }


    cudaFree(in);
    cudaFree(out);
    cudaFree(res);
    free(res_host);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
