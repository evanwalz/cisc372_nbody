#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "compute.h"

extern vector3 *hVel;
extern vector3 *hPos;
extern double  *mass;
extern vector3 *d_hVel;
extern vector3 *d_hPos;

#include <cuda_runtime.h>

static double   *d_mass   = NULL;
static vector3  *d_accels = NULL;

#define BLOCK_SIZE 256

__global__ void computeAccelsKernel(vector3 *pos, double *mass, vector3 *accels, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;

    int i = idx / n;   // target body
    int j = idx % n;   // source body

    if (i == j) {
        // zero acceleration from self
	accels[idx][0] = 0.0;
    	accels[idx][1] = 0.0;
        accels[idx][2] = 0.0;
    } else {
        double dx = pos[i][0] - pos[j][0];
        double dy = pos[i][1] - pos[j][1];
        double dz = pos[i][2] - pos[j][2];

        double magnitude_sq = dx*dx + dy*dy + dz*dz;
        double magnitude    = sqrt(magnitude_sq);

        double accelmag = -1.0 * GRAV_CONSTANT * mass[j] / magnitude_sq;

        accels[idx][0] = accelmag * dx / magnitude;
        accels[idx][1] = accelmag * dy / magnitude;
        accels[idx][2] = accelmag * dz / magnitude;
    }
}

__global__ void updateBodiesKernel(vector3 *pos,
                                   vector3 *vel,
                                   vector3 *accels,
                                   int      n)
{
    // One block handles ONE body i
    int i   = blockIdx.x;
    int tid = threadIdx.x;

    if (i >= n) return;

    // Shared memory to hold partial sums of acceleration for body i
    __shared__ double s_ax[BLOCK_SIZE];
    __shared__ double s_ay[BLOCK_SIZE];
    __shared__ double s_az[BLOCK_SIZE];

    // Each thread will accumulate a partial sum over some of the j's
    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;

    // Walk over j's in chunks: j = tid, tid + BLOCK_SIZE, ...
    for (int j = tid; j < n; j += BLOCK_SIZE) {
        int idx = i * n + j;   // row i, column j in the flattened matrix

        ax += accels[idx][0];
        ay += accels[idx][1];
        az += accels[idx][2];
    }

    // Store partial sums in shared memory
    s_ax[tid] = ax;
    s_ay[tid] = ay;
    s_az[tid] = az;

    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_ax[tid] += s_ax[tid + stride];
            s_ay[tid] += s_ay[tid + stride];
            s_az[tid] += s_az[tid + stride];
        }
        __syncthreads();
    }

    // After reduction, thread 0 has the total acceleration for body i
    if (tid == 0) {
        double ax_total = s_ax[0];
        double ay_total = s_ay[0];
        double az_total = s_az[0];

        // update velocity
        vel[i][0] += ax_total * INTERVAL;
        vel[i][1] += ay_total * INTERVAL;
        vel[i][2] += az_total * INTERVAL;

        // update position
        pos[i][0] += vel[i][0] * INTERVAL;
        pos[i][1] += vel[i][1] * INTERVAL;
        pos[i][2] += vel[i][2] * INTERVAL;
    }
}

extern "C" void compute() {
    int n = NUMENTITIES;

    if (d_hPos == NULL) {
        size_t vec3Size   = sizeof(vector3) * n;
        size_t massSize   = sizeof(double)  * n;
        size_t accelSize  = sizeof(vector3) * n * n;

        cudaMalloc((void**)&d_hPos,    vec3Size);
        cudaMalloc((void**)&d_hVel,    vec3Size);
        cudaMalloc((void**)&d_mass,    massSize);
        cudaMalloc((void**)&d_accels,  accelSize);
    }

    size_t vec3Size  = sizeof(vector3) * n;
    size_t massSize  = sizeof(double)  * n;

    // copy current host state to device
    cudaMemcpy(d_hPos,  hPos,  vec3Size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel,  hVel,  vec3Size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass,  mass,  massSize, cudaMemcpyHostToDevice);

    int threadsPerBlock1 = 256;
    int totalPairs       = n * n;
    int blocks1          = (totalPairs + threadsPerBlock1 - 1) / threadsPerBlock1;

    computeAccelsKernel<<<blocks1, threadsPerBlock1>>>(d_hPos, d_mass, d_accels, n);

    // for reduction kernel: one block per body (n blocks), BLOCK_SIZE threads per block (for partial sums + reduction)
    dim3 blocks2(n);
    dim3 threads2(BLOCK_SIZE);

    updateBodiesKernel<<<blocks2, threads2>>>(d_hPos, d_hVel, d_accels, n);

    // wait for GPU to finish before copying back
    cudaDeviceSynchronize();

    // copy updated state back to host for the next CPU timestep / printing
    cudaMemcpy(hPos, d_hPos, vec3Size, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, vec3Size, cudaMemcpyDeviceToHost);
}
