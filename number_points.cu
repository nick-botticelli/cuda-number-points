#include <iostream>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define N 1000000
#define SEED 72

#define BLOCKSIZE 1024

struct Point
{
    double x;
    double y;
};

void generatePoints(struct Point *data);
__global__ void number_points(struct Point *data, double *epsilonSqr, unsigned int *total);

int main(int argc, char *argv[])
{
    // Read epsilon distance from command line
    if (argc != 2)
    {
        printf("\nIncorrect number of input parameters. Please input an epsilon distance.\n");
        return 0;
    }    
    
    char inputEpsilon[20];
    strcpy(inputEpsilon, argv[1]);
    double epsilon = atof(inputEpsilon);

    // Generate random points:
    struct Point *data;
    data = (struct Point *) malloc(sizeof(struct Point) * N);
    generatePoints(data);

    omp_set_num_threads(1);
    
    cudaError_t errCode = cudaSuccess;
    if (errCode != cudaSuccess)
        std::cout << "\nLast error: " << errCode << std::endl;
    
    double *epsilonSqr = (double *) malloc(sizeof(double));
    unsigned int *total = (unsigned int *) malloc(sizeof(unsigned int));

    struct Point *dev_data;
    double *dev_epsilonSqr;
    unsigned int *dev_total;

    *epsilonSqr = epsilon * epsilon;
    *total = 0;

    // Allocate on the device
    errCode = cudaMalloc((struct Point **) &dev_data, sizeof(struct Point) * N);
    if(errCode != cudaSuccess)
        std::cout << "\nError: dev_data alloc error with code " << errCode << std::endl;

    errCode = cudaMalloc((unsigned int **) &dev_total, sizeof(unsigned int));
    if (errCode != cudaSuccess)
        std::cout << "\nError: dev_total alloc error with code " << errCode << std::endl;

    errCode = cudaMalloc((double **) &dev_epsilonSqr, sizeof(double));
        if (errCode != cudaSuccess)
                std::cout << "\nError: dev_epsilon alloc error with code " << errCode << std::endl;

    // Copy data over to device
    errCode = cudaMemcpy(dev_data, data, sizeof(struct Point) * N, cudaMemcpyHostToDevice);
    if (errCode != cudaSuccess)
        std::cout << "\nError: dev_data memcpy error with code " << errCode << std::endl;

    errCode = cudaMemcpy(dev_total, total, sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (errCode != cudaSuccess)
        std::cout << "\nError: dev_total memcpy error with code " << errCode << std::endl;

    errCode = cudaMemcpy(dev_epsilonSqr, epsilonSqr, sizeof(double), cudaMemcpyHostToDevice);
        if (errCode != cudaSuccess)
                std::cout << "\nError: dev_epsilonSqr memcpy error with code " << errCode << std::endl;


    // Calculate total blocks
    const unsigned int totalBlocks = ceil(N * 1.0 / BLOCKSIZE);

    double tstart=omp_get_wtime();

    // Execute kernel
    number_points <<< totalBlocks, BLOCKSIZE >>> (dev_data, dev_epsilonSqr, dev_total);
    cudaDeviceSynchronize();

    double tend = omp_get_wtime();

    if (errCode != cudaSuccess)
        std::cout << "\nError after kernel launch " << errCode << std::endl;
    
    printf("\nKernel time (s): %f", tend - tstart);

    // Copy data from device to host 
    errCode = cudaMemcpy(total, dev_total, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    if (errCode != cudaSuccess)
        fprintf(stderr, "\nError: \"%s\": %s", cudaGetErrorName(errCode), cudaGetErrorString(errCode));
    else
        printf("\nNumber of points within epsilon: %llu", (*total * 2) + N);

    cudaFree(dev_data);
    cudaFree(dev_epsilonSqr);
    cudaFree(dev_total);

    free(data);
    printf("\n");

    return 0;
}

/*
 * Kernel - Find the total number of points within epsilon distance of each point
 */
__global__ void number_points(struct Point *data, double *epsilonSqr, unsigned int *total)
{
    unsigned int i = threadIdx.x + (blockIdx.x * blockDim.x);

    if (i >= N - 1)
        return;

    __shared__ unsigned int totalInBlock[BLOCKSIZE];
    unsigned int totalInThread = 0;
    unsigned int tmpTotal;

    struct Point p1 = data[i];

    for (unsigned int j = i + 1; j < N; j++)
    {
        struct Point p2 = data[j];

        double xDiff = p1.x - p2.x;
        double yDiff = p1.y - p2.y;
        double distance = (xDiff * xDiff) + (yDiff * yDiff);

        // Increment thread total
        if (distance <= *epsilonSqr)
            totalInThread++;
    }

    // Put thread total into block total array
    totalInBlock[threadIdx.x] = totalInThread;

    __syncthreads();

    tmpTotal = 0;

    // Sum all block totals into final total
    if (threadIdx.x == 0)
    {
        for (unsigned int index = 0; index < BLOCKSIZE; index++)
            tmpTotal += totalInBlock[index];

        atomicAdd(total, tmpTotal);
    }

    return;
}

void generatePoints(struct Point *data)
{
    srand(time(0));

    for (unsigned int i = 0; i < N; i++)
    {
        data[i].x = 1000.0 * ((double)(rand()) / RAND_MAX);
        data[i].y = 1000.0 * ((double)(rand()) / RAND_MAX);
    }
}
