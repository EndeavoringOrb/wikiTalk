#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <iostream>
#include <chrono>

#define M 1
#define K 2
#define N 4

static uint32_t PCG_Hash(uint32_t input)
{
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/**
 * Generates a random floating-point value in the range [0, 1) using PCG
 * (Permuted Congruential Generator) algorithm.
 *
 * @param seed A 32-bit unsigned integer used as the seed for the PCG algorithm.
 * @return A random floating-point value in the range [0, 1].
 */
float randFloat(uint32_t &seed)
{
    seed = PCG_Hash(seed);
    return (float)seed / (float)std::numeric_limits<uint32_t>::max();
}

void initValues(float *values, int numValues, uint32_t &seed)
{
    for (int i = 0; i < numValues; i++)
    {
        // values[i] = randFloat(seed);
        values[i] = i;
    }
}

struct Clock
{
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;

    Clock() : startTime(std::chrono::high_resolution_clock::now()) {}

    double getElapsedTime() const
    {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current_time - startTime;
        return elapsed.count();
    }

    void restart()
    {
        startTime = std::chrono::high_resolution_clock::now();
    }
};

int main()
{
    // Settings
    uint32_t seed = 42;

    std::cout << "Initializing..." << std::endl;

    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate memory on host
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    // Initialize matrices (you should fill these with your data)
    initValues(h_A, M * K, seed);
    initValues(h_B, K * N, seed);
    initValues(h_C, M * N, seed);

    std::cout << "A: (" << M << ", " << K << ")" << std::endl;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            if (j > 0)
            {
                std::cout << ", ";
            }
            std::cout << h_A[i * K + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "B: (" << K << ", " << N << ")" << std::endl;
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (j > 0)
            {
                std::cout << ", ";
            }
            std::cout << h_B[i * N + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "C: (" << M << ", " << N << ")" << std::endl;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (j > 0)
            {
                std::cout << ", ";
            }
            std::cout << h_C[i * N + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Allocate memory on device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Doing calculation..." << std::endl;
    Clock clock;

    // Perform matrix multiplication: C = A * B
    float alpha = 1.0f;
    float beta = 1.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    float calculationTime = clock.getElapsedTime();

    std::cout << std::endl;
    std::cout << "C: (" << M << ", " << N << ")" << std::endl;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (j > 0)
            {
                std::cout << ", ";
            }
            std::cout << h_C[i * N + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cublasDestroy(handle);

    std::cout << "Finished (" << M << ", " << K << ") @ (" << K << ", " << N << ") matrix multiplication in " << calculationTime << "s." << std::endl;

    return 0;
}