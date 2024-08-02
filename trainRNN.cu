#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <fstream>
#include <stdint.h>
#include <limits>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <thread>
#include <algorithm>
#include <vector>
#include <string>

constexpr int MAX_THREADS_PER_BLOCK = 32;
constexpr float MAX_GRAD = 100.0f;           // This is the magnitude of what the gradient will be set to if we get -INF while getting loss (due to log(0)). This way we avoid -inf values messing up our grads/params
constexpr float PI = 3.14159265358979323846; // Mathematical constant pi. Used when generating random points from a normal distribution (randDist)

/**
 * Manipulates the input uint32_t so that the output is "random". It is not actually random: the same input will result in the same output, but given a certain input it is hard for an onlooker to predict the output without knowing the exact function.
 *
 * @param input A 32-bit unsigned integer
 * @return A new 32-bit unsigned integer
 */
static uint32_t PCG_Hash(uint32_t input)
{
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/**
 * Generates a random floating-point value in the range [0, 1]
 *
 * @param seed A reference to a 32-bit unsigned integer used as the seed for the PCG Hash. This is a reference so that we can change the seed within the function which means two subsequent calls to randFloat will return different values because the seed is changed without us having to manually update the seed.
 * @return A random floating-point value in the range [0, 1].
 */
float randFloat(uint32_t &seed)
{
    seed = PCG_Hash(seed);
    return (float)seed / (float)std::numeric_limits<uint32_t>::max();
}

/**
 * Generates a random floating-point value following a normal distribution
 *
 * @param mean A float, the mean of the normal distribution
 * @param std A float, the standard deviation of the normal distribution
 * @param seed A reference to a 32-bit unsigned integer used as the seed
 * @return A random floating-point value
 */
float randDist(const float mean, const float std, uint32_t &randSeed)
{
    // Generate two independent random numbers from a uniform distribution in the range (0,1)
    float u1 = randFloat(randSeed);
    float u2 = randFloat(randSeed);

    // Box-Muller transform to convert uniform random numbers to normal distribution
    float z0 = std * std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * PI * u2) + mean;

    // Return the generated random number
    return z0;
}

void clearLines(int numLines)
{
    for (int i = 0; i < numLines; i++)
    {
        std::cout << "\033[F\033[K";
    }
    std::flush(std::cout);
}

inline float absFloat(float value)
{
    return value < 0.0f ? -value : value;
}

struct Matrix
{
    float *data;
    int rows;
    int cols;
    int numValues;

    Matrix(int r, int c) : rows(r), cols(c), numValues(r * c)
    {
        // Create data
        float *tempData = new float[rows * cols];

        // Set data to 0
        for (int i = 0; i < numValues; ++i)
        {
            tempData[i] = 0.0f;
        }

        // Allocate memory on device
        cudaMalloc(&data, r * c * sizeof(float));

        // Copy data from host to device
        cudaMemcpy(data, tempData, r * c * sizeof(float), cudaMemcpyHostToDevice);

        // Delete temp data
        delete[] tempData;
    }

    ~Matrix()
    {
        cudaFree(data);
    }

    // Fills the matrix data with random values sampled from a normal distribution specified by mean and std
    void randomize(float mean, float std, uint32_t &randSeed)
    {
        // Create data
        float *tempData = new float[rows * cols];

        for (int i = 0; i < numValues; ++i)
        {
            tempData[i] = randDist(mean, std, randSeed);
        }

        // Copy data from host to device
        cudaMemcpy(data, tempData, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

        // Delete temp data
        delete[] tempData;
    }

    // Fills the matrix data with 0.0f
    void zeros()
    {
        cudaMemset(data, 0, numValues * sizeof(float));
    }

    // Fills the matrix data with 1.0f
    void ones()
    {
        cudaMemset(data, 1, numValues * sizeof(float));
    }

    // Gets the norm of the matrix
    float norm(cublasHandle_t &handle)
    {
        float normVal = 0.0f;
        cublasSnrm2(handle, numValues, data, 1, &normVal);
        return normVal;
    }

    // Gets the norm of a row in the matrix
    float norm(cublasHandle_t &handle, int rowNum)
    {
        float normVal = 0.0f;
        cublasSnrm2(handle, cols, data + rowNum * cols, 1, &normVal);
        return normVal;
    }

    void copy(Matrix &other)
    {
        cudaMemcpy(data, other.data, numValues * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // Prints the matrix to the terminal
    void print(std::string name, int rowNum = -1)
    {
        // Ensure kernels have finished executing
        cudaDeviceSynchronize();
        std::cout << std::fixed << std::setprecision(2);
        std::cout << name << ": (" << rows << ", " << cols << ")" << std::endl;
        float *tempValues = new float[numValues];

        cudaMemcpy(tempValues, data, numValues * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < rows; i++)
        {
            if (rowNum != -1 && rowNum != i)
            {
                continue;
            }
            for (int j = 0; j < cols; j++)
            {
                if (j > 0)
                {
                    std::cout << ", ";
                }
                std::cout << tempValues[i * cols + j];
            }
            std::cout << "\n";
        }

        delete[] tempValues;
    }

    // Checks if any values are nan
    int hasNan()
    {
        // Ensure kernels have finished executing
        cudaDeviceSynchronize();
        float *tempValues = new float[numValues];

        cudaMemcpy(tempValues, data, numValues * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < numValues; i++)
        {
            bool foundNan = std::isnan(tempValues[i]);
            if (foundNan)
            {
                delete[] tempValues;
                return i;
            }
        }

        delete[] tempValues;
        return -1;
    }
};

__global__ void preComputeKernel(float *hhVal0, float *hhVal1, float *hhVal2, float *hhVal3,
                                 float *ihVal0, const float *hh, const float *ih,
                                 const int hiddenDim)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < hiddenDim)
    {
        // Reset
        ihVal0[j] = 0.0f;

        // Compute norm and stateGradVal
        float normVal = 0.0f;
        float stateGradVal = 0.0f;

        for (int k = 0; k < hiddenDim; k++)
        {
            float hhValue = hh[j * hiddenDim + k];
            normVal += hhValue * hhValue;
            stateGradVal += hhValue;
            ihVal0[j] += ih[j * hiddenDim + k];
        }

        normVal = sqrt(normVal);
        float normVal2 = normVal * hiddenDim;

        // Compute val0, val1, val3
        hhVal0[j] = stateGradVal / normVal2;
        hhVal1[j] = 1.0f / normVal2;
        hhVal3[j] = 1.0f / (normVal2 * normVal2 * normVal);

        // Compute val2
        float val = stateGradVal * hiddenDim;
        for (int k = 0; k < hiddenDim; k++)
        {
            hhVal2[j * hiddenDim + k] = val * hh[j * hiddenDim + k];
        }
    }
}

__global__ void getdRKernel(int token, float *dR_dPCurrent, float *dR_dRPrev,
                            const float *activationGradVal, const float *hhVal0,
                            const float *hhVal1, const float *hhVal2, const float *hhVal3,
                            const float *inState, const float *ihVal0, const float *embedding,
                            int hiddenDim, int numParams, int hhIndex, int ihIndex,
                            int hiddenBiasIndex)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < hiddenDim && j < hiddenDim)
    {
        // activation
        const float gradVal = activationGradVal[i];

        // hiddenToHidden
        // Grad
        if (i == j)
        {
            dR_dRPrev[i * hiddenDim + i] = gradVal * hhVal0[i];
        }

        // embedding grad
        dR_dPCurrent[i * numParams + token * hiddenDim + j] = gradVal * ihVal0[j];

        // hh weight
        dR_dPCurrent[i * numParams + hhIndex + i * hiddenDim + j] =
            gradVal * (inState[i] * hhVal1[i] + inState[i] * hhVal3[i] * hhVal2[i * hiddenDim + j]);

        // inputToHidden
        dR_dPCurrent[i * numParams + ihIndex + j * hiddenDim + j] = gradVal * embedding[token * hiddenDim + j];

        if (j == 0)
        {
            // hiddenBias
            dR_dPCurrent[i * numParams + hiddenBiasIndex + i] = gradVal;
        }
    }
}

__global__ void getdLKernel(const int token, float *probs, float *dL_dY,
                            int vocabSize, int numParams)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vocabSize)
    {
        float x = probs[i]; // x = e^x / sumExp
        float gradVal;

        if (i == token)
        {
            // Derivative of the loss for the correct token
            gradVal = -1.0f / x + 1.0f;
        }
        else
        {
            // Derivative of the loss for an incorrect token
            gradVal = 1.0f / (1.0f - x) - 1.0f;
        }

        dL_dY[i] = gradVal * (x - x * x); // (e^x / sumExp) - (e^x * e^x) / (sumExp * sumExp)
    }
}

__global__ void computeActivationGradsKernel(float *activationGradVal, const float *newState, int hiddenDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hiddenDim)
    {
        float x = newState[idx] < 0.0f ? -newState[idx] : newState[idx];
        float term1 = x + 1.0f;
        float term2 = x * x + term1;
        activationGradVal[idx] = (x + term1) / (term2 * term2);
    }
}

// Kernel 2: Compute hhVal4
__global__ void computeHHVal4Kernel(float *hhVal4, const float *inState, const float *hhVal1, const float *hhVal2, const float *hhVal3, int hiddenDim)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < hiddenDim && k < hiddenDim)
    {
        const float term1 = inState[j] * hhVal1[j];
        const float term2 = inState[j] * hhVal3[j];
        hhVal4[j * hiddenDim + k] = term1 + term2 * hhVal2[j * hiddenDim + k];
    }
}

// Main kernel
__global__ void getdYCurrentKernel(int token, float *state, float *dY_dPCurrent, float *dY_dRPrev,
                                   const float *embedding, const float *activationGradVal, const float *hhVal0,
                                   const float *hhVal4, const float *ihVal0, int vocabSize, int hiddenDim,
                                   int numParams, int hhIndex, int ihIndex, int hiddenBiasIndex)
{
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    int k = blockIdx.z * blockDim.y + threadIdx.y;

    if (i < vocabSize && j < hiddenDim && k < hiddenDim)
    {
        float gradVal = embedding[i * hiddenDim + j];

        if (k == 0)
        {
            // logits
            dY_dPCurrent[i * numParams + i * hiddenDim + j] = state[j];

            // activation
            gradVal *= activationGradVal[j];

            // grad through hiddenToHidden
            dY_dRPrev[i * hiddenDim + j] = gradVal * hhVal0[j];

            // hiddenBias
            dY_dPCurrent[i * numParams + hiddenBiasIndex + j] = gradVal;

            // inputToHidden (accumulation)
            atomicAdd(&dY_dPCurrent[i * numParams + token * hiddenDim + j], ihVal0[j]);
        }

        // hh weight
        dY_dPCurrent[i * numParams + hhIndex + j * hiddenDim + k] = gradVal * hhVal4[j * hiddenDim + k];

        // inputToHidden
        dY_dPCurrent[i * numParams + ihIndex + j * hiddenDim + k] = gradVal * embedding[token * hiddenDim + j];
    }
}

__global__ void getdYdLogitsKernel(int token, float *state, float *dY_dPCurrent, float *dY_dRPrev, int vocabSize, int hiddenDim, int numParams)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < vocabSize && j < hiddenDim)
    {
        // Set embedding grad
        dY_dPCurrent[i * numParams + i * hiddenDim + j] = state[j];
    }
}

__global__ void activationKernel(const float *in, float *out, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        float x = in[index];
        float adjustedX = 2 * fabsf(x); // adjustedX = 2 * abs(x)
        float term1 = 1.0f + adjustedX + adjustedX * adjustedX * 0.5f;

        // Use conditional expression to select between positive and negative branches
        out[index] = copysignf((term1 - 1.0f) / (term1 + 1.0f), x);
    }
}

__global__ void vecMul(float *a, float *b, float *result, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        result[idx] = a[idx] * b[idx];
    }
}

struct RNNLanguageModel
{
    int vocabSize;
    int hiddenDim;
    int numParams;

    Matrix embedding;  // For token->tok_emb and state->logits
    Matrix ih;         // for tok_emb->hidden1
    Matrix hh;         // for state->hidden2
    Matrix hiddenBias; // For bias after (hidden1 + hidden2)

    int ihIndex;         // The index for the parameters of the ih matrix
    int hhIndex;         // The index for the parameters of the hh matrix
    int hiddenBiasIndex; // The index for the parameters of the hiddenBias matrix

    Matrix inState;  // input state from forward
    Matrix newState; // for intermediate hidden state

    // Matrices for Backpropagation Through Time
    Matrix dY_dRPrev;
    Matrix dY_dPCurrent;
    Matrix dL_dY;
    Matrix dL_dP;
    Matrix dR_dRPrev;
    Matrix dR_dPCurrent;
    Matrix delta;

    // Matrices for storing pre-computed values
    Matrix hhVal0;
    Matrix hhVal1;
    Matrix hhVal2;
    Matrix hhVal3;

    Matrix ihVal0;

    Matrix activationGradVal; // (1, hiddenDim), Used

    Matrix hhVal4; // (hiddenDim, hiddenDim), Used as a temp vector in getdY_dCurrent and hiddenToHidden

    RNNLanguageModel(int _vocabSize, int _hiddenDim)
        : embedding(_vocabSize, _hiddenDim),
          ih(_hiddenDim, _hiddenDim),
          hh(_hiddenDim, _hiddenDim),
          hiddenBias(1, _hiddenDim),

          inState(1, _hiddenDim),
          newState(1, _hiddenDim),

          dY_dRPrev(_vocabSize, _hiddenDim),
          dY_dPCurrent(_vocabSize, _vocabSize * _hiddenDim + 2 * _hiddenDim * _hiddenDim + _hiddenDim),
          dL_dY(1, _vocabSize),
          dL_dP(1, _vocabSize * _hiddenDim + 2 * _hiddenDim * _hiddenDim + _hiddenDim),
          dR_dRPrev(_hiddenDim, _hiddenDim),
          dR_dPCurrent(_hiddenDim, _vocabSize * _hiddenDim + 2 * _hiddenDim * _hiddenDim + _hiddenDim),
          delta(_hiddenDim, _vocabSize * _hiddenDim + 2 * _hiddenDim * _hiddenDim + _hiddenDim),

          hhVal0(1, _hiddenDim),
          hhVal1(1, _hiddenDim),
          hhVal2(_hiddenDim, _hiddenDim),
          hhVal3(1, _hiddenDim),

          ihVal0(1, _hiddenDim),

          activationGradVal(1, _hiddenDim),

          hhVal4(_hiddenDim, _hiddenDim)
    {
        vocabSize = _vocabSize;
        hiddenDim = _hiddenDim;
        numParams = _vocabSize * _hiddenDim + 2 * _hiddenDim * _hiddenDim + _hiddenDim;

        ihIndex = vocabSize * hiddenDim;
        hhIndex = ihIndex + hiddenDim * hiddenDim;
        hiddenBiasIndex = hhIndex + hiddenDim * hiddenDim;

        // Initialize matrix values
        uint32_t randSeed = 42;
        embedding.randomize(0.0f, 0.02f, randSeed);
        ih.randomize(0.0f, 0.02f, randSeed);
        hh.randomize(0.0f, 0.02f, randSeed);
        hiddenBias.randomize(0.0f, 0.02f, randSeed);
    }

    void preCompute(cublasHandle_t &handle)
    {
        int threadsPerBlock = std::min(MAX_THREADS_PER_BLOCK, hiddenDim);
        int blocksPerGrid = (hiddenDim + threadsPerBlock - 1) / threadsPerBlock;

        // Assuming all device pointers (d_hhVal0, d_hhVal1, etc.) are already allocated

        // Launch kernel
        preComputeKernel<<<blocksPerGrid, threadsPerBlock>>>(
            hhVal0.data, hhVal1.data, hhVal2.data, hhVal3.data, ihVal0.data, hh.data, ih.data, hiddenDim);

        // Check for errors
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        }
    }

    void getdR(int token)
    {
        dR_dPCurrent.zeros();

        int size = std::min(hiddenDim, 32);
        dim3 blockSize(size, size); // Adjust based on your GPU's capabilities
        dim3 gridSize((hiddenDim + size - 1) / size,
                      (hiddenDim + size - 1) / size);

        // Launch Kernel 1
        dim3 block1(std::min(hiddenDim, MAX_THREADS_PER_BLOCK));
        dim3 grid1((hiddenDim + block1.x - 1) / block1.x);
        computeActivationGradsKernel<<<grid1, block1>>>(activationGradVal.data, newState.data, hiddenDim);

        getdRKernel<<<gridSize, blockSize>>>(token, dR_dPCurrent.data, dR_dRPrev.data,
                                             activationGradVal.data, hhVal0.data, hhVal1.data, hhVal2.data, hhVal3.data,
                                             inState.data, ihVal0.data, embedding.data,
                                             hiddenDim, numParams, hhIndex, ihIndex, hiddenBiasIndex);
    }

    void getdL(const int token, Matrix &probs, cublasHandle_t handle)
    {
        int threadsPerBlock = std::min(MAX_THREADS_PER_BLOCK, vocabSize);
        int blocksPerGrid = (vocabSize + threadsPerBlock - 1) / threadsPerBlock;

        getdLKernel<<<blocksPerGrid, threadsPerBlock>>>(token, probs.data, dL_dY.data,
                                                        vocabSize, numParams);

        // Check for kernel launch errors
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "getdL_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        }

        // dL_dP += dL_dY @ dY_dP
        const float alpha = 1.0f;
        const float beta = 1.0f;

        cublasSgemv(handle, CUBLAS_OP_N,
                    numParams, vocabSize,
                    &alpha,
                    dY_dPCurrent.data, numParams,
                    dL_dY.data, 1,
                    &beta,
                    dL_dP.data, 1);
    }

    void getDelta(cublasHandle_t &handle)
    {
        // dR_dPCurrent += dR_dRPrev @ delta
        int M = hiddenDim;
        int K = hiddenDim;
        int N = numParams;

        float alpha = 1.0f;
        float beta = 1.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    delta.data, N,
                    dR_dRPrev.data, K,
                    &beta,
                    dR_dPCurrent.data, N);

        // delta = dR_dPCurrent
        delta.copy(dR_dPCurrent);
    }

    void getdY(cublasHandle_t &handle)
    {
        // dY_dPCurrent += dY_dRPrev @ delta
        int M = vocabSize;
        int K = hiddenDim;
        int N = numParams;

        float alpha = 1.0f;
        float beta = 1.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    delta.data, N,
                    dY_dRPrev.data, K,
                    &beta,
                    dY_dPCurrent.data, N);
    }

    void getdYCurrent(int token, Matrix &state)
    {
        dY_dPCurrent.zeros();

        // Launch Kernel 1
        dim3 block1(std::min(hiddenDim, MAX_THREADS_PER_BLOCK));
        dim3 grid1((hiddenDim + block1.x - 1) / block1.x);
        computeActivationGradsKernel<<<grid1, block1>>>(activationGradVal.data, newState.data, hiddenDim);

        // Launch Kernel 2
        int size = std::min(hiddenDim, 32);
        dim3 block2(size, size); // Adjust these values as needed
        dim3 grid2((hiddenDim + block2.x - 1) / block2.x, (hiddenDim + block2.y - 1) / block2.y);
        computeHHVal4Kernel<<<grid2, block2>>>(hhVal4.data, inState.data, hhVal1.data, hhVal2.data, hhVal3.data, hiddenDim);

        // Launch Main Kernel
        dim3 block3(std::min(hiddenDim, MAX_THREADS_PER_BLOCK), 1);
        dim3 grid3(vocabSize, (hiddenDim + block3.x - 1) / block3.x, hiddenDim);
        getdYCurrentKernel<<<grid3, block3>>>(token, state.data, dY_dPCurrent.data, dY_dRPrev.data, embedding.data,
                                              activationGradVal.data, hhVal0.data, hhVal4.data, ihVal0.data,
                                              vocabSize, hiddenDim, numParams, hhIndex, ihIndex, hiddenBiasIndex);
    }

    void getdYdLogits(int token, Matrix &state)
    {
        dY_dPCurrent.zeros();
        dY_dRPrev.zeros();

        dim3 blockSize(32, 32);
        dim3 gridSize((vocabSize + blockSize.x - 1) / blockSize.x, (hiddenDim + blockSize.y - 1) / blockSize.y);

        getdYdLogitsKernel<<<gridSize, blockSize>>>(token, state.data, dY_dPCurrent.data, dY_dRPrev.data, vocabSize, hiddenDim, numParams);
    }

    void updateParams(cublasHandle_t &handle)
    {
        const float alpha = -1.0f;

        // Update embedding
        cublasSaxpy(handle, vocabSize * hiddenDim, &alpha, dL_dP.data, 1, embedding.data, 1);

        // Update hh
        cublasSaxpy(handle, hiddenDim * hiddenDim, &alpha, dL_dP.data + hhIndex, 1, hh.data, 1);

        // Update ih
        cublasSaxpy(handle, hiddenDim * hiddenDim, &alpha, dL_dP.data + ihIndex, 1, ih.data, 1);

        // Update bias
        cublasSaxpy(handle, hiddenDim, &alpha, dL_dP.data + hiddenBiasIndex, 1, hiddenBias.data, 1);
    }

    void reset(cublasHandle_t &handle)
    {
        // Reset all values
        inState.zeros();
        newState.zeros();
        dY_dRPrev.zeros();
        dY_dPCurrent.zeros();
        dL_dY.zeros();
        dL_dP.zeros();
        dR_dRPrev.zeros();
        dR_dPCurrent.zeros();
        delta.zeros();

        // Pre-Compute values
        preCompute(handle);
    }

    void forward(Matrix &state, int token, cublasHandle_t &handle)
    {
        inState.copy(state);           // set inState
        inputToHidden(token, handle);  // do input transformation
        hiddenToHidden(state, handle); // do hidden transformation

        // add bias to newHidden
        const float alpha = 1.0f;
        cublasSaxpy(handle, hiddenDim, &alpha, hiddenBias.data, 1, newState.data, 1);

        // apply activation function to newState
        activation(newState, state);
    }

    // logits = embedding @ state
    void getLogits(Matrix &state, Matrix &logits, cublasHandle_t &handle)
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasSgemv(handle, CUBLAS_OP_N,
                    hiddenDim, vocabSize,
                    &alpha,
                    embedding.data, hiddenDim,
                    state.data, 1,
                    &beta,
                    logits.data, 1);
    }

    // newState = ih @ embedding[token]
    void inputToHidden(int token, cublasHandle_t &handle)
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasSgemv(handle, CUBLAS_OP_N,
                    hiddenDim, hiddenDim,
                    &alpha,
                    ih.data, hiddenDim,
                    embedding.data + token * hiddenDim, 1,
                    &beta,
                    newState.data, 1);
    }

    // newState += hh @ state
    void hiddenToHidden(Matrix &state, cublasHandle_t &handle)
    {
        // Perform element-wise multiplication of state and hhVal1
        // temp (hhVal4) = state * hhval1
        int blockSize = std::min(hiddenDim, 256);
        int numBlocks = (hiddenDim + blockSize - 1) / blockSize;
        vecMul<<<numBlocks, blockSize>>>(state.data, hhVal1.data, hhVal4.data, hiddenDim);

        // newState += hh @ temp
        const float alpha = 1.0f;
        const float beta = 1.0f;

        cublasSgemv(handle, CUBLAS_OP_N,
                    hiddenDim, hiddenDim,
                    &alpha,
                    hh.data, hiddenDim,
                    hhVal4.data, 1,
                    &beta,
                    newState.data, 1);
    }

    // Applies the activation function to the input matrix, storing the result in the output matrix
    // f(x) = 1 + x + (x^2 / 2)
    // out = (f(2 * in) - 1) / (f(2 * in) + 1)
    void activation(Matrix &in, Matrix &out)
    {
        int blockSize = std::min(MAX_THREADS_PER_BLOCK, hiddenDim); // Number of threads per block
        int numBlocks = (hiddenDim + blockSize - 1) / blockSize;

        // Launch the kernel
        activationKernel<<<numBlocks, blockSize>>>(in.data, out.data, hiddenDim);
    }
};

// Helper function to serialize a Matrix
void serializeMatrix(std::ofstream &out, const int rows, const int cols, const int numValues, const float *data)
{
    out.write(reinterpret_cast<const char *>(&rows), sizeof(int));
    out.write(reinterpret_cast<const char *>(&cols), sizeof(int));
    out.write(reinterpret_cast<const char *>(&numValues), sizeof(int));
    out.write(reinterpret_cast<const char *>(data), sizeof(float) * numValues);
}

// Serialize RNNLanguageModel to a file
void serializeRNNLanguageModel(const RNNLanguageModel &model, const std::string &filename)
{
    std::ofstream out(filename, std::ios::binary);
    if (!out)
    {
        std::cerr << "Error: Unable to open file for writing: " << filename << std::endl;
        return;
    }

    // Ensure kernels have finished executing
    cudaDeviceSynchronize();

    out.write(reinterpret_cast<const char *>(&model.vocabSize), sizeof(int));
    out.write(reinterpret_cast<const char *>(&model.hiddenDim), sizeof(int));

    int maxNumValues = std::max(model.embedding.numValues, model.ih.numValues);
    float *tempValues = new float[maxNumValues];

    cudaMemcpy(tempValues, model.embedding.data, model.embedding.numValues * sizeof(float), cudaMemcpyDeviceToHost);
    serializeMatrix(out, model.embedding.rows, model.embedding.cols, model.embedding.numValues, tempValues);

    cudaMemcpy(tempValues, model.ih.data, model.ih.numValues * sizeof(float), cudaMemcpyDeviceToHost);
    serializeMatrix(out, model.ih.rows, model.ih.cols, model.ih.numValues, tempValues);

    cudaMemcpy(tempValues, model.hh.data, model.hh.numValues * sizeof(float), cudaMemcpyDeviceToHost);
    serializeMatrix(out, model.hh.rows, model.hh.cols, model.hh.numValues, tempValues);

    cudaMemcpy(tempValues, model.hiddenBias.data, model.hiddenBias.numValues * sizeof(float), cudaMemcpyDeviceToHost);
    serializeMatrix(out, model.hiddenBias.rows, model.hiddenBias.cols, model.hiddenBias.numValues, tempValues);

    delete[] tempValues;

    out.close();
}

__global__ void getGradsKernel(float *grad, float *m, float *v, const float beta1, const float beta2, const float alpha, const float eps, const float beta1Minus, const float beta2Minus, const float mHatMul, const float vHatMul, const int nParams)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nParams)
    {
        // Compute m and mHat
        const float mVal = beta1 * m[i] + beta1Minus * grad[i];
        m[i] = mVal;
        const float mHatVal = mVal * mHatMul;

        // Compute v and vHat
        const float vVal = beta2 * v[i] + beta2Minus * grad[i] * grad[i];
        v[i] = vVal;
        const float vHatVal = vVal * vHatMul;

        // Compute new grad
        grad[i] = alpha * mHatVal / (sqrtf(vHatVal) + eps);
    }
}

struct AdamOptimizer
{
    int nParams;
    Matrix m;
    Matrix v;
    float alpha;      // the learning rate. good default value: 1e-2
    float beta1;      // good default value: 0.9
    float beta1Power; // used instead of calculating a std::pow every iteration
    float beta2;      // good default value: 0.999
    float beta2Power; // used instead of calculating a std::pow every iteration
    int t = 0;
    float eps;

    AdamOptimizer(int _nParams, float _alpha, float _beta1, float _beta2, float _eps) : m(1, _nParams),
                                                                                        v(1, _nParams)
    {
        nParams = _nParams;
        alpha = _alpha;
        beta1 = _beta1;
        beta2 = _beta2;
        beta1Power = beta1;
        beta2Power = beta2;
        eps = _eps;
    }

    void getGrads(Matrix &grad)
    {
        // Compute constants
        const float beta1Minus = 1.0f - beta1;
        const float beta2Minus = 1.0f - beta2;
        const float mHatMul = 1.0f / (1.0f - beta1Power);
        const float vHatMul = 1.0f / (1.0f - beta2Power);

        int blockSize = std::min(MAX_THREADS_PER_BLOCK, nParams);
        int numBlocks = (nParams + blockSize - 1) / blockSize;

        getGradsKernel<<<numBlocks, blockSize>>>(grad.data, m.data, v.data, beta1, beta2, alpha, eps, beta1Minus, beta2Minus, mHatMul, vHatMul, nParams);

        // Increase values
        beta1Power *= beta1;
        beta2Power *= beta2;
        t++;
    }
};

struct Page
{
    std::vector<uint8_t> title;
    std::vector<uint8_t> text;
    int titleSize;
    int textSize;
};

struct DataLoader
{
    std::ifstream file;
    bool fileOpen = false;
    uint32_t numTuples;
    uint32_t currentTuple = 0;

    DataLoader(const std::string &filename) : file(filename, std::ios::binary)
    {
        if (!file)
        {
            throw std::runtime_error("Unable to open file");
        }
        file.read(reinterpret_cast<char *>(&numTuples), sizeof(numTuples));
        fileOpen = true;
    }

    ~DataLoader()
    {
        if (fileOpen)
        {
            file.close();
        }
    }

    void readHeader()
    {
        if (!file)
        {
            throw std::runtime_error("No file is open");
        }
        file.read(reinterpret_cast<char *>(&numTuples), sizeof(numTuples));
        currentTuple = 0;
    }

    void openFile(const std::string &filename)
    {
        if (fileOpen)
        {
            file.close();
        }
        file.open(filename, std::ios::binary);
        readHeader();
    }

    bool nextPage(Page &page)
    {
        // If there are no more pages, return false
        if (currentTuple >= numTuples)
        {
            return false;
        }

        // Read the length of the title and page
        uint32_t len1, len2;
        file.read(reinterpret_cast<char *>(&len1), sizeof(len1));
        file.read(reinterpret_cast<char *>(&len2), sizeof(len2));

        // Set page title and text sizes
        page.titleSize = len1;
        page.textSize = len2;

        // Make sure the title and text vectors have enough room for the data
        if (page.title.size() < len1)
        {
            page.title.resize(len1);
        }
        if (page.text.size() < len2)
        {
            page.text.resize(len2);
        }

        // Read the data for the title and page
        file.read(reinterpret_cast<char *>(page.title.data()), len1);
        file.read(reinterpret_cast<char *>(page.text.data()), len2);

        return true; // We were able to load a page
    }
};

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

__global__ void softmaxKernel(float *logits, const int numValues, const float maxVal)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numValues)
    {
        logits[idx] = std::exp(logits[idx] - maxVal);
    }
}

// Applies softmax in place to the input logits
void softmax(Matrix &logits, cublasHandle_t &handle)
{
    // get max
    int maxValIndex;
    cublasIsamax(handle, logits.numValues, logits.data, 1, &maxValIndex);
    float maxVal;
    cudaMemcpy(&maxVal, logits.data + maxValIndex, sizeof(float), cudaMemcpyDeviceToHost);

    // Launch kernel to compute exponentials
    int threadsPerBlock = std::min(256, logits.numValues);
    int blocksPerGrid = (logits.numValues + threadsPerBlock - 1) / threadsPerBlock;
    softmaxKernel<<<blocksPerGrid, threadsPerBlock>>>(logits.data, logits.numValues, maxVal);

    // get sum
    float sumExp;
    cublasSasum(handle, logits.numValues, logits.data, 1, &sumExp);

    // divide by sum
    const float invSumExp = 1.0f / sumExp;
    cublasSscal(handle, logits.numValues, &invSumExp, logits.data, 1);
}

// Gets the loss of a prediction given the probability of the correct token
// x = the probability (a number: 0-1) of the token
// loss = -ln(x) + x - 1
float getLoss(Matrix &probs, int token)
{
    float x;
    cudaMemcpy(&x, probs.data + token, sizeof(float), cudaMemcpyDeviceToHost);
    float logVal = -log(x);
    if (logVal == INFINITY)
    {
        return MAX_GRAD;
    }
    else
    {
        return logVal + x - 1.0f;
    }
}

// Does one training step for a model.
// Does the forward pass and backward pass
// Gets the loss if bool train is true
// Does NOT update model parameters
void trainStep(int token, RNNLanguageModel &model, Matrix &state, Matrix &logits, bool train, int stepNum, cublasHandle_t &handle)
{
    if (train)
    {
        model.getLogits(state, logits, handle);

        if (stepNum == 0)
        {
            model.getdYdLogits(token, state); // if this is the first step, only the embedding matrix has been used so we only calculate grad for the embedding matrix
        }
        else
        {
            model.getdYCurrent(token, state);
        }

        model.getdY(handle);     // dY_dP = dY_dPCurrent + dY_dRPrev @ delta
        softmax(logits, handle); // logits -> probs
        model.getdL(token, logits, handle);
    }

    model.forward(state, token, handle);
    model.getdR(token);
    model.getDelta(handle); // delta = dR_dPCurrent + dR_dRPrev @ delta
}

int main()
{
    // Model parameters
    constexpr int vocabSize = 95;
    constexpr int hiddenDim = 256;

    // Learning parameters
    float learningRate = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    int numEpochs = 1;

    // Settings
    int numTokenFiles = 74;
    std::string dataFolder = "tokenData/";
    int logInterval = 1000;
    std::string savePath = "models/embedArticleCPP/model.bin";

    std::cout << "Initializing..." << std::endl;

    // Init model
    RNNLanguageModel model = RNNLanguageModel(vocabSize, hiddenDim);

    // Init optimizer
    AdamOptimizer optimizer = AdamOptimizer(model.numParams, learningRate, beta1, beta2, 1e-5);

    // Init state and logits
    Matrix state = Matrix(1, hiddenDim);
    Matrix logits = Matrix(1, vocabSize);

    // Init dataloader
    DataLoader dataLoader(dataFolder + "0.bin");
    Page page;

    // Init clock
    Clock clock;

    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    std::cout << "Vocab Size: " << vocabSize << std::endl;
    std::cout << "Hidden Size: " << hiddenDim << std::endl;
    std::cout << "# Embedding Params: " << vocabSize * hiddenDim << std::endl;
    std::cout << "# Input->Hidden Params: " << hiddenDim * hiddenDim << std::endl;
    std::cout << "# Hidden->Hidden Params: " << hiddenDim * hiddenDim << std::endl;
    std::cout << "# Hidden Bias Params: " << hiddenDim << std::endl;
    std::cout << "# Total Params: " << model.numParams << std::endl;

    std::cout << "\nTraining..." << std::endl;

    for (int i = 0; i < numEpochs; i++)
    {
        for (int j = 0; j < numTokenFiles; j++)
        {
            dataLoader.openFile(dataFolder + std::to_string(j) + ".bin");

            // Init trackers
            float loss = -1.0f;
            float gradNorm = -1.0f;
            int tokensPerSecond = -1.0f;

            for (int k = 0; k < dataLoader.numTuples; k++)
            {
                // Load page
                bool loadedPage = dataLoader.nextPage(page);

                if (!loadedPage)
                {
                    std::cout << "Failed to load page #" << k + 1 << " in file #" << j << "." << std::endl;
                    continue;
                }

                std::cout << "Epoch [" << i + 1 << "/" << numEpochs << "], ";
                std::cout << "File [" << j + 1 << "/" << numTokenFiles << "], ";
                std::cout << "Page [" << k + 1 << "/" << dataLoader.numTuples << "], ";
                std::cout << "Last Avg. Tok/Sec: " << tokensPerSecond << ", ";
                std::cout << "Last Loss: " << loss << ", ";
                std::cout << "Last Grad Norm: " << gradNorm << std::endl;

                // Reset
                state.zeros();
                model.reset(handle);
                clock.restart();

                // Get embedding
                std::cout << "Embedding" << std::endl;
                std::flush(std::cout);
                for (int l = 0; l < page.textSize; l++)
                {
                    trainStep(page.text[l], model, state, logits, false, l, handle);
                    if (l % logInterval == 0)
                    {
                        clearLines(1);
                        std::cout << "Embedding [" << l + 1 << "/" << page.textSize << "], " << (int)((float)l / clock.getElapsedTime()) << " tok/sec" << std::endl;
                        std::flush(std::cout);
                    }
                }
                clearLines(1);
                std::cout << "Embedding [" << page.textSize << "/" << page.textSize << "], " << (int)((float)page.textSize / clock.getElapsedTime()) << " tok/sec" << std::endl;
                std::flush(std::cout);

                // Reset clock
                const float timeElapsed = clock.getElapsedTime();
                clock.restart();

                // Init loss
                loss = 0.0f;

                // Evaluate
                std::cout << "Evaluating" << std::endl;
                std::flush(std::cout);
                for (int l = 0; l < page.textSize; l++)
                {
                    const uint8_t token = page.text[l];
                    trainStep(token, model, state, logits, true, l + page.textSize, handle);
                    loss += getLoss(logits, token);
                    if (l % logInterval == 0)
                    {
                        clearLines(1);
                        std::cout << "Evaluating [" << l + 1 << "/" << page.textSize << "], " << (int)((float)l / clock.getElapsedTime()) << " tok/sec" << std::endl;
                        std::flush(std::cout);
                    }
                }
                clearLines(1);
                std::cout << "Evaluating [" << page.textSize << "/" << page.textSize << "], " << (int)((float)page.textSize / clock.getElapsedTime()) << " tok/sec" << std::endl;
                std::flush(std::cout);

                // Update model parameters
                optimizer.getGrads(model.dL_dP);
                model.updateParams(handle);

                // Update trackers
                loss /= (float)page.textSize;
                gradNorm = model.dL_dP.norm(handle);
                tokensPerSecond = (int)((float)page.textSize / (clock.getElapsedTime() + timeElapsed));

                // Save model
                std::cout << "Saving model..." << std::endl;
                serializeRNNLanguageModel(model, savePath);

                // Clear text
                clearLines(4);
            }
        }
    }

    return 0;
}