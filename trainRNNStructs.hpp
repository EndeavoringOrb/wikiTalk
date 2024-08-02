#ifndef RNN_STRUCTS
#define RNN_STRUCTS

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

#include <immintrin.h>
#include <omp.h>

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

float absFloat(float value)
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
        data = new float[rows * cols];
        zeros();
    }

    ~Matrix()
    {
        delete[] data;
    }

    // Fills the matrix data with random values sampled from a normal distribution specified by mean and std
    void randomize(float mean, float std, uint32_t &randSeed)
    {
        for (int i = 0; i < numValues; ++i)
        {
            data[i] = randDist(mean, std, randSeed);
        }
    }

    // Fills the matrix data with 0.0f
    void zeros()
    {
        for (int i = 0; i < numValues; ++i)
        {
            data[i] = 0.0f;
        }
    }

    // Fills the matrix data with 1.0f
    void ones()
    {
        for (int i = 0; i < numValues; ++i)
        {
            data[i] = 1.0f;
        }
    }

    // Gets the norm of the matrix
    float norm()
    {
        float value = 0.0f;
        for (int i = 0; i < numValues; i++)
        {
            float x = data[i];
            value += x * x;
        }
        return sqrt(value);
    }

    // Gets the norm of a row in the matrix
    float norm(int rowNum)
    {
        float value = 0.0f;
        for (int i = rowNum * cols; i < rowNum * cols + cols; i++)
        {
            float x = data[i];
            value += x * x;
        }
        return sqrt(value);
    }

    void copy(Matrix &other)
    {
        std::copy(other.data, other.data + numValues, data);
    }

    // Prints the matrix to the terminal
    void print(std::string name)
    {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << name << ": (" << rows << ", " << cols << ")" << std::endl;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (j > 0)
                {
                    std::cout << ", ";
                }
                std::cout << data[i * cols + j];
            }
            std::cout << "\n";
        }
    }
};

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

    Matrix activationGradVal;

    Matrix hhVal4;

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
          dR_dRPrev(1, _hiddenDim),
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

    void preCompute()
    {
        // Reset
        hhVal0.zeros();
        hhVal1.zeros();
        hhVal2.zeros();
        hhVal3.zeros();
        ihVal0.zeros();

        // Compute
        for (int j = 0; j < hiddenDim; j++)
        {
            // val0
            const float normVal = hh.norm(j);
            const float normVal2 = normVal * hiddenDim;
            float stateGradVal = 0.0f;
            for (int k = 0; k < hiddenDim; k++)
            {
                stateGradVal += hh.data[j * hiddenDim + k];
            }
            hhVal0.data[j] = stateGradVal / normVal2;

            // val1
            hhVal1.data[j] = 1.0f / normVal2;

            // val2
            const float val = stateGradVal * hiddenDim;
            for (int k = 0; k < hiddenDim; k++)
            {
                hhVal2.data[j * hiddenDim + k] = val * hh.data[j * hiddenDim + k];
            }

            // val3
            hhVal3.data[j] = 1.0f / (normVal2 * normVal2 * normVal);

            for (int k = 0; k < hiddenDim; k++)
            {
                ihVal0.data[j] += ih.data[j * hiddenDim + k];
            }
        }
    }

    void getdR(int token, Matrix &state)
    {
        //dR_dPCurrent.zeros();
        // set non-token indices of embedding to 0
        for (int i = 0; i < hiddenDim; i++) {
            for (int j = 0; j < token * hiddenDim; j++) {
                dR_dPCurrent.data[i * numParams + j] = 0.0f;
            }
            for (int j = token * hiddenDim + hiddenDim; j < vocabSize * hiddenDim; j++) {
                dR_dPCurrent.data[i * numParams + j] = 0.0f;
            }
        }

        for (int i = 0; i < hiddenDim; i++)
        {
            // activation
            const float x = absFloat(state.data[i]);
            const float term1 = x + 1.0f;
            const float term2 = x * x + term1;
            const float gradVal = (x + term1) / (term2 * term2); // grad = grad * ..., but because this is the first backProp step here, grad is 1 so we can just set grad to ...

            // hiddenToHidden
            // Grad
            dR_dRPrev.data[i] = gradVal * hhVal0.data[i];

            for (int j = 0; j < hiddenDim; j++)
            {
                // hh weight
                dR_dPCurrent.data[i * numParams + hhIndex + i * hiddenDim + j] = gradVal * (inState.data[i] * hhVal1.data[i] + inState.data[i] * hhVal3.data[i] * hhVal2.data[i * hiddenDim + j]); // Set hh grad

                // inputToHidden
                dR_dPCurrent.data[i * numParams + token * hiddenDim + j] = gradVal * ihVal0.data[i];                              // Accumulate grad into embedding after inputToHidden
                dR_dPCurrent.data[i * numParams + ihIndex + j * hiddenDim + j] = gradVal * embedding.data[token * hiddenDim + j]; // Set ih grad
            }

            // hiddenBias
            dR_dPCurrent.data[i * numParams + hiddenBiasIndex + i] = gradVal; // Set bias grad
        }
    }

    void getdL1(const int token, Matrix &probs)
    {
        // Get dL_dY
        for (int i = 0; i < vocabSize; i++)
        {
            const float x = probs.data[i]; // x = e^x / sumExp
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

            // softmax backward
            // d/dx (e^x * (1/sumExp))
            // d/dx (e^x) * (1/sumExp) + e^x * d/dx (1/sumExp)
            // e^x * (1/sumExp) - e^x * (1/(sumExp ** 2)) * d/dx (sumExp)
            // e^x * (1/sumExp) - e^x * (1/(sumExp ** 2)) * e^x
            // (e^x / sumExp) - (e^x * e^x) / (sumExp * sumExp)
            gradVal *= x - x * x; // (e^x / sumExp) - (e^x * e^x) / (sumExp * sumExp)

            // dL_dP += dL_dY @ dY_dP
            for (int j = 0; j < numParams; j++)
            {
                dL_dP.data[j] += gradVal * dY_dPCurrent.data[i * numParams + j];
            }
        }
    }

    void getdL2(const int token, Matrix &probs)
    {
        // Get dL_dY
        for (int i = 0; i < vocabSize; i++)
        {
            const float x = probs.data[i]; // x = e^x / sumExp
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

            // softmax backward
            // d/dx (e^x * (1/sumExp))
            // d/dx (e^x) * (1/sumExp) + e^x * d/dx (1/sumExp)
            // e^x * (1/sumExp) - e^x * (1/(sumExp ** 2)) * d/dx (sumExp)
            // e^x * (1/sumExp) - e^x * (1/(sumExp ** 2)) * e^x
            // (e^x / sumExp) - (e^x * e^x) / (sumExp * sumExp)
            gradVal *= x - x * x; // (e^x / sumExp) - (e^x * e^x) / (sumExp * sumExp)

            // dL_dP += dL_dY @ (dY_dPCurrent + dY_dRPrev @ delta)
            for (int j = 0; j < hiddenDim; j++)
            {
                for (int k = 0; k < numParams; k++)
                {
                    dL_dP.data[k] += gradVal * (dY_dPCurrent.data[i * numParams + k] + dY_dRPrev.data[i * hiddenDim + j] * delta.data[j * numParams + k]);
                }
            }
        }
    }

    // dL_dP = dL_dY @ (dY_dRPrev @ delta + dY_dPCurrent) + dL_dP
    void getdL(const int token, const Matrix &probs)
    {
        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 neg_one = _mm256_set1_ps(-1.0f);
        const __m256 tokVal = _mm256_set1_ps(token);

        for (int i = 0; i < vocabSize; i += 8)
        {
            __m256 x = _mm256_loadu_ps(&probs.data[i]);

            // 1/x if i == token else 1/(1-x)
            __m256 temp0 = _mm256_cmp_ps(tokVal, _mm256_set_ps(i + 7, i + 6, i + 5, i + 4, i + 3, i + 2, i + 1, i), _CMP_EQ_OQ); // mask
            __m256 gradVal = _mm256_blendv_ps(
                _mm256_div_ps(one, _mm256_sub_ps(one, x)),
                _mm256_div_ps(neg_one, x),
                temp0);

            // gradVal = gradVal * (x - x * x)
            temp0 = _mm256_mul_ps(x, x);
            gradVal = _mm256_mul_ps(gradVal, _mm256_sub_ps(x, temp0));

            // dL_dP = dL_dY @ (dY_dRPrev @ delta + dY_dPCurrent) + dL_dP
            for (int j = 0; j < hiddenDim; j++)
            {
                __m256 temp1 = _mm256_set1_ps(dY_dRPrev.data[i * hiddenDim + j]);

                for (int k = 0; k < numParams; k += 8)
                {
                    __m256 dY_dPCurrent_ik = _mm256_loadu_ps(&dY_dPCurrent.data[i * numParams + k]);
                    __m256 delta_jk = _mm256_loadu_ps(&delta.data[j * numParams + k]);

                    dY_dPCurrent_ik = _mm256_fmadd_ps(temp1, delta_jk, dY_dPCurrent_ik); // dY_dP = dY_dRPrev @ delta + dY_dPCurrent
                    __m256 temp2 = _mm256_loadu_ps(&dL_dP.data[k]);
                    temp2 = _mm256_fmadd_ps(gradVal, dY_dPCurrent_ik, temp2); // dL_dP = dL_dY @ dY_dP + dL_dP

                    _mm256_storeu_ps(&dL_dP.data[k], temp1);
                }
            }
        }
    }

    void getDeltaOLD()
    {
        for (int i = 0; i < hiddenDim; i++)
        {
            for (int j = 0; j < hiddenDim; j++)
            {
                for (int k = 0; k < numParams; k++)
                {
                    dR_dPCurrent.data[i * numParams + k] += dR_dRPrev.data[i * hiddenDim + j] * delta.data[j * numParams + k];
                }
            }
        }
        for (int i = 0; i < hiddenDim * numParams; i++)
        {
            delta.data[i] = dR_dPCurrent.data[i];
        }
    }

    // Specifically for this model, dR_dRPrev is always 0 for values not on the diagonal, so we can optimize it a bit
    void getDelta()
    {
        const int blockSize = 8; // AVX2 processes 8 floats at a time

        for (int i = 0; i < hiddenDim; i++)
        {
            __m256 dR_dRPrev_vec = _mm256_set1_ps(dR_dRPrev.data[i]);
            for (int j = 0; j < numParams; j += blockSize)
            {
                __m256 delta_vec = _mm256_loadu_ps(&delta.data[i * numParams + j]);
                __m256 dR_dPCurrent_vec = _mm256_loadu_ps(&dR_dPCurrent.data[i * numParams + j]);

                delta_vec = _mm256_fmadd_ps(dR_dRPrev_vec, delta_vec, dR_dPCurrent_vec);

                _mm256_storeu_ps(&delta.data[i * numParams + j], delta_vec);

                // delta.data[i * numParams + j] = dR_dPCurrent.data[i * numParams + j] + dR_dRPrev.data[i] * delta.data[i * numParams + j];
            }
        }
    }

    void getdYOLD()
    {
        for (int i = 0; i < vocabSize; i++)
        {
            for (int j = 0; j < hiddenDim; j++)
            {
                for (int k = 0; k < numParams; k++)
                {
                    dY_dPCurrent.data[i * numParams + k] += dY_dRPrev.data[i * hiddenDim + j] * delta.data[j * numParams + k];
                }
            }
        }
    }

    void getdY()
    {
        const int blockSize = 8; // AVX2 processes 8 floats at a time

        for (int i = 0; i < vocabSize; i++)
        {
            for (int j = 0; j < hiddenDim; j++)
            {
                __m256 dY_dRPrev_vec = _mm256_set1_ps(dY_dRPrev.data[i * hiddenDim + j]);

                for (int k = 0; k < numParams; k += blockSize)
                {
                    __m256 delta_vec = _mm256_loadu_ps(&delta.data[j * numParams + k]);
                    __m256 dY_dPCurrent_vec = _mm256_loadu_ps(&dY_dPCurrent.data[i * numParams + k]);

                    __m256 result = _mm256_fmadd_ps(dY_dRPrev_vec, delta_vec, dY_dPCurrent_vec);

                    _mm256_storeu_ps(&dY_dPCurrent.data[i * numParams + k], result);
                }

                // Handle remaining elements if numParams is not divisible by blockSize
                for (int k = numParams - (numParams % blockSize); k < numParams; k++)
                {
                    dY_dPCurrent.data[i * numParams + k] += dY_dRPrev.data[i * hiddenDim + j] * delta.data[j * numParams + k];
                }
            }
        }
    }

    void getdYCurrent(int token, Matrix &state)
    {
        // dY_dPCurrent.zeros();
        // dont need to set embedding grad to zero, it will be overwritten
        // dont need to set hh grad to zero, it will be overwritten
        // dont need to set ih grad to zero, it will be overwritten
        // dont need to set bias grad either

        for (int j = 0; j < hiddenDim; j++)
        {
            // activationGradVal
            const float x = absFloat(newState.data[j]);
            const float term1 = x + 1.0f;
            const float term2 = x * x + term1;
            activationGradVal.data[j] = (x + term1) / (term2 * term2); // grad = grad * ..., but because this is the first backProp step here, grad is 1 so we can just set grad to ...

            // hhVal4
            const float term3 = inState.data[j] * hhVal1.data[j];
            const float term4 = inState.data[j] * hhVal3.data[j];
            for (int k = 0; k < hiddenDim; k++)
            {
                hhVal4.data[j * hiddenDim + k] = term3 + term4 * hhVal2.data[j * hiddenDim + k];
            }
        }

        for (int i = 0; i < vocabSize; i++)
        {
            for (int j = 0; j < hiddenDim; j++)
            {
                // logits
                float gradVal = embedding.data[i * hiddenDim + j];                    // Set grad after logits projection
                dY_dPCurrent.data[i * numParams + i * hiddenDim + j] = state.data[j]; // Set embedding grad

                // activation
                gradVal *= activationGradVal.data[j];

                // grad through hiddenToHidden
                dY_dRPrev.data[i * hiddenDim + j] = gradVal * hhVal0.data[j];

                for (int k = 0; k < hiddenDim; k++)
                {
                    // hh weight
                    dY_dPCurrent.data[i * numParams + hhIndex + j * hiddenDim + k] = gradVal * hhVal4.data[j * hiddenDim + k]; // Set hh grad

                    // inputToHidden
                    dY_dPCurrent.data[i * numParams + ihIndex + j * hiddenDim + k] = gradVal * embedding.data[token * hiddenDim + j]; // Set ih grad
                }

                // inputToHidden
                dY_dPCurrent.data[i * numParams + token * hiddenDim + j] = gradVal * ihVal0.data[j]; // Accumulate grad into embedding after inputToHidden

                // hiddenBias
                dY_dPCurrent.data[i * numParams + hiddenBiasIndex + j] = gradVal; // Set bias grad
            }
        }
    }

    void getdYdLogits(int token, Matrix &state)
    {
        dY_dPCurrent.zeros();
        dY_dRPrev.zeros();
        for (int i = 0; i < vocabSize; i++)
        {
            for (int j = 0; j < hiddenDim; j++)
            {
                // logits
                dY_dPCurrent.data[i * numParams + i * hiddenDim + j] = state.data[j]; // Set embedding grad
            }
        }
    }

    void updateParams()
    {
        // Update embedding
        for (int i = 0; i < vocabSize * hiddenDim; i++)
        {
            embedding.data[i] -= dL_dP.data[i];
        }

        // Update hh
        for (int i = 0; i < hiddenDim * hiddenDim; i++)
        {
            hh.data[i] -= dL_dP.data[i + hhIndex];
        }

        // Update ih
        for (int i = 0; i < hiddenDim * hiddenDim; i++)
        {
            ih.data[i] -= dL_dP.data[i + ihIndex];
        }

        // Update bias
        for (int i = 0; i < hiddenDim; i++)
        {
            hiddenBias.data[i] -= dL_dP.data[i + hiddenBiasIndex];
        }
    }

    void reset()
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
        preCompute();
    }

    void forward(Matrix &state, int token)
    {
        newState.zeros();      // reset newState
        inState.copy(state);   // set inState
        inputToHidden(token);  // do input transformation
        hiddenToHidden(state); // do hidden transformation

        // add bias to newHidden
        for (int i = 0; i < hiddenDim; i++)
        {
            newState.data[i] += hiddenBias.data[i];
        }

        // apply activation function to newState
        activation(newState, state);
    }

    // logits = state @ embedding.T
    void getLogits(Matrix &state, Matrix &logits)
    {
        logits.zeros();
        for (int i = 0; i < vocabSize; i++)
        {
            for (int j = 0; j < hiddenDim; j++)
            {
                logits.data[i] += state.data[j] * embedding.data[i * hiddenDim + j];
            }
        }
    }

    // newState = ih @ embedding[token]
    void inputToHidden(int token)
    {
        for (int i = 0; i < hiddenDim; i++)
        {
            for (int j = 0; j < hiddenDim; j++)
            {
                newState.data[j] += embedding.data[hiddenDim * token + i] * ih.data[i * hiddenDim + j];
            }
        }
    }

    // newState += hh @ state
    void hiddenToHidden(Matrix &state)
    {
        for (int i = 0; i < hiddenDim; i++)
        {
            const float mulVal = state.data[i] * hhVal1.data[i];
            for (int j = 0; j < hiddenDim; j++)
            {
                newState.data[j] += mulVal * hh.data[i * hiddenDim + j];
            }
        }
    }

    // Applies the activation function to the input matrix, storing the result in the output matrix
    // f(x) = 1 + x + (x^2 / 2)
    // out = (f(2 * in) - 1) / (f(2 * in) + 1)
    void activation(Matrix &in, Matrix &out)
    {
        for (int i = 0; i < hiddenDim; i++)
        {
            const float x = in.data[i]; // Get x
            const float adjustedX = 2 * fabsf(x);
            const float term1 = (1.0f + adjustedX + adjustedX * adjustedX * 0.5f);
            out.data[i] = copysignf((term1 - 1.0f) / (term1 + 1.0f), x);
        }
    }
};

// Helper function to serialize a Matrix
void serializeMatrix(std::ofstream &out, const Matrix &matrix)
{
    out.write(reinterpret_cast<const char *>(&matrix.rows), sizeof(int));
    out.write(reinterpret_cast<const char *>(&matrix.cols), sizeof(int));
    out.write(reinterpret_cast<const char *>(&matrix.numValues), sizeof(int));
    out.write(reinterpret_cast<const char *>(matrix.data), sizeof(float) * matrix.numValues);
}

// Helper function to deserialize a Matrix
void deserializeMatrix(std::ifstream &in, Matrix &matrix)
{
    in.read(reinterpret_cast<char *>(&matrix.rows), sizeof(int));
    in.read(reinterpret_cast<char *>(&matrix.cols), sizeof(int));
    in.read(reinterpret_cast<char *>(&matrix.numValues), sizeof(int));

    matrix.data = new float[matrix.numValues];
    in.read(reinterpret_cast<char *>(matrix.data), sizeof(float) * matrix.numValues);
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

    out.write(reinterpret_cast<const char *>(&model.vocabSize), sizeof(int));
    out.write(reinterpret_cast<const char *>(&model.hiddenDim), sizeof(int));

    serializeMatrix(out, model.embedding);
    serializeMatrix(out, model.ih);
    serializeMatrix(out, model.hh);
    serializeMatrix(out, model.hiddenBias);

    out.close();
}

// Deserialize RNNLanguageModel from a file
RNNLanguageModel deserializeRNNLanguageModel(const std::string &filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in)
    {
        std::cerr << "Error: Unable to open file for reading: " << filename << std::endl;
        exit(0);
    }

    int vocabSize;
    int hiddenDim;

    in.read(reinterpret_cast<char *>(&vocabSize), sizeof(int));
    in.read(reinterpret_cast<char *>(&hiddenDim), sizeof(int));

    RNNLanguageModel model = RNNLanguageModel(vocabSize, hiddenDim);

    deserializeMatrix(in, model.embedding);
    deserializeMatrix(in, model.ih);
    deserializeMatrix(in, model.hh);
    deserializeMatrix(in, model.hiddenBias);

    in.close();
    return model;
}

// Applies softmax in place to the input logits
void softmax(Matrix &logits)
{
    // get max
    float maxVal = -INFINITY;
    for (int i = 0; i < logits.numValues; i++)
    {
        if (logits.data[i] > maxVal)
        {
            maxVal = logits.data[i];
        }
    }

    // get sum and set vals
    float sumExp = 0.0f;
    for (int i = 0; i < logits.numValues; i++)
    {
        const float val = std::exp(logits.data[i] - maxVal);
        sumExp += val;
        logits.data[i] = val;
    }

    // divide by sum
    const float sumExpInv = 1.0f / sumExp;
    for (int i = 0; i < logits.numValues; i++)
    {
        logits.data[i] *= sumExpInv;
    }
}

// Gets the loss of a prediction given the probability of the correct token
// x = the probability (a number: 0-1) of the token
// loss = -ln(x) + x - 1
float getLoss(Matrix &probs, int token)
{
    const float x = probs.data[token];
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

#endif