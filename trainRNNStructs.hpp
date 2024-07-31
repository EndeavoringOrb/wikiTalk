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

    // Gets the norm of a row in the matrix
    float norm(int rowNum)
    {
        float value;
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
          activationGradVal(1, _hiddenDim)
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
        dR_dPCurrent.zeros();
        dR_dRPrev.zeros();
        for (int i = 0; i < hiddenDim; i++)
        {
            // activation
            const float x = absFloat(state.data[i]);
            const float term1 = x + 1.0f;
            const float term2 = x * x + term1;
            float gradVal = (x + term1) / (term2 * term2); // grad = grad * ..., but because this is the first backProp step here, grad is 1 so we can just set grad to ...

            // hiddenToHidden
            const float normVal = hh.norm(i);
            const float normVal2 = normVal * hiddenDim;
            // Grad
            float stateGradVal = 0.0f;
            for (int j = 0; j < hiddenDim; j++)
            {
                stateGradVal += hh.data[i * hiddenDim + j];
            }
            dR_dRPrev.data[i * hiddenDim + i] = gradVal * stateGradVal / normVal2;
            // hh weight
            for (int j = 0; j < hiddenDim; j++)
            {
                dR_dPCurrent.data[i * numParams + hhIndex + i * hiddenDim + j] = gradVal * (inState.data[i] / normVal2 + (inState.data[i] * stateGradVal * hiddenDim * hh.data[i * hiddenDim + j]) / (normVal2 * normVal2 * normVal)); // Set hh grad
            }

            // inputToHidden
            for (int j = 0; j < hiddenDim; j++)
            {
                dR_dPCurrent.data[i * numParams + token * hiddenDim + i] += ih.data[i * hiddenDim + j];                           // Accumulate grad into embedding after inputToHidden
                dR_dPCurrent.data[i * numParams + ihIndex + j * hiddenDim + j] = gradVal * embedding.data[token * hiddenDim + j]; // Set ih grad
            }

            // hiddenBias
            dR_dPCurrent.data[i * numParams + hiddenBiasIndex + i] = gradVal; // Set bias grad
        }
    }

    void getdL(const int token, Matrix &probs)
    {
        // Get dL_dY
        for (int i = 0; i < vocabSize; i++)
        {
            if (i == token)
            {
                // Derivative of the loss for the correct token
                dL_dY.data[i] = -1.0f / probs.data[i] + 1.0f;
            }
            else
            {
                // Derivative of the loss for an incorrect token
                dL_dY.data[i] = 1.0f / (1.0f - probs.data[i]) - 1.0f;
            }
        }

        // softmax backward
        // d/dx (e^x * (1/sumExp))
        // d/dx (e^x) * (1/sumExp) + e^x * d/dx (1/sumExp)
        // e^x * (1/sumExp) - e^x * (1/(sumExp ** 2)) * d/dx (sumExp)
        // e^x * (1/sumExp) - e^x * (1/(sumExp ** 2)) * e^x
        // (e^x / sumExp) - (e^x * e^x) / (sumExp * sumExp)
        for (int i = 0; i < vocabSize; i++)
        {
            const float x = probs.data[i]; // x = e^x / sumExp
            dL_dY.data[i] *= x - x * x;    // (e^x / sumExp) - (e^x * e^x) / (sumExp * sumExp)
        }

        // dL_dP += dL_dY @ dY_dP
        for (int i = 0; i < token; i++)
        {
            for (int j = 0; j < numParams; j++)
            {
                dL_dP.data[j] += dL_dY.data[i] * dY_dPCurrent.data[i * numParams + j];
            }
        }

        for (int i = token + 1; i < vocabSize; i++)
        {
            for (int j = 0; j < numParams; j++)
            {
                dL_dP.data[j] += dL_dY.data[i] * dY_dPCurrent.data[i * numParams + j];
            }
        }

        for (int j = 0; j < numParams; j++)
        {
            dL_dP.data[j] += dL_dY.data[token] * dY_dPCurrent.data[token * numParams + j];
        }
    }

    void getDelta()
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

    void getdY()
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

    void getdYCurrent(int token, Matrix &state)
    {
        dY_dPCurrent.zeros();
        dY_dRPrev.zeros();

        for (int j = 0; j < hiddenDim; j++)
        {
            const float x = absFloat(newState.data[j]);
            const float term1 = x + 1.0f;
            const float term2 = x * x + term1;
            activationGradVal.data[j] = (x + term1) / (term2 * term2); // grad = grad * ..., but because this is the first backProp step here, grad is 1 so we can just set grad to ...
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

                // hiddenToHidden
                // grad
                dY_dRPrev.data[i * hiddenDim + j] = gradVal * hhVal0.data[j];
                // hh weight
                for (int k = 0; k < hiddenDim; k++)
                {
                    dY_dPCurrent.data[i * numParams + hhIndex + j * hiddenDim + k] = gradVal * (inState.data[j] * hhVal1.data[j] + (inState.data[j] * hhVal2.data[j * hiddenDim + k]) * hhVal3.data[j]); // Set hh grad
                }

                // inputToHidden
                dY_dPCurrent.data[i * numParams + token * hiddenDim + j] = ihVal0.data[j]; // Accumulate grad into embedding after inputToHidden
                for (int k = 0; k < hiddenDim; k++)
                {
                    dY_dPCurrent.data[i * numParams + ihIndex + j * hiddenDim + k] = gradVal * embedding.data[token * hiddenDim + j]; // Set ih grad
                }

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

    // newState += embedding[token] @ ih
    void inputToHidden(int token)
    {
        for (int i = 0; i < hiddenDim; i++)
        {
            for (int j = 0; j < hiddenDim; j++)
            {
                newState.data[i] += embedding.data[hiddenDim * token + i] * ih.data[i * hiddenDim + j];
            }
        }
    }

    // newState += state @ hh
    void hiddenToHidden(Matrix &state)
    {
        for (int i = 0; i < hiddenDim; i++)
        {
            float rowSum = 0.0f;
            for (int j = 0; j < hiddenDim; j++)
            {
                rowSum += hh.data[i * hiddenDim + j];
            }
            // (state * rowSum) / (hh.norm(i) * hiddenDim)

            // STATE DERIVATIVE
            // d/dstate (state * rowSum) * (1 / (hh.norm(i) * hiddenDim)) + (state * rowSum) * d/dstate (1 / (hh.norm(i) * hiddenDim))
            // rowSum / (hh.norm(i) * hiddenDim) + 0

            // HH DERIVATIVE
            // d/dhh (state * rowSum) * (1 / (hh.norm(i) * hiddenDim)) + (state * rowSum) * d/dhh (1 / (hh.norm(i) * hiddenDim))
            // state / (hh.norm(i) * hiddenDim) + (state * rowSum * hiddenDim * hh[i, j]) / ((hh.norm(i) * hiddenDim) ** 2 * hh.norm(i))
            newState.data[i] += (state.data[i] * rowSum) / (hh.norm(i) * hiddenDim);
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

            if (x < 0.0f)
            {
                const float adjustedX = -x - x; // adjustedX = 2 * abs(x)
                const float term1 = (1.0f + adjustedX + adjustedX * adjustedX * 0.5f);
                out.data[i] = -(term1 - 1.0f) / (term1 + 1.0f);
            }
            else
            {
                const float adjustedX = x + x; // adjustedX = 2 * abs(x)
                const float term1 = 1.0f + adjustedX + adjustedX * adjustedX * 0.5f;
                out.data[i] = (term1 - 1.0f) / (term1 + 1.0f);
            }
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