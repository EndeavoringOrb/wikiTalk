#include <iostream>
#include <stdint.h>
#include <limits>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <thread>

constexpr bool DEBUG_PRINT = true;
constexpr bool CLIP_GRADS = true;

constexpr float PI = 3.14159265358979323846;

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
        for (int i = 0; i < numValues; i++)
        {
            data[i] = other.data[i];
        }
    }

    // Prints the matrix to the terminal
    void print(std::string name)
    {
        if (DEBUG_PRINT)
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

    Matrix dY_dRPrev;
    Matrix dY_dPCurrent;
    Matrix dL_dY;
    Matrix dL_dP;
    Matrix dR_dRPrev;
    Matrix dR_dPCurrent;
    Matrix delta;

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
          delta(_hiddenDim, _vocabSize * _hiddenDim + 2 * _hiddenDim * _hiddenDim + _hiddenDim)
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

    void getdR(int token, Matrix &state)
    {
        dR_dPCurrent.zeros();
        dR_dRPrev.zeros();
        for (int i = 0; i < hiddenDim; i++)
        {
            // activation
            const float x = absFloat(state.data[i]);
            const float term1 = x * x + x + x + 4.0f;           // x^2 + 2x + 4
            float gradVal = (8 * (x + 1.0f)) / (term1 * term1); // grad = grad * ..., but because this is the first backProp step here, grad is 1 so we can just set grad to ...

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

    void getdL(int token, Matrix &logits)
    {
        for (int i = 0; i < vocabSize; i++)
        {
            for (int j = 0; j < numParams; j++)
            {
                dL_dP.data[j] += ((i == token) ? -1 : 1) * dY_dPCurrent.data[i * numParams + j];
            }
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
        for (int i = 0; i < vocabSize; i++)
        {
            for (int j = 0; j < hiddenDim; j++)
            {
                // logits
                float gradVal = embedding.data[i * hiddenDim + j];                    // Set grad after logits projection
                dY_dPCurrent.data[i * numParams + i * hiddenDim + j] = state.data[j]; // Set embedding grad

                // activation
                const float x = absFloat(newState.data[j]);
                const float term1 = x * x + x + x + 4.0f; // x^2 + 2x + 4
                gradVal *= (8 * (x + 1.0f)) / (term1 * term1);

                // hiddenToHidden
                const float normVal = hh.norm(j);
                const float normVal2 = normVal * hiddenDim;
                // Grad
                float stateGradVal = 0.0f;
                for (int k = 0; k < hiddenDim; k++)
                {
                    stateGradVal += hh.data[j * hiddenDim + k];
                }
                dY_dRPrev.data[i * hiddenDim + j] = gradVal * stateGradVal / normVal2;
                // hh weight
                for (int k = 0; k < hiddenDim; k++)
                {
                    dY_dPCurrent.data[i * numParams + hhIndex + j * hiddenDim + k] = gradVal * (inState.data[j] / normVal2 + (inState.data[j] * stateGradVal * hiddenDim * hh.data[j * hiddenDim + k]) / (normVal2 * normVal2 * normVal)); // Set hh grad
                }

                // inputToHidden
                for (int k = 0; k < hiddenDim; k++)
                {
                    dY_dPCurrent.data[i * numParams + token * hiddenDim + j] += ih.data[j * hiddenDim + k];                           // Accumulate grad into embedding after inputToHidden
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

    void updateParams(const float learningRate)
    {
        // Update embedding
        for (int i = 0; i < vocabSize * hiddenDim; i++)
        {
            embedding.data[i] -= learningRate * dL_dP.data[i];
        }

        // Update hh
        for (int i = 0; i < hiddenDim * hiddenDim; i++)
        {
            hh.data[i] -= learningRate * dL_dP.data[i + hhIndex];
        }

        // Update ih
        for (int i = 0; i < hiddenDim * hiddenDim; i++)
        {
            ih.data[i] -= learningRate * dL_dP.data[i + ihIndex];
        }

        // Update bias
        for (int i = 0; i < hiddenDim; i++)
        {
            hiddenBias.data[i] -= learningRate * dL_dP.data[i + hiddenBiasIndex];
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

void trainStep(int token, RNNLanguageModel &model, Matrix &state, Matrix &logits, bool train, int stepNum)
{
    if (train)
    {
        model.getLogits(state, logits);
        logits.print("logits");
        if (stepNum == 0)
        {
            model.getdYdLogits(token, state); // if this is the first step, only the embedding matrix has been used so we only calculate grad for the embedding matrix
        }
        else
        {
            model.getdYCurrent(token, state);
        }

        model.getdY(); // dY_dP = dY_dPCurrent + dY_dRPrev @ delta
        model.getdL(token, logits);
        model.dL_dP.print("dL_dP");
    }

    model.forward(state, token);
    model.getdR(token, state);
    model.dR_dRPrev.print("dR_dRPrev");
    model.getDelta(); // delta = dR_dPCurrent + dR_dRPrev @ delta
    clearLines(7);
}

int main()
{
    int vocabSize = 4;
    int hiddenDim = 2;
    float learningRate = 0.1f;

    std::cout << "Initializing..." << std::endl;
    RNNLanguageModel model = RNNLanguageModel(vocabSize, hiddenDim);

    Matrix state = Matrix(1, hiddenDim);
    Matrix logits = Matrix(1, vocabSize);

    std::cout << "Training..." << std::endl;
    std::cout << logits.data[0] << std::endl;

    for (int i = 0; i < 100; i++)
    {
        state.zeros();
        model.reset();
        for (int j = 0; j < 10; j++)
        {
            trainStep(0, model, state, logits, true, j);
            clearLines(1);
            std::cout << logits.data[0] << std::endl;
        }

        model.updateParams(learningRate);
    }

    return 0;
}