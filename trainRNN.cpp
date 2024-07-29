#include <stdint.h>
#include <limits>
#include <cmath>

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

struct Matrix
{
    float *data;
    int rows;
    int cols;

    Matrix(int r, int c) : rows(r), cols(c)
    {
        data = new float[rows * cols];
    }

    ~Matrix()
    {
        delete[] data;
    }

    float &operator()(int row, int col)
    {
        return data[row * cols + col];
    }

    const float &operator()(int row, int col) const
    {
        return data[row * cols + col];
    }

    void randomize(float mean, float std, uint32_t &randSeed)
    {
        for (int i = 0; i < rows * cols; ++i)
        {
            data[i] = randDist(mean, std, randSeed);
        }
    }
};

struct RNNLanguageModel
{
    int vocabSize;
    int hiddenDim;

    Matrix embedding; // For token->tok_emb and state->logits
    Matrix ih;        // for tok_emb->hidden1
    Matrix hh;        // for state->hidden2

    RNNLanguageModel(int _vocabSize, int _hiddenDim)
        : embedding(_vocabSize, _hiddenDim), ih(_hiddenDim, _hiddenDim), hh(_hiddenDim, _hiddenDim)
    {
        vocabSize = _vocabSize;
        hiddenDim = _hiddenDim;

        // Initialize matrix values
        uint32_t randSeed = 42;
        embedding.randomize(0.0f, 0.02f, randSeed);
        ih.randomize(0.0f, 0.02f, randSeed);
        hh.randomize(0.0f, 0.02f, randSeed);
    }
};

int main()
{
    int vocabSize = 4;
    int hiddenDim = 2;

    return 0;
}