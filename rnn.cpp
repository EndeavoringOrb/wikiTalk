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

#include "trainRNNStructs.hpp"

const float e = 2.71828182845904523536028747135266249775724709369995;

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
        for (int i = 0; i < nParams; i++)
        {
            // Compute m and mHat
            const float mVal = beta1 * m.data[i] + beta1Minus * grad.data[i];
            m.data[i] = mVal;
            const float mHatVal = mVal * mHatMul;

            // Compute v and vHat
            const float vVal = beta2 * v.data[i] + beta2Minus * grad.data[i] * grad.data[i];
            v.data[i] = vVal;
            const float vHatVal = vVal * vHatMul;

            // Compute new grad
            grad.data[i] = alpha * mHatVal / (sqrtf(vHatVal) + eps);
        }

        // Increase values
        beta1Power *= beta1;
        beta2Power *= beta2;
        t++;
    }
};

float dot(Matrix &vector, Matrix &matrix, int row)
{
    const float *a = &vector.data[0];
    const float *b = &matrix.data[row * matrix.cols];

    __m256 sum = _mm256_setzero_ps();

    int i = 0;
    for (; i <= matrix.cols - 8; i += 8)
    {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vb = _mm256_load_ps(b + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
    }

    float result[8];
    _mm256_storeu_ps(result, sum);
    const float dotVal = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7];

    return dotVal;
}

float dot(Matrix &matrix0, int row0, Matrix &matrix1, int row1)
{
    const int numCols = matrix0.cols;
    const float *a = &matrix0.data[row0 * numCols];
    const float *b = &matrix1.data[row1 * numCols];

    __m256 sum = _mm256_setzero_ps();

    int i = 0;
    for (; i <= numCols - 8; i += 8)
    {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vb = _mm256_load_ps(b + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
    }

    float result[8];
    _mm256_storeu_ps(result, sum);
    const float dotVal = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7];

    return dotVal;
}

// Multiplies row of matrix by mulVal and adds the result to outRow of out
void vecMulAdd(const float mulVal, Matrix &matrix, int row, Matrix &out, int outRow)
{
    // std::cout << "vecMulAdd" << std::endl;
    const int numCols = matrix.cols;
    const float *a = &matrix.data[row * numCols];
    float *b = &out.data[outRow * numCols];

    __m256 val = _mm256_set1_ps(mulVal);

    for (int i = 0; i <= numCols - 8; i += 8)
    {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vb = _mm256_load_ps(b + i);
        __m256 result = _mm256_fmadd_ps(va, val, vb);
        _mm256_store_ps(b + i, result);
    }
}

// assumes x is in the range (-inf, 0]
float fastExp(float x)
{
    int n = trunc(x) - 1;    // this will be off by 1 if x is exactly equal to an integer, but that's pretty unlikely and floor() is slow so who cares
    float remainder = x - n; // a number [0, 1)

    // val = e ^ n
    float val = 1.0f;
    for (int i = 0; i < -n; i++)
    {
        val *= e;
    }
    val = 1.0f / val;

    // approximate (e ^ remainder), then
    // val = val * (e ^ remainder)
    if (remainder < 0.5f)
    {
        if (remainder < 0.25f)
        {
            if (remainder < 0.125f)
            {
                return val * (1.06491075 * remainder + 0.99863083);
            }
            else
            {
                return val * (1.20670246 * remainder + 0.98075915);
            }
        }
        else
        {
            if (remainder < 0.375f)
            {
                return val * (1.36737327 * remainder + 0.94042401);
            }
            else
            {
                return val * (1.54943692 * remainder + 0.87196039);
            }
        }
    }
    else
    {
        if (remainder < 0.75f)
        {
            if (remainder < 0.875f)
            {
                return val * (1.75574186 * remainder + 0.76859292);
            }
            else
            {
                return val * (1.98951581 * remainder + 0.62224061);
            }
        }
        else
        {
            if (remainder < 0.625f)
            {
                return val * (2.25441616 * remainder + 0.42328938);
            }
            else
            {
                return val * (2.55458771 * remainder + 0.16032663);
            }
        }
    }
}

struct TrainRNN
{
    int vocabSize;
    int hiddenDim;
    int numParams;

    Matrix embedding;
    Matrix embeddingGrad;
    Matrix hh;
    Matrix hhGrad;
    Matrix ih;
    Matrix ihGrad;
    Matrix bias;
    Matrix biasGrad;

    Matrix embedded;
    Matrix embeddingGradMul;
    float maxYVal;

    Matrix hhMulVals;
    Matrix hhGradRowMul;
    Matrix scaledHH;

    Matrix states;
    Matrix activationGradMuls;

    Matrix Y;
    Matrix grad;
    Matrix tempGrad;
    Matrix inputGrad;

    TrainRNN(int _vocabSize, int _hiddenDim, int _longestSequence)
        : vocabSize(_vocabSize),
          hiddenDim(_hiddenDim),
          numParams(_vocabSize * _hiddenDim + 2 * _hiddenDim * _hiddenDim + _hiddenDim),
          embedding(_vocabSize, _hiddenDim),
          embeddingGrad(_vocabSize, _hiddenDim),
          embedded(_vocabSize, _hiddenDim),
          embeddingGradMul(_vocabSize, _hiddenDim),

          hh(_hiddenDim, _hiddenDim),
          hhGrad(_hiddenDim, _hiddenDim),
          scaledHH(_hiddenDim, _hiddenDim),
          hhMulVals(1, _hiddenDim),
          hhGradRowMul(1, _hiddenDim),

          ih(_hiddenDim, _hiddenDim),
          ihGrad(_hiddenDim, _hiddenDim),

          bias(1, _hiddenDim),
          biasGrad(1, _hiddenDim),

          grad(1, _hiddenDim),
          tempGrad(1, _hiddenDim),
          inputGrad(1, _hiddenDim),

          Y(1, _vocabSize),

          states(_longestSequence, _hiddenDim),
          activationGradMuls(_longestSequence, _hiddenDim)
    {
        uint32_t randSeed = 42;
        embedding.randomize(0.0f, 0.02f, randSeed);
        hh.randomize(0.0f, 0.02f, randSeed);
        ih.randomize(0.0f, 0.02f, randSeed);
        bias.randomize(0.0f, 0.02f, randSeed);
    }

    void preCompute()
    {
        // set embedded
        embedded.copy(embedding);
        for (int i = 0; i < embedded.numValues; i++)
        {
            activation(embedded.data[i]);
        }

        // set embeddingGradMul
        for (int i = 0; i < embeddingGradMul.numValues; i++)
        {
            const float x = embeddingGradMul.data[i];
            const float term1 = x + 1.0f;
            const float term2 = x * x + term1;
            embeddingGradMul.data[i] = (x + term1) / (term2 * term2);
        }

        // set embeddingRowSums
        maxYVal = -INFINITY;
        for (int i = 0; i < vocabSize; i++)
        {
            float sumVal = 0.0f;
            for (int j = 0; j < hiddenDim; j++)
            {
                sumVal += embedding.data[i * hiddenDim + j];
            }
            maxYVal = std::max(maxYVal, sumVal);
        }

        // set hhMulVals & hhGradRowMul
        for (int i = 0; i < hiddenDim; i++)
        {
            float normVal = 0.0f;
            for (int j = 0; j < hiddenDim; j++)
            {
                const float x = hh.data[i * hiddenDim + j];
                normVal += x * x;
            }
            normVal = sqrt(normVal);

            const float hhMulVal = 1.0f / (normVal * hiddenDim);
            hhMulVals.data[i] = hhMulVal;

            hhGradRowMul.data[i] = -(hhMulVal * hhMulVal) * (hiddenDim / (normVal + normVal));
        }

        // set scaledHH
        for (int i = 0; i < hiddenDim; i++)
        {
            for (int j = 0; j < hiddenDim; j++)
            {
                scaledHH.data[i * hiddenDim + j] = hh.data[i * hiddenDim + j] * hhMulVals.data[i];
            }
        }
    }

    void forwardStep(
        Matrix &state,
        const int tokenIndex,
        const uint8_t token)
    {
        states.copyRow(state, tokenIndex);

        for (int i = 0; i < hiddenDim; i++)
        {
            float stateVal = dot(state, scaledHH, i);       // hh
            stateVal += dot(embedded, token, ih, i);        // ih
            stateVal += bias.data[i];                       // bias
            activationGrad(stateVal, state, i, tokenIndex); // activation
        }
    }

    float backwardStep(
        const int tokenIndex,
        const int titleSize,
        const uint8_t token,
        const uint8_t prevToken)
    {
        float lossVal = 0.0f;

        if (tokenIndex >= titleSize)
        {
            for (int i = 0; i < vocabSize; i++)
            {
                Y.data[i] = dot(states, tokenIndex, embedding, i);
            }

            // softmax(Y);
            softmaxY();

            const float tokenProb = Y.data[token];
            lossVal = -log(tokenProb);

            Y.data[token] -= 1.0f;

            for (int i = 0; i < vocabSize; i++)
            {
                for (int j = 0; j < hiddenDim; j++)
                {
                    // embedding grad
                    embeddingGrad.data[i * hiddenDim + j] += Y.data[i] * states.data[tokenIndex * hiddenDim + j];

                    // grad += dL_dY through embedding
                    grad.data[j] += Y.data[i] * embedding.data[i * hiddenDim + j];
                }
            }
        }

        if (tokenIndex == 0)
        {
            return lossVal;
        }

        const int prevTokenIndex = tokenIndex - 1;

        inputGrad.zeros();
        tempGrad.zeros();

        for (int i = 0; i < hiddenDim; i++)
        {
            // grad through activation
            grad.data[i] *= activationGradMuls.data[prevTokenIndex * hiddenDim + i];

            // bias grad
            biasGrad.data[i] += grad.data[i];

            const float hhMulVal = grad.data[i] * hhGradRowMul.data[i];

            // ih grad
            // vecMulAdd(grad.data[i], embedded, prevToken * hiddenDim, ihGrad, i);

            // grad through ih
            // vecMulAdd(grad.data[i], ih, i, inputGrad, 0);

            // hh grad
            // vecMulAdd(hhMulVal, states, prevTokenIndex, hhGrad, i);

            // grad through hh
            // vecMulAdd(grad.data[i], scaledHH, i, tempGrad, 0);

            for (int j = 0; j < hiddenDim; j++)
            {
                // ih grad
                ihGrad.data[i * hiddenDim + j] += grad.data[i] * embedded.data[prevToken * hiddenDim + j];

                // grad through ih
                inputGrad.data[j] += grad.data[i] * ih.data[i * hiddenDim + j];

                // hh grad
                hhGrad.data[i * hiddenDim + j] += hhMulVal * states.data[prevTokenIndex * hiddenDim + j];

                // grad through hh
                tempGrad.data[j] += grad.data[i] * scaledHH.data[i * hiddenDim + j];
            }
        }

        for (int i = 0; i < hiddenDim; i++)
        {
            // embedding grad
            embeddingGrad.data[prevToken * hiddenDim + i] += inputGrad.data[i] * embeddingGradMul.data[prevToken * hiddenDim + i];

            // grad through hh
            grad.data[i] = tempGrad.data[i];
        }

        return lossVal;
    }

    void softmaxY()
    {
        // get sum and set vals
        float sumExp = 0.0f;
        for (int i = 0; i < vocabSize; i++)
        {
            const float val = std::exp(Y.data[i] - maxYVal);
            sumExp += val;
            Y.data[i] = val;
        }

        // divide by sum
        const float sumExpInv = 1.0f / sumExp;
        for (int i = 0; i < vocabSize; i++)
        {
            Y.data[i] *= sumExpInv;
        }
    }

    void activationGrad(const float x, Matrix &state, const int stateIndex, const int tokenIndex)
    {
        const float absX = absFloat(x);
        const float term1 = absX + 1.0f;
        const float term2 = absX * absX + term1;
        activationGradMuls.data[tokenIndex * hiddenDim + stateIndex] = (absX + term1) / (term2 * term2);
        state.data[stateIndex] = copysignf((term2 - 1) / (term2), x);
    }

    void activation(float &x)
    {
        const float absX = absFloat(x);
        const float term2 = absX * absX + absX + 1.0f;
        x = copysignf((term2 - 1) / (term2), x);
    }

    void reset()
    {
        embeddingGrad.zeros();
        hhGrad.zeros();
        ihGrad.zeros();
        biasGrad.zeros();
    }

    void updateParams()
    {
        for (int i = 0; i < embedding.numValues; i++)
        {
            embedding.data[i] -= embeddingGrad.data[i];
        }
        for (int i = 0; i < hh.numValues; i++)
        {
            hh.data[i] -= hhGrad.data[i];
        }
        for (int i = 0; i < ih.numValues; i++)
        {
            ih.data[i] -= ihGrad.data[i];
        }
        for (int i = 0; i < bias.numValues; i++)
        {
            bias.data[i] -= biasGrad.data[i];
        }
    }
};

// Serialize RNNLanguageModel to a file
void serializeTrainRNN(const TrainRNN &model, const std::string &filename)
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
    serializeMatrix(out, model.bias);

    out.close();
}

// Deserialize RNNLanguageModel from a file
void deserializeTrainRNN(const std::string &filename, TrainRNN &model)
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

    deserializeMatrix(in, model.embedding);
    deserializeMatrix(in, model.ih);
    deserializeMatrix(in, model.hh);
    deserializeMatrix(in, model.bias);

    in.close();
}

int main()
{
    // Model parameters
    constexpr int vocabSize = 96;
    constexpr int hiddenDim = 32;

    // Learning parameters
    float learningRate = 0.0001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    int numEpochs = 1;

    // Settings
    int numTokenFiles = 74;
    std::string dataFolder = "tokenData/";
    int logInterval = 1000;
    std::string savePath = "models/embedArticleCPP/model.bin";

    std::cout << "Initializing..." << std::endl;

    uint32_t longestSequence = findLongestSequence(dataFolder, numTokenFiles);
    // longestSequence = std::min((uint32_t)1000, longestSequence);

    TrainRNN model = TrainRNN(vocabSize, hiddenDim, longestSequence);
    AdamOptimizer embeddingOptimizer = AdamOptimizer(vocabSize * hiddenDim, learningRate, beta1, beta2, 1e-8f);
    AdamOptimizer hhOptimizer = AdamOptimizer(hiddenDim * hiddenDim, learningRate, beta1, beta2, 1e-8f);
    AdamOptimizer ihOptimizer = AdamOptimizer(hiddenDim * hiddenDim, learningRate, beta1, beta2, 1e-8f);
    AdamOptimizer biasOptimizer = AdamOptimizer(hiddenDim, learningRate, beta1, beta2, 1e-8f);

    Matrix state = Matrix(1, hiddenDim);

    // Init dataloader
    DataLoader dataLoader = DataLoader(dataFolder + "0.bin");
    Page page;

    // Init clock
    Clock clock;

    std::cout << "Vocab Size: " << vocabSize << std::endl;
    std::cout << "Hidden Size: " << hiddenDim << std::endl;
    std::cout << "# Embedding Params: " << vocabSize * hiddenDim << std::endl;
    std::cout << "# Input->Hidden Params: " << hiddenDim * hiddenDim << std::endl;
    std::cout << "# Hidden->Hidden Params: " << hiddenDim * hiddenDim << std::endl;
    std::cout << "# Hidden Bias Params: " << hiddenDim << std::endl;
    std::cout << "# Total Params: " << vocabSize * hiddenDim + 2 * hiddenDim * hiddenDim + hiddenDim << std::endl;

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
                model.reset();
                model.preCompute();
                state.zeros();
                clock.restart();

                // Forward
                std::cout << "Doing forward pass" << std::endl;
                for (int l = 0; l < page.titleSize; l++)
                {
                    model.forwardStep(state, l, page.title[l]);
                }
                model.forwardStep(state, page.titleSize, vocabSize - 1);
                for (int l = 0; l < page.textSize; l++)
                {
                    model.forwardStep(state, page.titleSize + 1 + l, page.text[l]);
                }

                loss = 0.0f;

                // Backward
                std::cout << "Doing backward pass" << std::endl;
                for (int l = page.titleSize + page.textSize; l > page.titleSize; l--)
                {
                    loss += model.backwardStep(l, page.titleSize, page.text[l - page.titleSize - 1], page.text[l - page.titleSize - 1]);
                }
                loss += model.backwardStep(page.titleSize, page.titleSize, vocabSize - 1, page.title[page.titleSize - 1]);
                for (int l = page.titleSize - 1; l > -1; l--)
                {
                    loss += model.backwardStep(l, page.titleSize, page.text[l], page.text[l - 1]);
                }

                // Reset clock
                const float timeElapsed = clock.getElapsedTime();

                // Update model parameters
                embeddingOptimizer.getGrads(model.embeddingGrad);
                hhOptimizer.getGrads(model.hhGrad);
                ihOptimizer.getGrads(model.ihGrad);
                biasOptimizer.getGrads(model.biasGrad);
                model.updateParams();

                // Update trackers
                loss /= (float)page.textSize;
                tokensPerSecond = (int)((float)page.textSize / (clock.getElapsedTime() + timeElapsed));

                // Save model
                serializeTrainRNN(model, savePath);

                // Clear text
                clearLines(3);
            }
        }
    }

    return 0;
}