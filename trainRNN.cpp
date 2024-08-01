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

// Does one training step for a model.
// Does the forward pass and backward pass
// Gets the loss if bool train is true
// Does NOT update model parameters
void trainStep(int token, RNNLanguageModel &model, Matrix &state, Matrix &logits, bool train, int stepNum)
{
    if (train)
    {
        model.getLogits(state, logits);
        if (stepNum == 0)
        {
            model.getdYdLogits(token, state); // if this is the first step, only the embedding matrix has been used so we only calculate grad for the embedding matrix
        }
        else
        {
            model.getdYCurrent(token, state);
        }

        model.getdY();   // dY_dP = dY_dPCurrent + dY_dRPrev @ delta
        softmax(logits); // logits -> probs
        model.getdL(token, logits);
    }

    model.forward(state, token);
    model.getdR(token, state);
    model.getDelta(); // delta = dR_dPCurrent + dR_dRPrev @ delta
}

int main()
{
    // Model parameters
    constexpr int vocabSize = 95;
    constexpr int hiddenDim = 16;

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
                std::cout << "Last Loss: " << loss << ", ";
                std::cout << "Last Grad Norm: " << gradNorm << std::endl;

                // Reset
                state.zeros();
                model.reset();
                clock.restart();

                // Get embedding
                std::cout << "Embedding" << std::endl;
                std::flush(std::cout);
                for (int l = 0; l < page.textSize; l++)
                {
                    trainStep(page.text[l], model, state, logits, false, l);
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
                clock.restart();

                // Init loss
                loss = 0.0f;

                // Evaluate
                std::cout << "Evaluating" << std::endl;
                std::flush(std::cout);
                for (int l = 0; l < page.textSize; l++)
                {
                    const uint8_t token = page.text[l];
                    trainStep(token, model, state, logits, true, l + page.textSize);
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
                model.updateParams();

                // Update trackers
                loss /= (float)page.textSize;
                gradNorm = model.dL_dP.norm();

                // Save model
                serializeRNNLanguageModel(model, savePath);

                // Clear text
                clearLines(3);
            }
        }
    }

    return 0;
}