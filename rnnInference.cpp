#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

#include <immintrin.h>
#include <memory>

void clearLines(int numLines)
{
    for (int i = 0; i < numLines; i++)
    {
        std::cout << "\033[F\033[K";
    }
    std::flush(std::cout);
}

struct Matrix
{
    float *data;
    int rows;
    int cols;
    int numValues = -1;

    Matrix() {}

    Matrix(Matrix &other)
    {
        if (numValues != -1)
        {
            delete[] data;
        }
        rows = other.rows;
        cols = other.cols;
        numValues = other.numValues;

        size_t size = rows * cols; // Number of floats
        size_t alignment = 32;
        size_t space = size * sizeof(float) + alignment;
        void *ptr = operator new(space);
        void *aligned_ptr = std::align(alignment, size * sizeof(float), ptr, space);
        data = static_cast<float *>(aligned_ptr);

        copy(other);
    }

    Matrix(int r, int c) : rows(r), cols(c), numValues(r * c)
    {
        // data = new float[rows * cols];

        size_t size = rows * cols; // Number of floats
        size_t alignment = 32;
        size_t space = size * sizeof(float) + alignment;
        void *ptr = operator new(space);
        void *aligned_ptr = std::align(alignment, size * sizeof(float), ptr, space);
        data = static_cast<float *>(aligned_ptr);

        zeros();
    }

    ~Matrix()
    {
        delete[] data;
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
        for (int i = 0; i < numValues; i++)
        {
            data[i] = other.data[i];
        }
    }

    void copyRow(Matrix &other, int row)
    {
        for (int i = 0; i < cols; i++)
        {
            data[row * cols + i] = other.data[i];
        }
    }

    // Prints the matrix to the terminal
    void print(std::string name)
    {
        // std::cout << std::fixed << std::setprecision(5);
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

struct RNNModel
{
    int vocabSize;
    int hiddenDim;

    Matrix embedding;
    Matrix ih;
    Matrix hh;
    Matrix bias;

    Matrix embedded;
    Matrix scaledHH;

    Matrix newState;

    std::vector<int> kVals;
    std::vector<int> jVals;

    void activation(float &x)
    {
        float xAbs = fabsf(x);
        xAbs = 1 - (1.0f / (xAbs * xAbs + xAbs + 1.0f));
        x = copysignf(xAbs, x);
    }

    void preCompute()
    {
        // Compute scaledHH
        scaledHH.zeros();
        for (int i = 0; i < hiddenDim; i++)
        {
            // Compute norm
            float rowNorm = 0.0f;
            for (int j = 0; j < hiddenDim; j++)
            {
                const float x = hh.data[i * hiddenDim + j];
                rowNorm += x * x;
            }
            rowNorm = sqrt(rowNorm) * hiddenDim;
            const float invRowNorm = 1.0f / rowNorm;

            // Scale hh
            for (int j = 0; j < hiddenDim; j++)
            {
                hh.data[i * hiddenDim + j] *= invRowNorm;
            }
        }

        // Compute embedded
        embedded.zeros();
        for (int i = 0; i < vocabSize; i++)
        {
            for (int j = 0; j < hiddenDim; j++)
            {
                for (int k = 0; k < hiddenDim; k++)
                {
                    embedded.data[i * hiddenDim + j] += embedding.data[i * hiddenDim + k] * ih.data[k * hiddenDim + j];
                }
                embedded.data[i * hiddenDim + j] += bias.data[j];
                activation(embedded.data[i * hiddenDim + j]);
            }
        }

        kVals.clear();
        kVals.reserve(hiddenDim / 8);
        for (int k = 0; k < hiddenDim; k += 8)
        {
            kVals.emplace_back(k);
        }

        jVals.clear();
        jVals.reserve(hiddenDim / 8);
        for (int j = 0; j < hiddenDim; j++)
        {
            jVals.emplace_back(j);
        }
    }

    void embed(Matrix &state, const std::vector<uint8_t> &tokens)
    {
        for (const int j : jVals)
        {
            newState.data[j] = 0.0f;
            state.data[j] = 0.0f;
        }
        //__m256 ones = _mm256_set1_ps(1.0f);
        //__m256 zeros = _mm256_setzero_ps();
        //__m256 sign_bit = _mm256_set1_ps(-0.0f);
        for (const uint8_t token : tokens)
        {
            // newState = state @ scaledHH
            for (const int j : jVals)
            {
                __m256 stateVal = _mm256_set1_ps(state.data[j]);
                for (const int k : kVals)
                {
                    // state @ scaledHH
                    // newState.data[k] += state.data[j] * scaledHH.data[j * hiddenDim + k];
                    float *newStateK = &newState.data[k];
                    __m256 temp = _mm256_load_ps(&scaledHH.data[j * hiddenDim + k]);
                    __m256 result = _mm256_load_ps(newStateK);
                    result = _mm256_fmadd_ps(stateVal, temp, result);
                    _mm256_store_ps(newStateK, result);
                }
            }

            for (const int j : jVals)
            {
                // newState += embedded[token]
                state.data[j] = newState.data[j] + embedded.data[token * hiddenDim + j];
                //__m256 newStateVals = _mm256_load_ps(&newState.data[j]);
                //__m256 embeddedVals = _mm256_load_ps(&embedded.data[token * hiddenDim + j]);
                //_mm256_store_ps(&state.data[j], _mm256_add_ps(newStateVals, embeddedVals));

                // state = activation(newState)
                activation(state.data[j]);
                //__m256 x = _mm256_load_ps(&state.data[j]);
                //__m256 xAbs = _mm256_andnot_ps(sign_bit, x);
                // xAbs = _mm256_add_ps(_mm256_fmadd_ps(xAbs, xAbs, xAbs), ones);
                //__m256 result = _mm256_sub_ps(ones, _mm256_div_ps(ones, xAbs));
                //__m256 sign_mask = _mm256_and_ps(x, _mm256_set1_ps(-0.0f));
                //_mm256_store_ps(&state.data[j], _mm256_or_ps(result, sign_mask));

                // Reset newState for next token
                newState.data[j] = 0.0f;
                //_mm256_store_ps(&newState.data[j], zeros);
            }
        }
    }
};

RNNModel deserializeModel(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Unable to open file");
    }

    RNNModel model;

    uint32_t numMatrices, rows, cols;
    // Read # of matrices
    file.read(reinterpret_cast<char *>(&numMatrices), sizeof(uint32_t));

    // Read matrix size
    file.read(reinterpret_cast<char *>(&rows), sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(&cols), sizeof(uint32_t));
    model.embedding.rows = rows;
    model.embedding.cols = cols;
    model.embedding.numValues = rows * cols;
    model.embedding.data = new float[model.embedding.numValues];
    model.embedded.rows = rows;
    model.embedded.cols = cols;
    model.embedded.numValues = rows * cols;
    model.embedded.data = new float[model.embedded.numValues];
    model.vocabSize = rows;
    model.hiddenDim = cols;
    // Load matrix data
    file.read(reinterpret_cast<char *>(model.embedding.data), sizeof(float) * model.embedding.numValues);

    // Read matrix size
    file.read(reinterpret_cast<char *>(&rows), sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(&cols), sizeof(uint32_t));
    model.ih.rows = rows;
    model.ih.cols = cols;
    model.ih.numValues = rows * cols;
    model.ih.data = new float[model.ih.numValues];
    // Load matrix data
    file.read(reinterpret_cast<char *>(model.ih.data), sizeof(float) * model.ih.numValues);

    // Read matrix size
    file.read(reinterpret_cast<char *>(&rows), sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(&cols), sizeof(uint32_t));
    model.hh.rows = rows;
    model.hh.cols = cols;
    model.hh.numValues = rows * cols;
    model.hh.data = new float[model.hh.numValues];
    model.scaledHH.rows = rows;
    model.scaledHH.cols = cols;
    model.scaledHH.numValues = rows * cols;
    model.scaledHH.data = new float[model.scaledHH.numValues];
    // Load matrix data
    file.read(reinterpret_cast<char *>(model.hh.data), sizeof(float) * model.hh.numValues);

    // Read matrix size
    file.read(reinterpret_cast<char *>(&rows), sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(&cols), sizeof(uint32_t));
    model.bias.rows = rows;
    model.bias.cols = cols;
    model.bias.numValues = rows * cols;
    model.bias.data = new float[model.bias.numValues];
    model.newState.rows = rows;
    model.newState.cols = cols;
    model.newState.numValues = rows * cols;
    model.newState.data = new float[model.newState.numValues];
    // Load matrix data
    file.read(reinterpret_cast<char *>(model.bias.data), sizeof(float) * model.bias.numValues);

    return model;
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

void embedWiki(int numTokenFiles, DataLoader &dataLoader, std::string &dataFolder, Page &page, RNNModel &textModel, std::string &saveFolder, Matrix &state)
{
    Clock clock;
    uint64_t numTokensProcessed = 0;

    for (int j = 0; j < numTokenFiles; j++)
    {
        dataLoader.openFile(dataFolder + std::to_string(j) + ".bin");

        std::string savePath = saveFolder + "/" + std::to_string(j) + ".emb";
        std::ofstream out(savePath, std::ios::binary);
        if (!out)
        {
            std::cerr << "Error: Unable to open file for writing: " << savePath << std::endl;
            return;
        }

        // Write # of pages and embedding size
        out.write(reinterpret_cast<const char *>(&dataLoader.numTuples), sizeof(uint32_t));
        out.write(reinterpret_cast<const char *>(&textModel.hiddenDim), sizeof(uint32_t));

        std::cout << std::endl;

        for (int k = 0; k < dataLoader.numTuples; k++)
        {
            // Load page
            bool loadedPage = dataLoader.nextPage(page);

            if (!loadedPage)
            {
                std::cout << "Failed to load page #" << k + 1 << " in file #" << j << "." << std::endl;
                continue;
            }

            numTokensProcessed += page.textSize;

            // Embed
            textModel.embed(state, page.text);

            // Write embedding
            out.write(reinterpret_cast<const char *>(state.data), sizeof(float) * textModel.hiddenDim);

            if (k % 1 == 0)
            {
                // Log
                clearLines(1);
                std::cout << "File [" << j + 1 << "/" << numTokenFiles << "], ";
                std::cout << "Page [" << k + 1 << "/" << dataLoader.numTuples << "], ";
                std::cout << (uint64_t)(numTokensProcessed / clock.getElapsedTime()) << " Tok/Sec" << std::endl;
            }
        }

        // Log
        clearLines(1);
        std::cout << "File [" << j + 1 << "/" << numTokenFiles << "], ";
        std::cout << "Page [" << dataLoader.numTuples << "/" << dataLoader.numTuples << "], ";
        std::cout << (uint64_t)(numTokensProcessed / clock.getElapsedTime()) << " Tok/Sec" << std::endl;
    }

    const float elapsed = clock.getElapsedTime();
    std::cout << "Finished embedding " << numTokensProcessed << " tokens in " << elapsed << "s. " << (uint64_t)(numTokensProcessed / elapsed) << " Tok/Sec" << std::endl;
}

int main()
{
    // Settings
    std::string modelPath = "models/embedArticle/0";
    int numTokenFiles = 74;
    std::string dataFolder = "tokenData/";
    std::string saveFolder = "embedWiki";

    // Load
    // RNNModel titleModel = deserializeModel(modelPath + "/titleModel.bin");
    RNNModel textModel = deserializeModel(modelPath + "/textModel.bin");
    textModel.preCompute();

    // Init dataloader
    DataLoader dataLoader = DataLoader(dataFolder + "0.bin");
    Page page;

    Matrix state = Matrix(textModel.hiddenDim, 1);
    embedWiki(numTokenFiles, dataLoader, dataFolder, page, textModel, saveFolder, state);

    return 0;
}