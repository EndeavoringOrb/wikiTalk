#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <fstream>
#include <vector>
#include <string>

uint32_t findLongestSequence(const std::string &folder, const int numFiles)
{
    uint32_t maxLength = 0;

    for (int i = 0; i < numFiles; i++)
    {
        std::cout << i + 1 << "/" << numFiles << "\r";
        std::string filename = folder + "/" + std::to_string(i) + ".bin";
        try
        {
            std::ifstream file;
            file.open(filename, std::ios::binary);

            uint32_t numTuples;

            file.read(reinterpret_cast<char *>(&numTuples), sizeof(numTuples));

            for (int j = 0; j < numTuples; j++)
            {
                // Read the length of the title and page
                uint32_t len1, len2;
                file.read(reinterpret_cast<char *>(&len1), sizeof(len1));
                file.read(reinterpret_cast<char *>(&len2), sizeof(len2));

                file.seekg(len1 + len2, std::ios::cur);

                maxLength = std::max(maxLength, len1 + len2);
            }

            file.close();
        }
        catch (const std::exception &e)
        {
            // If there's an error opening or reading a file, we'll just skip it and continue with the next one
            std::cerr << "Error processing file " << filename << ": " << e.what() << std::endl;
        }
    }

    return maxLength;
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
    std::string filename;
    bool fileOpen = false;
    uint32_t numTuples;
    uint32_t currentTuple = 0;

    DataLoader(const std::string &_filename)
        : file(_filename, std::ios::binary),
          filename(_filename)
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

    void openFile(const std::string &_filename)
    {
        if (fileOpen)
        {
            file.close();
        }
        filename = _filename;
        file.open(filename, std::ios::binary);
        readHeader();
    }

    void closeFile()
    {
        if (fileOpen)
        {
            file.close();
        }
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

int getNumBits(int num)
{
    if (num == 0)
    {
        return 1; // Special case: 0 can be represented with a single bit
    }

    int n = 0;
    while ((1 << n) <= num)
    {
        n++;
    }

    return n;
}

std::vector<bool> getBits(int n)
{
    if (n == 0)
    {
        return {0};
    }

    std::vector<bool> binaryVector;

    while (n > 0)
    {
        binaryVector.push_back(n % 2);
        n /= 2;
    }

    // The bits are stored in reverse order (LSB first), so reverse the vector if you want MSB first.
    std::reverse(binaryVector.begin(), binaryVector.end());

    return binaryVector;
}

struct SequenceItem
{
    float *probs;
    uint8_t nextToken;

    void resize(int newSize)
    {
        delete[] probs;
        probs = new float[newSize];
    }

    ~SequenceItem()
    {
        delete[] probs;
    }

    void reset(int size)
    {
        for (int i = 0; i < size; i++)
        {
            probs[i] = 0.0f;
        }
    }

    void normalize(int size)
    {
        // Get sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            sum += probs[i];
        }

        // Get inverse
        sum = 1.0f / sum;

        // Normalize probs
        for (int i = 0; i < size; i++)
        {
            probs[i] *= sum;
        }
    }

    int getNumProbs(int size)
    {
        int numProbs = 0;
        for (int i = 0; i < size; i++)
        {
            if (probs[i] != 0.0f)
            {
                numProbs++;
            }
        }
        return numProbs;
    }
};

struct Sequence
{
    int length;
    int vocabSize;
    std::vector<SequenceItem> items;

    Sequence(const int _vocabSize, const int _length)
        : length(_length),
          vocabSize(_vocabSize)
    {
        items.reserve(length);
        for (int i = 0; i < length; i++)
        {
            SequenceItem item;
            item.resize(vocabSize);
            items.emplace_back(item);
        }
    }

    void set(const std::vector<uint8_t> &data)
    {
        for (int i = 0; i < data.size(); i++)
        {
            SequenceItem &item = items[i];
            item.reset(vocabSize);
            item.nextToken = data[i];
        }
        length = data.size();
    }

    void add(const int index, const uint8_t value)
    {
        items[index].probs[value]++;
    }

    void normalize()
    {
        for (int i = 0; i < length; i++)
        {
            items[i].normalize(vocabSize);
        }
    }

    int getSequenceNumBits()
    {
        int numBits = 0;
        int bitsPerToken = getNumBits(vocabSize);
        for (int i = 0; i < length; i++)
        {
            SequenceItem &item = items[i];
            int numProbs = 0;
            for (int j = 0; j < vocabSize; j++)
            {
                if (item.probs[j] != 0.0f)
                {
                    numProbs++;
                }
            }
            if (numProbs == 1)
            {
                numBits += bitsPerToken;
            }
            else
            {
                numBits += numProbs * (32 + bitsPerToken);
            }
        }
        return numBits;
    }
};

int main()
{
    std::cout << "Initializing..." << std::endl;
    int vocabSize = 95;
    int numTokenFiles = 74;
    std::string tokenFolder = "tokenData";
    std::string saveFolder = "tokenPredData";
    int maxSaveFileSize = 50000000;

    int saveFileNum = 0;
    int saveFileNumBytes = 0;
    int bitsPerToken = getNumBits(vocabSize);
    std::ofstream out(saveFolder + "/" + std::to_string(saveFileNum) + ".bin", std::ios::binary);
    if (!out)
    {
        std::cerr << "Error: Unable to open file for writing: " << saveFolder + "/" + std::to_string(saveFileNum) + ".bin" << std::endl;
        exit(0);
    }

    std::cout << "Finding longest sequence..." << std::endl;
    uint32_t longestSequence = findLongestSequence(tokenFolder, numTokenFiles);
    std::cout << "Longest sequence has " << longestSequence << " tokens." << std::endl;
    Sequence sequence = Sequence(vocabSize, longestSequence);
    DataLoader dataLoader = DataLoader(tokenFolder + "/0.bin");
    Page page;
    DataLoader otherDataLoader = DataLoader(tokenFolder + "/0.bin");
    Page otherPage;

    for (int i = 0; i < numTokenFiles; i++)
    {
        dataLoader.openFile(tokenFolder + "/" + std::to_string(i) + ".bin");

        for (int j = 0; j < dataLoader.numTuples; j++)
        {
            // Load page
            bool loadedPage = dataLoader.nextPage(page);

            if (!loadedPage)
            {
                std::cout << "Failed to load page #" << j + 1 << " in file #" << i << "." << std::endl;
                continue;
            }

            // Set sequence
            std::cout << "Setting sequence " << j << std::endl;
            sequence.set(page.text);

            // Get sequence probs
            int stepNum = 0;
            for (int ii = 0; ii < numTokenFiles; ii++)
            {
                otherDataLoader.openFile(tokenFolder + "/" + std::to_string(ii) + ".bin");

                std::cout << "Getting Probs " << ii + 1 << "/" << numTokenFiles << "\r";

                for (int jj = 0; jj < otherDataLoader.numTuples; jj++)
                {
                    // Load page
                    bool loadedPage = otherDataLoader.nextPage(otherPage);

                    if (!loadedPage)
                    {
                        std::cout << "Failed to load page #" << j + 1 << " in file #" << i << "." << std::endl;
                        continue;
                    }

                    for (int k = 0; k < std::min(page.textSize, otherPage.textSize); k++)
                    {
                        if (otherPage.text[k] == page.text[k])
                        {
                            sequence.add(k, otherPage.text[k]);
                        }
                        else
                        {
                            break;
                        }
                    }
                    stepNum++;
                }
            }

            // Normalize sequence
            sequence.normalize();

            // If length of new sequence will make file too large, make new file
            int numNewBytes = sequence.getSequenceNumBits();
            if (numNewBytes + saveFileNumBytes > maxSaveFileSize)
            {
                out.close();
                saveFileNum++;
                saveFileNumBytes = numNewBytes;
                out = std::ofstream(saveFolder + "/" + std::to_string(saveFileNum) + ".bin", std::ios::binary);
                if (!out)
                {
                    std::cerr << "Error: Unable to open file for writing: " << saveFolder + "/" + std::to_string(saveFileNum) + ".bin" << std::endl;
                    exit(0);
                }
            }

            // Write sequence length
            out.write(reinterpret_cast<const char *>(&sequence.length), sizeof(int));

            // Find and write the first unique token index
            int firstUniqueIndex = 0;
            for (int i = 0; i < sequence.length; i++)
            {
                if (sequence.items[i].getNumProbs(sequence.vocabSize) == 1)
                {
                    firstUniqueIndex = i;
                    break;
                }
            }
            out.write(reinterpret_cast<const char *>(&firstUniqueIndex), sizeof(int));

            // Write probs and tokens
            char buffer = 0;
            for (int i = 0; i < sequence.length; i++)
            {
                SequenceItem &item = sequence.items[i];

                if (i < firstUniqueIndex)
                {
                    // Write number of probs
                    int numProbs = item.getNumProbs(sequence.vocabSize);
                    out.write(reinterpret_cast<const char *>(&numProbs), sizeof(int));

                    // Write {tok, prob} pairs
                    for (int i = 0; i < sequence.vocabSize; i++)
                    {
                        if (item.probs[i] != 0.0f)
                        {
                            out.write(reinterpret_cast<const char *>(&i), sizeof(uint8_t));
                            out.write(reinterpret_cast<const char *>(&item.probs[i]), sizeof(float));
                        }
                    }

                    // Write next token
                    out.write(reinterpret_cast<const char *>(&item.nextToken), sizeof(uint8_t));
                }
                else
                {
                    // Write next token
                    out.write(reinterpret_cast<const char *>(&item.nextToken), sizeof(uint8_t));
                }
            }
        }
    }
    return 0;
}