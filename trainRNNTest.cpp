#include "trainRNNStructs.hpp"

int main()
{
    std::string loadPath = "models/embedArticleCPP/model.bin";

    RNNLanguageModel model = deserializeRNNLanguageModel(loadPath);
}