g++ -c -g -O3 prepareTokenData.cpp
g++ prepareTokenData.o -o prepareTokenData -Wall -Wextra
del prepareTokenData.o

g++ -c -g -O3 -mavx2 -mfma rnnInference.cpp
g++ rnnInference.o -o rnnInference -Wall -Wextra
del rnnInference.o