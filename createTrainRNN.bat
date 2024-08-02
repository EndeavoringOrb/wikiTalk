g++ -c -g -O3 -mavx2 -mfma trainRNN.cpp
g++ trainRNN.o -o trainRNN -Wall -Wextra
del trainRNN.o

nvcc trainRNN.cu -o trainRNNCuda.exe -lineinfo -lcublas