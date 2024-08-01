g++ -c -g -O3 trainRNN.cpp
g++ trainRNN.o -o trainRNN -Wall -Wextra
del trainRNN.o

nvcc trainRNN.cu -o trainRNNCuda.exe -lineinfo -lcublas