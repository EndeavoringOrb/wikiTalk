g++ -c -g -O3 -mavx2 -mfma -fopenmp trainRNN.cpp
g++ trainRNN.o -o trainRNN -Wall -Wextra -fopenmp
del trainRNN.o

nvcc trainRNN.cu -o trainRNNCuda.exe -lineinfo -lcublas