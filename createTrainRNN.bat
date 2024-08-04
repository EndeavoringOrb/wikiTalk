g++ -c -g -O3 -mavx2 -mfma -fopenmp trainRNN.cpp
g++ trainRNN.o -o trainRNN -Wall -Wextra -fopenmp
del trainRNN.o

g++ -c -g -O3 -mavx2 -mfma -fopenmp rnn.cpp
g++ rnn.o -o rnn -Wall -Wextra -fopenmp
del rnn.o

nvcc trainRNN.cu -o trainRNNCuda.exe -lineinfo -lcublas