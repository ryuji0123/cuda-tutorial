#include <iostream>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void matrixMultiplicationKernel(int* A,int* B,int* C,int N)
{
	int ROW = blockIdx.y*blockDim.y+threadIdx.y;
	int COL = blockIdx.x*blockDim.x+threadIdx.x;
	float tmp_sum = 0.0f;

	if(ROW < N && COL < N){
		for(int i=0;i<N;i++){
			tmp_sum += A[ROW*N+i] *B[i*N+COL];
		}
	}
	C[ROW*N+COL] = tmp_sum;
}

void matrixMultiplication(int* A,int* B,int* C,int N);

int main()
{
	int N = 16;

	//Host i/o vectors
	int *h_A;
	int *h_B;
	int *h_C;

	//Device i/o vector
	int *d_A;
	int *d_B;
	int *d_C;

	size_t bytes = N*N*sizeof(int);

	h_A = (int*)malloc(bytes);
	h_B = (int*)malloc(bytes);
	h_C = (int*)malloc(bytes);

	cudaMalloc(&d_A,bytes);
	cudaMalloc(&d_B,bytes);
	cudaMalloc(&d_C,bytes);
	
	// Initialize matricies on the host
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			h_A[i*N+j] = 2;
			h_B[i*N+j] = 3;
		}
	}

	//Copy host vectors to device
	cudaMemcpy(d_A,h_A,bytes,cudaMemcpyHostToDevice);	
	cudaMemcpy(d_B,h_B,bytes,cudaMemcpyHostToDevice);

	matrixMultiplication(d_A,d_B,d_C,N);

	cudaMemcpy(h_C,d_C,bytes,cudaMemcpyDeviceToHost);
	
	//check result
	for(int i=0;i<4;i++){
		for(int j=0;j<4;j++){
			printf(" %d",h_C[i*N+j]);
		}
		printf("\n");
	}
	// Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
	//free host memory
	free(h_A);
	free(h_B);
	free(h_C);
	return 0;
}


void matrixMultiplication(int* A,int* B,int* C,int N)
{
	dim3 threadsPerBlock(N,N);
	dim3 blocksPerGrid(1,1);
	if(N*N>512){
		threadsPerBlock.x = 512;
		threadsPerBlock.y = 512;
		blocksPerGrid.x = ceil(int(N)/double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(int(N)/double(threadsPerBlock.y));
	}
	matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(A,B,C,N);
}


