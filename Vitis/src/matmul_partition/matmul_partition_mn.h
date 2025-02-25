#define MAX_DIM_A 2
#define MAX_DIM_B 2
#define MAX_DIM_C 2 // figure out how to make this max(A,B)

void matmul_partition_mn(int* in1, int* in2, int* out_r, int m, int shared_dim, int n);