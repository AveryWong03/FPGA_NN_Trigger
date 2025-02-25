/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

// Includes
#include "matmul_partition_mn.h"

// TRIPCOUNT identifier
const unsigned int c_dim = MAX_DIM_A;

void matmul_partition_mn(int* in1, int* in2, int* out_r, int m, int shared_dim, int n) { // Matrix Dimension. Assuming Square Matrix
#pragma HLS INTERFACE m_axi port = in1 depth = 256
#pragma HLS INTERFACE m_axi port = in2 depth = 256
#pragma HLS INTERFACE m_axi port = out_r depth = 256
#pragma HLS INTERFACE s_axilite port=return



int A[MAX_DIM_A * MAX_DIM_A];
int B[MAX_DIM_B * MAX_DIM_B];
int C[MAX_DIM_C * MAX_DIM_C];

// Cyclic Partition for A as matrix multiplication needs row-wise parallel
// access
#pragma HLS ARRAY_PARTITION variable = A dim = 1 cyclic factor = 16
// Block Partition for B as matrix multiplication needs column-wise parallel
// access
#pragma HLS ARRAY_PARTITION variable = B dim = 1 block factor = 16

// As A and B Matrix are partitioned with the factor of MAX_DIM, so to get
// parallel row/column access, input square matrix[dimXdim] should be written
// into local Array in MATRIX[MAX_DIM * MAX_DIM] format

// Burst read for matrix A
// Auto-pipeline is going to apply pipeline to these loops


// i = row, j = col

readA:
    for (int itr = 0, i = 0, j = 0; itr < m * shared_dim; itr++, j++) {
#pragma HLS LOOP_TRIPCOUNT min = c_dim* c_dim max = c_dim * c_dim
        if (j == shared_dim)  { // maybe set to m/n? 
            j = 0;
            i++;
        }
        A[i * MAX_DIM_A + j] = in1[itr];
    }

// Burst read for matrix B
readB:
    for (int itr = 0, i = 0, j = 0; itr < shared_dim * n; itr++, j++) {
#pragma HLS LOOP_TRIPCOUNT min = c_dim* c_dim max = c_dim * c_dim
        if (j == n) {
            j = 0;
            i++;
        }
        B[i * MAX_DIM_B + j] = in2[itr];
    }


lreorder1:
    for (int i = 0; i < m; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_dim max = c_dim
    // As A and B are partition correctly so loop pipelining is applied
    // at 2nd level loop and which will eventually unroll the lower loop
    lreorder2:
        for (int j = 0; j < n; j++) {
#pragma HLS LOOP_TRIPCOUNT min = c_dim max = c_dim
            int result = 0;
        lreorder3:
            for (int k = 0; k < MAX_DIM_C; k++) {
                //#pragma HLS LOOP_TRIPCOUNT min = c_dim max = c_dim
                result += A[i * MAX_DIM_A + k] * B[k * MAX_DIM_B + j];
            }
            C[i * MAX_DIM_C + j] = result;
        }
    }

// Burst write from output matrices to global memory
// Burst write from matrix C
writeC:
    for (int itr = 0, i = 0, j = 0; itr < m * n; itr++, j++) {
#pragma HLS LOOP_TRIPCOUNT min = c_dim* c_dim max = c_dim * c_dim
        if (j == n) {
            j = 0;
            i++;
        }
        out_r[itr] = C[i * MAX_DIM_C + j];
    }
}
