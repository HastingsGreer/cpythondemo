#include "cuda.h"
#include <stdio.h>

__global__ void mandel(double* ref_real_array,
                     double* ref_imag_array, 
                     double* dc_real_array, 
                     double* dc_imag_array, 
                     int depth,
                     int* count_array)
{
  unsigned int i = threadIdx.x + 512 * blockIdx.x;


  double dc_real = dc_real_array[i];
  double dc_imag = dc_imag_array[i];

  int count = 0;
  double d_real = 0;
  double d_imag = 0;
  double d_real_temp;
  while((d_real + ref_real_array[count]) * (d_real + ref_real_array[count]) + 
                   (d_imag + ref_imag_array[count]) * (d_imag + ref_imag_array[count]) < 4 && 
                   count < depth){
    
    double z_real = ref_real_array[count];
    double z_imag = ref_imag_array[count];
    d_real_temp = 2 * z_real * d_real - 2 * z_imag * d_imag + d_real * d_real - d_imag * d_imag + dc_real;
    d_imag = 2 * z_real * d_imag + 2 * z_imag * d_real + 2 * d_real * d_imag + dc_imag;
    d_real = d_real_temp;
    count ++;
  }
  count_array[i] = count;
  
}
extern "C" int cu_mandel(double* ref_real_array,
                     double* ref_imag_array, 
                     double* dc_real_array, 
                     double* dc_imag_array, 
                     int depth,
                     int* count_array, 
                     int l_ref){
    printf("entering function");

    double *dev_real_ref, *dev_imag_ref, *dev_dc_real, *dev_dc_imag;
    int *dev_counts;
    cudaMalloc((void**)&dev_real_ref, l_ref * sizeof(double));
    cudaMalloc((void**)&dev_imag_ref, l_ref * sizeof(double));
    cudaMalloc((void**)&dev_dc_real, 512 * 512 *sizeof(double));
    cudaMalloc((void**)&dev_dc_imag, 512 * 512 *sizeof(double));
    cudaMalloc((void**)&dev_counts, 512 * 512 *sizeof(int));

    cudaMemcpy(dev_real_ref, ref_real_array, l_ref * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_imag_ref, ref_imag_array, l_ref * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dc_real, dc_real_array, 512 * 512 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dc_imag, dc_imag_array, 512 * 512 * sizeof(double), cudaMemcpyHostToDevice);
    printf("calling kernel");

    mandel<<<512, 512>>>(dev_real_ref, dev_imag_ref, dev_dc_real, dev_dc_imag, depth, dev_counts);


    cudaMemcpy(count_array, dev_counts, 512 * 512 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("a count: %d\n", count_array[0]);
    cudaFree(dev_real_ref);
    cudaFree(dev_imag_ref);
    cudaFree(dev_dc_real);
    cudaFree(dev_dc_imag);
    cudaFree(dev_counts);
    return 0;

}