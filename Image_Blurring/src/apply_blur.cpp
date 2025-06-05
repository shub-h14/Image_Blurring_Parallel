#include "blur_utils.h"

__global__ void blurKernel(const uchar4 *input, uchar4 *output, int width, int height, int blurSize);

void printHelp() {
    printf("Enhanced Image Blur - CUDA Implementation\n");
    printf("Usage: ./image_blur [options] <input_image> <output_image>\n\n");
    printf("Options:\n");
    printf("  -t <type>        Blur type (0=box, 1=gaussian, 2=median) [default: %d]\n", BLUR_BOX);
    printf("  -r <radius>      Blur radius [default: %d]\n", DEFAULT_BLUR_SIZE);
    printf("  -s <sigma>       Sigma value for Gaussian blur [default: 1.0]\n");
    printf("  -p <passes>      Number of blur passes [default: %d]\n", DEFAULT_BLUR_PASSES);
    printf("  -b <block_size>  CUDA block size [default: %d]\n", DEFAULT_BLOCK_SIZE);
    printf("  -m <mode>        Memory mode (0=global, 1=shared) [default: 1]\n");
    printf("  -w <width>       Output image width (0=use input width) [default: 0]\n");
    printf("  -h <height>      Output image height (0=use input height) [default: 0]\n");
}

void applyBlur(const uchar4 *input, uchar4 *output, int width, int height,
               int blurSize, float sigma, BlurType blurType,
               int passes, bool useSharedMemory, int blockSize) {
    uchar4 *d_input, *d_output, *d_temp;
    size_t size = width * height * sizeof(uchar4);

    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);
    cudaMalloc((void **)&d_temp, size);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid((width + blockSize - 1) / blockSize,
                       (height + blockSize - 1) / blockSize);

    for (int p = 0; p < passes; p++) {
        blurKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height, blurSize);

        if (p < passes - 1) {
            cudaMemcpy(d_temp, d_output, size, cudaMemcpyDeviceToDevice);
            uchar4 *temp = d_input;
            d_input = d_temp;
            d_temp = temp;
        }
    }

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
}
