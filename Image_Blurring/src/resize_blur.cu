#include "blur_utils.h"

__global__ void resizeAndBlurKernel(const uchar4 *input, uchar4 *output,
                                    int inputWidth, int inputHeight,
                                    int outputWidth, int outputHeight,
                                    int blurSize) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    if (outCol < outputWidth && outRow < outputHeight) {
        float inX = (float)outCol * inputWidth / outputWidth;
        float inY = (float)outRow * inputHeight / outputHeight;

        int inCol = (int)inX;
        int inRow = (int)inY;

        float dx = inX - inCol;
        float dy = inY - inRow;

        uchar4 p00 = input[min(inRow, inputHeight-1) * inputWidth + min(inCol, inputWidth-1)];
        uchar4 p01 = input[min(inRow, inputHeight-1) * inputWidth + min(inCol+1, inputWidth-1)];
        uchar4 p10 = input[min(inRow+1, inputHeight-1) * inputWidth + min(inCol, inputWidth-1)];
        uchar4 p11 = input[min(inRow+1, inputHeight-1) * inputWidth + min(inCol+1, inputWidth-1)];

        if (blurSize > 0) {
            float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            int count = 0;

            for (int i = -blurSize; i <= blurSize; i++) {
                for (int j = -blurSize; j <= blurSize; j++) {
                    int curY = inRow + i;
                    int curX = inCol + j;

                    if (curY >= 0 && curY < inputHeight && curX >= 0 && curX < inputWidth) {
                        uchar4 pixel = input[curY * inputWidth + curX];
                        sum.x += pixel.x;
                        sum.y += pixel.y;
                        sum.z += pixel.z;
                        count++;
                    }
                }
            }

            if (count > 0) {
                p00.x = p01.x = p10.x = p11.x = (unsigned char)(sum.x / count);
                p00.y = p01.y = p10.y = p11.y = (unsigned char)(sum.y / count);
                p00.z = p01.z = p10.z = p11.z = (unsigned char)(sum.z / count);
            }
        }

        uchar4 result;
        result.x = (unsigned char)((1-dx)*(1-dy)*p00.x + dx*(1-dy)*p01.x + (1-dx)*dy*p10.x + dx*dy*p11.x);
        result.y = (unsigned char)((1-dx)*(1-dy)*p00.y + dx*(1-dy)*p01.y + (1-dx)*dy*p10.y + dx*dy*p11.y);
        result.z = (unsigned char)((1-dx)*(1-dy)*p00.z + dx*(1-dy)*p01.z + (1-dx)*dy*p10.z + dx*dy*p11.z);
        result.w = 255;

        output[outRow * outputWidth + outCol] = result;
    }
}

void resizeAndBlur(const uchar4 *input, uchar4 *output,
                   int inputWidth, int inputHeight,
                   int outputWidth, int outputHeight,
                   int blurSize, int blockSize) {
    uchar4 *d_input, *d_output;

    size_t inputSize = inputWidth * inputHeight * sizeof(uchar4);
    size_t outputSize = outputWidth * outputHeight * sizeof(uchar4);

    cudaMalloc((void **)&d_input, inputSize);
    cudaMalloc((void **)&d_output, outputSize);

    cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid((outputWidth + blockSize - 1) / blockSize,
                       (outputHeight + blockSize - 1) / blockSize);

    resizeAndBlurKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_output, inputWidth, inputHeight, outputWidth, outputHeight, blurSize
    );

    cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
