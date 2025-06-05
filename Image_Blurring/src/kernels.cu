#include "blur_utils.h"

__global__ void blurKernel(const uchar4 *input, uchar4 *output, int width, int height, int blurSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        int count = 0;

        for (int i = -blurSize; i <= blurSize; i++) {
            for (int j = -blurSize; j <= blurSize; j++) {
                int curY = row + i;
                int curX = col + j;

                if (curY >= 0 && curY < height && curX >= 0 && curX < width) {
                    uchar4 pixel = input[curY * width + curX];
                    sum.x += pixel.x;
                    sum.y += pixel.y;
                    sum.z += pixel.z;
                    count++;
                }
            }
        }

        uchar4 result;
        result.x = (unsigned char)(sum.x / count);
        result.y = (unsigned char)(sum.y / count);
        result.z = (unsigned char)(sum.z / count);
        result.w = 255;

        output[row * width + col] = result;
    }
}
