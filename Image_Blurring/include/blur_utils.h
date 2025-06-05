#ifndef BLUR_UTILS_H
#define BLUR_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#define DEFAULT_BLUR_SIZE 5
#define DEFAULT_BLUR_PASSES 1
#define DEFAULT_BLOCK_SIZE 16
#define DEFAULT_OUTPUT_WIDTH 0
#define DEFAULT_OUTPUT_HEIGHT 0

typedef enum {
    BLUR_BOX = 0,
    BLUR_GAUSSIAN = 1,
    BLUR_MEDIAN = 2
} BlurType;

typedef struct {
    unsigned char x, y, z, w;
} uchar4;

void printHelp();
uchar4* readImage(const char *filename, int *width, int *height);
void writeImage(const uchar4 *image, const char *filename, int width, int height);

void applyBlur(const uchar4 *input, uchar4 *output, int width, int height,
               int blurSize, float sigma, BlurType blurType,
               int passes, bool useSharedMemory, int blockSize);

void resizeAndBlur(const uchar4 *input, uchar4 *output,
                   int inputWidth, int inputHeight,
                   int outputWidth, int outputHeight,
                   int blurSize, int blockSize);

#endif // BLUR_UTILS_H
