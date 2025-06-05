#include "blur_utils.h"

int main(int argc, char **argv) {
    int xDimension, yDimension;
    char inputImageFile[256] = {0};
    char outputImageFile[256] = {0};
    uchar4 *hostImage = NULL;

    int blurSize = DEFAULT_BLUR_SIZE;
    int blockSize = DEFAULT_BLOCK_SIZE;
    int passes = DEFAULT_BLUR_PASSES;
    float sigma = 1.0f;
    BlurType blurType = BLUR_BOX;
    bool useSharedMemory = true;
    int outputWidth = DEFAULT_OUTPUT_WIDTH;
    int outputHeight = DEFAULT_OUTPUT_HEIGHT;

    int arg = 1;
    while (arg < argc && argv[arg][0] == '-') {
        switch (argv[arg][1]) {
            case 't': if (++arg < argc) blurType = (BlurType)atoi(argv[arg]); break;
            case 'r': if (++arg < argc) blurSize = atoi(argv[arg]); break;
            case 's': if (++arg < argc) sigma = atof(argv[arg]); break;
            case 'p': if (++arg < argc) passes = atoi(argv[arg]); break;
            case 'b': if (++arg < argc) blockSize = atoi(argv[arg]); break;
            case 'm': if (++arg < argc) useSharedMemory = (atoi(argv[arg]) == 1); break;
            case 'w': if (++arg < argc) outputWidth = atoi(argv[arg]); break;
            case 'h': 
                if (strcmp(argv[arg], "-h") == 0) { printHelp(); return 0; }
                if (++arg < argc) outputHeight = atoi(argv[arg]);
                break;
            default: printf("Unknown option: %s\n", argv[arg]); printHelp(); return 1;
        }
        arg++;
    }

    if (arg >= argc) { printf("Error: Input image file not specified.\n"); printHelp(); return 1; }
    strcpy(inputImageFile, argv[arg++]);
    if (arg >= argc) { printf("Error: Output image file not specified.\n"); printHelp(); return 1; }
    strcpy(outputImageFile, argv[arg]);

    hostImage = readImage(inputImageFile, &xDimension, &yDimension);
    if (!hostImage) return 1;

    int finalWidth = (outputWidth > 0) ? outputWidth : xDimension;
    int finalHeight = (outputHeight > 0) ? outputHeight : yDimension;

    uchar4 *outputImage = (uchar4 *)malloc(finalWidth * finalHeight * sizeof(uchar4));
    if (!outputImage) {
        printf("Error: Failed to allocate memory for output image.\n");
        free(hostImage);
        return 1;
    }

    printf("Processing image: %s -> %s\n", inputImageFile, outputImageFile);
    printf("Input dimensions: %d x %d\n", xDimension, yDimension);

    if (finalWidth != xDimension || finalHeight != yDimension) {
        printf("Output dimensions: %d x %d (resizing enabled)\n", finalWidth, finalHeight);
        resizeAndBlur(hostImage, outputImage, xDimension, yDimension,
                      finalWidth, finalHeight, blurSize, blockSize);
    } else {
        printf("Output dimensions: %d x %d\n", finalWidth, finalHeight);
        applyBlur(hostImage, outputImage, xDimension, yDimension,
                  blurSize, sigma, blurType, passes, useSharedMemory, blockSize);
    }

    writeImage(outputImage, outputImageFile, finalWidth, finalHeight);

    free(hostImage);
    free(outputImage);
    return 0;
}
