#include "blur_utils.h"

uchar4* readImage(const char *filename, int *width, int *height) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        return NULL;
    }

    char magic[3];
    int maxval;
    if (fscanf(fp, "%2s\n%d %d\n%d\n", magic, width, height, &maxval) != 4 ||
        strcmp(magic, "P6") != 0 || maxval != 255) {
        printf("Error: Invalid PPM file format (must be P6 with maxval 255)\n");
        fclose(fp);
        return NULL;
    }

    size_t num_pixels = (*width) * (*height);
    uchar4 *image = (uchar4*)malloc(num_pixels * sizeof(uchar4));
    unsigned char *temp = (unsigned char*)malloc(num_pixels * 3);

    if (fread(temp, 1, num_pixels * 3, fp) != num_pixels * 3) {
        printf("Error: Failed to read image data\n");
        free(temp); free(image); fclose(fp);
        return NULL;
    }

    for (size_t i = 0; i < num_pixels; i++) {
        image[i].x = temp[i*3];     // R
        image[i].y = temp[i*3 + 1]; // G
        image[i].z = temp[i*3 + 2]; // B
        image[i].w = 255;           // A
    }

    free(temp);
    fclose(fp);
    printf("Loaded image: %s (%d x %d)\n", filename, *width, *height);
    return image;
}

void writeImage(const uchar4 *image, const char *filename, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error: Cannot create file %s\n", filename);
        return;
    }

    fprintf(fp, "P6\n%d %d\n255\n", width, height);

    size_t num_pixels = width * height;
    unsigned char *temp = (unsigned char*)malloc(num_pixels * 3);

    for (size_t i = 0; i < num_pixels; i++) {
        temp[i*3] = image[i].x;
        temp[i*3 + 1] = image[i].y;
        temp[i*3 + 2] = image[i].z;
    }

    fwrite(temp, 1, num_pixels * 3, fp);
    free(temp);
    fclose(fp);
    printf("Wrote image: %s (%d x %d)\n", filename, width, height);
}
