NVCC = nvcc
CXX = g++
CFLAGS = -Iinclude -O2 -fopenmp
NVFLAGS = -Iinclude -O2

SRC = src
INCLUDE = include

OBJS = $(SRC)/main.cu \
       $(SRC)/image_io.cpp \
       $(SRC)/apply_blur.cpp \
       $(SRC)/kernels.cu \
       $(SRC)/resize_blur.cu

TARGET = image_blur

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)
