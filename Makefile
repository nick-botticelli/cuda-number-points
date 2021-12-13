NVCC = nvcc
NVCCFLAGS = -lcuda

CXXFLAGS = -fopenmp,-Ofast

all: number_points

number_points: number_points.cu
	$(NVCC) $(CXXFLAGS) -Xcompiler $(CXXFLAGS) $^ -o $@

clean:
	rm -f *.o ./number_points
