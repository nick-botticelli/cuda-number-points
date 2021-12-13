# Number Points CUDA Demo

This program accepts a distance (epsilon) as input to find all
points within epsilon distance of every point. To do this, the
program utilizes an NVIDIA GPU via CUDA to accelerate
computation.

## Using

1. Build the program with the command `make`. This project
requires NVCC and OpenMP. Refer to the
[CI workflow](./.github/workflows/nvcc.yml) for more
information.
2. Run the program with the command `./number_points`. You must
have a CUDA-compatible NVIDIA GPU for this program to run.

# License

The MIT License (MIT)
