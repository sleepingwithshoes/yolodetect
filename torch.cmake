list(APPEND CMAKE_PREFIX_PATH "/usr/local/lib/libtorch")
list(APPEND CMAKE_CUDA_ARCHITECTURES "native")
list(APPEND CMAKE_CUDA_COMPILER  "/usr/local/cuda/bin/nvcc")
list(APPEND CAFFE2_USE_CUDNN 1)
list(APPEND PYTHON_EXECUTABLE "/usr/bin/python3")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
