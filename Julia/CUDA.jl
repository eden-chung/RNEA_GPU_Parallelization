using Pkg
Pkg.add("CUDA")

using CUDA
CUDA.versioninfo()

using CUDA

function time_matmul_cuda(size)
    # Ensure that CUDA is available
    if !CUDA.functional()
        error("CUDA is not available or your GPU is not supported.")
    end
    
    # Create random matrices on the GPU
    A_gpu = CUDA.rand(size, size)
    B_gpu = CUDA.rand(size, size)
    
    # Warm up (important to get accurate timing and avoid including compilation time)
    CUDA.@sync A_gpu * B_gpu
    
    # Measure the execution time of matrix multiplication on the GPU
    start_time = CUDA.@elapsed begin
        C_gpu = A_gpu * B_gpu
        CUDA.synchronize()  # Ensure the multiplication is complete
    end
    
    return start_time  # Time in seconds
end

sizes = [100, 300, 500, 700, 900, 1300, 2000, 4000, 6000, 8000]
cuda_times = [time_matmul_cuda(size) for size in sizes]

# Print the results
for (i, size) in enumerate(sizes)
    println("Matrix size: $size x $size, GPU Time taken: $(cuda_times[i]) seconds")
end



# function time_matmul_julia(size)
#     A = rand(size, size)
#     B = rand(size, size)
#     start_time = time()  # Capture the start time
#     C = A * B  # Perform matrix multiplication
#     end_time = time()  # Capture the end time
#     return end_time - start_time  # Calculate the elapsed time
# end

# sizes = [100, 300, 500, 700, 900, 1300, 2000, 4000, 6000, 8000]
# julia_times = [time_matmul_julia(size) for size in sizes]

# # Print the results
# for (i, size) in enumerate(sizes)
#     println("Matrix size: $size x $size, Time taken: $(julia_times[i]) seconds")
# end
