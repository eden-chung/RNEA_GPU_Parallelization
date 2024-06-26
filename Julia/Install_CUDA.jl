#using Pkg
#Pkg.add("CUDA")

using CUDA, LinearAlgebra
#CUDA.versioninfo()


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


function time_matmul_julia(size)
    A = rand(size, size)
    B = rand(size, size)
    start_time = time()  # Capture the start time
    C = A * B # Perform matrix multiplication
    end_time = time()  # Capture the end time
    return end_time - start_time  # Calculate the elapsed time
end


function time_dot_product_julia(size)
    A = rand(size)
    B = rand(size)

    start_time = time()
    C = dot(A, B)
    end_time = time()
    return end_time - start_time
end

function time_dot_product_cuda(size)
    if !CUDA.functional()
        error("CUDA is not available or your GPU is not supported.")
    end

    print("size is", size)

    A = rand(size)
    B = rand(size)
    
    #A_gpu = CUDA.rand(size)
    #B_gpu = CUDA.rand(size)

    A_gpu = CUDA.CuArray(A)
    B_gpu = CUDA.CuArray(B)

    

    try
        result = dot(A_gpu, B_gpu)
        print("result is", result, "\n")
    catch ex
        println("Error during dot production  comptuation",  ex)
    end


    # start_time = CUDA.@elapsed begin
    #     C_gpu = dot(A_gpu, B_gpu)
    #     CUDA.synchronize()
    # end

    #return start_time
end

function time_matrix_inversion_julia(size)
    A = rand(size, size)

    start_time = time()
    inv(A)
    end_time = time()
    return end_time - start_time
end

function time_matrix_inversion_cuda(size)
    if !CUDA.functional()
        error("CUDA is not available or your GPU is not supported.")
    end
    
    A_gpu = CUDA.rand(size, size)

    CUDA.@sync inv(A_gpu)

    start_time = CUDA.@elapsed begin
        inv(A_gpu)
        CUDA.synchronize()
    end

    return start_time
end




sizes_matmul = [100, 300, 500, 700, 900, 1300, 2000, 4000, 6000, 8000]

println("sizes_matmul: ", sizes_matmul)

julia_times_matmul = [time_matmul_julia(size) for size in sizes_matmul]
print("julia_times_matmuml: ", julia_times_matmul)

# for (i, size) in enumerate(sizes_matmul)
#     println("Matrix size: $size x $size, Time taken for matrix multiplication Julia: $(julia_times_matmul[i]) seconds")
# end

cuda_times_matmul = [time_matmul_cuda(size) for size in sizes_matmul]
print("cuda_times_matmuml: ", cuda_times_matmul)



# for (i, size) in enumerate(sizes_matmul)
#     println("Matrix size: $size x $size, Time taken for matrix multiplication CUDA: $(cuda_times_matmul[i]) seconds")
# end


##now do dot product

# sizes_dot = [1000, 10000, 1_000_000, 2_000_000, 4_000_000, 6_000_000, 8_000_000, 10_000_000,
#          12_000_000, 14_000_000, 16_000_000, 18_000_000, 20_000_000]

# julia_times_dot = [time_dot_product_julia(size) for size in sizes_dot]

# println("sizes dot: ", sizes_dot)
# println("julia_times_dot: " julia_times_dot)

# for (i, size) in enumerate(sizes_dot)
#     println("Vector size: $size, Time taken for dot product Julia: $(julia_times_dot[i]) seconds")
# end

# cuda_times_dot = [time_dot_product_cuda(size) for size in sizes_dot]


# for (i, size) in enumerate(sizes_dot)
#     println("Vector size: $size, Time taken for dot product CUDA: $(cuda_times_dot[i]) seconds")
# end


sizes_inversion = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 4000, 7000, 10000]

println("sizes_inversion: ", sizes_inversion)

julia_times_inversion = [time_matrix_inversion_julia(size) for size in sizes_inversion]
println("julia_times_inversion", julia_times_inversion)

# for (i, size) in enumerate(sizes_dot)
#     println("Vector size: $size, Time taken for matrix inversion Julia: $(julia_times_inversion[i]) seconds")
# end

cuda_times_inversion = [time_matrix_inversion_cuda(size) for size in sizes_inversion]

println("cuda_times_inversion", cuda_times_inversion)

# for (i, size) in enumerate(sizes_dot)
#     println("Vector size: $size, Time taken for matrix inversion Julia: $(cuda_times_inversion[i]) seconds")
# end

