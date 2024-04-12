function time_matmul_julia(size)
    A = rand(size, size)
    B = rand(size, size)
    start_time = time()  # Capture the start time
    C = A * B  # Perform matrix multiplication
    end_time = time()  # Capture the end time
    return end_time - start_time  # Calculate the elapsed time
end

sizes = [100, 300, 500, 700, 900, 1300, 2000, 4000, 6000, 8000]
julia_times = [time_matmul_julia(size) for size in sizes]

# Print the results
for (i, size) in enumerate(sizes)
    println("Matrix size: $size x $size, Time taken: $(julia_times[i]) seconds")
end
