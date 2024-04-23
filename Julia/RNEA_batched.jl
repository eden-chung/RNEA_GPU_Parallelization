batch_size = 1000

q = [-0.3369  1.2966 -0.6775 -1.4218 -0.7067 -0.135  -1.1495]
qd = [ 0.433  -0.4216 -0.6454 -1.8605 -0.0131 -0.4583  0.7412]
u = [ 0.7418  1.9284 -0.9039  0.0334  1.1799 -1.946   0.3287]
n = 7

parent_id_arr = [-1, 0, 1, 2, 3, 4, 5]

S_arr = [[0. 0. 1. 0. 0. 0.]
[0. 0. 1. 0. 0. 0.]
[0. 0. 1. 0. 0. 0.]
[0. 0. 1. 0. 0. 0.]
[0. 0. 1. 0. 0. 0.]
[0. 0. 1. 0. 0. 0.]
[0. 0. 1. 0. 0. 0.]]

# Imat_arr = [
#     [
#         [ 0.2091  0.0  0.0  0.0  -0.6912 -0.1728]
#         [ 0.0  0.1989 -0.0003  0.6912  0.0  0.0]
#         [ 0.0 -0.0003  0.0227  0.1728  0.0  0.0]
#         [ 0.0  0.6912  0.1728  5.76    0.0  0.0]
#         [-0.6912  0.0  0.0  0.0  5.76    0.0]
#         [-0.1728  0.0  0.0  0.0  0.0  5.76  ]
#     ]
#     [
#         [ 0.0971 -0.0 -0.0  0.0  -0.2667  0.3746]
#         [-0.0  0.0528 -0.0  0.2667  0.0  -0.0019]
#         [-0.0 -0.0  0.0552 -0.3746  0.0019  0.0]
#         [ 0.0  0.2667 -0.3746  6.35    0.0  0.0]
#         [-0.2667  0.0  0.0019  0.0  6.35    0.0]
#         [ 0.3746 -0.0019  0.0  0.0  0.0  6.35  ]
#     ]
#     [
#         [ 0.1496  0.0  0.0  0.0  -0.455   0.105 ]
#         [ 0.0  0.1421  0.0003  0.455   0.0  0.0]
#         [ 0.0  0.0003  0.014  -0.105   0.0  0.0]
#         [ 0.0  0.455  -0.105   3.5     0.0  0.0]
#         [-0.455   0.0  0.0  0.0  3.5     0.0]
#         [ 0.105   0.0  0.0  0.0  0.0  3.5   ]
#     ]
#     [
#         [ 0.0566  0.0  0.0  0.0  -0.119   0.2345]
#         [ 0.0  0.0245  0.0  0.119   0.0  0.0]
#         [ 0.0  0.0  0.0374 -0.2345  0.0  0.0]
#         [ 0.0  0.119  -0.2345  3.5     0.0  0.0]
#         [-0.119   0.0  0.0  0.0  3.5     0.0]
#         [ 0.2345  0.0  0.0  0.0  0.0  3.5   ]
#     ]
#     [
#         [ 0.0536 -0.0  0.0  0.0  -0.266   0.0735]
#         [-0.0  0.0491  0.0  0.266   0.0  -0.0003]
#         [ 0.0  0.0  0.0075 -0.0735  0.0003  0.0]
#         [ 0.0  0.266  -0.0735  3.5     0.0  0.0]
#         [-0.266   0.0  0.0003  0.0  3.5     0.0]
#         [ 0.0735 -0.0003  0.0  0.0  0.0  3.5   ]
#     ]
#     [
#         [ 0.0049  0.0  0.0  0.0  -0.0007  0.0011]
#         [ 0.0  0.0047 -0.0  0.0007  0.0  0.0]
#         [ 0.0 -0.0  0.0036 -0.0011  0.0  0.0]
#         [ 0.0  0.0007 -0.0011  1.8     0.0  0.0]
#         [-0.0007  0.0  0.0  0.0  1.8     0.0]
#         [ 0.0011  0.0  0.0  0.0  0.0  1.8   ]
#     ]
#     [
#         [ 0.006   0.0  0.0  0.0  -0.024   0.    ]
#         [ 0.0  0.006   0.0  0.024   0.0  0.    ]
#         [ 0.0  0.0  0.005   0.0  0.0  0.    ]
#         [ 0.0  0.024   0.0  1.2     0.0  0.0]
#         [-0.024   0.0  0.0  0.0  1.2     0.0]
#         [ 0.0  0.0  0.0  0.0  0.0  1.2   ]
#     ]
# ]

imat1 = reshape([0.2091  0.0  0.0  0.0  -0.6912 -0.1728;
0.0  0.1989 -0.0003  0.6912  0.0  0.0;
0.0 -0.0003  0.0227  0.1728  0.0  0.0;
0.0  0.6912  0.1728  5.76    0.0  0.0;
-0.6912  0.0  0.0  0.0  5.76    0.0;
-0.1728  0.0  0.0  0.0  0.0  5.76  ], (6,6,1))

imat2 = reshape([0.0971 -0.0 -0.0  0.0  -0.2667  0.3746;
-0.0  0.0528 -0.0  0.2667  0.0  -0.0019;
-0.0 -0.0  0.0552 -0.3746  0.0019  0.0;
0.0  0.2667 -0.3746  6.35    0.0  0.0;
-0.2667  0.0  0.0019  0.0  6.35    0.0;
0.3746 -0.0019  0.0  0.0  0.0  6.35  ], (6,6,1))

imat3 = reshape([0.1496  0.0  0.0  0.0  -0.455   0.105;
0.0  0.1421  0.0003  0.455   0.0  0.0;
0.0  0.0003  0.014  -0.105   0.0  0.0;
0.0  0.455  -0.105   3.5     0.0  0.0;
-0.455   0.0  0.0  0.0  3.5     0.0;
0.105   0.0  0.0  0.0  0.0  3.5   ], (6,6,1))

imat4 = reshape([0.0566  0.0  0.0  0.0  -0.119   0.2345;
0.0  0.0245  0.0  0.119   0.0  0.0;
0.0  0.0  0.0374 -0.2345  0.0  0.0;
0.0  0.119  -0.2345  3.5     0.0  0.0;
-0.119   0.0  0.0  0.0  3.5     0.0;
0.2345  0.0  0.0  0.0  0.0  3.5   ], (6,6,1))

imat5 = reshape([0.0536 -0.0  0.0  0.0  -0.266   0.0735;
-0.0  0.0491  0.0  0.266   0.0  -0.0003;
0.0  0.0  0.0075 -0.0735  0.0003  0.0;
0.0  0.266  -0.0735  3.5     0.0  0.0;
-0.266   0.0  0.0003  0.0  3.5     0.0;
0.0735 -0.0003  0.0  0.0  0.0  3.5   ], (6,6,1))

imat6 = reshape([0.0049  0.0  0.0  0.0  -0.0007  0.0011;
0.0  0.0047 -0.0  0.0007  0.0  0.0;
0.0 -0.0  0.0036 -0.0011  0.0  0.0;
0.0  0.0007 -0.0011  1.8     0.0  0.0;
-0.0007  0.0  0.0  0.0  1.8     0.0;
0.0011  0.0  0.0  0.0  0.0  1.8   ], (6,6,1))

imat7 = reshape([0.006   0.0  0.0  0.0  -0.024   0.    ;
0.0  0.006   0.0  0.024   0.0  0.    ;
0.0  0.0  0.005   0.0  0.0  0.    ;
0.0  0.024   0.0  1.2     0.0  0.0;
-0.024   0.0  0.0  0.0  1.2     0.0;
0.0  0.0  0.0  0.0  0.0  1.2   ], (6,6,1))

Imat_arr = cat(imat1, imat2, imat3, imat4, imat5, imat6, imat7, dims=3)
Imat_arr = permutedims(Imat_arr, (3, 1, 2))

print("Imat_arr shape: ", size(Imat_arr), "\n")


xmat1 = reshape([0.4089 -0.9126 0.0 0.0 0.0 0.0;
         0.9126 0.4089 0.0 0.0 0.0 0.0;
         0.0 0.0 1.0 0.0 0.0 0.0;
         0.1437 0.0644 0.0 0.4089 -0.9126 0.0;
         -0.0644 0.1437 0.0 0.9126 0.4089 0.0;
         0.0 0.0 0.0 0.0 0.0 1.0], (6, 6, 1))

xmat2 = reshape([-0.4089 0.0 -0.9126 0.0 0.0 0.0;
         -0.9126 0.0 0.4089 0.0 0.0 0.0;
         0.0 1.0 0.0 0.0 0.0 0.0;
         0.0 -0.0828 0.0 -0.4089 0.0 -0.9126;
         0.0 -0.1848 0.0 -0.9126 0.0 0.4089;
         -0.2025 0.0 0.0 0.0 1.0 0.0], (6, 6, 1))

xmat3 = reshape([-0.4089 0.0 -0.9126 0.0 0.0 0.0;
        -0.9126 0.0 0.4089 0.0 0.0 0.0;
        0.0 1.0 0.0 0.0 0.0 0.0;
        -0.1866 0.0 0.0836 -0.4089 0.0 -0.9126;
        0.0836 0.0 0.1866 -0.9126 0.0 0.4089;
        0.0 0.0 0.0 0.0 1.0 0.0], (6, 6, 1))

xmat4 = reshape([0.4089 0.0 -0.9126 0.0 0.0 0.0;
        0.9126 0.0 0.4089 0.0 0.0 0.0;
        0.0 -1.0 0.0 0.0 0.0 0.0;
        0.0 0.0881 0.0 0.4089 0.0 -0.9126;
        0.0 0.1967 0.0 0.9126 0.0 0.4089;
        0.2155 0.0 0.0 0.0 -1.0 0.0], (6, 6, 1))

xmat5 = reshape([-0.4089 0.0 -0.9126 0.0 0.0 0.0;
    -0.9126 0.0 0.4089 0.0 0.0 0.0;
    0.0 1.0 0.0 0.0 0.0 0.0;
    -0.1684 0.0 0.0754 -0.4089 0.0 -0.9126;
    0.0754 0.0 0.1684 -0.9126 0.0 0.4089;
    0.0 0.0 0.0 0.0 1.0 0.0], (6, 6, 1))

xmat6 = reshape([0.4089 0.0 -0.9126 0.0 0.0 0.0;
    0.9126 0.0 0.4089 0.0 0.0 0.0;
    0.0 -1.0 0.0 0.0 0.0 0.0;
    0.0 0.0881 0.0 0.4089 0.0 -0.9126;
    0.0 0.1967 0.0 0.9126 0.0 0.4089;
    0.2155 0.0 0.0 0.0 -1.0 0.0], (6, 6, 1))

xmat7 = reshape([-0.4089 0.0 -0.9126 0.0 0.0 0.0;
    -0.9126 0.0 0.4089 0.0 0.0 0.0;
    0.0 1.0 0.0 0.0 0.0 0.0;
    -0.0739 0.0 0.0331 -0.4089 0.0 -0.9126;
    0.0331 0.0 0.0739 -0.9126 0.0 0.4089;
    0.0 0.0 0.0 0.0 1.0 0.0], (6, 6, 1))


xmat_func_arr = cat(xmat1, xmat2, xmat3, xmat4, xmat5, xmat6, xmat7, dims=3)
xmat_func_arr = permutedims(xmat_func_arr, (3, 1, 2))

print("xmat_func_arr shape: ", size(xmat_func_arr))


h_vec_batched = ones(6, batch_size)
h_I_batched = ones(6, 6, batch_size)
h_output_batched = zeros(6, batch_size)
h_mxS_output_batched = zeros(6, batch_size) 


# Now repeat along the fourth dimension only
h_xmat_func_arr_batched = repeat(xmat_func_arr, outer=[1, 1, 1, batch_size])

print("h_xmat_func_arr_batched shape: ", size(h_xmat_func_arr_batched))


h_S_arr_batched = repeat(S_arr, outer=[1, 1, batch_size])
h_Imat_arr_batched = repeat(Imat_arr, outer=[1, 1, 1, batch_size])
h_q_batched = repeat(reshape(q, :, 1), outer=[1, batch_size])
h_qd_batched = repeat(reshape(qd, :, 1), outer=[1, batch_size])

print("h_S_arr_batched shape: ", size(h_S_arr_batched), "\n")
print("h_Imat_arr_batched shape: ", size(h_Imat_arr_batched), "\n")
print("h_q_batched shape: ", size(h_q_batched), "\n")
print("h_qd_batched shape: ", size(h_qd_batched), "\n")

h_crOp_output_batched = zeros(6, 6, batch_size)
h_mxS_output_batched = zeros(6, batch_size)
h_vxIv_output_batched = zeros(6, batch_size)

print("h_crOp_output_batched shape: ", size(h_crOp_output_batched), "\n")
print("h_mxS_output_batched shape: ", size(h_mxS_output_batched), "\n")
print("h_vxIv_output_batched shape: ", size(h_vxIv_output_batched), "\n")

#########

using LinearAlgebra

#cross operator
function cross_operator_batched(d_vec, d_output)
    for k in 1:size(d_vec, 2)
        d_output[1, 2, k] = -d_vec[2, k]
        d_output[1, 3, k] = d_vec[1, k]
        d_output[2, 1, k] = d_vec[2, k]
        d_output[2, 3, k] = -d_vec[1, k]
        d_output[3, 1, k] = -d_vec[2, k]
        d_output[3, 2, k] = d_vec[1, k]

        d_output[4, 2, k] = -d_vec[5, k]
        d_output[4, 3, k] = d_vec[4, k]
        d_output[4, 5, k] = -d_vec[2, k]
        d_output[4, 6, k] = d_vec[1, k]
        d_output[5, 1, k] = d_vec[5, k]
        d_output[5, 3, k] = -d_vec[4, k]
        d_output[5, 4, k] = d_vec[2, k]
        d_output[5, 6, k] = -d_vec[1, k]
        d_output[6, 1, k] = -d_vec[5, k]
        d_output[6, 2, k] = d_vec[4, k]
        d_output[6, 4, k] = -d_vec[2, k]
        d_output[6, 5, k] = d_vec[1, k]
    end
end

function benchmark_cross_operator(batch_size, alpha, repetitions)
    h_vec_batched = ones(Float64, 6, batch_size)
    h_output_batched = zeros(Float64, 6, 6, batch_size)

    cross_operator_batched(h_vec_batched, h_output_batched)
    println("Cross operator output shape: ", size(h_output_batched))

    start_time = time()
    for i in 1:repetitions
        cross_operator_batched(h_vec_batched, h_output_batched)
    end
    elapsed_time = time() - start_time
    println("Benchmark time for $repetitions repetitions: $elapsed_time seconds")
end

function mxS(S, vec, vec_output, mxS_output, alpha=1)
    print("dimensions of S", size(S), "\n")
    cross_operator_batched(vec, vec_output)
    for i in 1:size(vec_output, 3)
        if ndims(S) == 3
            mxS_output[:, i] .= alpha * (vec_output[:, :, i] * S[:, :, i])
        else
            mxS_output[:, i] .= alpha * (vec_output[:, :, i] * S[:, i])
        end
    end
end


function benchmark_mxS(batch_size, alpha, repetitions)
    h_vec_batched = ones(6, batch_size)
    h_s_vec_batched = repeat(ones(6, 1), 1, 1, batch_size)
    h_output_batched = zeros(6, 6, batch_size)
    h_mxS_output_batched = zeros(6, batch_size)

    # Timing
    start_time = time()
    for i in 1:repetitions
        mxS(h_s_vec_batched, h_vec_batched, h_output_batched, h_mxS_output_batched, alpha)
    end
    end_time = time()

    elapsed_time = end_time - start_time
    println("Benchmark time for $repetitions repetitions: $elapsed_time seconds")
end

function vxIv(vec, Imat, res, batch_size)
    temp = sum(Imat .* vec, dims=1)  # Element-wise multiplication and summation along the first dimension

    vecXIvec = zeros(Float64, 6, batch_size)

    for i in 1:batch_size
        vecXIvec[1, i] = -vec[3, 1, i] * temp[1, 1, i] + vec[2, 1, i] * temp[1, 1, i] - vec[3+3, 1, i] * temp[1, 1, i] + vec[2+3, 1, i] * temp[1, 1, i]
        vecXIvec[2, i] = vec[3, 1, i] * temp[1, 1, i] - vec[1, 1, i] * temp[1, 1, i] + vec[3+3, 1, i] * temp[1, 1, i] - vec[1+3, 1, i] * temp[1, 1, i]
        vecXIvec[3, i] = -vec[2, 1, i] * temp[1, 1, i] + vec[1, 1, i] * temp[1, 1, i] - vec[2+3, 1, i] * temp[1, 1, i] + vec[1+3, 1, i] * temp[1, 1, i]
        vecXIvec[4, i] = -vec[3, 1, i] * temp[1, 1, i] + vec[2, 1, i] * temp[1, 1, i]
        vecXIvec[5, i] = vec[3, 1, i] * temp[1, 1, i] - vec[1, 1, i] * temp[1, 1, i]
        vecXIvec[6, i] = -vec[2, 1, i] * temp[1, 1, i] + vec[1, 1, i] * temp[1, 1, i]
    end

    res .= vecXIvec
end

function benchmark_vxIv(batch_size, alpha, repetitions)
    # COMPARING to batched
    h_vec_batched = vec = ones(Float64, 6, 1, batch_size)#ones(Float64, 6, batch_size)
    h_I_batched = ones(Float64, 6, 6, batch_size)
    h_output_batched = zeros(Float64, 6, batch_size)

    # CPU/with numpy
    @time vxIv(h_vec_batched, h_I_batched, h_output_batched, batch_size) # warm-up once
    println("vxIV shape: ", size(h_output_batched))
    # testing in loop of 100
    startnext = time()
    for i in 1:repetitions
        vxIv(h_vec_batched, h_I_batched, h_output_batched, batch_size)
    end
    println("CPU without jit Batched vxIv: ", time() - startnext)
end

function rnea_fpass(num_joints, parent_id_arr, xmat_func_arr, S_arr, Imat_arr, crOp_output, mxS_output, vxIv_output, batch_size, q, qd, qdd = nothing, GRAVITY = -9.81)
    n = num_joints

    v = zeros(6, n, batch_size)
    a = zeros(6, n, batch_size)
    f = zeros(6, n, batch_size)

    gravity_vec = zeros(6, batch_size)
    gravity_vec[6, :] .= GRAVITY

    for ind in 1:n
        parent_ind = parent_id_arr[ind]

        print("shape of xmat_Func_arr", size(xmat_func_arr))
        Xmat = xmat_func_arr[ind, :, :, :]

        println("Shape of S_arr: ", size(S_arr))
        S = S_arr[ind, :, :]

        if parent_ind == -1
            print("shape of Xmat: ", size(Xmat))
            print("shape of gravity_vec: ", size(gravity_vec))
            #a[:, ind, :] .= Xmat * gravity_vec
            gravity_vec_3d = reshape(gravity_vec, 1, size(gravity_vec, 1), size(gravity_vec, 2))
            a[:, ind, :] .= dropdims(sum(Xmat .* gravity_vec_3d, dims=1), dims=1)
            print("shape of a: ", size(a), "\n")
        else
            v[:, ind, :] .= Xmat * v[:, parent_ind + 1, :]
            a[:, ind, :] .= Xmat * a[:, parent_ind + 1, :]
        end

        v[:, ind, :] .+= S .* qd[ind]

        print("S size", size(S), "\n")
        print("v size", size(v), "\n")
        print("v[:, ind, :] size", size(v[:, ind, :]), "\n")
        print("cr0p output", size(crOp_output), "\n")
        print("mxS output", size(mxS_output), "\n")
        print("qd size", size(qd), "\n")
        print("qd[ind] size", size(qd[ind]), "\n")

        mxS(S, v[:, ind, :], crOp_output, mxS_output, qd[ind])

        print("test1")
        a[:, ind, :] .+= mxS_output

        if qdd !== nothing
            a[:, ind, :] .+= S .* qdd[ind]
        end

        Imat = Imat_arr[ind]
        temp = Imat * a[:, ind, :]
        vxIv(v[:, ind, :], Imat, vxIv_output, batch_size)
        f[:, ind, :] .= temp + vxIv_output
    end

    return v, a, f
end

function benchmark_rnea_fpass(n, parent_id_arr, xmat_func_arr, S_arr, Imat_arr, crOp_output, mxS_output, vxIv_output, batch_size, q, qd, itr, qdd=nothing, GRAVITY=-9.81)
    v, a, f = rnea_fpass(n, parent_id_arr, xmat_func_arr, S_arr, Imat_arr, crOp_output, mxS_output, vxIv_output, batch_size, q, qd, qdd, GRAVITY)

    elapsed_time = @elapsed for i in 1:itr
        rnea_fpass(n, parent_id_arr, xmat_func_arr, S_arr, Imat_arr, crOp_output, mxS_output, vxIv_output, batch_size, q, qd, qdd, GRAVITY)
    end

    println("CPU batched fpass with Julia: $elapsed_time seconds")
end


function main()
    alpha = 0.1
    repetitions = 100
    benchmark_cross_operator(batch_size, alpha, repetitions)
    benchmark_mxS(batch_size, alpha, repetitions)
    benchmark_vxIv(batch_size, alpha, repetitions)
    benchmark_rnea_fpass(n, parent_id_arr, h_xmat_func_arr_batched, h_S_arr_batched, h_Imat_arr_batched, h_crOp_output_batched, h_mxS_output_batched, h_vxIv_output_batched, batch_size, h_q_batched, h_qd_batched, 100)
end

main()
