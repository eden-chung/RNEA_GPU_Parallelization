import numpy as np

import matplotlib.pyplot as plt

# Data for plotting
sizes = [100, 300, 500, 700, 900, 1300, 2000, 4000, 6000, 8000]

# Times for Julia
times_julia_matmul = [
    0.0002028942108154297, 0.0015261173248291016, 0.0018839836120605469,
    0.003946065902709961, 0.008172035217285156, 0.020615816116333008,
    0.07111001014709473, 0.5325109958648682, 2.074859142303467, 4.900615215301514
]

# Times for CUDA
times_cuda_matmul = [
    0.000108512, 0.000299264, 0.00027811198, 0.00085958396,
    0.001239648, 0.002941984, 0.008219328, 0.06722493,
    0.21636927, 0.5181641
]

times_python_matmul = [0.0003440380096435547, 0.010555028915405273, 0.022825956344604492, 0.05473208427429199, 0.0564577579498291, 0.12842774391174316, 0.5433192253112793, 2.6279523372650146, 8.354299068450928, 19.75766134262085]
times_python_cuda_matmul = [0.16113495826721191, 0.004086971282958984, 0.00036525726318359375, 9.799003601074219e-05, 0.00013327598571777344, 9.965896606445312e-05, 0.0003452301025390625, 0.0004994869232177734, 0.0006930828094482422, 0.0006248950958251953]

times_python_dot = [1.9788742065429688e-05, 1.33514404296875e-05, 0.0011932849884033203, 0.0021626949310302734, 0.004592180252075195, 0.012362241744995117, 0.008512496948242188, 0.014724493026733398, 0.013254642486572266, 0.0151824951171875, 0.025162220001220703, 0.027042388916015625, 0.028426647186279297]
times_torch_dot = [7.581710815429688e-05, 2.4557113647460938e-05, 9.393692016601562e-05, 7.653236389160156e-05, 0.00010514259338378906, 9.608268737792969e-05, 9.775161743164062e-05, 9.942054748535156e-05, 9.226799011230469e-05, 9.799003601074219e-05, 9.799003601074219e-05, 0.00011968612670898438, 0.00010228157043457031]


times_python_inversion = [0.0061724185943603516, 0.013892650604248047, 0.03299260139465332, 0.0608518123626709, 0.12616896629333496, 0.18459510803222656, 0.24437355995178223, 0.36903953552246094, 0.4780538082122803, 0.6630184650421143, 1.14896559715271, 5.802893161773682, 19.580580472946167, 56.60026240348816]
times_torch_inversion = [0.3084089756011963, 0.012573480606079102, 0.024840593338012695, 0.04392433166503906, 0.06574106216430664, 0.07049250602722168, 0.07761263847351074, 0.07291364669799805, 0.10363364219665527, 0.14080452919006348, 0.2320852279663086, 0.8887021541595459, 4.272075891494751, 12.337721347808838]


# Plotting the results
plt.figure(figsize=(10,6))
plt.plot(sizes, times_julia_matmul, marker='o', label='Julia', color='red')
plt.plot(sizes, times_cuda_matmul, marker='o', label='CUDA.jl', color='green')
# plt.plot(sizes, times_python_matmul, marker='o', label='Python', color='tab:blue')
# plt.plot(sizes, times_python_cuda_matmul, marker='o', label='PyTorch', color='orange')

# Title and labels
plt.title('Matrix Multiplication Performance: Julia, CUDA.jl')
plt.xlabel('Matrix Size')
plt.ylabel('Time (seconds)')

# Legend
plt.legend()

# Grid and show plot
plt.grid(True)
plt.show()


# sizes_inversion = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 4000, 7000, 10000]
# julia_times_inversion = [0.0020530223846435547, 0.003587961196899414, 0.008840799331665039, 0.01601696014404297, 0.028993844985961914, 0.04532885551452637, 1.2533419132232666, 0.08825993537902832, 0.1355760097503662, 0.18045997619628906, 0.3117859363555908, 1.2057139873504639, 6.009594202041626, 17.65435791015625]
# cuda_times_inversion = [0.001162752, 0.00580368, 0.006961728, 0.011603456, 0.018463807, 0.022008032, 0.030136127, 0.035900127, 0.046443038, 0.056463744, 0.09003046, 0.25004572, 1.094886, 2.764174]

# # Plotting the results
# plt.figure(figsize=(10,6))
# plt.plot(sizes_inversion, julia_times_inversion, marker='o', label='Julia', color='red')
# plt.plot(sizes_inversion, cuda_times_inversion, marker='o', label='CUDA.jl', color='green')
# # plt.plot(sizes_inversion, times_python_inversion, marker='o', label='Python', color='tab:blue')
# # plt.plot(sizes_inversion, times_torch_inversion, marker='o', label='PyTorch', color='orange')


# # Title and labels
# #plt.title('Matrix Inversion Performance: Python, PyTorch, Julia, CUDA.jl')
# plt.title('Matrix Inversion Performance: Julia, CUDA.jl')

# plt.xlabel('Matrix Size')
# plt.ylabel('Time (seconds)')

# # Legend
# plt.legend()

# # Grid and show plot
# plt.grid(True)
# plt.show()


# import matplotlib.pyplot as plt

# # Data for plotting
# labels = ['mxS', 'vxIv', 'fpass', 'bpass', 'RNEA']

# numpy_times = [0.0073125471, 0.0132675189, 3.710159997, 0.6121313864, 3.54439054]
# pytorch_times = [0.0633323941, 0.0929346393, 0.2108642872, 0.1830261935, 0.3851471845]

# # Plotting the results
# plt.figure(figsize=(10,6))
# x = np.arange(len(labels))
# width = 0.35
# plt.bar(x - width/2, numpy_times, width, label='NumPy on CPU')
# plt.bar(x + width/2, pytorch_times, width, label='PyTorch on GPU')

# # Title and labels
# plt.title('Performance Comparison: NumPy vs PyTorch')
# plt.xlabel('Operation')
# plt.ylabel('Time (seconds)')

# # X-axis ticks and labels
# plt.xticks(x, labels, rotation=45, ha='right', fontsize=8)

# # Legend
# plt.legend()

# # Grid and show plot
# plt.grid(True)
# plt.show()


# Data for plotting
labels = ['cross_product', 'mxS']

julia_times = [0.4101729393005371, 4.354753017425537]
cuda_times = [2.9041030406951904, 18.969295978546143]

# Plotting the results
plt.figure(figsize=(10,6))
x = np.arange(len(labels))
width = 0.35
plt.bar(x - width/2, julia_times, width, label='Julia on CPU', color='red')
plt.bar(x + width/2, cuda_times, width, label='CUDA.jl on GPU', color='green')

# Title and labels
plt.title('Performance Comparison: Julia vs CUDA.jl')
plt.xlabel('Operation')
plt.ylabel('Time (seconds)')

# X-axis ticks and labels
plt.xticks(x, labels, rotation=45, ha='right', fontsize=8)

# Legend
plt.legend()

# Grid and show plot
plt.grid(True)
plt.show()
