## Parallelizing Incremental Batch Gradient Descent with Hogwild!

Stochastic gradient descent (SGD) was initially thought to be an inherently serial
algorithm: each thread must wait for another to update before moving on to the
next iteration of the steepest descent computation. To make the algorithm parallel,
an obvious approach would be to lock each update step (i.e. each thread locks the
current estimate of the solution before updating the parameter, and unlocks once
the update is done). One quickly sees that the overhead from numerous lock
allocations and scheduling supersedes the actual times spent on updates — this is
where the Hogwild! algorithm shines: by removing all thread locks from parallel SGD
code, the asynchronous parallelization has shown to be mathematically efficient and
resulted in magnitudes of speedup over the vanilla version. Here we study the
speedup and accuracy of the original Hogwild! algorithm as applied to a batch SGD
by varying configurations (i.e. batch sizes and design matrix dimensions) using a
serial implementation in C++ as a baseline compared against a CUDA-parallized
version leveraging two GeForce RTX 2080 Ti GPUs (11 GB memory each) on the cuda2
CIMS server. For fixed matrix size, smaller batch sizes led to greater speedups than
those for larger batch sizes. On the other hand, the test loss for the parallel version is
greater than that of the serial version.

[Final Report](https://github.com/gcalbertini/hpc_cuda_nyu/blob/6a4a7586bb2e74d3be8f1548b43354680d24fc14/report_cuda_hogwild.pdf) \
[Slide Presentation and Summary of Results](https://github.com/gcalbertini/hpc_cuda_nyu/blob/6a4a7586bb2e74d3be8f1548b43354680d24fc14/slides_cuda_hogwild.pptx)

Paper of reference: https://arxiv.org/abs/1106.5730
