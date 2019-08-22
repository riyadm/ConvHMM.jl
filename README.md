# ConvHMM.jl
Implementation of Convolved HMM (Markov-Switching Gaussian Process) described in [this paper](https://arxiv.org/abs/1710.06613). Check my [Master's thesis](https://purl.stanford.edu/km001pf4033) for examples of application to seismic inverse problem.

## Installation
Get the latest stable Julia binaries [here](https://julialang.org/downloads/). If you are new to Julia, I suggest installing the [JuliaPro](https://juliacomputing.com/products/juliapro.html) distribution. Currently, the package can be installed through Julia's package manager:
```julia
]add https://github.com/riyadm/ConvHMM.jl
```

## Example

```julia
using ConvHMM

# domain size
n = 100

# emission moments
μ = [-1., 0., 1.]
σ = [.3, .2, .3]

# stochastic matrix
P = [.8 .1 .1;
     .1 .8 .1;
     .1 .1 .8]
	 
# Gaussian correlogram
ρ(h) = exp(-h^2/2)

# convolution kernel (wavelet)
ω = Ricker(60., 0.001)

# noise variance
σ² = 1e-4

# model definition
hmm = ConvolvedHMM(P, μ, σ, ρ, ω, n, σ²)

# approximate posterior
approximator = forward(hmm, obs, Projection)

# 1000 MCMC samples
samples = sample(approximator, 1000)
```