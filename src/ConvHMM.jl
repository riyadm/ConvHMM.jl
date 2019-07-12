module ConvHMM

using LinearAlgebra
using Distributions
using Parameters
import Distributions: logpdf, loglikelihood

include("utils.jl")
include("convolvedhmm.jl")
include("likelihoods.jl")
include("approximator.jl")
include("forward.jl")
include("wavelets.jl")



export
  ConvolvedHMM,
  logpdf,
  loglikelihood,
  sample,
  mcmc,

  # wavelets
  kernelmatrix,
  Ricker,
  Ormsby,
  
  # forward-backward
  forward,
  viterbi,

  # likelihood approximations
  Truncation,
  Projection
end