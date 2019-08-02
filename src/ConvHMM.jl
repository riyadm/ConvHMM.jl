module ConvHMM

using LinearAlgebra
using Distributions
using Parameters
using Random: shuffle
using Requires

import Distributions: logpdf, loglikelihood

include("utils.jl")
include("wavelets.jl")
include("convolvedhmm.jl")
include("likelihoods.jl")
include("approximator.jl")
include("forward.jl")

# optionally load GeoStats.jl API
# function __init__()
#     @require GeoStatsBase="323cb8eb-fbf6-51c0-afd0-f8fba70507b2" include("geostats.jl")
# end

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
  LikelihoodApproximation,
  Truncation,
  Projection
end