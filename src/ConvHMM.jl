module ConvHMM

using LinearAlgebra
using Distributions
using Parameters
using SparseArrays
import Distributions: logpdf
import Base: rand

include("utils.jl")
include("convolvedhmm.jl")
include("approximator.jl")
include("baumwelch.jl")

export
  ConvolvedHMM,
  llhmm,
  ll,
  forward,
  backward,
  propose,
  randpair

end