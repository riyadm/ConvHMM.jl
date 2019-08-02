abstract type HMM end

"""
    ConvolvedHMM(P, μ, σ, Σᵨ, W, σₙ²)
    ConvolvedHMM(P, μ, σ, ρ, ω, σₙ², n)

A convolved HMM model. If functions `ρ` and `ω` are used to define correlation and wavelet kernels, sequence length `n` has to be provided. For matrix definition `size(Σᵨ,1) == size(W,1) == n` is assumed.

## Arguments

* `P` - stochastic matrix (`KxK`)
* `μ` - vector of emission means (`Kx1`)
* `σ` - vector of emission standrd deviations (`Kx1`)

* `Σᵨ` - emission correlation matrix (`nxn`)
* `ρ` - stationary correlation function or a callable object of the form ``ρ(h)``

* `W` - transformation matrix from emissions to observations
* `ω` - stationary wavelet or a callable object of the form ``ω(h)``

* `σₙ²` - observation noise variance
* `n` - latent sequence length
"""
struct ConvolvedHMM{V,S,M} <: HMM
    μ::V
    σ::V
    W::M
    D::M
    Σᵨ::M
    Σₐ::M
    logP::M
    logπ₀::V
end

function ConvolvedHMM(P::AbstractMatrix, μ::Vector, σ::Vector, Σᵨ::AbstractMatrix, W::AbstractMatrix, σₙ²::Real, D = nothing)
  n = size(W, 1)
    
  @assert (length(μ) == length(σ) && size(Σᵨ, 1) == n) "dimensions mismatch"
  @assert isposdef(Σᵨ) "correlation matrix not positive definite"
  @assert isstochastic(P) "invalid stochastic matrix P"
    
  if D == nothing
      @warn "differential matrix defaulted to identity"
      D = diagm(0 => ones(n))
  end
  Σₐ = diagm(0 => fill(σₙ², n)) 
  e = eigen(Matrix(P'))
  π₀ = abs.(normalize(e.vectors[:, findfirst(e.values .≈ 1.)], 1))
    
  V = promote_type(typeof(μ), typeof(σ))
  S = promote_type(typeof(σₙ²), eltype(σ))
  M = promote_type(typeof(W), typeof(Σᵨ), typeof(P))
  ConvolvedHMM{V,S,M}(μ, σ, W, D, Σᵨ, Σₐ, log.(P), log.(π₀))
end

function ConvolvedHMM(P::AbstractMatrix, μ::Vector, σ::Vector, ρ, ω, n::Int, σₙ²::Real, D = nothing)
  W  = kernelmatrix(ω, n)
  Σᵨ = Matrix(Symmetric(kernelmatrix(ρ, n)))
  
  ConvolvedHMM(P, μ, σ, Σᵨ, W, σₙ², D)
end

"""
    nstates(hmm)

Return the hidden state space size of `hmm`.
"""
nstates(hmm::ConvolvedHMM) = length(hmm.μ)

"""
    sample(hmm)

Sample `(state sequence, emissions, observations)` triple from `hmm`. 
"""
function sample(hmm::ConvolvedHMM)
  @unpack Σᵨ, logP, logπ₀ = hmm
  n = size(Σᵨ, 1)
  
  s = sampleMarkovChain(n, exp.(logP), exp.(logπ₀))
  m, d = sample(hmm, s)
  
  s, m, d
end

"""
    sample(hmm, s)

Sample tuple `(emissions, observations)` from `hmm`, conditioned to a state sequence `s`.
"""
function sample(hmm::ConvolvedHMM, s)
  @unpack μ, σ, W, D, Σᵨ, Σₐ, logP, logπ₀ = hmm
  n = size(Σᵨ, 1)
  Σₛ = Diagonal(σ[s])
  μₛ = μ[s]
  m = rand(MvNormal(μₛ, Symmetric(Σₛ * Σᵨ * Σₛ)))
  d = rand(MvNormal(W * D * m, Σₐ))
  m, d
end

# log-likelihood function p(d|s)
function loglikelihood(s, d, μ, σ, Σₐ, H, Σᵨ)
  Σₛ = Diagonal(σ[s])
  Σd = H * Σₛ * Σᵨ * Σₛ * H' .+ Σₐ
  μd = H * μ[s]
  logpdf(MvNormal(μd, Matrix(Hermitian(Σd))), d)
end

"""
    logpdf(hmm, s, obs)

Evaluate the logarithm of the joint probability density ``p(s|d)`` of `hmm` provided latent sequence `s` and observation vector `obs`.
"""
function Distributions.logpdf(hmm::ConvolvedHMM, s::AbstractArray, obs::Vector)
    @unpack μ, σ, W, D, Σᵨ, Σₐ, logP, logπ₀ = hmm
    l = logπ₀[s[1]]
    @inbounds for i = 2:length(s)
      l += logP[s[i-1], s[i]]
    end
    
    l += loglikelihood(s, obs, μ, σ, Σₐ, W * D, Σᵨ)
    l
end

"""
    isstochastic(P)

Check if `P` is a valid stochastic matrix.
"""
@inline isstochastic(P::AbstractMatrix) = all(sum(P, dims=2) .≈ 1.) && isequal(size(P)...)