struct MarkovChain end

abstract type HMM end

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

function ConvolvedHMM(μ::Vector, σ::Vector, σₙ²::Real, W::AbstractMatrix, Σᵨ::AbstractMatrix, P::AbstractMatrix, D = nothing)
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

function ConvolvedHMM(μ::Vector, σ::Vector, σₙ²::Real, ω, ρ, n::Int, P::AbstractMatrix, D = nothing)
  W  = kernelmatrix(ω, n)
  Σᵨ = Matrix(Symmetric(kernelmatrix(ρ, n)))
  
  ConvolvedHMM(μ, σ, σₙ², W, Σᵨ, P, D)
end

nstates(hmm::ConvolvedHMM) = length(hmm.μ)

function sample(hmm::ConvolvedHMM)
  @unpack Σᵨ, logP, logπ₀ = hmm
  n = size(Σᵨ, 1)
  
  s = sampleMarkovChain(n, exp.(logP), exp.(logπ₀))
  m, d = sample(hmm, s)
  
  s, m, d
end

function sample(hmm::ConvolvedHMM, s)
  @unpack μ, σ, W, D, Σᵨ, Σₐ, logP, logπ₀ = hmm
  n = size(Σᵨ, 1)
  Σₛ = Diagonal(σ[s])
  μₛ = μ[s]
  m = rand(MvNormal(μₛ, Symmetric(Σₛ * Σᵨ * Σₛ)))
  d = rand(MvNormal(W * D * m, Σₐ))
  m, d
end

# likelihood function p(d|s)
function loglikelihood(s, d, μ, σ, Σₐ, H, Σᵨ)
  Σₛ = Diagonal(σ[s])
  Σd = H * Σₛ * Σᵨ * Σₛ * H' .+ Σₐ
  μd = H * μ[s]
  logpdf(MvNormal(μd, Matrix(Hermitian(Σd))), d)
end

function Distributions.logpdf(hmm::ConvolvedHMM, s::AbstractArray, data::Vector)
    @unpack μ, σ, W, D, Σᵨ, Σₐ, logP, logπ₀ = hmm
    l = logπ₀[s[1]]
    @inbounds for i = 2:length(s)
      l += logP[s[i-1], s[i]]
    end
    
    l += loglikelihood(s, data, μ, σ, Σₐ, W * D, Σᵨ)
    l
end

@inline isstochastic(P::AbstractMatrix) = all(sum(P, dims=2) .≈ 1.) && isequal(size(P)...)