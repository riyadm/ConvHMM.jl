struct ConvolvedHMM{V,S,M}
    μs::V
    σs::V
    σₐ::S
    W::M
    Σᵨ::M
    Σₐ::M
    logP::M
    logπ₀::V
end

function ConvolvedHMM(μs::Vector, σs::Vector, σₐ::Real, W::Matrix, Σᵨ::Matrix, P::AbstractArray)
    n = size(W, 1)
    
    length(μs) == length(σs) && size(Σᵨ, 1) == n || throw(ArgumentError("dimensions mismatch"))
    isposdef(Σᵨ) || throw(ArgumentError("covariance matrix not positive definite"))
    
    Σₐ = Matrix(diagm(0 => fill(σₐ^2, n))) 
    e = eigen(Matrix(P'))
    π₀ = abs.(normalize(e.vectors[:, findfirst(e.values .≈ 1.)], 1))
    
    ConvolvedHMM{typeof(μs), typeof(σₐ), typeof(W)}(μs, σs, σₐ, W, Σᵨ, Σₐ, log.(P), log.(π₀))
end

function randpair(hmm::ConvolvedHMM)
  @unpack μs, σs, σₐ, W, Σᵨ, Σₐ, logP, logπ₀ = hmm
  n = size(Σᵨ, 1)
  
  k = sampleMarkov(n, exp.(logP), exp.(logπ₀))
  d = rand(MvNormal(W * μs[k], Matrix(Hermitian(Diagonal(σs[k]) * Σᵨ * Diagonal(σs[k]) + Diagonal(σₐ * ones(n))))))
  d, k
end

function randpair(hmm::ConvolvedHMM, s)
  @unpack μs, σs, σₐ, W, Σᵨ, Σₐ, logP, logπ₀ = hmm
  n = size(Σᵨ, 1)
  
  m = rand(MvNormal(μs[s], Matrix(Hermitian(Diagonal(σs[s]) * Σᵨ * Diagonal(σs[s])))))
  d = rand(MvNormal(W * m, diagm(0 => σₐ * ones(n))))
  d, m
end

# likelihood function p(d|s)
function llhmm(s, d, μs, σs, Σₐ, W, Σᵨ)
  Σₛ = Diagonal(σs[s])
  Σd = W * Σₛ * Σᵨ * Σₛ * W' .+ Σₐ
  μd = W * μs[s]
  logpdf(MvNormal(μd, Matrix(Hermitian(Σd))), d)
end

function Distributions.logpdf(hmm::ConvolvedHMM, s::Vector, data::Vector)
    @unpack μs, σs, σₐ, W, Σᵨ, Σₐ, logP, logπ₀ = hmm
    l = logπ₀[s[1]]
    @inbounds for i = 2:length(s)
      l += logP[s[i-1], s[i]]
    end
    
    l += llhmm(s, data, μs, σs, Σₐ, W, Σᵨ)
    l
end

function Distributions.mode(hmm::ConvolvedHMM, s::Vector)
  @unpack μs, σs, σₐ, W, Σᵨ, Σₐ, logP, logπ₀ = hmm
  n = length(s)
  m = mode(MvNormal(μs[s], Diagonal(σs[s]) * Σᵨ * Diagonal(σs[s])))
  d = rand(MvNormal(W * m, Diagonal(σₐ * ones(n))))
  d, m
end