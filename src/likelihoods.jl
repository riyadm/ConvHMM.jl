abstract type LikelihoodApproximation{T} end

struct Truncation{T<:HMM} <: LikelihoodApproximation{T}
  hmm::T
end

function (ll::Truncation)(seq::Vector, obs::Vector, range)
  @unpack μ, σ, Σᵨ, Σₐ, W, D = ll.hmm
  H  = view(W,  range, range) * view(D, range, range)
  Sᵨ = view(Σᵨ, range, range)          
  Sₐ = view(Σₐ, range, range)
  d  = view(obs, range)
  loglikelihood(seq, d, μ, σ, Sₐ, H, Sᵨ) #/ length(range)
end

struct Projection{T<:HMM,V<:AbstractVector,M<:AbstractMatrix} <: LikelihoodApproximation{T}
  hmm::T
  μ₁::V
  μ₂::V
  Σ₁₁::M
  Σ₂₂::M
  Σ₂₁::M
end

function Projection(hmm::T) where {T<:HMM}
  @unpack W, D, Σₐ = hmm
  H = W * D
    
  # response approximation
  μ₁, Σ₁₁ = project(hmm)
  
  # observation moments
  μ₂ = H * μ₁ 
  Σ₂₂ = Symmetric(H * Σ₁₁ * H' + Σₐ)
  
  # observation-response covariance
  Σ₂₁ = H * Σ₁₁
    
  Projection{T,typeof(μ₁),typeof(Σ₁₁)}(hmm, μ₁, μ₂, Σ₁₁, Σ₂₂, Σ₂₁) 
end

function (ll::Projection)(seq::Vector, obs::Vector, range)
  @unpack μ₁, μ₂, Σ₁₁, Σ₂₂, Σ₂₁ = ll
  @unpack μ, σ, Σᵨ = ll.hmm
    
  # extract slices for conditioning
  Σ₂₁ₖ = view(Σ₂₁,     :, range)
  Σ₁₁ₖ = view(Σ₁₁, range, range)
  Σᵨₖ  = view(Σᵨ,  range, range)
  μ₁ₖ  = view(μ₁,  range)
  
  # values to condition on
  μₛ  = μ[seq]
  S  = Diagonal(σ[seq])
  Σₛ = S * Σᵨₖ * S

  # approximate moments
  C = Σ₂₁ₖ * inv(Σ₁₁ₖ)
  μₚ = μ₂ + C * (μₛ - μ₁ₖ)
  Σₚ = Σ₂₂ - C * Σ₂₁ₖ' + C * Σₛ * C'  
  
  logpdf(MvNormal(μₚ, Symmetric(Σₚ)), obs)
end


"""
    project(hmm)

Computes Gaussian mixture approximation to `p(m)`. Returns mean and covariance of n-dimensional marginal Gaussian.
"""
function project(hmm::ConvolvedHMM)
  @unpack μ, σ, logP, Σᵨ, logπ₀ = hmm
  n = size(Σᵨ, 1)
  nk = length(μ)
  
  # transition and stationary distributions
  π₀ = exp.(logπ₀)
  P = exp.(logP)
  
  # point-wise projected mean
  μₘ = μ' * π₀
  μₚ = μₘ * ones(n)
  
  # vector of covariances c(|s-t|) 
  c = Vector{eltype(σ)}(undef, n)
  
  # diagonal elements for lag 0
  c[1] = sum((σ.^2 .+ (μ .- μₘ).^2) .* π₀)
  
  # initialize n-step transition matrix
  Pⁿ = P
  
  # lags 1 to n-1
  for h in 1:n-1
    c[h+1] = 0
    for k₁ in 1:nk, k₂ in 1:nk
      c[h+1] += (σ[k₁] * σ[k₂] * Σᵨ[1,h+1] + (μ[k₁] - μₘ) * (μ[k₂] - μₘ)) * π₀[k₁] * Pⁿ[k₁,k₂]
    end
    
    # increment lag
    Pⁿ *= P
  end
  
  # build covariance matrix
  Σₚ = Matrix{eltype(σ)}(undef, n, n)
  for i in 1:n, j in 1:n
    Σₚ[i,j] = c[abs(j-i)+1]
  end
  
  μₚ, Σₚ
end