mutable struct HMMApproximator{T<:HMM}
  hmm::T
  obs::Vector
  zi::Vector
  llfw::Vector
  z::Vector
  n::Int
  nk::Int
  k::Int
end


function sample(approx::HMMApproximator)
  @unpack hmm, zi, llfw, k, n, z, nk = approx
  @unpack logP = hmm
    
  p = Vector{Float64}(undef, nk)  
  s = Vector{Int}(undef, n)
  r = k - 1
  
  ll_n = 0
    
  # last (k-1) elements
  idx = rand(Categorical(exp.(z)))
  s[n-r+1:n] = decode(idx, nk, r)
  ll_n += zi[end][idx]
    
  # intermediate elements
  for i = n-r:-1:2  
    for s1 = 1:nk
      s[i] = s1
      p[s1] = logP[s[i+r-1], s[i+r]] + llfw[i][encode(s[i:i+r], nk)] .- zi[i][encode(s[i+1:i+r], nk)]
    end
    idx = rand(Categorical(exp.(p)))
    s[i] = idx
    ll_n += p[idx]
  end
    
  # first element
  for s1 = 1:nk
    s[1] = s1
    p[s1] = llfw[1][encode(s[1:k], nk)] - zi[1][encode(s[2:k], nk)]
  end
  idx = rand(Categorical(exp.(p))) 
  s[1] = idx
  ll_n += p[idx]

  s, ll_n
end

function sample(approx::HMMApproximator, nsamples::Int)
  samples = Vector{Vector{Int}}()
  lls     = Vector{Float64}()
  
  for i in 1:nsamples
    s, ll = sample(approx)
    push!(samples, s)
    push!(lls, ll)
  end
  
  samples, lls
end

function viterbi(approx::HMMApproximator)
  @unpack hmm, zi, llfw, k = approx
  @unpack logP = hmm
  
  nk = size(logP, 1)
  r = k - 1
  n = length(zi) + r
  z = zi[end] .- logsumexp(zi[end])
    
  p = Vector{Float64}(undef, nk)
  s = Vector{Int}(undef, n)
  ll = 0
    
  # last (k-1) elements
  idx = argmax(z)
  s[n-r+1:n] = decode(idx, nk, r)
  ll += zi[end][idx]

  for i = n-r:-1:2  
    for s1 = 1:nk
      s[i] = s1
      p[s1] = logP[s[i+r-1], s[i+r]] + llfw[i][encode(s[i:i+r], nk)] - zi[i][encode(s[i+1:i+r], nk)]
    end
    idx = argmax(p)
    s[i] = idx
    ll += p[idx]
  end

  # first element
  for s1 = 1:nk
    s[1] = s1
    p[s1] = llfw[1][encode(s[1:k], nk)] - zi[1][encode(s[2:k], nk)]
  end
  idx = argmax(p) 
  s[1] = idx
  ll += p[idx]

  s, ll
end

function mcmc!(samples::Vector, approx::HMMApproximator, nIter=1000)
  @unpack hmm, obs = approx
  n = length(obs)
    
  @assert length(samples) == nIter "dimension mismatch"
  @assert length(first(samples)) == n "dimension mismatch"
    
  s_prev, p_a_prev = sample(approx)
  p_prev = logpdf(hmm, s_prev, obs)
  α = 0
    
  for j in 1:nIter
    s, p_a = sample(approx)
    p = logpdf(hmm, s, obs)
    αⱼ = min(1, exp(p - p_a + p_a_prev - p_prev))
    s_prev = rand() < αⱼ ? s : s_prev
    samples[j] = s_prev
    α = ((j-1)*α + αⱼ)/j
  end
  
  samples, α
end

function mcmc(approx::HMMApproximator, nIter=1000)
  n = length(approx.obs)
  samples = [Vector{Int}(undef, n) for _ in 1:nIter]
  mcmc!(samples, approx, nIter)
end

function Base.show(io::IO, h::HMMApproximator)
    print(io, "HMMApproximator(k=$(h.k))")
end
