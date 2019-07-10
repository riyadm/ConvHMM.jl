mutable struct HMMApproximator{T<:HMM}
  hmm::T
  zi::Vector
  llfw::Vector
  k::Int
end

function sample(approx::HMMApproximator, nsamples::Int=1)
  @unpack hmm, zi, llfw, k = approx
  @unpack logP = hmm
    
  nk = size(logP, 1)
  r = k - 1
  n = length(zi) + r
  z = zi[end] .- logsumexp(zi[end])
    
  p = Vector{Float64}(undef, nk)
  samples = [Vector{Int}(undef, n) for _ in 1:nsamples]
  sample_lls = Vector{Float64}(undef, nsamples)
  
  
  for ns = 1:nsamples
    s = Vector{Int}(undef, n)
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
    sample_lls[ns] = ll_n
    samples[ns] = s
  end

  samples, sample_lls
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

function Base.show(io::IO, h::HMMApproximator)
    print(io, "HMMApproximator(k=$(h.k))")
end
