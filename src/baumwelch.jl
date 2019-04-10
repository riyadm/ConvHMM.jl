function forward(hmm::ConvolvedHMM, data::Vector, k::Int)
  @unpack μs, σs, σₐ, W, Σᵨ, Σₐ, logP, logπ₀ = hmm
  n = length(data)
  nk = length(μs)
  r = k - 1
  # iterator over Nkʳ subsequences
  subseqs = Iterators.product([1:nk for _ in 1:r]...)
    
  # normalization constants
  zi = Vector{Vector}(undef, n-r)
  ll = Vector{Vector}(undef, n-r)
  q = Vector{Float64}(undef, nk)
  
  # initialization
  d = data[1:k]
  zi[1] = Vector{Float64}(undef, nk^r)
  ll[1] = Vector{Float64}(undef, nk^k)
  seq = Vector{Int}(undef, k)
  for s in subseqs
    for s1 = 1:nk
      seq = vcat(s1, s...)
      q[s1] = llhmm(seq[1:1], d[1:1], μs, σs, Σₐ[1,1], 1, 1) / k + logπ₀[s1]
      for i = 2:k
        Wi = view(W, 1:i, 1:i)
        Σᵨi = view(Σᵨ, 1:i, 1:i)          
        Σₐi = view(Σₐ, 1:i, 1:i)     
        q[s1] += logP[seq[i-1], seq[i]] + llhmm(seq[1:i], d[1:i], μs, σs, Σₐi, Wi, Σᵨi) / k
      end
      ll[1][encode(seq,nk)] = q[s1]
    end
    zi[1][encode(s,nk)] = logsumexp(q)
  end
  
  # intermediate subsequences    
  

  for t = 2:(n-k)
    zi[t] = Vector{Float64}(undef, nk^r)
    ll[t] = Vector{Float64}(undef, nk^k)
    d = data[t:t+r]
    Wk = view(W, t:t+r, t:t+r)
    Σᵨk = view(Σᵨ, t:t+r, t:t+r)         
    Σₐk = view(Σₐ, t:t+r, t:t+r)
    llfun(s, d) = llhmm(s, d, μs, σs, Σₐk, Wk, Σᵨk)
    for s in subseqs
      for s1 = 1:nk    
        seq = vcat(s1, s...)
        q[s1] = llfun(seq, d) / k + zi[t-1][encode(seq[1:end-1],nk)]
        ll[t][encode(seq,nk)] = q[s1]
      end
      zi[t][encode(s, nk)] = logsumexp(q) + logP[s[end-1], s[end]]   
    end
  end
  
  # last element
  d = data[end-r:end]
  zi[end] = Vector{Float64}(undef, nk^r)
  ll[end] = Vector{Float64}(undef, nk^k)
  for s in subseqs
    for s1 = 1:nk
      seq = vcat(s1, s...)
      q[s1] = llhmm(seq[end:end], d[end:end], μs, σs, Σₐ[end,end], W[end,end], Σᵨ[end,end]) / k
      
      for i = 1:k-1
        rng = n-k+i:n
        Wi = view(W, rng, rng)
        Σᵨi = view(Σᵨ, rng, rng)     
        Σₐi = view(Σₐ, rng, rng)
        q[s1] += llhmm(seq[i:end], d[i:end], μs, σs, Σₐi, Wi, Σᵨi) / k
      end
      q[s1] += zi[end-1][encode(seq[1:end-1],nk)]
      ll[end][encode(seq,nk)] = q[s1]
    end
    zi[end][encode(s,nk)] = logsumexp(q) + logP[s[end-1], s[end]]
  end
  
  HMMApproximator(hmm, zi, ll, k)
end

# function backward(hmm::ConvolvedHMM, zi)
function propose(approx::HMMApproximator, nsamples::Int=1)
  @unpack hmm, zi, llfw, k = approx
  @unpack μs, σs, σₐ, W, Σᵨ, Σₐ, logP = hmm
  nk = length(μs)
  r = k - 1
  n = length(zi) + r
  c = logsumexp(zi[end])
    
  subseqs = Iterators.product([1:nk for _ in 1:r]...)
  p = Vector{Float64}(undef, nk)
  samples = Vector{Vector}(undef, nsamples)
  sample_lls = Vector{Float64}(undef, nsamples)
  for ns = 1:nsamples
    s = Vector{Int}(undef, n)
    ll_n = 0
    
    # last (k-1) elements
    z = zi[end] .- c
    sum(exp.(z))
    idx = rand(Categorical(exp.(z)))
    s[n-r+1:n] = decode(idx, nk, r)
    ll_n += zi[end][idx]
     
    # (n-k+1)st element    
    for s1 = 1:nk
      s[n-k+1] = s1
      p[s1] = logP[s[end-1], s[end]] + llfw[end][encode(s[n-k+1:end], nk)] - zi[end][idx]
    end
    idx = rand(Categorical(exp.(p)))
    s[n-k+1] = idx
    ll_n += p[idx]
    
    for i = n-k:-1:2  
      for s1 = 1:nk
        s[i] = s1
        p[s1] = logP[s[i+k-2], s[i+k-1]] + llfw[i][encode(s[i:i+k-1], nk)] .- zi[i][encode(s[i+1:i+k-1], nk)]
      end
      idx = rand(Categorical(exp.(p)))
      s[i] = idx
      ll_n += p[idx]
    end
    
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