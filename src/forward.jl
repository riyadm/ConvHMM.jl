# TODO: non-allocating version of `forward`
function forward(hmm::T, obs::Vector, k::Int, approximation::Type{L}) where {T<:HMM,L<:LikelihoodApproximation}
  @unpack logP, logπ₀ = hmm
  
  n = length(obs)
  nk = nstates(hmm)
  r = k - 1
    
  # initialize likelihood approximation
  likelihood = approximation(hmm)
    
  # iterator over subsequences of length (k-1)
  subseqs = Iterators.product([1:nk for _ in 1:r]...)
    
  # normalization constants and likelihoods allocation
  zi = [Vector{Float64}(undef, nk^r) for _ in 1:n-r]
  ll = [Vector{Float64}(undef, nk^k) for _ in 1:n-r]
  
  q   = Vector{Float64}(undef, nk)
  seq = Vector{Int}(undef, k)
    
  # top sequence
  for s in subseqs
    for s1 = 1:nk
      seq = vcat(s1, s...)
      q[s1] = logπ₀[s1] + likelihood(seq[1:1], obs, 1:1) / k
      for i = 2:k
        q[s1] += logP[seq[i-1], seq[i]] + likelihood(seq[1:i], obs, 1:i) / k
      end
      ll[1][encode(seq, nk)] = q[s1]
    end
    zi[1][encode(s, nk)] = logsumexp(q)
  end
  
  # intermediate subsequences    
  for t = 2:(n-k)
    range = t:t+r
    for s in subseqs
      for s1 = 1:nk    
        seq = vcat(s1, s...)
        q[s1] = likelihood(seq, obs, range) / k + zi[t-1][encode(seq[1:end-1], nk)]
        ll[t][encode(seq, nk)] = q[s1]
      end

      zi[t][encode(s, nk)] = logsumexp(q) + logP[s[end-1], s[end]] 
    end
  end
  
  # last element
  for s in subseqs
    for s1 = 1:nk
      seq = vcat(s1, s...)
      q[s1] = likelihood(seq[end:end], obs, n:n) / k
      
      for i = 1:r
        range = n-k+i:n
        q[s1] += likelihood(seq[i:end], obs, range) / k
      end
            
      q[s1] += zi[end-1][encode(seq[1:end-1], nk)]
      ll[end][encode(seq, nk)] = q[s1]
    end
    
    zi[end][encode(s, nk)] = logsumexp(q) + logP[s[end-1], s[end]]
  end
  
  HMMApproximator(hmm, zi, ll, k)
end