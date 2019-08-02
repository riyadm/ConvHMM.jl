function sampleMarkovChain(n::Int, P::AbstractMatrix, πₚ::AbstractVector)
    s = Vector{Int}(undef, n)
    sampleMarkovChain!(s, n, P, πₚ)
end

function sampleMarkovChain!(s, n::Int, P::AbstractMatrix, πₚ::AbstractVector)
    s[1] = rand(Categorical(πₚ))
    @inbounds for i = 2:n
        s[i] = rand(Categorical(P[s[i-1], :]))
    end
    s
end

# categorical sequence encoding/decoding
@inline function encode(arr::Union{Vector{Int},Tuple}, k::Int)
  s = 1
  n = 1
  @inbounds for i = 1:length(arr)
    s += (arr[i] - 1) * n
    n *= k
  end
  s
end

@inline function decode(s::Int, k::Int, l::Int)
  arr = []
  n = k^(l-1)
  c = s - 1
  for i = 1:l
    pushfirst!(arr, div(c, n) + 1)
    c = mod(c, n)
    n = n // k
  end  
  arr
end

@inline function logsumexp(x::Vector{T}) where T<:Real
  xmax = maximum(x)
  y = map(x -> exp.(x - xmax), x)

  log(sum(y)) + xmax
end
