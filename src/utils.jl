function sampleMarkov(n::Int, P::AbstractMatrix, πₚ::AbstractVector)
    s = Vector{Int}(undef, n)
    sampleMarkov!(s, n, P, πₚ)
end

function sampleMarkov!(s, n::Int, P::AbstractMatrix, πₚ::AbstractVector)
    s[1] = rand(Categorical(πₚ))
    @inbounds for i = 2:n
        s[i] = rand(Categorical(P[s[i-1], :]))
    end
    s
end

# kernel (spatial correlation function)
ρₘ(h::Int, range) = exp(-(h/range)^2)
Σᵨ(n::Int, range) = map(x->ρₘ(abs(x[1]-x[2]), range), CartesianIndices((n,n)))
Σᵨ!(X::Matrix, n::Int, range::Real) = map!(x->ρₘ(abs(x[1]-x[2]), range), X, CartesianIndices((n,n)))
Σᵨtrunc(n::Int, k) = map(x->abs(x[1]-x[2]) <= k ? ρₘ(abs(x[1]-x[2])) : 0, CartesianIndices((n,n)))

# convolution matrix
ω(τ::Int, σ::Float64) = (1 / √(2σ*π)) * exp(-.5(τ / σ)^2)
ωᵣ(t::Int, λ, γ) = γ * (1 - (t / λ)^2) * exp(-0.5(t / λ)^2)
W(n::Int, σ) = map(x->ω(abs(x[1]-x[2]), σ), CartesianIndices((n,n)))
W!(X::Matrix, n::Int, σ) = map!(x->ω(abs(x[1]-x[2]), σ), X, CartesianIndices((n,n)))
Wᵣ(n::Int, λ, γ) = map(x->ωᵣ(x[1]-x[2], λ, γ), CartesianIndices((n,n)))
Wtrunc(n::Int, σ::Float64, k) = map(x->abs(x[1]-x[2]) <= k ? ω(abs(x[1]-x[2]), σ) : 0, CartesianIndices((n,n)))
Wᵣtrunc(n::Int, k) = map(x->abs(x[1]-x[2]) <= k ? ωᵣ(abs(x[1]-x[2])) : 0, CartesianIndices((n,n)))

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


logsumexp(xs::Vector{T}) where T<:Real = begin
  largest = maximum(xs)
  ys = map(x -> exp.(x - largest), xs)

  log(sum(ys)) + largest
end