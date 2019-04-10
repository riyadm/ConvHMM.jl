struct HMMApproximator{T}
  hmm::ConvolvedHMM
  zi::T
  llfw::T
  k::Int
end


function Base.show(io::IO, h::HMMApproximator)
    print(io, "HMMApproximator(k=$(h.k))")
end
