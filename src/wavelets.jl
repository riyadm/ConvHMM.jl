abstract type Wavelet end

kernelmatrix(w, n::Int, m::Int = n) = map(i -> w(abs(i[1]-i[2])), CartesianIndices((n,m)))

function kernelmatrix(v::Vector, n::Int)
    m = []
    idx = ceil(Int, length(v)/2)
    for i in 1:n
        if i <= idx
            push!(m, vcat(v[idx-i+1:end], zeros(n-length(v[idx-i+1:end]))))
        elseif i <= n - idx
            vs = vcat(zeros(i - idx), v)
            push!(m, vcat(vs, zeros(n - length(vs))))
        else
            push!(m, vcat(zeros(i - idx), v[1:n-i+idx]))
        end
    end
    
    Matrix(hcat(m...)')
end


struct Ricker <: Wavelet
  f::Real
  dt::Real
end

function (w::Ricker)(h::Int)
  t = w.dt * h
    
  x = (π * w.f * t) .^ 2
  A = (1 .- 2x) .* exp.(-x) 
end


struct Ormsby <: Wavelet
  f::NTuple{4,Real}
  dt::Real
end

function (w::Ormsby)(h::Int)
  t = w.dt * h
  f = collect(w.f)
  
  z = π * f
  s = @. sinc(z * t)^2 
  A = (s[4] * z[4]^2 - s[3] * z[3]^2)/(z[4]-z[3]) - (s[2] * z[2]^2 - s[1] * z[1]^2)/(z[2]-z[1]) 
end

