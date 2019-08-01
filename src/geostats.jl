using .GeoStatsBase
import .GeoStatsBase: preprocess, solve_single

export HMM


"""
    HMM((var₁,=>param)

Markov-Switching linear Gaussian simulation. Allows joint simulation of facies and petrophysical properties through inference from observed signal. 

## Parameters
* `variogram`   - theoretical variogram (default to GaussianVariogram())
* `transitions` - stochastic matrix defining facies transitions
* `linop`       - linear operator (matrix) transforming variable into the observable
* `means`       - class-conditional observation means
* `vars`        - class-conditional observation variances
* `noise`       - observation noise variance
* `approx`      - likelihood approximation to use. Currently supported: `Projection` (default) and `Truncation`
* `range`       - likelihood approximation range
### References

Rimstad 2013. *Approximate posterior distributions for convolutional two-level hidden
Markov models.*
"""
@simsolver HMM begin
  @param variogram = GaussianVariogram()
  @param transitions = nothing
  @param linop = nothing
  @param means = nothing
  @param vars = nothing
  @param noise = 1e-6
  @param approx = Projection
  @param range = 3
end

function preprocess(problem::SimulationProblem, solver::HMM)

  # result of preprocessing
  preproc = Dict{Symbol,NamedTuple}()

  for (var, V) in variables(problem)
  end

  preproc
end

function solve_single(problem::SimulationProblem, var::Symbol,
                      solver::HMM, preproc)
  # retrieve domain size
  sz = size(domain(problem))

  # determine result type
  V = variables(problem)[var]
end