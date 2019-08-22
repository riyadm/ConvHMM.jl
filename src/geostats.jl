using .GeoStatsBase
import .GeoStatsBase: preprocess, solve

export HMMSim


"""
    HMMSim((var₁,=>param)

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
@simsolver HMMSim begin
  @param variogram
  @param transitions
  @param linop
  @param means
  @param stds
  @param noise = 1e-6
  @param approx = Projection
  @param range = 3
end

function preprocess(problem::SimulationProblem, solver::HMMSim)

  # result of preprocessing
  preproc = Dict{Symbol,NamedTuple}()

  params = first(values(solver.params))
  n = size(params.linop, 1)
  Σ = kernelmatrix(params.variogram, n)
  hmm = ConvolvedHM(params.transitions,
                    params.means,
		    params.stds,
                    params.Σ,
                    params.linop,
                    params.noise)

  preproc
end

function solve(problem::SimulationProblem, solver::HMMSim)
  # retrieve domain size
  sz = size(domain(problem))

  
end
