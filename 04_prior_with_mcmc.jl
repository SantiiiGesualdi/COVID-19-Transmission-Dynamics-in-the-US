using ComponentArrays
using DifferentialEquations, DelayDiffEq
using Turing
using FlexiChains: summarystats
using ForwardDiff: value
using LogExpFunctions: logistic, logit, softplus
using Random, CSV, Parquet2, DataFrames
using DataInterpolations
using Plots, StatsPlots


# ===========================================================================
# LOAD DATA & PRE-CALCULATED DYNAMIC PARAMETERS
# ===========================================================================
const STATE_NAME = "California"
const TOTAL_POP  = 39_355_309.0
state_lower = lowercase(STATE_NAME)

println("Loading data and pre-calculated neural network dynamics...")

# Load Obs Data
cases_deaths = DataFrame(Parquet2.readfile("data/cases_deaths.parquet"))
df_state = filter(row -> row.State == STATE_NAME, cases_deaths)
sort!(df_state, :ID_Period)

t_obs = Float64.(df_state.ID_Period)
C_obs = Float64.(df_state.Accumulated_Cases) ./ TOTAL_POP
D_obs = Float64.(df_state.Accumulated_Deaths) ./ TOTAL_POP

tspan = (minimum(t_obs), maximum(t_obs))

# Load Dynamics Paramas from CSV
df_dyn = CSV.read("parameters/$state_lower/dynamic_parameters_$state_lower.csv", DataFrame)

# Create Interpolations
β_i_itp = LinearInterpolation(df_dyn.beta_i_symptomatic, df_dyn.Observation_Day)
β_e_itp = LinearInterpolation(df_dyn.beta_e_asymptomatic, df_dyn.Observation_Day)
ϕ_e_itp = LinearInterpolation(df_dyn.phi_e_prop_asymptomatic, df_dyn.Observation_Day)
ϕ_r_itp = LinearInterpolation(df_dyn.phi_r_recovery, df_dyn.Observation_Day)
ϕ_d_itp = LinearInterpolation(df_dyn.phi_d_mortality, df_dyn.Observation_Day)
v_itp   = LinearInterpolation(df_dyn.v_vaccination_rate, df_dyn.Observation_Day)

# Initial state 
u_0 = [1.0 - C_obs[1] - D_obs[1], 0.0, C_obs[1], 0.0, 0.0, D_obs[1], C_obs[1], 0.0]
hist_func = (p, t; idxs=nothing) -> isnothing(idxs) ? u_0 : u_0[idxs]

# ===========================================================================
# INTERPOLATED DYNAMICS
# ===========================================================================
function SEIRVD_Dynamics_Interpolated(u, h, p, t)
    # Map parameters
    η_raw, ω_raw, τ_raw = p.η, p.ω, p.τ
    
    # Calculate time delay
    τ_safe = softplus(τ_raw) + 1.0
    t_past = t - τ_safe
    
    # Soft shield to prevent negative values
    S = (u[1] + sqrt(u[1]^2 + 1e-5)) / 2.0
    E = (u[2] + sqrt(u[2]^2 + 1e-5)) / 2.0
    I = (u[3] + sqrt(u[3]^2 + 1e-5)) / 2.0
    R = u[4]
    V = u[5]

    # Map Turing variables to physical limits
    eps_val = 1e-7
    ω_c = 0.01 * logistic(ω_raw) + eps_val
    η_c = 0.01 * logistic(η_raw) + eps_val

    # Query the interpolators for TODAY
    t_clamp = clamp(t, tspan[1], tspan[2])
    β_i_c = β_i_itp(t_clamp)
    β_e_c = β_e_itp(t_clamp)
    ϕ_e_c = ϕ_e_itp(t_clamp)
    ϕ_r_c = ϕ_r_itp(t_clamp)
    ϕ_d_c = ϕ_d_itp(t_clamp)
    
    # Handle Vaccination start time
    t_init_vac = 260.0 
    v_mask = t > t_init_vac - 10 ? 1.0 : 0.0
    v_c    = v_mask * v_itp(t_clamp)

    infections_today = S * (β_i_c * I + β_e_c * E)

    # Historical values
    h_S = h(p, t_past; idxs=1)
    h_E = h(p, t_past; idxs=2)
    h_I = h(p, t_past; idxs=3)

    S_past = (h_S + sqrt(h_S^2 + 1e-5)) / 2.0
    E_past = (h_E + sqrt(h_E^2 + 1e-5)) / 2.0
    I_past = (h_I + sqrt(h_I^2 + 1e-5)) / 2.0

    # Query the interpolators for THE PAST
    t_past_clamp = clamp(t_past, tspan[1], tspan[2])
    β_i_c_past = β_i_itp(t_past_clamp)
    β_e_c_past = β_e_itp(t_past_clamp)

    finish_incubation_today = S_past * (β_i_c_past * I_past + β_e_c_past * E_past)
    
    asymptomatic_today = ϕ_e_c * finish_incubation_today
    symptomatic_today  = (1.0 - ϕ_e_c) * finish_incubation_today

    dS = -infections_today - v_c * S + ω_c * R + η_c * V
    dE =  infections_today - finish_incubation_today
    dI =  symptomatic_today - ϕ_r_c * I - ϕ_d_c * I
    dR =  asymptomatic_today + ϕ_r_c * I - ω_c * R
    dV =  v_c * S - η_c * V
    dD =  ϕ_d_c * I
    dC =  symptomatic_today
    dV_acc = v_c * S

    return [dS, dE, dI, dR, dV, dD, dC, dV_acc]
end

# Base DDE problem setup
dummy_p = ComponentArray(η = 0.0, ω = 0.0, τ = 5.0)
prob_base = DDEProblem(SEIRVD_Dynamics_Interpolated, u_0, hist_func, tspan, dummy_p, constant_lags=[6.0])

# ===========================================================================
# BAYESIAN MODEL DEFINITION
# ===========================================================================
@model function bayesian_covid_model(t_obs, C_obs, D_obs, prob)
    # Priors
    η ~ Normal(-3.89, 1.5) 
    ω ~ Normal(-2.19, 1.5) 
    τ ~ Normal(5.0, 1.5)   
    
    # Observation noise
    σ_C ~ InverseGamma(2, 3)
    σ_D ~ InverseGamma(2, 3)

    p_current = ComponentArray(η = η, ω = ω, τ = τ)
    τ_safe = softplus(τ) + 1.0
    
    # constant_lags only locate solver discontinuities and must be plain Float64;
    # τ's gradient still flows through t_past = t - τ_safe inside the dynamics.
    _prob = remake(prob, p=p_current, constant_lags=[value(τ_safe)])
    
    # ForwardDiff handles standard Tsit5 cleanly
    sol = solve(_prob, MethodOfSteps(Tsit5()), 
                saveat = t_obs, 
                abstol = 1e-6, reltol = 1e-4)

    if sol.retcode == ReturnCode.Success && size(Array(sol), 2) == length(t_obs)
        pred = Array(sol)
        C_pred = max.(pred[7, :], 1e-9)
        D_pred = max.(pred[6, :], 1e-9)

        # Likelihood
        for i in 1:length(t_obs)
            C_obs[i] ~ LogNormal(log(C_pred[i]), σ_C)
            D_obs[i] ~ LogNormal(log(D_pred[i]), σ_D)
        end
    else
        Turing.@addlogprob! -Inf
    end
end

# ===========================================================================
# EXECUTE MCMC SAMPLING
# ===========================================================================
println("\n🚀 Starting Hamiltonian Monte Carlo (NUTS)...")
model = bayesian_covid_model(t_obs, C_obs, D_obs, prob_base)

# HMC is now evaluating a standard DDE without neural network overhead.
# target_accept=0.9 takes smaller steps → far fewer divergences on this stiff DDE.
# N=10 was a smoke test; real inference needs hundreds–thousands of draws.
sampler = NUTS(0.9; adtype=AutoForwardDiff())
chain = sample(model, sampler, 300, discard_initial=300, progress=true)

println("\n✅ MCMC Sampling Finished.")

# ===========================================================================
# EXPORT AND PLOTTING
# ===========================================================================
mkpath("figs/$state_lower/mcmc")

# Extract to Physical Values
η_physical = 0.01 .* logistic.(Array(chain[:η]))
ω_physical = 0.01 .* logistic.(Array(chain[:ω]))
τ_physical = softplus.(Array(chain[:τ])) .+ 1.0

p_density = density(η_physical, label="η (Vaccine Loss)", lw=2, xlabel="Physical Value", ylabel="Density")
density!(p_density, ω_physical, label="ω (Immunity Loss)", lw=2)
savefig(p_density, "figs/$state_lower/mcmc/physical_rates_posterior.png")

p_tau_density = density(τ_physical, label="τ (Incubation Delay)", lw=2, color=:forestgreen)
savefig(p_tau_density, "figs/$state_lower/mcmc/physical_tau_posterior.png")

# ===========================================================================
# DIAGNOSTICS: convergence summary + trace plots
# ===========================================================================
η_s, ω_s, τ_s = Array(chain[:η]), Array(chain[:ω]), Array(chain[:τ])

# Summary table = the primary numeric result: mean/std + ess (effective sample
# size) and rhat (between/within-chain variance ratio; want ≈1.00).
println("\n=== Posterior summary (sampled scale) ===")
show(stdout, MIME("text/plain"), summarystats(chain)); println()

# Trace = sampled value vs iteration. Healthy = stationary "fuzzy caterpillar",
# no drift or long flat stretches (stuck sampler).
p_trace = plot(
    plot(η_s, title="η", legend=false),
    plot(ω_s, title="ω", legend=false),
    plot(τ_s, title="τ", legend=false);
    layout=(3, 1), xlabel="Iteration", ylabel="Value", size=(700, 600),
)
savefig(p_trace, "figs/$state_lower/mcmc/trace.png")

# ===========================================================================
# POSTERIOR PREDICTIVE CHECK: does the fitted model reproduce the data?
# ===========================================================================

p_ppc_C = scatter(t_obs, C_obs .* TOTAL_POP, ms=3, color=:black, label="Observed",
                  xlabel="Day", ylabel="Accumulated cases", title="Posterior predictive: Cases", legend=:topleft)
p_ppc_D = scatter(t_obs, D_obs .* TOTAL_POP, ms=3, color=:black, label="Observed",
                  xlabel="Day", ylabel="Accumulated deaths", title="Posterior predictive: Deaths", legend=:topleft)

for i in eachindex(η_s)
    p_i = ComponentArray(η=η_s[i], ω=ω_s[i], τ=τ_s[i])
    prob_i = remake(prob_base, p=p_i, constant_lags=[softplus(τ_s[i]) + 1.0])
    sol_i  = solve(prob_i, MethodOfSteps(Tsit5()), saveat=t_obs, abstol=1e-6, reltol=1e-4)
    if sol_i.retcode == ReturnCode.Success && size(Array(sol_i), 2) == length(t_obs)
        arr = Array(sol_i)
        lbl = i == 1 ? "Posterior draws" : ""
        plot!(p_ppc_C, t_obs, max.(arr[7, :], 1e-9) .* TOTAL_POP, color=:steelblue, alpha=0.35, label=lbl)
        plot!(p_ppc_D, t_obs, max.(arr[6, :], 1e-9) .* TOTAL_POP, color=:firebrick, alpha=0.35, label=lbl)
    end
end
savefig(p_ppc_C, "figs/$state_lower/mcmc/ppc_cases.png")
savefig(p_ppc_D, "figs/$state_lower/mcmc/ppc_deaths.png")
println("\n📊 Saved figures to figs/$state_lower/mcmc/")