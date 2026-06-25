using ComponentArrays
using DifferentialEquations, DelayDiffEq, OrdinaryDiffEq
using OrdinaryDiffEqSDIRK, ADTypes
using Lux, SciMLSensitivity
using ReverseDiff, Zygote
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LogExpFunctions: logistic, logit, softplus
using LineSearches
using Random
using Printf
using Plots
using Statistics
using DataFrames, Parquet2, Dates, CSV, JLD2

const Optimisers = OptimizationOptimisers.Optimisers

# Physical incubation/infectious delay τ ∈ [3, 9] days. SHARED across the whole pipeline
delay_from_raw(τ_raw) = 3.0 + 6.0 * logistic(0.1 * τ_raw)

# ===========================================================================
# DATA LOADING FUNCTION
# ===========================================================================
function load_data(state_name, data_path, vac_path, total_pop, t_vac_default)
    println("Loading data for $state_name...")

    if !isfile(data_path) || !isfile(vac_path)
        error("❌ File not found: Please check $data_path and $vac_path.")
    end

    cases_deaths = DataFrame(Parquet2.readfile(data_path))
    vac          = DataFrame(Parquet2.readfile(vac_path))

    df_vac   = filter(row -> row.State == state_name, vac)
    df_state = filter(row -> row.State == state_name, cases_deaths)

    sort!(df_state, :ID_Period)
    sort!(df_vac, :ID_Period)

    t_obs     = Float64.(df_state.ID_Period)
    C_obs     = Float64.(df_state.Accumulated_Cases)  ./ total_pop
    D_obs     = Float64.(df_state.Accumulated_Deaths) ./ total_pop

    valid_idx  = .!isnan.(C_obs) .& .!isnan.(D_obs)
    valid_idx .&= isfinite.(C_obs) .& isfinite.(D_obs)

    if sum(.!valid_idx) > 0
        t_obs = t_obs[valid_idx]
        C_obs = C_obs[valid_idx]
        D_obs = D_obs[valid_idx]
    end

    # Data clean
    C_obs     = accumulate(max, C_obs)
    D_obs     = accumulate(max, D_obs)
    New_C_obs = vcat(C_obs[1], diff(C_obs)) # Useful to control the peaks on the new waves

    tspan = (minimum(t_obs), maximum(t_obs))
    println("✓ Data loaded: $(length(t_obs)) points in [$(tspan[1]), $(tspan[2])]")

    clean_state_dates = Date.(df_state.Report_Date)
    date_to_t = Dict(clean_state_dates .=> df_state.ID_Period)

    t_vac_list = Float64[]
    V_acc_obs_list = Float64[]

    for row in eachrow(df_vac)
        if haskey(date_to_t, row.Date)
            push!(t_vac_list, date_to_t[row.Date])
            push!(V_acc_obs_list, row.People_at_least_one_dose / total_pop)
        end
    end

    t_vac = Float64.(t_vac_list)
    V_acc_obs = Float64.(V_acc_obs_list)
    t_init_vac = isempty(t_vac) ? Float64(t_vac_default) : Float64(t_vac[1])

    println("✓ Asynchronous vaccination data loaded: $(length(t_vac)) useful points.")

    return t_obs, C_obs, D_obs, New_C_obs, t_vac, V_acc_obs, t_init_vac, tspan
end


# ===========================================================================
# 2. PHYSICAL DYNAMICS & NEURAL NETWORK
# ===========================================================================

function SEIRVD_Dynamics(u, t, t_past, h, p_const, p_time, p_time_past, eps, t_init_vac)
    β_i, β_e, ϕ_e, ϕ_r, ϕ_d, v = p_time
    β_i_past_nn, β_e_past_nn, _, _, _, _ = p_time_past

    # Soft shield: pushes any negative to quasi 0 without breaking the derivative.
    u_safe = (u .+ sqrt.(u.^2 .+ 1e-12)) ./ 2.0
    S, E, I, R, V, D, C = u_safe[1], u_safe[2], u_safe[3], u_safe[4], u_safe[5], u_safe[6], u_safe[7]

    ω_c = 0.01 * logistic(p_const.ω) + eps
    η_c = 0.01 * logistic(p_const.η) + eps

    β_i_c = 0.75  * logistic(β_i) + eps
    β_e_c = 0.75 * logistic(β_e) + eps
    ϕ_e_c = 0.5  * logistic(ϕ_e) + eps
    ϕ_r_c = 0.5  * logistic(ϕ_r) + eps
    ϕ_d_c = 0.01 * logistic(ϕ_d) + eps

    # Infection rates IN THE PAST
    β_i_c_past = 0.75 * logistic(β_i_past_nn) + eps
    β_e_c_past = 0.75 * logistic(β_e_past_nn) + eps

    v_mask = t > t_init_vac - 10 ? 1.0 : 0.0
    v_c    = v_mask * (0.04 * logistic(v) + eps)

    infections_today = S * (β_i_c * I + β_e_c * E)

    h_S = h(nothing, t_past; idxs=1)
    h_E = h(nothing, t_past; idxs=2)
    h_I = h(nothing, t_past; idxs=3)

    S_past = (h_S + sqrt(h_S^2 + 1e-12)) / 2.0
    E_past = (h_E + sqrt(h_E^2 + 1e-12)) / 2.0
    I_past = (h_I + sqrt(h_I^2 + 1e-12)) / 2.0

    # Application of past infection rates
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

function fourier_features(t)
    freqs = [0.5, 1.0, 2.0, 4.0] .* 2π
    return vcat(t, sin.(freqs .* t), cos.(freqs .* t))
end

# Informed initialization for the output-layer bias: with the weights shrunk, each rate starts
# at a physically plausible CONSTANT (the net then learns time-variation). Values = logit(target/scale):
#   β_i→0.20 (·0.75)  β_e→0.10 (·0.75)  φ_e→0.40 (·0.5)  φ_r→0.15 (·0.5)  φ_d→1e-3 (·0.01)  v→5e-3 (·0.04)
const OUT_BIAS = Float32[-1.01, -1.87, 1.386, -0.847, -2.197, -1.946]

const nn = Chain(
    Dense(9 =>64, swish),
    Dense(64 =>64, swish),
    Dense(64 => 32, swish),
    Dense(32 => 6)
)

function f_neural_dde(u, h, p_global, t, t0_val, T_val, t_init_vac, st)
    # Current time parameters (t)
    t_scaled = (t - t0_val) / (T_val - t0_val)
    features_mat = reshape(fourier_features(t_scaled), 9, 1)
    p_time, _ = nn(features_mat, p_global.nn, st)

    # Calculate past time
    τ_safe = delay_from_raw(p_global.ode.τ)
    t_past = t - τ_safe

    # Past time parameters (t - τ)
    t_scaled_past =  (t_past - t0_val) / (T_val - t0_val)
    features_mat_past = reshape(fourier_features(t_scaled_past), 9, 1)
    p_time_past, _ = nn(features_mat_past, p_global.nn, st)

    return SEIRVD_Dynamics(u, t, t_past, h, p_global.ode, p_time, p_time_past, 1e-7, t_init_vac)
end
# ===========================================================================
# PREDICTION AND LOSS FUNCTION
# ===========================================================================

function predict(θ_actual, t_max, t_obs_data, prob_base, u_base, t0_val)
    τ_safe_val = Zygote.ignore_derivatives(() -> delay_from_raw(θ_actual.ode.τ))
    lags       = Zygote.ignore_derivatives(() -> [τ_safe_val])

    # Binary search
    idx_max = Zygote.ignore_derivatives(() -> searchsortedlast(t_obs_data, t_max))
    t_obs_window = Zygote.ignore_derivatives(() -> @view(t_obs_data[1:idx_max]))
    current_tspan = (t0_val, t_max)

    _prob = remake(prob_base, p=θ_actual, tspan=current_tspan,
                   u0=u_base, constant_lags=lags)

    # Explicit Tsit5
    sol = solve(_prob, MethodOfSteps(Tsit5()),
                saveat   = t_obs_window,
                dt       = 0.1,
                abstol   = 1e-8, reltol = 1e-6,
                maxiters = 10_000,
                sensealg = ReverseDiffAdjoint())
    return Array(sol), t_obs_window
end

function loss_function(θ_actual, t_max, t_obs_data, C_obs_data, D_obs_data, New_C_obs_data,
                       prob_base, u_base, t_vac_data, V_acc_obs_data, t0_val, hp, features_grid)
    pred, t_obs_window = predict(θ_actual, t_max, t_obs_data, prob_base, u_base, t0_val)

    is_bad = Zygote.ignore_derivatives() do
        size(pred, 2) != length(t_obs_window) ||
        any(isnan, pred) ||
        any(isinf, pred) ||
        any(sum(pred[1:6, :], dims=1) .> 1.1) ||
        any(sum(pred[1:6, :], dims=1) .< 0.9)
    end

    if is_bad
        return 1.0e4 + 0.1 * sum(abs2, θ_actual), 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4
    end

    idx_max = length(t_obs_window)

    D_obs_slice = @view D_obs_data[1:idx_max]
    C_obs_slice = @view C_obs_data[1:idx_max]

    D_pred = pred[6, :]
    C_pred = pred[7, :]

    # Smooth |·| for the log
    C_pred_smooth = sqrt.(C_pred.^2 .+ 1e-18)
    D_pred_smooth = sqrt.(D_pred.^2 .+ 1e-18)

    # Per-series epsilons
    loss_C = mean((log.(C_pred_smooth .+ hp.eps_C) .- log.(C_obs_slice .+ hp.eps_C)).^2)
    loss_D = mean((log.(D_pred_smooth .+ hp.eps_D) .- log.(D_obs_slice .+ hp.eps_D)).^2)

    # --- INCIDENCE (new cases per period = first difference of cumulative cases) ---
    if idx_max >= 2
        NewC_obs_slice  = @view New_C_obs_data[2:idx_max]
        inc_pred        = C_pred[2:end] .- C_pred[1:end-1]
        inc_pred_smooth = sqrt.(inc_pred.^2 .+ 1e-18)
        loss_NC = mean((log.(inc_pred_smooth .+ hp.eps_NC) .- log.(NewC_obs_slice .+ hp.eps_NC)).^2)
    else
        loss_NC = 0.0
    end

    idx_vac_max = Zygote.ignore_derivatives(() -> searchsortedlast(t_vac_data, t_max))
    if idx_vac_max > 0
        t_vac_window     = Zygote.ignore_derivatives(() -> @view(t_vac_data[1:idx_vac_max]))
        V_acc_obs_window = Zygote.ignore_derivatives(() -> @view(V_acc_obs_data[1:idx_vac_max]))

        vac_indices = Zygote.ignore_derivatives() do
            [clamp(searchsortedfirst(t_obs_window, t), 1, length(t_obs_window)) for t in t_vac_window]
        end

        V_acc_pred        = pred[8, vac_indices]
        V_acc_pred_smooth = sqrt.(V_acc_pred.^2 .+ 1e-18)
        loss_V = mean((log.(V_acc_pred_smooth .+ hp.eps_V) .- log.(V_acc_obs_window .+ hp.eps_V)).^2)
    else
        loss_V = 0.0
    end

    # --- TEMPORAL SMOOTHNESS PENALTY (per parameter) ---
    # Penalize variation of the slow biological rates (φ_e, φ_r, φ_d → rows 3,4,5) far more than
    # the transmission/vaccination rates (β_i, β_e, v → rows 1,2,6). Improves identifiability
    # while keeping all 6 outputs time-varying.
    nn_outputs, _ = nn(features_grid, θ_actual.nn, st)
    diffs    = nn_outputs[:, 2:end] .- nn_outputs[:, 1:end-1]
    smooth_w = [1.0, 1.0, 10.0, 10.0, 10.0, 1.0]
    row_msd  = vec(sum(abs2, diffs, dims=2)) ./ size(diffs, 2)
    loss_smooth = sum(smooth_w .* row_msd) / sum(smooth_w)

    regularization = 1e-6 * sum(abs2, θ_actual.nn)

    total_loss = loss_C + (hp.w_D * loss_D) + (hp.w_V * loss_V) +
                 (hp.w_NC * loss_NC) + (hp.w_S * loss_smooth) + regularization

    return total_loss, loss_C, loss_D, loss_V, loss_NC, loss_smooth
end

# ===========================================================================
# TRAINING FUNCTION
# ===========================================================================

function train_model(θ_initial, t_obs_data, C_obs_data, D_obs_data, New_C_obs_data,
                     t_vac_data, V_acc_obs_data, features_grid, training_schedule,
                     prob_base, u_base, t0_val, hp)
    loss_hist  = Float64[]
    param_hist = Vector{Float64}[]

    θ_local = deepcopy(θ_initial)
    println("\n🚀 Starting Hybrid Curriculum Training...")

    for (i, stage) in enumerate(training_schedule)
        t_max, iter_adam, iter_lbfgs = stage

        println("\n" * "═"^60)
        println("📈 PHASE $i: Training from t=0 to $t_max")
        println("═"^60)

        opt_loss = (ps, _) -> begin
            l_tot, _, _, _, _, _ = loss_function(ps, t_max, t_obs_data, C_obs_data, D_obs_data,
                New_C_obs_data, prob_base, u_base, t_vac_data, V_acc_obs_data, t0_val, hp, features_grid)
            return l_tot
        end

        optf = OptimizationFunction(opt_loss, Optimization.AutoZygote())

        cb = (state, loss) -> begin
            ps = state.u
            l_tot, l_C, l_D, l_V, l_NC, l_smooth = loss_function(ps, t_max, t_obs_data, C_obs_data,
                D_obs_data, New_C_obs_data, prob_base, u_base, t_vac_data, V_acc_obs_data, t0_val, hp, features_grid)
            push!(loss_hist, l_tot)

            val_η  = 0.01 * logistic(ps.ode.η)
            val_ω  = 0.01 * logistic(ps.ode.ω)
            τ_real = delay_from_raw(ps.ode.τ)
            push!(param_hist, [val_η, val_ω, τ_real])

            if rand() < 0.1
            @printf("[Win: %d | Epoch: %d] TOT: %.4e | C: %.4e | D: %.4e | V: %.4e | NC: %.4e | SM: %.4e | Delay: %.2f\n",
                       t_max, length(loss_hist), l_tot, l_C, l_D, l_V, l_NC, l_smooth, τ_real)
            end
            return false
        end

        if iter_adam > 0
            # Learning-rate schedule: higher LR on the short early windows.
            lr = max(1e-4, 5e-4 * 0.85^(i - 1))
            println("   ▶️  Running Adam (lr=$(round(lr, sigdigits=2)), ClipNorm)...")
            optprob_adam = OptimizationProblem(optf, θ_local)
            adam_clipped = Optimisers.OptimiserChain(
                Optimisers.ClipNorm(1.0),
                Optimisers.Adam(lr)
            )
            res_adam = solve(optprob_adam, adam_clipped, callback = cb, maxiters = iter_adam)
            θ_local = res_adam.u
        end

        if iter_lbfgs > 0
            println("   ▶️  Running L-BFGS...")
            optprob_lbfgs = OptimizationProblem(optf, θ_local)
            res_lbfgs = solve(optprob_lbfgs,
                              OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking()),
                              callback = cb, maxiters = iter_lbfgs)
            θ_local = res_lbfgs.u
        end
    end

    println("\n✅ Training Finished. Final Loss: $(round(loss_hist[end], digits=6))")
    return θ_local, loss_hist, param_hist
end

function make_history(initial_state)
    return (p, t; idxs=nothing) -> typeof(idxs) <: Number ? initial_state[idxs] : initial_state
end
# ===========================================================================
# MAIN EXECUTION SCRIPT
# ===========================================================================

const DATA_PATH  = "data/cases_deaths.parquet"
const VAC_PATH   = "data/doses_admin.parquet"
const STATE_NAME = "California"
const TOTAL_POP  = 39_355_309.0

# Execute data loading
t_obs, C_obs, D_obs, New_C_obs, t_vac, V_acc_obs, t_init_vac, tspan =
    load_data(STATE_NAME, DATA_PATH, VAC_PATH, TOTAL_POP, 260)

# Extract time limits for safe passing
t0_val = Float64(tspan[1])
T_val  = Float64(tspan[2])

# Lux state (empty for this plain MLP) — built once, shared across seeds.
rng0 = MersenneTwister(42)
_, st_temp = Lux.setup(rng0, nn)
const st = Lux.testmode(st_temp)

const delay = 6.0
# Base trainable constants (raw space). τ = 0 ⇒ delay_from_raw = 6 days.
const θ_0 = (η = logit(0.0015),
             ω = logit(0.1),
             τ = -10 * log(6.0 / (delay - 3.0) - 1))

# Build θ for a given seed: random NN shrunk to a near-constant map, then the output-layer bias
# set to physically plausible constants (informed init).
function init_params(seed)
    rng = MersenneTwister(seed)
    nn_ps, _ = Lux.setup(rng, nn)
    θ_tmp = ComponentArray(nn = nn_ps, ode = θ_0)
    θ = ComponentArray(Float64.(getdata(θ_tmp)), getaxes(θ_tmp))
    θ.nn .= θ.nn .* 0.01
    θ.nn.layer_4.bias .= OUT_BIAS
    return θ
end

θ = init_params(42)

# ODE Initial State. Warm-star
E0 = C_obs[1]
I0 = C_obs[1]
u_0 = [1.0 - E0 - I0 - D_obs[1],
       E0,
       I0,
       0.0,
       0.0,
       D_obs[1],
       C_obs[1],
       0.0]

hist_func = make_history(u_0)

# Closure to inject global and constant variables into the DDEProblem safely
f_dde_closed = (u, h, p_global, t) -> f_neural_dde(u, h, p_global, t, t0_val, T_val, t_init_vac, st)

prob_dde = DDEProblem(f_dde_closed, u_0, hist_func, tspan, θ;
                      constant_lags=[delay_from_raw(θ.ode.τ)])

# Training configuration
training_schedule = [
    (20.0,  450, 50),  # 500
    (40.0,  300, 50),  # 850
    (80.0,  300, 50),  # 1200
    (150.0, 500, 100), # 1800
    (250.0, 400, 100), # 2300
    (350.0, 300, 50),  # 2650
    (500.0, 500, 100), # 3250
    (700.0, 500, 150), # 3900
    (T_val, 1200, 600) #Fine tuning #5700
]

const NUM_GRID_POINTS = 100
t_grid_scaled = range(0.0, 1.0, length=NUM_GRID_POINTS)
const features_grid_static = hcat([fourier_features(t) for t in t_grid_scaled]...)


const HP = (w_D = 1.0, w_V = 0.5, w_NC = 0.3, w_S = 0.10,
            eps_C = 1e-6, eps_D = 1e-7, eps_V = 1e-6, eps_NC = 1e-5)

# START TRAINING — multi-start over seeds
const SEEDS = [42]

θ_optimal     = init_params(SEEDS[1])
loss_history  = Float64[]
param_history = Vector{Float64}[]
best_loss     = Inf

for seed in SEEDS
    println("\n" * "#"^60)
    println("# MULTI-START — seed $seed")
    println("#"^60)
    θ_seed = init_params(seed)
    θ_s, lh_s, ph_s = train_model(
        θ_seed, t_obs, C_obs, D_obs, New_C_obs, t_vac, V_acc_obs,
        features_grid_static, training_schedule, prob_dde, u_0, t0_val, HP)
    if !isempty(lh_s) && lh_s[end] < best_loss
        global best_loss     = lh_s[end]
        global θ_optimal     = θ_s
        global loss_history  = lh_s
        global param_history = ph_s
    end
end

# ===========================================================================
#  PLOTS (INDIVIDUAL PNG/PDF EXPORTS)
# ===========================================================================

state_lower = lowercase(STATE_NAME)
output_png = "figs/$state_lower/png"
output_pdf = "figs/$state_lower/pdf"

mkpath(output_png)
mkpath(output_pdf)

pred_final, _ = predict(θ_optimal, T_val, t_obs, prob_dde, u_0, t0_val)
C_pred_final = pred_final[7, :]
D_pred_final = pred_final[6, :]
V_acc_pred_final = pred_final[8, :]

prob_final = remake(prob_dde, p=θ_optimal, tspan=tspan,
                    constant_lags=[delay_from_raw(θ_optimal.ode.τ)])
continuous_sol = solve(prob_final, MethodOfSteps(Tsit5()), saveat=1.0)

# ---------------------------------------------------------------------------
# Loss History Plot
# ---------------------------------------------------------------------------
p1 = plot(loss_history, yscale=:log10,
          xlabel="Iteraciones", ylabel="Costo", color=:purple, linewidth=2, legend=false)

savefig(p1, joinpath(output_png, "1_loss_history_$state_lower.png"))
savefig(p1, joinpath(output_pdf, "1_loss_history_$state_lower.pdf"))

# ---------------------------------------------------------------------------
# Tracker Variables Fit Plots - Separated
# ---------------------------------------------------------------------------
p_C = plot(xlabel="Días",ylabel="Casos Confirmados (C)")
scatter!(p_C, t_obs, C_obs, label="Datos", color=:steelblue, alpha=0.5, markersize=4, markerstrokewidth=0)
plot!(p_C, t_obs, C_pred_final, label="Ajuste del modelo", color=:navy, linewidth=2.5)

savefig(p_C, joinpath(output_png, "2_data_adjust_cases_$state_lower.png"))
savefig(p_C, joinpath(output_pdf, "2_data_adjust_cases_$state_lower.pdf"))

p_D = plot(xlabel="Días", ylabel="Muertes (D)")
scatter!(p_D, t_obs, D_obs, label="Datos", color=:firebrick, alpha=0.5, markersize=4, markerstrokewidth=0)
plot!(p_D, t_obs, D_pred_final, label="Ajuste del modelo", color=:darkred, linewidth=2.5)

savefig(p_D, joinpath(output_png, "2_data_adjust_deaths_$state_lower.png"))
savefig(p_D, joinpath(output_pdf, "2_data_adjust_deaths_$state_lower.pdf"))

p_V = plot(xlabel="Días", ylabel="Vacunación (V)")
scatter!(p_V, t_vac, V_acc_obs, label="Datos", color=:mediumseagreen, alpha=0.5, markersize=4, markerstrokewidth=0)
plot!(p_V, t_obs, V_acc_pred_final, label="Ajuste del modelo", color=:darkgreen, linewidth=2.5)

savefig(p_V, joinpath(output_png, "2_data_adjust_vaccination_$state_lower.png"))
savefig(p_V, joinpath(output_pdf, "2_data_adjust_vaccination_$state_lower.pdf"))

# ---------------------------------------------------------------------------
# Physical Dynamics Plots - Separated
# ---------------------------------------------------------------------------
p_macro = plot(continuous_sol, idxs=[1, 4, 5],
            ylabel="Proporción", linewidth=2.5,
            label=["Suceptibles (S)" "Recuperados (R)" "Vacunados (V)"])

savefig(p_macro, joinpath(output_png, "3_sirvd_dynamics_macro_$state_lower.png"))
savefig(p_macro, joinpath(output_pdf, "3_sirvd_dynamics_macro_$state_lower.pdf"))

# Assuming Exposed is index 2 and Deaths is index 6
p_micro = plot(continuous_sol, idxs=[2, 3, 6],
            xlabel="Días", ylabel="Proporción", linewidth=2.5,
            label=["Expuestos (E)" "Infectados (I)" "Muertes (D)"],
            color=[:orange :red :black])

savefig(p_micro, joinpath(output_png, "3_sirvd_dynamics_micro_$state_lower.png"))
savefig(p_micro, joinpath(output_pdf, "3_sirvd_dynamics_micro_$state_lower.pdf"))

# ---------------------------------------------------------------------------
# Constant Parameters Plots - Separated
# ---------------------------------------------------------------------------
η_hist = [p[1] for p in param_history]
ω_hist = [p[2] for p in param_history]
τ_hist = [p[3] for p in param_history]

p4_rates = plot(η_hist, label="η (Pérdida de vacunación)", color=:dodgerblue, linewidth=2)
plot!(p4_rates, ω_hist, label="ω (Pérdida de inmunidad)", color=:darkorange, linewidth=2)
plot!(p4_rates, xlabel="Iteraciones", ylabel="Días")

savefig(p4_rates, joinpath(output_png, "4_constant_parameters_rates_$state_lower.png"))
savefig(p4_rates, joinpath(output_pdf, "4_constant_parameters_rates_$state_lower.pdf"))

p4_tau = plot(τ_hist, label="Retardo (Días)", color=:forestgreen, linewidth=2)
plot!(p4_tau, xlabel="Iteraciones", ylabel="Días")

savefig(p4_tau, joinpath(output_png, "4_constant_parameters_tau_$state_lower.png"))
savefig(p4_tau, joinpath(output_pdf, "4_constant_parameters_tau_$state_lower.pdf"))

# ---------------------------------------------------------------------------
# Dynamic Parameters (Neural Network) - Separated
# ---------------------------------------------------------------------------
p_time_matrix = zeros(6, length(t_obs))
for (i, t) in enumerate(t_obs)
    t_scaled = (t - t0_val) / (T_val - t0_val)

    features_mat = reshape(fourier_features(t_scaled), 9, 1)
    out, _ = nn(features_mat, θ_optimal.nn, st)

    p_time_matrix[1, i] = 0.75  * logistic(out[1]) + 1e-7 # β_i
    p_time_matrix[2, i] = 0.75  * logistic(out[2]) + 1e-7 # β_e
    p_time_matrix[3, i] = 0.5  * logistic(out[3]) + 1e-7 # ϕ_e
    p_time_matrix[4, i] = 0.5  * logistic(out[4]) + 1e-7 # ϕ_r
    p_time_matrix[5, i] = 0.01 * logistic(out[5]) + 1e-7 # ϕ_d
    p_time_matrix[6, i] = 0.01 * logistic(out[6]) + 1e-7 # v
end

p5_rates_high = plot(t_obs, p_time_matrix[1, :], label="β_i (Infección de sintomaticos)", linewidth=2, color=:red)
plot!(p5_rates_high, t_obs, p_time_matrix[2, :], label="β_e (Infección de asintomáticos)", linewidth=2, color=:orange)
plot!(p5_rates_high, t_obs, p_time_matrix[3, :], label="ϕ_e (Proporción de asintomáticos)", linewidth=2, color=:purple)
plot!(p5_rates_high, t_obs, p_time_matrix[4, :], label="ϕ_r (Taza de recuperación)", linewidth=2, color=:green,)
plot!(p5_rates_high, ylabel="Valor")
plot!(p5_rates_high, xlabel="Días")
plot!(p5_rates_high,legend=:topleft)

savefig(p5_rates_high, joinpath(output_png, "5_nn_parameter_high_$state_lower.png"))
savefig(p5_rates_high, joinpath(output_pdf, "5_nn_parameter_high_$state_lower.pdf"))

p5_rates_low = plot(t_obs, p_time_matrix[5, :], label="ϕ_d (Mortalidad)", linewidth=2, color=:black)
plot!(p5_rates_low, t_obs, p_time_matrix[6, :], label="v (Vacunación)", linewidth=2, color=:teal)
plot!(p5_rates_low, xlabel="Días", ylabel="Valor")

savefig(p5_rates_low, joinpath(output_png, "5_nn_parameter_low_$state_lower.png"))
savefig(p5_rates_low, joinpath(output_pdf, "5_nn_parameter_low_$state_lower.pdf"))

println("✅ Base training finished and individual PDF/PNG plots saved!")

# ===========================================================================
# EXPORT PARAMETERS AND NEURAL NETWORK
# ===========================================================================

out_params  = "parameters/$state_lower"
mkpath(out_params)

println("\n💾 Saving parameters to: $out_params")

# ---------------------------------------------------------------------------
# Save Constant Parameters
# ---------------------------------------------------------------------------
val_η_final = 0.01 * logistic(θ_optimal.ode.η)
val_ω_final = 0.01 * logistic(θ_optimal.ode.ω)
val_τ_final = delay_from_raw(θ_optimal.ode.τ)


df_constants = DataFrame(
    Parameter = ["eta_vaccine_loss", "omega_immunity_loss", "tau_incubation_delay"],
    Physical_Value = [val_η_final, val_ω_final, val_τ_final],
    Raw_Optimized_Value = [θ_optimal.ode.η, θ_optimal.ode.ω, θ_optimal.ode.τ]
)

path_const = joinpath(out_params, "constant_parameters_$state_lower.csv")
CSV.write(path_const, df_constants)

# ---------------------------------------------------------------------------
# Save Dynamic Parameters (Time Evolution)
# ---------------------------------------------------------------------------
df_dynamics = DataFrame(
    Observation_Day = t_obs,
    beta_i_symptomatic = p_time_matrix[1, :],
    beta_e_asymptomatic = p_time_matrix[2, :],
    phi_e_prop_asymptomatic = p_time_matrix[3, :],
    phi_r_recovery = p_time_matrix[4, :],
    phi_d_mortality = p_time_matrix[5, :],
    v_vaccination_rate = p_time_matrix[6, :]
)

path_dynamics = joinpath(out_params, "dynamic_parameters_$state_lower.csv")
CSV.write(path_dynamics, df_dynamics)

# ---------------------------------------------------------------------------
# Save Neural Network (Structure and Weights)
# ---------------------------------------------------------------------------
path_nn = joinpath(out_params, "neural_network_weights_$state_lower.jld2")

jldsave(path_nn;
    optimal_weights = θ_optimal.nn,
    lux_state = st,
    architecture = nn
)

println("✅ Data export completed successfully!")
