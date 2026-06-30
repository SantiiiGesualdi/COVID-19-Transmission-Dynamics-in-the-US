using DifferentialEquations, DelayDiffEq, OrdinaryDiffEq
using Lux, ComponentArrays, JLD2
using LogExpFunctions: logistic
using DataFrames, Parquet2, CSV, Dates
using Statistics, Printf
using Plots

# ===========================================================================
# LIGHTWEIGHT STATISTICS
# ===========================================================================
# Population skewness / excess kurtosis (StatsBase defaults).
function skewness(x::AbstractVector)
    m = mean(x); s = std(x; corrected=false)
    s == 0 && return NaN
    return mean(((x .- m) ./ s) .^ 3)
end
function kurtosis(x::AbstractVector)
    m = mean(x); m2 = mean((x .- m) .^ 2)
    m2 == 0 && return NaN
    return mean((x .- m) .^ 4) / m2^2 - 3.0
end

# Average ranks 
function rankavg(v::AbstractVector)
    n = length(v); p = sortperm(v); r = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && v[p[j+1]] == v[p[i]]
            j += 1
        end
        avg = (i + j) / 2
        for k in i:j
            r[p[k]] = avg
        end
        i = j + 1
    end
    return r
end
function spearman_cor(x::AbstractVector, y::AbstractVector)
    (std(x) == 0 || std(y) == 0) && return NaN
    return cor(rankavg(x), rankavg(y))
end


function autocor_lag(x::AbstractVector, k::Int)
    n = length(x); k >= n && return NaN
    m = mean(x); c0 = sum((x .- m) .^ 2)
    c0 == 0 && return NaN
    ck = sum((x[1:n-k] .- m) .* (x[1+k:n] .- m))
    return ck / c0
end

function ljung_box(x::AbstractVector, m::Int)
    n = length(x); q = 0.0
    for k in 1:m
        ρk = autocor_lag(x, k)
        isnan(ρk) && continue
        q += ρk^2 / (n - k)
    end
    return n * (n + 2) * q
end

function norminvcdf(p::Real)
    a = (-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00)
    d = ( 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00)
    plow = 0.02425; phigh = 1 - plow
    if p < plow
        q = sqrt(-2 * log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    elseif p <= phigh
        q = p - 0.5; r = q * q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6]) * q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
    else
        q = sqrt(-2 * log(1 - p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    end
end

# ===========================================================================
# 1. CONFIG 
# ===========================================================================
const STATE_NAME    = "California"
const TOTAL_POP     = 39_355_309.0
const DATA_PATH     = "data/cases_deaths.parquet"
const VAC_PATH      = "data/doses_admin.parquet"
const T_VAC_DEFAULT = 260
const EPS_DYN       = 1e-7      # eps inside SEIRVD_Dynamics (02 uses 1e-7)
const SMOOTH_ABS    = 1e-18     # smooth |·| constant inside the log (02)

const HP = (w_D = 1.0, w_V = 0.5, w_NC = 0.3, w_S = 0.10,
            eps_C = 1e-6, eps_D = 1e-7, eps_V = 1e-6, eps_NC = 1e-5)

state_lower = lowercase(STATE_NAME)

# Physical incubation/infectious delay τ ∈ [3, 9] days (02 line 18).
delay_from_raw(τ_raw) = 3.0 + 6.0 * logistic(0.1 * τ_raw)

# Fourier time features (02 line 139).
function fourier_features(t)
    freqs = [0.5, 1.0, 2.0, 4.0] .* 2π
    return vcat(t, sin.(freqs .* t), cos.(freqs .* t))
end

# Neural-network architecture (02 lines 149-154). Redefined locally so we only
# load the *weights* from JLD2 — robust against type-reconstruction issues.
const nn = Chain(
    Dense(9 => 64, swish),
    Dense(64 => 64, swish),
    Dense(64 => 32, swish),
    Dense(32 => 6),
)

# ===========================================================================
# DATA LOADING  
# ===========================================================================
function load_data(state_name, data_path, vac_path, total_pop, t_vac_default)
    println("Loading observed data for $state_name...")
    if !isfile(data_path) || !isfile(vac_path)
        error("❌ File not found: check $data_path and $vac_path.")
    end

    cases_deaths = DataFrame(Parquet2.readfile(data_path))
    vac          = DataFrame(Parquet2.readfile(vac_path))

    df_vac   = filter(row -> row.State == state_name, vac)
    df_state = filter(row -> row.State == state_name, cases_deaths)
    sort!(df_state, :ID_Period)
    sort!(df_vac, :ID_Period)

    t_obs = Float64.(df_state.ID_Period)
    C_obs = Float64.(df_state.Accumulated_Cases)  ./ total_pop
    D_obs = Float64.(df_state.Accumulated_Deaths) ./ total_pop

    valid_idx  = .!isnan.(C_obs) .& .!isnan.(D_obs)
    valid_idx .&= isfinite.(C_obs) .& isfinite.(D_obs)
    if sum(.!valid_idx) > 0
        t_obs = t_obs[valid_idx]
        C_obs = C_obs[valid_idx]
        D_obs = D_obs[valid_idx]
    end

    # Same cleaning as training: enforce monotone cumulative series.
    C_obs = accumulate(max, C_obs)
    D_obs = accumulate(max, D_obs)
    New_C_obs = vcat(C_obs[1], diff(C_obs))

    tspan = (minimum(t_obs), maximum(t_obs))
    println("✓ Data loaded: $(length(t_obs)) points in [$(tspan[1]), $(tspan[2])]")

    clean_state_dates = Date.(df_state.Report_Date)
    date_to_t = Dict(clean_state_dates .=> df_state.ID_Period)

    t_vac_list = Float64[]; V_acc_obs_list = Float64[]
    for row in eachrow(df_vac)
        if haskey(date_to_t, row.Date)
            push!(t_vac_list, date_to_t[row.Date])
            push!(V_acc_obs_list, row.People_at_least_one_dose / total_pop)
        end
    end
    t_vac      = Float64.(t_vac_list)
    V_acc_obs  = Float64.(V_acc_obs_list)
    t_init_vac = isempty(t_vac) ? Float64(t_vac_default) : Float64(t_vac[1])
    println("✓ Vaccination data: $(length(t_vac)) useful points (t_init_vac = $t_init_vac).")

    return t_obs, C_obs, D_obs, New_C_obs, t_vac, V_acc_obs, t_init_vac, tspan
end

# ===========================================================================
# EXACT TRAINED DYNAMICS (faithful to 02.f_neural_dde + SEIRVD_Dynamics)
# ===========================================================================

function make_nn_dynamics(nn_ps, st, t0_val, T_val, τ_safe, η_c, ω_c, t_init_vac; eps=EPS_DYN)
    span = T_val - t0_val
    return function seirvd_nn!(u, h, p, t)
        # Soft shield (02 uses 1e-12).
        u_safe = (u .+ sqrt.(u .^ 2 .+ 1e-12)) ./ 2.0
        S, E, I, R, V = u_safe[1], u_safe[2], u_safe[3], u_safe[4], u_safe[5]

        # Present-time NN rates.
        t_scaled = (t - t0_val) / span
        out, _   = nn(reshape(fourier_features(t_scaled), 9, 1), nn_ps, st)
        β_i_c = 0.75 * logistic(out[1]) + eps
        β_e_c = 0.75 * logistic(out[2]) + eps
        ϕ_e_c = 0.5  * logistic(out[3]) + eps
        ϕ_r_c = 0.5  * logistic(out[4]) + eps
        ϕ_d_c = 0.01 * logistic(out[5]) + eps
        v_raw = 0.04 * logistic(out[6]) + eps           # correct scale (02:109)

        v_mask = t > t_init_vac - 10 ? 1.0 : 0.0
        v_c    = v_mask * v_raw

        infections_today = S * (β_i_c * I + β_e_c * E)

        # Delayed state and delayed (past) NN infection rates.
        t_past = t - τ_safe
        h_S = h(p, t_past; idxs=1)
        h_E = h(p, t_past; idxs=2)
        h_I = h(p, t_past; idxs=3)
        S_past = (h_S + sqrt(h_S^2 + 1e-12)) / 2.0
        E_past = (h_E + sqrt(h_E^2 + 1e-12)) / 2.0
        I_past = (h_I + sqrt(h_I^2 + 1e-12)) / 2.0

        t_scaled_past = (t_past - t0_val) / span
        out_p, _      = nn(reshape(fourier_features(t_scaled_past), 9, 1), nn_ps, st)
        β_i_c_past = 0.75 * logistic(out_p[1]) + eps
        β_e_c_past = 0.75 * logistic(out_p[2]) + eps

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
end

# ===========================================================================
# METRICS
# ===========================================================================

function fit_metrics(obs::AbstractVector, pred::AbstractVector; eps_loss)
    n     = length(obs)
    resid = pred .- obs                       # residual = predicted - observed

    me    = mean(resid)                       # bias
    mae   = mean(abs.(resid))
    medae = median(abs.(resid))
    mse   = mean(resid .^ 2)
    rmse  = sqrt(mse)

    ss_res = sum(resid .^ 2)
    ss_tot = sum((obs .- mean(obs)) .^ 2)
    r2  = ss_tot > 0 ? 1 - ss_res / ss_tot : NaN     # = Nash–Sutcliffe efficiency
    r   = (std(obs) > 0 && std(pred) > 0) ? cor(obs, pred) : NaN
    ρ   = spearman_cor(obs, pred)

    # Percentage / relative errors.
    nz    = abs.(obs) .> 0
    mape  = any(nz) ? mean(abs.(resid[nz] ./ obs[nz])) * 100 : NaN
    smape = mean(2 .* abs.(resid) ./ (abs.(obs) .+ abs.(pred) .+ eps())) * 100
    wape  = sum(abs.(obs)) > 0 ? sum(abs.(resid)) / sum(abs.(obs)) * 100 : NaN  # robust MAPE


    pred_smooth = sqrt.(pred .^ 2 .+ SMOOTH_ABS)
    log_mse = mean((log.(pred_smooth .+ eps_loss) .- log.(obs .+ eps_loss)) .^ 2)
    rmsle   = sqrt(mean((log1p.(abs.(pred)) .- log1p.(abs.(obs))) .^ 2))

    # Normalized RMSE.
    rng         = maximum(obs) - minimum(obs)
    nrmse_range = rng > 0 ? rmse / rng * 100 : NaN
    nrmse_mean  = mean(obs) != 0 ? rmse / mean(obs) * 100 : NaN

    # Willmott's index of agreement d ∈ [0,1].
    ō       = mean(obs)
    denom_d = sum((abs.(pred .- ō) .+ abs.(obs .- ō)) .^ 2)
    willmott_d = denom_d > 0 ? 1 - ss_res / denom_d : NaN

    # Kling–Gupta efficiency 
    α   = std(obs) > 0 ? std(pred) / std(obs) : NaN   # variability ratio
    β   = mean(obs) != 0 ? mean(pred) / mean(obs) : NaN  # bias ratio
    kge = (isnan(r) || isnan(α) || isnan(β)) ? NaN :
          1 - sqrt((r - 1)^2 + (α - 1)^2 + (β - 1)^2)

    # Theil's U2 vs naive persistence forecast 
    theil_u2 = NaN
    if n >= 2
        num = sqrt(sum((pred[2:end] .- obs[2:end]) .^ 2))
        den = sqrt(sum((obs[2:end] .- obs[1:end-1]) .^ 2))
        theil_u2 = den > 0 ? num / den : NaN
    end

    # Residual autocorrelation / normality diagnostics.
    dw  = ss_res > 0 ? sum(diff(resid) .^ 2) / ss_res : NaN
    ac1 = autocor_lag(resid, 1)
    lb  = ljung_box(resid, min(10, n - 1))
    sk  = skewness(resid); ku = kurtosis(resid)
    jb  = n / 6 * (sk^2 + ku^2 / 4)           # Jarque–Bera (χ² with 2 dof)

    return (; n, ME_bias=me, MAE=mae, MedAE=medae, MSE=mse, RMSE=rmse,
            R2=r2, Pearson_r=r, Spearman_rho=ρ,
            MAPE=mape, sMAPE=smape, WAPE=wape,
            NRMSE_range=nrmse_range, NRMSE_mean=nrmse_mean,
            logMSE=log_mse, RMSLE=rmsle,
            Willmott_d=willmott_d, KGE=kge, KGE_r=r, KGE_alpha=α, KGE_beta=β,
            Theil_U2=theil_u2, DurbinWatson=dw, ACF_lag1=ac1, LjungBox_Q=lb,
            Resid_skew=sk, Resid_kurt=ku, JarqueBera=jb)
end

residual_stats(resid) = (; mean=mean(resid), std=std(resid),
                         skewness=skewness(resid), kurtosis=kurtosis(resid),
                         min=minimum(resid), max=maximum(resid))

# ===========================================================================
# RUN — load data, NN, constants, then solve the exact trained DDE
# ===========================================================================
t_obs, C_obs, D_obs, New_C_obs, t_vac, V_acc_obs, t_init_vac, tspan =
    load_data(STATE_NAME, DATA_PATH, VAC_PATH, TOTAL_POP, T_VAC_DEFAULT)
t0_val = Float64(tspan[1]); T_val = Float64(tspan[2])

# ---- Load the trained neural network (weights + Lux state) ----------------
path_nn = "parameters/$state_lower/neural_network_weights_$state_lower.jld2"
isfile(path_nn) || error("❌ Trained NN not found: $path_nn (run 02 first).")
nn_ps, st = jldopen(path_nn, "r") do f
    f["optimal_weights"], f["lux_state"]
end
println("✓ Loaded trained NN ($(length(nn_ps)) weights).")

# ---- Rebuild η, ω, τ from the raw optimized constants (exact transforms) ---
path_const = "parameters/$state_lower/constant_parameters_$state_lower.csv"
df_const = CSV.read(path_const, DataFrame)
getraw(name) = df_const[df_const.Parameter .== name, :Raw_Optimized_Value][1]
η_c    = 0.01 * logistic(getraw("eta_vaccine_loss"))    + EPS_DYN
ω_c    = 0.01 * logistic(getraw("omega_immunity_loss")) + EPS_DYN
τ_safe = delay_from_raw(getraw("tau_incubation_delay"))
@printf("✓ Constants  η=%.4e  ω=%.4e  τ=%.4f días\n", η_c, ω_c, τ_safe)

# ---- Initial condition IDENTICAL to 02 (E0 = I0 = C_obs[1]) ----------------
E0 = C_obs[1]; I0 = C_obs[1]
u_0 = [1.0 - E0 - I0 - D_obs[1], E0, I0, 0.0, 0.0, D_obs[1], C_obs[1], 0.0]
hist_func = (p, t; idxs=nothing) -> isnothing(idxs) ? u_0 : u_0[idxs]

f_dyn = make_nn_dynamics(nn_ps, st, t0_val, T_val, τ_safe, η_c, ω_c, t_init_vac)
prob  = DDEProblem(f_dyn, u_0, hist_func, tspan, nothing; constant_lags=[τ_safe])

println("\n🔄 Solving the exact trained SEIRVD-DDE (NN-driven) ...")
sol = solve(prob, MethodOfSteps(Tsit5()),
            saveat=t_obs, dt=0.1, abstol=1e-8, reltol=1e-6, maxiters=1_000_000)
println("✓ Solved. retcode = $(sol.retcode)")
pred = Array(sol)
size(pred, 2) == length(t_obs) ||
    @warn "Solver returned $(size(pred,2)) points but expected $(length(t_obs))."

C_pred     = pred[7, :]
D_pred     = pred[6, :]
V_acc_pred = pred[8, :]

# Vaccination predictions aligned to observed vaccination days (as in training).
vac_indices       = clamp.([searchsortedfirst(t_obs, t) for t in t_vac], 1, length(t_obs))
V_acc_pred_at_vac = V_acc_pred[vac_indices]

# Incidence (first differences of the cumulative series) — the honest,
# non-monotone signal. Compared on t_obs[2:end].
t_inc      = t_obs[2:end]
incC_obs   = diff(C_obs);  incC_pred = diff(C_pred)
incD_obs   = diff(D_obs);  incD_pred = diff(D_pred)

# ===========================================================================
# COMPUTE METRICS PER OBSERVABLE
# ===========================================================================
m_C    = fit_metrics(C_obs,     C_pred;            eps_loss=HP.eps_C)
m_D    = fit_metrics(D_obs,     D_pred;            eps_loss=HP.eps_D)
m_V    = fit_metrics(V_acc_obs, V_acc_pred_at_vac; eps_loss=HP.eps_V)
m_incC = fit_metrics(incC_obs,  incC_pred;         eps_loss=HP.eps_NC)
m_incD = fit_metrics(incD_obs,  incD_pred;         eps_loss=HP.eps_D)

series   = ["Cases_C", "Deaths_D", "Vaccination_Vacc", "Incidence_Cases", "Incidence_Deaths"]
all_mets = [m_C, m_D, m_V, m_incC, m_incD]

metric_names = collect(keys(m_C))
df_metrics = DataFrame(Metric = String.(metric_names))
for (name, m) in zip(series, all_mets)
    df_metrics[!, name] = [getfield(m, k) for k in metric_names]
end

# ===========================================================================
# 7. CONSOLE REPORT
# ===========================================================================
hr = "─"^78
println("\n" * "═"^78)
println("  GOODNESS-OF-FIT REPORT (NN ground truth) — SEIRVD-UDE — $(STATE_NAME)")
println("═"^78)
for (name, m) in zip(series, all_mets)
    println("\n▶ $name   (n = $(m.n))")
    println(hr)
    @printf("  MSE       = %.6e     R² (NSE)      = %+.6f\n", m.MSE, m.R2)
    @printf("  RMSE      = %.6e     Pearson r     = %+.6f\n", m.RMSE, m.Pearson_r)
    @printf("  MAE       = %.6e     Spearman ρ    = %+.6f\n", m.MAE, m.Spearman_rho)
    @printf("  MedAE     = %.6e     Willmott d    = %+.6f\n", m.MedAE, m.Willmott_d)
    @printf("  ME (bias) = %+.6e    KGE           = %+.6f\n", m.ME_bias, m.KGE)
    @printf("  logMSE    = %.6e     KGE(r,α,β)    = %.3f, %.3f, %.3f\n",
            m.logMSE, m.KGE_r, m.KGE_alpha, m.KGE_beta)
    @printf("  RMSLE     = %.6e     Theil U2      = %.4f\n", m.RMSLE, m.Theil_U2)
    @printf("  MAPE      = %8.3f %%    sMAPE         = %8.3f %%\n", m.MAPE, m.sMAPE)
    @printf("  WAPE      = %8.3f %%    NRMSE(range)  = %8.3f %%\n", m.WAPE, m.NRMSE_range)
    @printf("  NRMSE(mn) = %8.3f %%    Durbin–Watson = %.4f\n", m.NRMSE_mean, m.DurbinWatson)
    @printf("  ACF(lag1) = %+.4f       Ljung–Box Q   = %.3f\n", m.ACF_lag1, m.LjungBox_Q)
    @printf("  skew      = %+.4f       kurt(excess)  = %+.4f   JB = %.3f\n",
            m.Resid_skew, m.Resid_kurt, m.JarqueBera)
end

println("\n" * "═"^78)
println("  RESIDUAL DISTRIBUTION (pred − obs)")
println("═"^78)
for (name, resid) in zip(series, (C_pred .- C_obs, D_pred .- D_obs,
                                  V_acc_pred_at_vac .- V_acc_obs,
                                  incC_pred .- incC_obs, incD_pred .- incD_obs))
    rs = residual_stats(resid)
    @printf("  %-18s mean=%+.3e  std=%.3e  skew=%+.3f  kurt=%+.3f\n",
            name, rs.mean, rs.std, rs.skewness, rs.kurtosis)
end

# ---- Reconstruct the EXACT training loss (data + smoothness + L2) ----------
NUM_GRID = 100
tg = range(0.0, 1.0, length=NUM_GRID)
features_grid = hcat([fourier_features(t) for t in tg]...)
nn_outputs, _ = nn(features_grid, nn_ps, st)
diffs    = nn_outputs[:, 2:end] .- nn_outputs[:, 1:end-1]
smooth_w = [1.0, 1.0, 10.0, 10.0, 10.0, 1.0]
row_msd  = vec(sum(abs2, diffs, dims=2)) ./ size(diffs, 2)
loss_smooth = sum(smooth_w .* row_msd) / sum(smooth_w)
reg         = 1e-6 * sum(abs2, nn_ps)

data_loss  = m_C.logMSE + HP.w_D * m_D.logMSE + HP.w_V * m_V.logMSE + HP.w_NC * m_incC.logMSE
total_loss = data_loss + HP.w_S * loss_smooth + reg

println("\n" * "═"^78)
println("  TRAINING-OBJECTIVE RECONSTRUCTION (compare with the last training log line)")
println("═"^78)
@printf("  loss_C        = %.6e\n", m_C.logMSE)
@printf("  w_D·loss_D    = %.6e   (w_D = %.2f)\n", HP.w_D * m_D.logMSE, HP.w_D)
@printf("  w_V·loss_V    = %.6e   (w_V = %.2f)\n", HP.w_V * m_V.logMSE, HP.w_V)
@printf("  w_NC·loss_NC  = %.6e   (w_NC = %.2f)\n", HP.w_NC * m_incC.logMSE, HP.w_NC)
@printf("  w_S·loss_S    = %.6e   (w_S = %.2f)\n", HP.w_S * loss_smooth, HP.w_S)
@printf("  L2 reg        = %.6e\n", reg)
@printf("  ── data-loss  = %.6e\n", data_loss)
@printf("  ══ TOTAL loss = %.6e\n", total_loss)

# ===========================================================================
# EXPORT CSVs
# ===========================================================================
out_params = "parameters/$state_lower"
mkpath(out_params)

path_metrics = joinpath(out_params, "goodness_of_fit_nn_$state_lower.csv")
CSV.write(path_metrics, df_metrics)
println("\n💾 Metrics   -> $path_metrics")

df_resid = DataFrame(
    Observation_Day = t_obs,
    C_obs = C_obs, C_pred = C_pred, C_resid = C_pred .- C_obs,
    D_obs = D_obs, D_pred = D_pred, D_resid = D_pred .- D_obs,
)
path_resid = joinpath(out_params, "residuals_nn_$state_lower.csv")
CSV.write(path_resid, df_resid)
println("💾 Residuals -> $path_resid")

# ===========================================================================
# DIAGNOSTIC PLOTS
# ===========================================================================
const FIG_DIR = "figs/$state_lower/goodness_of_fit_nn"
const FIG_PNG = joinpath(FIG_DIR, "png")
const FIG_PDF = joinpath(FIG_DIR, "pdf")
mkpath(FIG_PNG); mkpath(FIG_PDF)
save_fig(plt, name) = begin
    savefig(plt, joinpath(FIG_PNG, "$(name)_$state_lower.png"))
    savefig(plt, joinpath(FIG_PDF, "$(name)_$state_lower.pdf"))
end

C_res = C_pred .- C_obs
D_res = D_pred .- D_obs
V_res = V_acc_pred_at_vac .- V_acc_obs

# ---- Fit on log scale (reveals the tails the loss optimizes) -----------
function plot_logfit(t_d, obs, t_p, pred, c_data, c_model, ylab)
    plt = plot(t_d, max.(obs, 1e-9), yscale=:log10, label="Datos", color=c_data,
               lw=2, size=(720, 450), xlabel="Días", ylabel=ylab, legend=:bottomright)
    plot!(plt, t_p, max.(pred, 1e-9), label="Modelo", color=c_model, lw=2, ls=:dash)
    return plt
end
save_fig(plot_logfit(t_obs, C_obs,     t_obs, C_pred,     :steelblue,      :navy,      "C (log)"), "logfit_cases")
save_fig(plot_logfit(t_obs, D_obs,     t_obs, D_pred,     :firebrick,      :darkred,   "D (log)"), "logfit_deaths")
save_fig(plot_logfit(t_vac, V_acc_obs, t_obs, V_acc_pred, :mediumseagreen, :darkgreen, "V (log)"), "logfit_vaccination")

# ---- Residuals vs time, with ±2σ bands ---------------------------------
function plot_residual_time(t, r, col, name, x)
    μ, σ = mean(r), std(r)
    plt = plot(t, r, color=col, lw=1.5, legend=false, size=(720, 450),
               xlabel="Días", ylabel="Residuos", label = name)
    hline!(plt, [0.0], color=:black, ls=:dash, label="\"0\" de referencia")
    hline!(plt, [μ + 2σ, μ - 2σ], color=:gray, ls=:dot,label = "̄$x ± 2σ")
    return plt
end
save_fig(plot_residual_time(t_obs, C_res, :navy, "Casos", "C"), "residual_cases")
save_fig(plot_residual_time(t_obs, D_res, :darkred   , "Muertes", "M"), "residual_deaths")
save_fig(plot_residual_time(t_vac, V_res, :darkgreen, "Vacunación", "V"),  "residual_vaccination")

# ---- Predicted vs observed scatter (with y = x) ------------------------
function plot_scatter_yx(obs, pred, col)
    lo, hi = min(minimum(obs), minimum(pred)), max(maximum(obs), maximum(pred))
    plt = scatter(obs, pred, color=col, alpha=0.5, ms=3, markerstrokewidth=0,
                  legend=false, size=(560, 520), xlabel="Observado", ylabel="Predicho")
    plot!(plt, [lo, hi], [lo, hi], color=:black, ls=:dash)
    return plt
end
save_fig(plot_scatter_yx(C_obs, C_pred,                :navy),      "scatter_cases")
save_fig(plot_scatter_yx(D_obs, D_pred,                :darkred),   "scatter_deaths")
save_fig(plot_scatter_yx(V_acc_obs, V_acc_pred_at_vac, :darkgreen), "scatter_vaccination")

# ---- Residual histogram with fitted normal density ---------------------
function plot_resid_hist(r, col)
    plt = histogram(r, bins=40, color=col, alpha=0.6, legend=false, normalize=:pdf,
                    size=(640, 440), xlabel="Residuos", ylabel="Densidad")
    μ, σ = mean(r), std(r)
    if σ > 0
        xs = range(minimum(r), maximum(r), length=200)
        ys = @. exp(-((xs - μ)^2) / (2σ^2)) / (σ * sqrt(2π))
        plot!(plt, xs, ys, color=:black, lw=2)
    end
    return plt
end
save_fig(plot_resid_hist(C_res, :navy),      "hist_cases")
save_fig(plot_resid_hist(D_res, :darkred),   "hist_deaths")
save_fig(plot_resid_hist(V_res, :darkgreen), "hist_vaccination")

# ---- Normal Q-Q plot of standardized residuals -------------------------
function plot_qq(r, col)
    σ = std(r)
    σ == 0 && return plot(title="QQ no disponible (σ=0)")
    z = (r .- mean(r)) ./ σ
    s = sort(z); n = length(s)
    theo = [norminvcdf((i - 0.5) / n) for i in 1:n]
    plt = scatter(theo, s, color=col, alpha=0.6, ms=3, markerstrokewidth=0,
                  legend=false, size=(560, 520),
                  xlabel="Cuantiles teóricos N(0,1)", ylabel="Cuantiles de residuos")
    lo, hi = minimum(theo), maximum(theo)
    plot!(plt, [lo, hi], [lo, hi], color=:black, ls=:dash)
    return plt
end
save_fig(plot_qq(C_res, :navy),      "qq_cases")
save_fig(plot_qq(D_res, :darkred),   "qq_deaths")
save_fig(plot_qq(V_res, :darkgreen), "qq_vaccination")

# ---- ACF of residuals, with ±1.96/√n white-noise bands -----------------
function plot_acf(r, col; maxlag=30)
    n = length(r); K = min(maxlag, n - 1)
    lags = collect(1:K)
    ac   = [autocor_lag(r, k) for k in lags]
    band = 1.96 / sqrt(n)
    plt = bar(lags, ac, color=col, alpha=0.7, legend=false, size=(720, 450),
              xlabel="Rezago (lag)", ylabel="ACF de residuos")
    hline!(plt, [band, -band], color=:red, ls=:dash)
    hline!(plt, [0.0], color=:black)
    return plt
end
save_fig(plot_acf(C_res, :navy),      "acf_cases")
save_fig(plot_acf(D_res, :darkred),   "acf_deaths")
save_fig(plot_acf(V_res, :darkgreen), "acf_vaccination")

# ---- Incidence overlay (first differences) — the honest, non-monotone view
function plot_incidence(t, obs_inc, pred_inc, c_d, c_m, ylab)
    plt = plot(t, obs_inc, label="Datos (incidencia)", color=c_d, lw=1.5,
               size=(720, 450), xlabel="Días", ylabel=ylab, legend=:topright)
    plot!(plt, t, pred_inc, label="Modelo (incidencia)", color=c_m, lw=2, ls=:dash)
    return plt
end
save_fig(plot_incidence(t_inc, incC_obs, incC_pred, :steelblue, :navy,    "ΔC por período"), "incidence_cases")
save_fig(plot_incidence(t_inc, incD_obs, incD_pred, :firebrick, :darkred, "ΔD por período"), "incidence_deaths")

println("\n✅ Goodness-of-fit (NN) analysis finished.")
println("   20 figures (one per plot) saved as PNG and PDF under:")
println("   $FIG_PNG")
println("   $FIG_PDF")
