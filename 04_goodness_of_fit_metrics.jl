# ===========================================================================
# 04 - GOODNESS-OF-FIT METRICS FOR THE SEIRVD-UDE MODEL
# ===========================================================================
# Reconstructs the *trained* SEIRVD delay model directly from the exported
# parameters and quantifies how well it fits the observed data.
#
#   * Dynamic (time-varying) parameters  -> dynamic_parameters_<state>.csv
#   * Constant parameters (η, ω, τ)      -> constant_parameters_<state>.csv
#
# The neural network is NOT needed here: the six time-varying rates are read
# from the CSV and linearly interpolated (same idea as 03_prior_with_mcmc.jl),
# while η, ω and τ are rebuilt from their raw optimized values using the EXACT
# transformations of the training script (02_seirvd_ude_ff.jl) so the dynamics
# reproduced here match the ones that were actually fitted.
#
# Outputs:
#   - Console report with all estimators per observable.
#   - parameters/<state>/goodness_of_fit_<state>.csv   (metrics table)
#   - parameters/<state>/residuals_<state>.csv         (per-day residuals)
#   - figs/<state>/goodness_of_fit/{png,pdf}/*  (one file per diagnostic plot)
# ===========================================================================

using DifferentialEquations, DelayDiffEq, OrdinaryDiffEq
using LogExpFunctions: logistic
using DataFrames, Parquet2, CSV, Dates
using Statistics, Printf
using Plots

# Population skewness / excess kurtosis (matches StatsBase defaults), defined
# locally so the script depends only on what 02_seirvd_ude_ff.jl already needs.
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

# ---------------------------------------------------------------------------
# CONFIG (must match 02_seirvd_ude_ff.jl)
# ---------------------------------------------------------------------------
const STATE_NAME    = "California"
const TOTAL_POP     = 39_355_309.0
const DATA_PATH     = "data/cases_deaths.parquet"
const VAC_PATH      = "data/doses_admin.parquet"
const T_VAC_DEFAULT = 260
const EPS_LOSS      = 1e-3   # same eps used inside loss_function (log-space loss)
const EPS_DYN       = 1e-7   # same eps used inside SEIRVD_Dynamics

state_lower = lowercase(STATE_NAME)

# ===========================================================================
# 1. DATA LOADING  (mirrors load_data of 02_seirvd_ude_ff.jl, + New_C_obs)
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

    t_obs     = Float64.(df_state.ID_Period)
    C_obs     = Float64.(df_state.Accumulated_Cases)  ./ total_pop
    D_obs     = Float64.(df_state.Accumulated_Deaths) ./ total_pop
    New_C_obs = Float64.(df_state.New_Cases)          ./ total_pop

    valid_idx  = .!isnan.(C_obs) .& .!isnan.(D_obs) .& .!isnan.(New_C_obs)
    valid_idx .&= isfinite.(C_obs) .& isfinite.(D_obs) .& isfinite.(New_C_obs)
    if sum(.!valid_idx) > 0
        t_obs     = t_obs[valid_idx]
        C_obs     = C_obs[valid_idx]
        D_obs     = D_obs[valid_idx]
        New_C_obs = New_C_obs[valid_idx]
    end

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
    t_vac      = Float64.(t_vac_list)
    V_acc_obs  = Float64.(V_acc_obs_list)
    t_init_vac = isempty(t_vac) ? Float64(t_vac_default) : Float64(t_vac[1])
    println("✓ Vaccination data: $(length(t_vac)) useful points (t_init_vac = $t_init_vac).")

    return t_obs, C_obs, D_obs, New_C_obs, t_vac, V_acc_obs, t_init_vac, tspan
end

# ===========================================================================
# 2. LIGHTWEIGHT LINEAR INTERPOLATION (flat extrapolation outside the grid)
#    Equivalent to DataInterpolations.LinearInterpolation + clamp, but with
#    no extra dependency.
# ===========================================================================
struct LinInterp
    t::Vector{Float64}
    y::Vector{Float64}
end
function (itp::LinInterp)(x::Real)
    t, y = itp.t, itp.y
    x <= t[1]   && return y[1]
    x >= t[end] && return y[end]
    i = searchsortedlast(t, x)          # t[i] <= x < t[i+1]
    w = (x - t[i]) / (t[i+1] - t[i])
    return (1 - w) * y[i] + w * y[i+1]
end

# ===========================================================================
# 3. INTERPOLATED SEIRVD DELAY DYNAMICS (faithful to 02_seirvd_ude_ff.jl)
#    The six dynamic rates come straight from the CSV (already in physical
#    units, eps already baked in). η, ω, τ are passed pre-computed (physical).
# ===========================================================================
function make_dynamics(itps, ω_c, η_c, τ_safe, t_init_vac)
    β_i_itp, β_e_itp, ϕ_e_itp, ϕ_r_itp, ϕ_d_itp, v_itp = itps
    return function seirvd_interp!(u, h, p, t)
        # Soft shield: pushes negatives to ~0 keeping the derivative smooth
        u_safe = (u .+ sqrt.(u .^ 2 .+ 1e-5)) ./ 2.0
        S, E, I, R, V = u_safe[1], u_safe[2], u_safe[3], u_safe[4], u_safe[5]

        # Present-time rates
        β_i_c = β_i_itp(t)
        β_e_c = β_e_itp(t)
        ϕ_e_c = ϕ_e_itp(t)
        ϕ_r_c = ϕ_r_itp(t)
        ϕ_d_c = ϕ_d_itp(t)

        v_mask = t > t_init_vac - 10 ? 1.0 : 0.0
        v_c    = v_mask * v_itp(t)

        infections_today = S * (β_i_c * I + β_e_c * E)

        # Delayed (past) state and past infection rates
        t_past = t - τ_safe
        h_S = h(p, t_past; idxs=1)
        h_E = h(p, t_past; idxs=2)
        h_I = h(p, t_past; idxs=3)
        S_past = (h_S + sqrt(h_S^2 + 1e-5)) / 2.0
        E_past = (h_E + sqrt(h_E^2 + 1e-5)) / 2.0
        I_past = (h_I + sqrt(h_I^2 + 1e-5)) / 2.0

        β_i_c_past = β_i_itp(t_past)
        β_e_c_past = β_e_itp(t_past)
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
# 4. METRICS
# ===========================================================================
"""
    fit_metrics(obs, pred; eps_loss)

Full battery of goodness-of-fit estimators for one observable.
`logMSE` reproduces the per-series term actually minimized during training
(smooth |·| + log with eps_loss), so it is directly comparable to the
`C:`, `D:`, `V:` numbers printed by the training loop.
"""
function fit_metrics(obs::AbstractVector, pred::AbstractVector; eps_loss=EPS_LOSS)
    n     = length(obs)
    resid = pred .- obs                       # residual = predicted - observed

    mse  = mean(resid .^ 2)
    rmse = sqrt(mse)
    mae  = mean(abs.(resid))
    me   = mean(resid)                        # mean error (bias)
    medae = median(abs.(resid))

    ss_res = sum(resid .^ 2)
    ss_tot = sum((obs .- mean(obs)) .^ 2)
    r2  = 1 - ss_res / ss_tot                 # = Nash–Sutcliffe efficiency
    r   = (std(obs) > 0 && std(pred) > 0) ? cor(obs, pred) : NaN

    # Percentage errors (guard against division by ~0 at the series start)
    nz    = abs.(obs) .> 0
    mape  = any(nz) ? mean(abs.(resid[nz] ./ obs[nz])) * 100 : NaN
    smape = mean(2 .* abs.(resid) ./ (abs.(obs) .+ abs.(pred) .+ eps())) * 100

    # Log-space MSE — the real training objective (smooth abs as in the code)
    pred_smooth = sqrt.(pred .^ 2 .+ 1e-7)
    log_mse = mean((log.(pred_smooth .+ eps_loss) .- log.(obs .+ eps_loss)) .^ 2)

    # RMSLE (symmetric, robust to scale)
    rmsle = sqrt(mean((log1p.(abs.(pred)) .- log1p.(abs.(obs))) .^ 2))

    # Normalized RMSE
    rng         = maximum(obs) - minimum(obs)
    nrmse_range = rng > 0 ? rmse / rng * 100 : NaN
    nrmse_mean  = mean(obs) != 0 ? rmse / mean(obs) * 100 : NaN

    # Durbin–Watson: residual autocorrelation (≈2 ⇒ none, <2 ⇒ positive)
    dw = ss_res > 0 ? sum(diff(resid) .^ 2) / ss_res : NaN

    return (; n, MSE=mse, RMSE=rmse, MAE=mae, MedAE=medae, ME_bias=me,
            R2=r2, Pearson_r=r, MAPE=mape, sMAPE=smape,
            logMSE=log_mse, RMSLE=rmsle,
            NRMSE_range=nrmse_range, NRMSE_mean=nrmse_mean, DurbinWatson=dw)
end

residual_stats(resid) = (; mean=mean(resid), std=std(resid),
                         skewness=skewness(resid), kurtosis=kurtosis(resid),
                         min=minimum(resid), max=maximum(resid))

# ===========================================================================
# 5. RUN
# ===========================================================================
t_obs, C_obs, D_obs, New_C_obs, t_vac, V_acc_obs, t_init_vac, tspan =
    load_data(STATE_NAME, DATA_PATH, VAC_PATH, TOTAL_POP, T_VAC_DEFAULT)

# ---- Load exported parameters --------------------------------------------
path_dyn   = "parameters/$state_lower/dynamic_parameters_$state_lower.csv"
path_const = "parameters/$state_lower/constant_parameters_$state_lower.csv"
df_dyn   = CSV.read(path_dyn, DataFrame)
df_const = CSV.read(path_const, DataFrame)
println("✓ Loaded $(nrow(df_dyn)) dynamic-parameter rows and $(nrow(df_const)) constants.")

# Build interpolators from the dynamic CSV (values already physical).
#
# ⚠ SCALE FIX FOR THE VACCINATION RATE:
# The training dynamics use   v_c = 0.04*logistic(out)+eps   (02_..ff.jl:103),
# but the CSV export wrote it as 0.01*logistic(out)+eps (02_..ff.jl:510), i.e.
# at 1/4 of the value actually used while fitting. We invert the export to
# recover the TRUE rate:  v_true = 4*(v_csv - eps) + eps.
# Without this, S is under-vaccinated, infections blow up and C diverges past 1.
const V_EXPORT_SCALE = 0.01
const V_DYNAMIC_SCALE = 0.04
v_csv  = Float64.(df_dyn.v_vaccination_rate)
v_true = (V_DYNAMIC_SCALE / V_EXPORT_SCALE) .* (v_csv .- EPS_DYN) .+ EPS_DYN

tgrid = Float64.(df_dyn.Observation_Day)
itps = (
    LinInterp(tgrid, Float64.(df_dyn.beta_i_symptomatic)),
    LinInterp(tgrid, Float64.(df_dyn.beta_e_asymptomatic)),
    LinInterp(tgrid, Float64.(df_dyn.phi_e_prop_asymptomatic)),
    LinInterp(tgrid, Float64.(df_dyn.phi_r_recovery)),
    LinInterp(tgrid, Float64.(df_dyn.phi_d_mortality)),
    LinInterp(tgrid, v_true),
)

# Rebuild η, ω, τ from raw optimized values with the EXACT training transforms
getraw(name) = df_const[df_const.Parameter .== name, :Raw_Optimized_Value][1]
η_raw  = getraw("eta_vaccine_loss")
ω_raw  = getraw("omega_immunity_loss")
τ_raw  = getraw("tau_incubation_delay")
η_c    = 0.01 * logistic(η_raw) + EPS_DYN
ω_c    = 0.01 * logistic(ω_raw) + EPS_DYN
τ_safe = 6.0  * logistic(τ_raw * 0.1) + 3.0
@printf("✓ Constants  η=%.3e  ω=%.3e  τ=%.4f días\n", η_c, ω_c, τ_safe)

# ---- Solve the reconstructed DDE over the whole horizon -------------------
u_0 = [1.0 - C_obs[1] - D_obs[1], 0.0, C_obs[1], 0.0, 0.0, D_obs[1], C_obs[1], 0.0]
hist_func = (p, t; idxs=nothing) -> isnothing(idxs) ? u_0 : u_0[idxs]
f_dyn = make_dynamics(itps, ω_c, η_c, τ_safe, t_init_vac)

prob = DDEProblem(f_dyn, u_0, hist_func, tspan, nothing; constant_lags=[τ_safe])
println("\n🔄 Solving reconstructed SEIRVD-DDE ...")
sol = solve(prob, MethodOfSteps(Tsit5()),
            saveat=t_obs, abstol=1e-8, reltol=1e-6, maxiters=1_000_000)

pred = Array(sol)
if size(pred, 2) != length(t_obs)
    @warn "Solver returned $(size(pred,2)) points but expected $(length(t_obs)); aligning by saved time grid."
end
println("✓ Solved. retcode = $(sol.retcode)")

C_pred     = pred[7, :]
D_pred     = pred[6, :]
V_acc_pred = pred[8, :]

# Vaccination predictions aligned to the observed vaccination days
vac_indices       = [searchsortedfirst(t_obs, t) for t in t_vac]
vac_indices       = clamp.(vac_indices, 1, length(t_obs))
V_acc_pred_at_vac = V_acc_pred[vac_indices]

# Modeled incidence (new cases per period) vs observed New_Cases — diagnostic
# only: the model fits *cumulative* C, not incidence directly.
New_C_pred = vcat(C_pred[1], diff(C_pred))

# ===========================================================================
# 6. COMPUTE METRICS PER OBSERVABLE
# ===========================================================================
m_C   = fit_metrics(C_obs, C_pred)
m_D   = fit_metrics(D_obs, D_pred)
m_V   = fit_metrics(V_acc_obs, V_acc_pred_at_vac)
m_inc = fit_metrics(New_C_obs, New_C_pred)

series   = ["Cases_C", "Deaths_D", "Vaccination_Vacc", "Incidence_NewCases"]
all_mets = [m_C, m_D, m_V, m_inc]

metric_names = collect(keys(m_C))
df_metrics = DataFrame(Metric = String.(metric_names))
for (name, m) in zip(series, all_mets)
    df_metrics[!, name] = [getfield(m, k) for k in metric_names]
end

# ---- Console report -------------------------------------------------------
hr = "─"^78
println("\n" * "═"^78)
println("  GOODNESS-OF-FIT REPORT — SEIRVD-UDE — $(STATE_NAME)")
println("═"^78)
for (name, m) in zip(series, all_mets)
    println("\n▶ $name   (n = $(m.n))")
    println(hr)
    @printf("  MSE        = %.6e        R²            = %.6f\n", m.MSE, m.R2)
    @printf("  RMSE       = %.6e        Pearson r     = %.6f\n", m.RMSE, m.Pearson_r)
    @printf("  MAE        = %.6e        MAPE          = %.3f %%\n", m.MAE, m.MAPE)
    @printf("  MedAE      = %.6e        sMAPE         = %.3f %%\n", m.MedAE, m.sMAPE)
    @printf("  ME (bias)  = %+.6e       NRMSE(range)  = %.3f %%\n", m.ME_bias, m.NRMSE_range)
    @printf("  logMSE     = %.6e        NRMSE(mean)   = %.3f %%\n", m.logMSE, m.NRMSE_mean)
    @printf("  RMSLE      = %.6e        Durbin–Watson = %.4f\n", m.RMSLE, m.DurbinWatson)
end

println("\n" * "═"^78)
println("  RESIDUAL DISTRIBUTION (pred − obs)")
println("═"^78)
for (name, resid) in zip(series, (C_pred .- C_obs, D_pred .- D_obs,
                                  V_acc_pred_at_vac .- V_acc_obs,
                                  New_C_pred .- New_C_obs))
    rs = residual_stats(resid)
    @printf("  %-20s mean=%+.3e  std=%.3e  skew=%+.3f  kurt=%+.3f\n",
            name, rs.mean, rs.std, rs.skewness, rs.kurtosis)
end

# Cross-check vs the training objective (loss_C + w_D·loss_D + w_V·loss_V)
w_D, w_V = 1.0, 1.0
proxy_train_loss = m_C.logMSE + w_D * m_D.logMSE + w_V * m_V.logMSE
@printf("\nℹ Reconstructed data-loss (logMSE_C + logMSE_D + logMSE_V) = %.6e\n", proxy_train_loss)
println("  (compare with the C/D/V terms in the training log; smoothness &")
println("   regularization terms are excluded.)")

# ===========================================================================
# 7. EXPORT CSVs
# ===========================================================================
out_params = "parameters/$state_lower"
mkpath(out_params)

path_metrics = joinpath(out_params, "goodness_of_fit_$state_lower.csv")
CSV.write(path_metrics, df_metrics)
println("\n💾 Metrics  -> $path_metrics")

# Per-day residuals for the cumulative observables (full t_obs grid)
df_resid = DataFrame(
    Observation_Day = t_obs,
    C_obs = C_obs, C_pred = C_pred, C_resid = C_pred .- C_obs,
    D_obs = D_obs, D_pred = D_pred, D_resid = D_pred .- D_obs,
)
path_resid = joinpath(out_params, "residuals_$state_lower.csv")
CSV.write(path_resid, df_resid)
println("💾 Residuals -> $path_resid")

# ===========================================================================
# 8. DIAGNOSTIC PLOTS  — ONE FILE PER PLOT, IN A DEDICATED FOLDER
# ===========================================================================
# Every goodness-of-fit figure is written individually (no combined panels)
# to a single dedicated directory. Change METRICS_FIG_DIR to relocate them.
const METRICS_FIG_DIR = "figs/$state_lower/goodness_of_fit"
const METRICS_PNG     = joinpath(METRICS_FIG_DIR, "png")
const METRICS_PDF     = joinpath(METRICS_FIG_DIR, "pdf")
mkpath(METRICS_PNG); mkpath(METRICS_PDF)

save_fig(plt, name) = begin
    savefig(plt, joinpath(METRICS_PNG, "$(name)_$state_lower.png"))
    savefig(plt, joinpath(METRICS_PDF, "$(name)_$state_lower.pdf"))
end

# Residuals (pred − obs), each on its own time grid
C_res = C_pred .- C_obs
D_res = D_pred .- D_obs
V_res = V_acc_pred_at_vac .- V_acc_obs

# --- 8.1 Residuals vs time -------------------------------------------------
function plot_residual(t, r, col)
    plt = plot(t, r, color=col, lw=1.5, legend=false, size=(720, 450),
               xlabel="Días", ylabel="pred − obs")
    hline!(plt, [0.0], color=:black, ls=:dash)
    return plt
end
save_fig(plot_residual(t_obs, C_res, :navy),      "residual_cases")
save_fig(plot_residual(t_obs, D_res, :darkred),   "residual_deaths")
save_fig(plot_residual(t_vac, V_res, :darkgreen), "residual_vaccination")

# --- 8.2 Predicted vs observed scatter (with y = x) ------------------------
function plot_scatter_yx(obs, pred, col)
    lo, hi = min(minimum(obs), minimum(pred)), max(maximum(obs), maximum(pred))
    plt = scatter(obs, pred, color=col, alpha=0.5, ms=3, markerstrokewidth=0,
                  legend=false, size=(560, 520),
                  xlabel="Observado", ylabel="Predicho")
    plot!(plt, [lo, hi], [lo, hi], color=:black, ls=:dash)
    return plt
end
save_fig(plot_scatter_yx(C_obs, C_pred,                :navy),      "scatter_cases")
save_fig(plot_scatter_yx(D_obs, D_pred,                :darkred),   "scatter_deaths")
save_fig(plot_scatter_yx(V_acc_obs, V_acc_pred_at_vac, :darkgreen), "scatter_vaccination")

# --- 8.3 Fit on log scale (reveals the tails the loss optimizes) -----------
function plot_logfit(t_d, obs, t_p, pred, c_data, c_model, ylab)
    plt = plot(t_d, max.(obs, 1e-9), yscale=:log10, label="Datos", color=c_data,
               lw=2, size=(720, 450), xlabel="Días", ylabel=ylab)
    plot!(plt, t_p, max.(pred, 1e-9), label="Modelo", color=c_model, lw=2, ls=:dash)
    return plt
end
save_fig(plot_logfit(t_obs, C_obs,     t_obs, C_pred,      :steelblue,      :navy,      "C (log)"), "logfit_cases")
save_fig(plot_logfit(t_obs, D_obs,     t_obs, D_pred,      :firebrick,      :darkred,   "D (log)"), "logfit_deaths")
save_fig(plot_logfit(t_vac, V_acc_obs, t_obs, V_acc_pred,  :mediumseagreen, :darkgreen, "V (log)"), "logfit_vaccination")

# --- 8.4 Residual histograms -----------------------------------------------
plot_resid_hist(r, col) =
    histogram(r, bins=40, color=col, alpha=0.7, legend=false, size=(640, 440),
              xlabel="pred − obs", ylabel="Frecuencia")
save_fig(plot_resid_hist(C_res, :navy),      "hist_cases")
save_fig(plot_resid_hist(D_res, :darkred),   "hist_deaths")
save_fig(plot_resid_hist(V_res, :darkgreen), "hist_vaccination")

println("\n✅ Goodness-of-fit analysis finished.")
println("   12 figures (one per plot) saved as PNG and PDF under:")
println("   $METRICS_PNG")
println("   $METRICS_PDF")
