using ComponentArrays
using DifferentialEquations, DelayDiffEq, OrdinaryDiffEq
using OrdinaryDiffEqSDIRK, ADTypes
using LogExpFunctions: logistic, logit, softplus
using LineSearches
using Lux
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Plots
using Statistics
using Zygote
using Random
using Printf
using DataFrames
using Parquet2
using SciMLSensitivity
using ReverseDiff
using Dates

const Optimisers = OptimizationOptimisers.Optimisers

const DATA_PATH  = "data/clean_dataset.parquet"
const VAC_PATH   = "data/doses_admin.parquet"
const STATE_NAME = "California"
const POB_TOTAL  = 39_355_309.0

const EPS = 1e-7         
const w_D = 1.0
t_init_vac = 260.0

println("Cargando datos para $STATE_NAME...")

if isfile(DATA_PATH)
    data = DataFrame(Parquet2.readfile(DATA_PATH))
    vac  = DataFrame(Parquet2.readfile(VAC_PATH))
    df_vac   = filter(row -> row.State == STATE_NAME, vac)
    df_state = filter(row -> row.State == STATE_NAME, data)
    sort!(df_state, :ID_Period)

    t_obs     = Float64.(df_state.ID_Period)
    C_obs     = Float64.(df_state.Accumulated_Cases)  ./ POB_TOTAL
    D_obs     = Float64.(df_state.Accumulated_Deaths) ./ POB_TOTAL
    New_C_obs = Float64.(df_state.New_Cases)          ./ POB_TOTAL

    valid_idx  = .!isnan.(C_obs) .& .!isnan.(D_obs) .& .!isnan.(New_C_obs)
    valid_idx .&= isfinite.(C_obs) .& isfinite.(D_obs) .& isfinite.(New_C_obs)

    if sum(.!valid_idx) > 0
        t_obs     = t_obs[valid_idx]
        C_obs     = C_obs[valid_idx]
        D_obs     = D_obs[valid_idx]
        New_C_obs = New_C_obs[valid_idx]
    end

    M_obs = length(t_obs)
    tspan = (minimum(t_obs), maximum(t_obs))
    println("✓ Datos cargados: $M_obs puntos en [$(tspan[1]), $(tspan[2])]")

    fechas_limpias_estado = Date.(df_state.Report_Date)
    date_to_t = Dict(fechas_limpias_estado .=> df_state.ID_Period)

    t_vac_list = Float64[]
    V_obs_list = Float64[]
    for row in eachrow(df_vac)
        if haskey(date_to_t, row.Date)
            push!(t_vac_list, date_to_t[row.Date])
            push!(V_obs_list, row.Doses_admin / POB_TOTAL)
        end
    end

    global t_vac_global = Float64.(t_vac_list)
    global V_obs_global = Float64.(V_obs_list)

    global t_init_vac = isempty(t_vac_global) ? 260.0 : Float64(t_vac_global[1])

    println("✓ Datos de vacunación asíncronos cargados: $(length(t_vac_global)) puntos útiles.")
else
    error("❌ Archivo no encontrado: $DATA_PATH.")
end

# Constantes de tiempo global
const t0_val = Float64(tspan[1])
const T_val  = Float64(tspan[2])

# ---------------------------------------------------------------------------
# Dinámica SEIRVD con retardo
# ---------------------------------------------------------------------------
function SEIRVD_Dynamics(u, t, h, p_const, p_time, eps)
    β_i, β_e, ϕ_e, ϕ_r, ϕ_d, v = p_time

    # 🛡️ Escudo suave: empuja cualquier negativo a ~0 sin romper derivada
    u_safe = (u .+ sqrt.(u.^2 .+ 1e-5)) ./ 2.0
    S, E, I, R, V, D, C = u_safe[1], u_safe[2], u_safe[3], u_safe[4], u_safe[5], u_safe[6], u_safe[7]

    ω_c = 0.01 * logistic(p_const.ω) + eps
    η_c = 0.01 * logistic(p_const.η) + eps

    β_i_c = 0.5  * logistic(β_i) + eps
    β_e_c = 0.5  * logistic(β_e) + eps
    ϕ_e_c = 0.5  * logistic(ϕ_e) + eps
    ϕ_r_c = 0.5  * logistic(ϕ_r) + eps
    ϕ_d_c = 0.01 * logistic(ϕ_d) + eps

    v_mask = t > t_init_vac ? 1.0 : 0.0
    v_c    = v_mask * (0.01 * logistic(v) + eps)

    contagios_hoy = S * (β_i_c * I + β_e_c * E)

    τ_safe = softplus(p_const.τ) + 1.0

    h_S = h(nothing, t - τ_safe; idxs=1)
    h_E = h(nothing, t - τ_safe; idxs=2)
    h_I = h(nothing, t - τ_safe; idxs=3)

    S_pasado = (h_S + sqrt(h_S^2 + 1e-5)) / 2.0
    E_pasado = (h_E + sqrt(h_E^2 + 1e-5)) / 2.0
    I_pasado = (h_I + sqrt(h_I^2 + 1e-5)) / 2.0

    terminan_incubar_hoy = S_pasado * (β_i_c * I_pasado + β_e_c * E_pasado)

    asintomaticos_hoy = ϕ_e_c * terminan_incubar_hoy
    sintomaticos_hoy  = (1.0 - ϕ_e_c) * terminan_incubar_hoy

    dS = -contagios_hoy - v_c * S + ω_c * R + η_c * V
    dE =  contagios_hoy - terminan_incubar_hoy
    dI =  sintomaticos_hoy - ϕ_r_c * I - ϕ_d_c * I
    dR =  asintomaticos_hoy + ϕ_r_c * I - ω_c * R
    dV =  v_c * S - η_c * V
    dD =  ϕ_d_c * I
    dC =  sintomaticos_hoy

    return [dS, dE, dI, dR, dV, dD, dC]
end

# ---------------------------------------------------------------------------
# Red neuronal
# ---------------------------------------------------------------------------
const nn = Chain(
    Dense(1  => 64, sin),
    Dense(64 => 64, sin),
    Dense(64 => 32, sin),
    Dense(32 => 6)
)

seed = 42
rng  = MersenneTwister(seed)
nn_ps, st_temp = Lux.setup(rng, nn)
const st = Lux.testmode(st_temp)

# Parámetros ODE entrenables
θ_0 = (
    η = logit(0.1),
    ω = logit(0.1),
    τ = 7.0
)

# armado de ComponentArray y forzamos eltype Float64
θ_tmp = ComponentArray(nn = nn_ps, ode = θ_0)
θ = ComponentArray(Float64.(getdata(θ_tmp)), getaxes(θ_tmp))
θ.nn .= θ.nn .* 0.01   # cool nn

function f_neural_dde(u, h, p_global, t)
    # escala tiempo para que las 'sin' puedan oscilar entre olas.
    t_scaled = 15.0 * (t - t0_val) / (T_val - t0_val)
    p_time, _ = nn([t_scaled;;], p_global.nn, st) 
    return SEIRVD_Dynamics(u, t, h, p_global.ode, p_time, EPS)
end

u_0 = [
    1.0 - C_obs[1] - D_obs[1], 0.0, C_obs[1], 0.0, 0.0, D_obs[1], C_obs[1]
]

function hist(p, t; idxs=nothing)
    return typeof(idxs) <: Number ? u_0[idxs] : u_0
end


prob_dde = DDEProblem(f_neural_dde, u_0, hist, tspan, θ;
                      constant_lags=[softplus(θ.ode.τ) + 1.0])

# ===========================================================================
# PREDICCIÓN + PÉRDIDA POR VENTANAS (CURRICULUM LEARNING)
# ===========================================================================
function predict(θ_actual, t_max)
    τ_safe_val = Zygote.ignore_derivatives(() -> softplus(θ_actual.ode.τ) + 1.0)
    lags       = Zygote.ignore_derivatives(() -> [τ_safe_val])

    t_obs_window = filter(t -> t <= t_max, t_obs)
    current_tspan = (t0_val, t_max)

    _prob = remake(prob_dde, p=θ_actual, tspan=current_tspan,
                   u0=u_0, constant_lags=lags)

    # Tsit5 explícito → gradiente limpio. Tolerancias finas
    sol = solve(_prob, MethodOfSteps(Tsit5()),
                saveat   = t_obs_window,
                dt       = 0.1,
                abstol   = 1e-8, reltol = 1e-6,
                maxiters = 10_000,
                sensealg = ReverseDiffAdjoint())   
    return Array(sol), t_obs_window
end

const eps_loss = 1e-3

function loss_function(θ_actual, t_max)
    pred, t_obs_window = predict(θ_actual, t_max)

    is_bad = Zygote.ignore_derivatives() do
        size(pred, 2) != length(t_obs_window) ||
        any(isnan, pred) ||
        any(isinf, pred) ||
        any(sum(pred[1:6, :], dims=1) .> 1.1) ||
        any(sum(pred[1:6, :], dims=1) .< 0.9)
    end

    if is_bad
        # con Float64 + clipping esto casi nunca debería dispararse.
        return 1.0e4 + 0.1 * sum(abs2, θ_actual), 1.0e4, 1.0e4
    end

    idx_max       = length(t_obs_window)
    D_obs_recorte = D_obs[1:idx_max]
    C_obs_recorte = C_obs[1:idx_max]

    D_pred = pred[6, :]
    C_pred = pred[7, :]

    # valor absoluto suave
    C_pred_smooth = sqrt.(C_pred.^2 .+ 1e-7)
    D_pred_smooth = sqrt.(D_pred.^2 .+ 1e-7)

    loss_C = mean((log.(C_pred_smooth .+ eps_loss) .- log.(C_obs_recorte .+ eps_loss)).^2)
    loss_D = mean((log.(D_pred_smooth .+ eps_loss) .- log.(D_obs_recorte .+ eps_loss)).^2)

    loss_V = 0.0

    regularizacion = 1e-10 * sum(abs2, θ_actual.nn)

    loss_total = loss_C + w_D * loss_D + regularizacion + loss_V
    return loss_total, loss_C, loss_D
end

# ===========================================================================
# ENTRENAMIENTO HÍBRIDO (ADAM con ClipNorm + L-BFGS)
# ===========================================================================
loss_history  = Float64[]
param_history = Vector{Float64}[]
global θ_actual = θ

rutina_entrenamiento = [
    (20.0,  450, 50),
    (40.0,  300, 50),
    (80.0,  300, 50),
    (150.0, 500, 100),
    (250.0, 400, 100),
    (350.0, 300, 50),
    (500.0, 500, 100),
    (700.0, 500, 150),
    (T_val, 600, 170)
]

println("\n🚀 Iniciando Entrenamiento Híbrido por Currículum...")

for (i, etapa) in enumerate(rutina_entrenamiento)
    t_max, iter_adam, iter_lbfgs = etapa

    println("\n" * "═"^60)
    println("📈 FASE $i: Entrenando desde t=0 hasta el día $t_max")
    println("   Configuración: Adam ($iter_adam iters) | L-BFGS ($iter_lbfgs iters)")
    println("═"^60)

    opt_loss = (ps, _) -> begin
        l_tot, _, _ = loss_function(ps, t_max)
        return l_tot
    end

    optf = OptimizationFunction(opt_loss, Optimization.AutoZygote())

    cb = (state, loss) -> begin
        ps = state.u
        l_tot, l_C, l_D = loss_function(ps, t_max)
        push!(loss_history, l_tot)

        val_η  = 0.01 * logistic(ps.ode.η)
        val_ω  = 0.01 * logistic(ps.ode.ω)
        τ_real = softplus(ps.ode.τ) + 1.0
        push!(param_history, [val_η, val_ω, τ_real])

        if rand() < 0.1
            @printf("[Día %.0f] L_TOT: %.3e | L_C: %.3e | L_D: %.3e | τ: %.2f\n",
                    t_max, l_tot, l_C, l_D, τ_real)
        end
        return false
    end

    # ---------- SUB-FASE A: ADAM con gradient clipping ----------
    if iter_adam > 0
        println("   ▶️ Ejecutando Adam (con ClipNorm)...")
        optprob_adam = OptimizationProblem(optf, θ_actual)
        adam_clipped = Optimisers.OptimiserChain(
            Optimisers.ClipNorm(1.0),     # <- el fix anti-blow-up
            Optimisers.Adam(5e-4)
        )
        res_adam = solve(optprob_adam, adam_clipped,
                         callback = cb,
                         maxiters = iter_adam)
        global θ_actual = res_adam.u
    end

    # ---------- SUB-FASE B: L-BFGS (line search conservador) ----------
    if iter_lbfgs > 0
        println("   ▶️ Ejecutando L-BFGS...")
        optprob_lbfgs = OptimizationProblem(optf, θ_actual)
        res_lbfgs = solve(optprob_lbfgs,
                          OptimizationOptimJL.LBFGS(
                              linesearch = LineSearches.BackTracking()),
                          callback = cb,
                          maxiters = iter_lbfgs)
        global θ_actual = res_lbfgs.u
    end
end

println("\n✅ Entrenamiento Híbrido Finalizado. Loss final: $(round(loss_history[end], digits=6))")

# ===========================================================================
# GRÁFICOS
# ===========================================================================
carpeta_salida = "figuras/curriculum_adam"
mkpath(carpeta_salida)

pred_final, _ = predict(θ_actual, T_val)
C_pred_final = pred_final[7, :]
D_pred_final = pred_final[6, :]

prob_final = remake(prob_dde, p=θ_actual, tspan=tspan,
                    constant_lags=[softplus(θ_actual.ode.τ) + 1.0])
sol_continua = solve(prob_final, MethodOfSteps(Rosenbrock23()), saveat=1.0)

p1 = plot(loss_history, yscale=:log10,
          title="Decaimiento del Loss (Curriculum Adam)",
          xlabel="Iteraciones Globales", ylabel="Loss Total",
          label="Adam+LBFGS", color=:purple, linewidth=2, size=(800, 500))
savefig(p1, joinpath(carpeta_salida, "1_loss_history.png"))

p_C = plot(title="Ajuste UDE: Casos Confirmados (C)",
           ylabel="Proporción (Casos)", legend=:topleft)
scatter!(p_C, t_obs, C_obs, label="Datos Casos", color=:steelblue,
         alpha=0.5, markersize=4, markerstrokewidth=0)
plot!(p_C, t_obs, C_pred_final, label="Modelo Casos", color=:navy, linewidth=2.5)

p_D = plot(title="Ajuste UDE: Fallecidos Acumulados (D)",
           xlabel="Días", ylabel="Proporción (Muertes)", legend=:topleft)
scatter!(p_D, t_obs, D_obs, label="Datos Muertes", color=:firebrick,
         alpha=0.5, markersize=4, markerstrokewidth=0)
plot!(p_D, t_obs, D_pred_final, label="Modelo Muertes", color=:darkred, linewidth=2.5)

p2 = plot(p_C, p_D, layout=(2, 1), size=(900, 700), margin=5Plots.mm)
savefig(p2, joinpath(carpeta_salida, "2_ajuste_datos.png"))

p3 = plot(sol_continua, idxs=[1,3,4,5,6],
          title="Dinámica de las Variables (SIRVD)",
          xlabel="Días", ylabel="Proporción de Población",
          label=["Susceptibles (S)" "Infectados (I)" "Recuperados (R)" "Vacunados (V)" "Fallecidos (D)"],
          linewidth=2.5, size=(900, 600))
savefig(p3, joinpath(carpeta_salida, "3_dinamica_sirvd.png"))

params_mat = reduce(hcat, param_history)'
p4 = plot(params_mat,
          title="Convergencia de Parámetros ODE",
          xlabel="Iteraciones Globales", ylabel="Valor del Parámetro",
          label=["η (Pérdida Vacuna)" "ω (Pérdida Inmunidad)" "τ (Retardo)"],
          linewidth=2, size=(800, 500))
savefig(p4, joinpath(carpeta_salida, "4_parametros_ode.png"))

println("✅ ¡Entrenamiento base terminado y gráficos guardados!")


params_mat = reduce(hcat, param_history)'
p4 = plot(params_mat,
          title="Convergencia de Parámetros ODE",
          xlabel="Iteraciones Globales", ylabel="Valor del Parámetro",
          label=["η (Pérdida Vacuna)" "ω (Pérdida Inmunidad)"],
          linewidth=2, size=(800, 500))
savefig(p4, joinpath(carpeta_salida, "4_parametros_ode.png"))


