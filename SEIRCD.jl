######################################################################
# MODELO SEIR-UDE (Neural ODE) PARA SERIES EPIDEMIOLÓGICAS MULTI-ESTADO
#
# Versión optimizada: Paralela + Zero allocations
######################################################################
# ====================================================================
# 1. DEPENDENCIAS
# ====================================================================
using DifferentialEquations
using SciMLSensitivity
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using Lux
using ComponentArrays
using Random
using Zygote
using LinearAlgebra
using StaticArrays
using Plots
using Printf
using DataFrames
using Parquet2
using Dates
using JLD2
using Base.Threads
using ForwardDiff

gr()
ENV["GKSwstype"] = "100"

# Enable multi-threading info
println("🔢 Número de hilos disponibles: $(Threads.nthreads())")
flush(stdout)

# ====================================================================
# 2. CONFIGURACIÓN
# ====================================================================

Base.@kwdef struct ModelConfig
    eps_tasa::Float64     = 1e-5
    n_norm::Float64       = 1.0
    penalty_base::Float64 = 1e6
    penalty_reg::Float64  = 1e4
    w_deaths::Float64     = 5.0
    min_beta::Float64     = 1e-8
    max_beta::Float64     = 1.0
end

Base.@kwdef struct TrainConfig
    seed::Int            = 666
    lr_phase1::Float64   = 0.01
    lr_phase2::Float64   = 0.001
    lr_phase3::Float64   = 0.0001
    maxiters_phase1::Int = 200
    maxiters_phase2::Int = 300
    maxiters_phase3::Int = 500
    verbose_every::Int   = 20
    n_states::Int        = 10
    pob_total::Float64   = 1_000_000.0
    data_path::String    = "data/clean_dataset.parquet"
    use_parallel::Bool   = true  # Enable parallel computation
end

# ====================================================================
# 3. CARGA DE DATOS
# ====================================================================

function load_top_states(file_path::String, n::Int)
    println("  Cargando archivo: $file_path")
    flush(stdout)
    
    if !isfile(file_path)
        error("❌ Archivo no encontrado: $file_path")
    end
    
    data = DataFrame(Parquet2.readfile(file_path))
    println("  ✓ Datos cargados: $(size(data, 1)) filas")
    flush(stdout)

    summary = combine(groupby(data, :State), nrow => :Num_Periods)
    sort!(summary, :Num_Periods, rev = true)
    
    println("\n  📊 Top 20 estados por número de períodos:")
    for (i, row) in enumerate(eachrow(first(summary, 20)))
        println("    $i. $(row.State): $(row.Num_Periods) períodos")
    end
    flush(stdout)
    
    top_states = first(summary, n).State
    df_top  = filter(row -> row.State in top_states, data)
    grouped = groupby(df_top, :State)
    dfs = DataFrame.(collect(grouped))
    
    println("\n  ✓ $(length(dfs)) estados seleccionados:")
    for (i, df) in enumerate(dfs)
        println("    $i. $(df.State[1]) ($(size(df, 1)) períodos)")
    end
    flush(stdout)
    
    return dfs
end

function normalize_data(dfs, pob_total::Real)
    println("\nNormalizando datos (población total = $pob_total)...")
    flush(stdout)
    
    normalized = map(dfs) do df
        (
            State              = df.State[1],
            ID_Period          = Float64.(df.ID_Period),
            Accumulated_Cases  = Float64.(df.Accumulated_Cases)  ./ pob_total,
            New_Cases          = Float64.(df.New_Cases)          ./ pob_total,
            Accumulated_Deaths = Float64.(df.Accumulated_Deaths) ./ pob_total,
        )
    end
    
    println("  ✓ Datos normalizados")
    flush(stdout)
    return normalized
end

function initial_conditions(ds, ::Type{T}) where {T}
    I0 = T(ds.Accumulated_Cases[1])
    E0 = T(ds.New_Cases[1])
    R0 = T(0.0)
    C0 = T(ds.Accumulated_Cases[1])
    D0 = T(ds.Accumulated_Deaths[1])
    S0 = max(T(1.0) - (I0 + E0 + R0), T(0.0))
    
    # Changed from SVector{6,T} to a standard Vector
    return T[S0, E0, I0, R0, C0, D0] 
end

# ====================================================================
# 4. RED NEURONAL
# ====================================================================

@inline f_activation(v) = one(v) / (one(v) + exp(-v))

function build_nn(rng::AbstractRNG)
    nn = Chain(
        Dense(6 => 32, f_activation),
        Dense(32 => 32, f_activation),
        Dense(32 => 16, f_activation),
        Dense(16 => 4),
    )
    ps_raw, st = Lux.setup(rng, nn)
    ps = ComponentArray{Float64}(ps_raw)
    println("  ✓ Red neuronal creada: $(sum(length, ps)) parámetros")
    flush(stdout)
    return nn, ps, st
end

# ====================================================================
# 5. DINÁMICA DEL SISTEMA (Optimizado - Zero allocations)
# ====================================================================

@inline function sier_rates(nn, u, p, st, cfg::ModelConfig)
    raw, st_new = nn(u, p, st)
    β = clamp(abs(raw[1]) + cfg.eps_tasa, cfg.min_beta, cfg.max_beta)
    σ = clamp(abs(raw[2]) + cfg.eps_tasa, 1e-6, 1.0)
    γ = clamp(abs(raw[3]) + cfg.eps_tasa, 1e-6, 1.0)
    μ = clamp(abs(raw[4]) + cfg.eps_tasa, 1e-8, 0.1)
    return β, σ, γ, μ, st_new
end

function make_dynamics(nn, st, cfg::ModelConfig)
    return function (du, u, p, t)
        S, E, I, R, C, D = u
        β, σ, γ, μ, _ = sier_rates(nn, u, p, st, cfg)

        @inbounds begin
            du[1] = -β * S * I / cfg.n_norm
            du[2] =  β * S * I / cfg.n_norm - σ * E
            du[3] =  σ * E - (γ + μ) * I
            du[4] =  γ * I
            du[5] =  σ * E
            du[6] =  μ * I
        end
    end
end

# ====================================================================
# 6. FUNCIÓN DE PÉRDIDA (Paralela y Optimizada)
# ====================================================================

function dataset_loss(θ, ds, dynamics, cfg::ModelConfig)
    t     = ds.ID_Period
    tspan = (first(t), last(t))
    
    u0 = initial_conditions(ds, eltype(θ))

    prob = ODEProblem(dynamics, u0, tspan, θ)
    
    sol  = solve(prob, Tsit5(), saveat = t,
                 sensealg = QuadratureAdjoint(autojacvec = ZygoteVJP()),
                 abstol = 1e-7, reltol = 1e-5, 
                 verbose = false,
                 maxiters = 1000)

    if sol.retcode != ReturnCode.Success
        return cfg.penalty_base + cfg.penalty_reg * sum(abs2, θ)
    end

    pred_mat = Array(sol)

    C_pred = @view pred_mat[5, :]
    D_pred = @view pred_mat[6, :]

    n = length(C_pred)
    
    loss_C   = sum(abs2, C_pred .- ds.Accumulated_Cases) / n
    loss_D   = sum(abs2, D_pred .- ds.Accumulated_Deaths) / n
    
    if n > 1
        ΔC_pred = @views C_pred[2:end] .- C_pred[1:end-1]
        ΔC_real = @views ds.New_Cases[2:end] .- ds.New_Cases[1:end-1]
        loss_new = sum(abs2, ΔC_pred .- ΔC_real) / (n - 1)
    else
        loss_new = zero(eltype(θ))
    end

    return loss_C + cfg.w_deaths * loss_D + loss_new
end

function total_loss_parallel(θ, p)
    dynamics = make_dynamics(p.nn, p.st, p.cfg)
    
    if p.use_parallel && length(p.dfs) > 1 && Threads.nthreads() > 1
        losses = Vector{eltype(θ)}(undef, length(p.dfs))
        Threads.@threads for i in eachindex(p.dfs)
            losses[i] = dataset_loss(θ, p.dfs[i], dynamics, p.cfg)
        end
        return sum(losses)
    else
        return sum(ds -> dataset_loss(θ, ds, dynamics, p.cfg), p.dfs)
    end
end

function total_loss(θ, p)
    return total_loss_parallel(θ, p)
end
# ====================================================================
# 7. ENTRENAMIENTO (SIN MUTACIONES EN CALLBACK)
# ====================================================================

function train_one_phase(θ0, p, opt_func, lr, maxiters, verbose_every, phase_name)
    println("\n📌 $phase_name (LR=$lr, $maxiters iters)")
    flush(stdout)
    
    loss_history = Float64[]
    iter_count = Ref(0)
    
    callback = function (state, loss_val)
        iter_count[] += 1
        Base.append!(loss_history, [loss_val])
        
        if iter_count[] % verbose_every == 0
            @printf("  %s Iter %4d | Pérdida = %.6e\n", phase_name, iter_count[], loss_val)
            flush(stdout)
        end
        return false
    end

    prob = OptimizationProblem(opt_func, θ0, p)
    res = solve(prob, OptimizationOptimisers.Adam(lr),
                callback = callback, maxiters = maxiters,
                verbose = false,
                progress = true)
    
    return res, loss_history
end

function train_improved(df_list, mcfg::ModelConfig, tcfg::TrainConfig)
    println("\n🚀 ENTRENAMIENTO MEJORADO (3 FASES - PARALELO)")
    flush(stdout)
    
    rng = MersenneTwister(tcfg.seed)
    nn, θ0, st = build_nn(rng)

    train_data = normalize_data(df_list, tcfg.pob_total)
    p = (dfs = train_data, nn = nn, st = st, cfg = mcfg, use_parallel = tcfg.use_parallel)

    opt_func = OptimizationFunction(total_loss, Optimization.AutoZygote())

    # Fase 1
    res1, history1 = train_one_phase(θ0, p, opt_func, tcfg.lr_phase1, 
                                      tcfg.maxiters_phase1, tcfg.verbose_every, "Fase 1")
    
    # Fase 2
    res2, history2 = train_one_phase(res1.u, p, opt_func, tcfg.lr_phase2,
                                      tcfg.maxiters_phase2, tcfg.verbose_every, "Fase 2")
    
    # Fase 3
    res3, history3 = train_one_phase(res2.u, p, opt_func, tcfg.lr_phase3,
                                      tcfg.maxiters_phase3, tcfg.verbose_every, "Fase 3")

    loss_history = vcat(history1, history2, history3)
    
    println("\n✅ Entrenamiento completado")
    println("   Pérdida final: $(res3.objective)")
    println("   Total iteraciones: $(length(loss_history))")
    flush(stdout)
    
    return (params = res3.u, loss_history = loss_history,
            nn = nn, st = st, cfg = mcfg, train_data = train_data)
end

# ====================================================================
# 8. DEBUGGING
# ====================================================================

function debug_model(θ, ds, nn, st, cfg::ModelConfig)
    println("\n🔍 DEBUGGING DEL MODELO - $(ds.State)")
    println("="^60)
    
    dynamics = make_dynamics(nn, st, cfg)
    t = ds.ID_Period
    u0 = Vector{eltype(θ)}(initial_conditions(ds, eltype(θ)))
    
    println("Condiciones iniciales:")
    println("  S0 = $(u0[1]), E0 = $(u0[2]), I0 = $(u0[3])")
    println("  R0 = $(u0[4]), C0 = $(u0[5]), D0 = $(u0[6])")
    
    println("\nTasas en t=0:")
    β, σ, γ, μ, _ = sier_rates(nn, u0, θ, st, cfg)
    println("  β = $β, σ = $σ, γ = $γ, μ = $μ")
    
    prob = ODEProblem(dynamics, u0, (first(t), last(t)), θ)
    sol = solve(prob, Tsit5(), saveat = t)
    
    println("\nResultados:")
    println("  Casos reales:    $(ds.Accumulated_Cases[end])")
    println("  Casos predichos: $(sol[5, end])")
    println("  Muertes reales:    $(ds.Accumulated_Deaths[end])")
    println("  Muertes predichas: $(sol[6, end])")
    
    return sol
end

# ====================================================================
# 9. VISUALIZACIÓN
# ====================================================================

function plot_loss(history; outfile = "loss_history.png")
    println("\n📊 Generando gráfica de pérdida...")
    flush(stdout)
    
    plt = plot(history, 
               xlabel = "Iteración", 
               ylabel = "Pérdida",
               yscale = :log10, 
               lw = 2, 
               legend = false,
               title = "Evolución de la pérdida",
               size = (800, 600),
               color = :blue)
    
    savefig(plt, outfile)
    display(plt)
    println("  ✓ Guardado: $outfile")
    flush(stdout)
    return plt
end

function plot_fit(θ, ds, nn, st, cfg::ModelConfig; outdir = ".")
    println("  Procesando: $(ds.State)...")
    flush(stdout)
    
    dynamics = make_dynamics(nn, st, cfg)
    t  = ds.ID_Period
    u0 = Vector{eltype(θ)}(initial_conditions(ds, eltype(θ)))

    prob = ODEProblem(dynamics, u0, (first(t), last(t)), θ)
    sol  = solve(prob, Tsit5(), saveat = t,
                 abstol = 1e-8, reltol = 1e-6)

    if sol.retcode != ReturnCode.Success
        @warn "  ⚠️  Solución ODE falló para $(ds.State)"
        return nothing
    end

    p1 = plot(t, ds.Accumulated_Cases, 
              label = "Reales", lw = 2, marker = :circle, markersize = 3,
              color = :blue)
    plot!(p1, t, sol[5, :], label = "Predichos", lw = 2, ls = :dash, color = :red)
    xlabel!(p1, "Tiempo")
    ylabel!(p1, "Casos acumulados")
    title!(p1, "Casos")

    p2 = plot(t, ds.Accumulated_Deaths, 
              label = "Reales", lw = 2, marker = :circle, markersize = 3,
              color = :blue)
    plot!(p2, t, sol[6, :], label = "Predichos", lw = 2, ls = :dash, color = :red)
    xlabel!(p2, "Tiempo")
    ylabel!(p2, "Muertes acumuladas")
    title!(p2, "Muertes")

    plt = plot(p1, p2, layout = (2, 1), size = (800, 700),
               title = "Ajuste - $(ds.State)", legend = :topleft)

    safe_name = replace(ds.State, " " => "_", "/" => "_", "\\" => "_")
    outfile = joinpath(outdir, "fit_$(safe_name).png")
    savefig(plt, outfile)
    display(plt)
    
    println("    ✓ Guardado: $outfile")
    flush(stdout)
    return plt
end

function save_results(resultado; outfile = "trained_model.jld2")
    println("\n💾 Guardando resultados...")
    flush(stdout)
    
    @save outfile params=Array(resultado.params) loss_history=resultado.loss_history
    println("  ✓ Guardado: $outfile")
    flush(stdout)
    return outfile
end

# ====================================================================
# 10. EJECUCIÓN PRINCIPAL
# ====================================================================

function main(; fast_mode = false, n_states = nothing)
    println("="^70)
    println("MODELO SEIR-UDE - Neural ODE")
    println("="^70)
    flush(stdout)
    
    mcfg = ModelConfig()
    tcfg = TrainConfig()
    
    if !isnothing(n_states)
        tcfg = TrainConfig(n_states = n_states)
        println("\n⚙️  Configuración: $n_states estados")
    end
    flush(stdout)

    if !isfile(tcfg.data_path)
        error("❌ Archivo no encontrado: $(tcfg.data_path)")
    end

    println("\n📁 Cargando datos...")
    flush(stdout)
    df_list = load_top_states(tcfg.data_path, tcfg.n_states)
    
    println("\n🏋️  Entrenando...")
    flush(stdout)
    resultado = train_improved(df_list, mcfg, tcfg)

    save_results(resultado)

    println("\n📈 Generando visualizaciones...")
    flush(stdout)
    
    plot_loss(resultado.loss_history, outfile = "loss_history.png")
    
    # Debuggear primer estado
    println("\n🔍 Debugging...")
    debug_model(resultado.params, resultado.train_data[1], 
                resultado.nn, resultado.st, resultado.cfg)
    
    for (i, ds) in enumerate(resultado.train_data)
        println("\n📊 Gráfica $i/$(length(resultado.train_data)): $(ds.State)")
        plot_fit(resultado.params, ds, resultado.nn, resultado.st, mcfg)
    end

    println("\n" * "="^70)
    println("✅ COMPLETADO")
    println("="^70)
    
    return resultado
end

# Función rápida para testing
function train_fast(df_list, mcfg::ModelConfig, tcfg::TrainConfig; 
                    maxiters = 300, lr = 0.001)
    println("\n🚀 ENTRENAMIENTO RÁPIDO (PARALELO)")
    flush(stdout)
    
    rng = MersenneTwister(tcfg.seed)
    nn, θ0, st = build_nn(rng)
    train_data = normalize_data(df_list, tcfg.pob_total)
    p = (dfs = train_data, nn = nn, st = st, cfg = mcfg, use_parallel = tcfg.use_parallel)

    loss_history = Float64[]
    iter_count = Ref(0)

    callback = function (state, loss_val)
        iter_count[] += 1
        Base.append!(loss_history, [loss_val])
        
        if iter_count[] % 20 == 0
            @printf("  Iter %4d | Pérdida = %.6e\n", iter_count[], loss_val)
            flush(stdout)
        end
        return false
    end

    opt_func = OptimizationFunction(total_loss, Optimization.AutoZygote())
    prob = OptimizationProblem(opt_func, θ0, p)
    
    res = solve(prob, OptimizationOptimisers.Adam(lr),
                callback = callback, maxiters = maxiters,
                verbose = false,
                progress = true)

    println("\n✅ Completado")
    println("   Pérdida final: $(res.objective)")
    flush(stdout)
    
    return (params = res.u, loss_history = loss_history,
            nn = nn, st = st, cfg = mcfg, train_data = train_data)
end

# ====================================================================
# EJECUCIÓN
# ====================================================================

#global resultado = main(fast_mode = false, n_states = 2)
global resultado = main(fast_mode = true, n_states = 5)  # Para testing

global pesos_optimos = resultado.params

println("\n📊 Resumen:")
println("  - Parámetros: $(length(pesos_optimos))")
println("  - Pérdida final: $(resultado.loss_history[end])")
println("  - Estados: $(length(resultado.train_data))")
flush(stdout)