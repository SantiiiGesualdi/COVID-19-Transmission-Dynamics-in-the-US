using CSV
using DataFrames
using Downloads
using Parquet2
using Dates

url = "https://raw.githubusercontent.com/govex/COVID-19/refs/heads/master/data_tables/vaccine_data/us_data/time_series/time_series_covid19_vaccine_us.csv"
println("Descargando datos...")
data = CSV.read(Downloads.download(url), DataFrame)

# Initial Clean Up 
df_limpio = dropmissing(data, [:Date, :Province_State, :People_at_least_one_dose])
select!(df_limpio, :Date, :Province_State => :State, :People_at_least_one_dose)
df_limpio.Date = Date.(df_limpio.Date)

sort!(df_limpio, [:State, :Date])

# New Administration by State
gdf_state = groupby(df_limpio, :State)

lag0(x) = vcat(x[1], x[2:end] .- x[1:end-1])

transform!(gdf_state, :People_at_least_one_dose => lag0 => :New_admin)

# If there any negative it will force to be 0
df_limpio.New_admin = max.(0, df_limpio.New_admin)

# Periods
transform!(gdf_state, :New_admin => (x -> x .> 0) => :IS_Report)
transform!(gdf_state, :IS_Report => cumsum => :ID_Period)

gdf_period = groupby(df_limpio, [:State, :ID_Period])

df_final = combine(gdf_period,
    :Date => minimum => :Start_Date,
    :Date => maximum => :Date, # Llamado :Date para que el script ODE lo lea nativamente
    nrow => :Days_In_Period,         
    :New_admin => sum => :New_admin,
    :People_at_least_one_dose => maximum => :People_at_least_one_dose # Fundamental para la UDE
)

filter!(row -> row.New_admin > 0, df_final)

# Final Clean Up
sort!(df_final, [:State, :Date])
select!(df_final, Not(:Start_Date)) # Opcional: remover columnas que no usarás en la ODE

# Safe File
mkpath("data") 
out_dir = joinpath(pwd(), "data/doses_admin.parquet")
Parquet2.writefile(out_dir, df_final)

println("✅ File saved successfully: $out_dir")