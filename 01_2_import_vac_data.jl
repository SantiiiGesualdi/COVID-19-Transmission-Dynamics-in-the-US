using CSV
using DataFrames
using Downloads

url = "https://raw.githubusercontent.com/govex/COVID-19/refs/heads/master/data_tables/vaccine_data/us_data/time_series/time_series_covid19_vaccine_us.csv"

df_vacunas = CSV.read(Downloads.download(url), DataFrame)

first(df_vacunas, 5)

df_limpio = dropmissing(df_vacunas, [:Doses_admin, :Province_State])

select!(df_limpio,:Date,:Province_State=>:State,:Doses_admin)

# Save as Parquet file
out_dir = joinpath(pwd(), "data/doses_admin.parquet")
Parquet2.writefile(out_dir, df_limpio)