using DataFrames
using Parquet2
using Dates

file_path = "data/cases_wrapped.parquet"

data = DataFrame(Parquet2.readfile(file_path))


select!(data, :State, :Last_Update, :Accumulated_Cases, 
              :Accumulated_Deaths, :Active, :Recovered,
              :People_Hospitalized)

lag0(x) = vcat(0, x[1:end-1]) 

gdf_state = groupby(data, :State)



# Calculate the New_Cases and New_Deaths columns
transform!(gdf_state, 
    :Accumulated_Cases => (x -> x .- lag0(x)) => :New_Cases,
    :Accumulated_Deaths => (x -> x .- lag0(x)) => :New_Deaths
)

# Detect the upload period
transform!(gdf_state, :New_Cases => (x -> x .!= 0) => :IS_Report)
transform!(gdf_state, :IS_Report => cumsum => :ID_Period)

# Create the new incidents columns
gdf_period = groupby(data, [:State, :ID_Period])

df = combine(gdf_period,
    :Last_Update => minimum => :Start_Date,
    :Last_Update => maximum => :Report_Date,
    nrow => :Days_In_Period,         
    :New_Deaths => sum => :New_Deaths,
    :New_Cases => sum => :New_Cases,
    :Accumulated_Cases => maximum => :Accumulated_Cases,
    :Accumulated_Deaths => maximum => :Accumulated_Deaths,
    :Active => maximum => :Active,
    :Recovered => maximum => :Recovered,
    :People_Hospitalized => maximum => :People_Hospitalized
)

transform!(df, 
    [:Accumulated_Deaths, :Accumulated_Cases] => ByRow((d, c) -> c == 0 ? 0.0 : d / c) => :Accumulated_Deaths_Rate,
    [:New_Deaths, :New_Cases] => ByRow((d, c) -> c == 0 ? 0.0 : d / c) => :New_Deaths_Rate
)


sort!(df, [:State, :Start_Date])
# Save cleaned dataset as Parquet file
out_dir = joinpath(pwd(), "data/clean_dataset.parquet")
Parquet2.writefile(out_dir, df)
