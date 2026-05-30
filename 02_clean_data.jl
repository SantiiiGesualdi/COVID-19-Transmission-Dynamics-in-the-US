using DataFrames
using Parquet2
using Dates

file_path = "data/Cases_Wrapped.parquet"

data = DataFrame(Parquet2.readfile(file_path))
data = select(data, :State, :Last_Update, :Accumulated_Cases, :Accumulated_Deaths);

lag0(x) = vcat(0, x[1:end-1]) 
rev_cumsum(x) = reverse(cumsum(reverse(x))) 

gdf_state = groupby(data, :State)

# Calculate the New_Cases and New_Deaths columns
transform!(gdf_state, 
    :Accumulated_Cases => (x -> x .- lag0(x)) => :New_Cases,
    :Accumulated_Deaths => (x -> x .- lag0(x)) => :New_Deaths
)

# Detect the upload period
transform!(gdf_state, :New_Cases => (x -> x .!= 0) => :is_report)
transform!(gdf_state, :is_report => rev_cumsum => :id_period)



gdf_period = groupby(data, [:State, :id_period])

df = combine(gdf_period,
    :Last_Update => minimum => :Start_Date,
    :Last_Update => maximum => :Report_Date,
    nrow => :Days_In_Period,         
    :New_Deaths => sum => :New_Deaths,
    :New_Cases => sum => :New_Cases,
    :Accumulated_Cases => maximum => :Accumulated_Cases,
    :Accumulated_Deaths => maximum => :Accumulated_Deaths
)

# Calculate the Rates of deaths (depending on cases amount)
df.Accumulated_Deaths_Rate = df.Accumulated_Deaths ./ df.Accumulated_Cases
df.New_Deaths_Rate = df.New_Deaths ./ df.New_Cases


# Delete missing values and sort
dropmissing!(df, :Start_Date) 
sort!(df, [:State, :Start_Date]);


out_dir = "data/"
Parquet2.writefile(joinpath(out_dir, "Claen_dataset.parquet"), df)