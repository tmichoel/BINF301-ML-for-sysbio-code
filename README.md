# BINF301-code

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> BINF301-code

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "BINF301-code"
```
which auto-activate the project and enable local path handling from DrWatson.

## Downloading datasets

To run the notebooks in the `notebooks` folder locally, you need to download and save several datasets in the right location. The datasets are stored on [JuliaHub](https://juliahub.com) and can be downloaded by running the `download_processed_data.jl` script in the `src` folder. After downloading, the datasets will be saved in the `data/processed` folder.


