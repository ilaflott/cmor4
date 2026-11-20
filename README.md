# CMOR

**Climate Model Output Rewriter (CMOR), version 4**

CMOR is a Python package for creating CF-compliant climate model output in NetCDF format. It validates metadata against project tables (CMIP7, obs4MIPs, DRCDP) and writes standards-compliant NetCDF files from Python metadata and data objects.

## Key Features

- **Pythonic API**: Create datasets using Python objects instead of JSON configuration files
- **Project table validation**: Ensures compliance with projects using CMOR table formats, currently supporting [CMIP7](https://github.com/WCRP-CMIP/cmip7-cmor-tables), [obs4MIPs](https://github.com/PCMDI/obs4MIPs-cmor-tables), and [DRCDP](https://github.com/PCMDI/DRCDP)
- **xarray integration**: Built on xarray for modern, Pythonic data handling
- **Minimal metadata entry**: Only specify variable names and data - CF attributes are applied automatically from tables
- **CF compliance**: Produces Climate and Forecast conventions-compliant NetCDF files
- **Incremental writes**: Write large datasets one time chunk at a time with `DatasetWriter` ([guide](https://mauzey1.github.io/cmor4/usage/incremental_writing.html))

## Installation

CMOR requires Python 3.11 or newer. Install from source:

```bash
git clone https://github.com/mauzey1/cmor4.git
cd cmor4
pip install -e .
```

The project tables are Git submodules. To fetch the CMIP7 tables used in the example below, run this from the repository root:

```bash
git submodule update --init project_tables/cmip7-cmor-tables
```

See the [installation guide](https://mauzey1.github.io/cmor4/usage/installation.html) for details.

## Example

Run from the repository root:

```python
import numpy as np
import cmor

# Load the CMIP7 project tables
project = cmor.ProjectTables.from_directory(
    "project_tables/cmip7-cmor-tables",
    cv_file="tables-cvs/cmor-cvs.json",
    variable_tables=["tables/CMIP7_atmos.json"],
)

# Describe the dataset; other global attributes are filled in from the CV
dataset = project.dataset_info({
    "activity_id": "CMIP",
    "experiment_id": "amip",
    "forcing_index": "f1",
    "grid_label": "g101",
    "initialization_index": "i1",
    "institution_id": "CNRM-CERFACS",
    "license_id": "CC-BY-4.0",
    "nominal_resolution": "100 km",
    "physics_index": "p1",
    "realization_index": "r1",
    "region": "glb",
    "source_id": "CNRM-ESM2-1e",
    "outpath": "./output",
})

# Select the variable by its CMIP7 branded name; CF attributes come from the table
variable = project.variable("tas_tavg-h2m-hxy-u")

# Define the coordinates (the 2 m height coordinate is added automatically)
lat_bounds = np.linspace(-90, 90, 181)
lon_bounds = np.linspace(0, 360, 361)
axes = [
    project.axis("time", calendar="360_day", frequency="mon",
                 values=[15.0, 45.0], bounds=[[0.0, 30.0], [30.0, 60.0]],
                 units="days since 2000-01-01"),
    project.axis("latitude", values=(lat_bounds[:-1] + lat_bounds[1:]) / 2,
                 bounds=np.column_stack([lat_bounds[:-1], lat_bounds[1:]])),
    project.axis("longitude", values=(lon_bounds[:-1] + lon_bounds[1:]) / 2,
                 bounds=np.column_stack([lon_bounds[:-1], lon_bounds[1:]])),
]

# Data in (time, lat, lon) order
data = 288.0 + np.random.randn(2, 180, 360).astype("f4")

# Validate and write the CF-compliant NetCDF file
path = cmor.cmorize(dataset, variable, axes, data)
print(f"Created: {path}")
```

## Documentation

- [Documentation site](https://mauzey1.github.io/cmor4/): user guide (grids, vertical and time coordinates, incremental writes, controlled vocabularies) and API reference
- [Example notebooks](./notebooks/): Jupyter notebooks with visualizations
- [Example scripts](./examples/): standalone CMIP7 example programs

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the development setup, tests, and code style checks.

## Support

- **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/mauzey1/cmor4/issues)

## Acknowledgments

CMOR is developed and maintained by the Program for Climate Model Diagnosis and Intercomparison ([PCMDI](https://pcmdi.llnl.gov/)) at [Lawrence Livermore National Laboratory](https://www.llnl.gov/).
