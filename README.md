# Phenology-Niche-SAM2
Official code and data repository for the paper: **" Cross-Scale Object-Level Mapping of Giant Panda Staple Bamboo Species using Phenology–Niche Consistency and SAM2 "**.

## Workflow & Code Structure
The workflow consists of Google Earth Engine (GEE) data preparation and local Python processing. 

* **Part 1: GEE Data Prep (Run in [GEE Code Editor](https://code.earthengine.google.com/))**
  * `01_GEE_Extract_Reference_NDVI.js`: Extracts monthly reference NDVI curves for bamboo species.
  * `02_GEE_Export_Composite_Image.js`: Exports Sentinel-2 time-series + NASADEM composite image (10m).
* **Part 2: Python Processing (Run locally)**
  * `03_Phenology_Niche_Seed_Extraction.py`: Extracts high-confidence bamboo seed prompts using HANTS and niche-weighted TW-DTW.
  * `04_SAM2_Object_Delineation.py`: Generates object-level bamboo maps using SAM2 on high-res PlanetScope imagery (3m).

 ## Data Availability
  *  **Input Data:**
  * **Validation Samples (`validation_samples.csv`):** Provided directly in the `/data/` folder of this repository (contains field-collected sample locations and species labels).
  * **PlanetScope Image (`PlanetScope_2025.tif`):** Accessed via the [Planet Official Website](https://www.planet.com/). Users need an approved education/research account to download the corresponding scenes.
  * **Sentinel & DEM Composite (`2025.tif`):** Generated directly via the provided GEE scripts (`01` and `02`). It utilizes publicly available Copernicus Sentinel-1/2 and NASADEM datasets hosted on Google Earth Engine.
* **Final Results (`/results/`):** 
  * The multi-year 3m object-level bamboo species distribution maps (Shapefiles for **2018, 2020, 2023, and 2025**) are provided in the `/results/` folder.
