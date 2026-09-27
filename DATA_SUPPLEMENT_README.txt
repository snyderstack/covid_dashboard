COVID-19 County Outcomes Analysis Platform - Data Supplement v2.0
==================================================================

This archive holds the data files that are too large for the GitHub
repository. With the repository and these files, every dashboard feature
works.

How to install
--------------
1. Clone the repository:
     git clone https://github.com/snyderstack/covid_dashboard.git
2. Put covid_data_supplement_v2.zip in the repository root (the folder
   that contains app.py) and unzip it there:
     unzip covid_data_supplement_v2.zip
   The archive already contains the data/ folder paths, so each file
   lands where the loaders look for it.
3. Install and run (see README.md):
     pip install -r requirements.txt
     python -m streamlit run app.py

Files in this archive
---------------------
File name and exact path (relative to the repository root), size, and
what it enables:

  data/COVID-19_Vaccinations_in_the_United_States,County_20260623.csv
      635 MB. CDC county vaccination data, Dec 2020 - May 2023.
      Read by vaccination_loader.py. Enables all vaccination features:
      map metrics, rollout charts, vaccination factors and models.
      The file name must match exactly, including the date suffix.

  data/AHRF_2020-2021_SAS/AHRF2021.sas7bdat
      184 MB. HRSA Area Health Resources File, 2020-2021 SAS release.
      Read by ahrf_loader.py. Supplementary 2019-2021 AHRF columns.

  data/AHRF2020.asc
      98 MB. HRSA Area Health Resources File, 2019-2020 ASCII release.
      Read by ahrf_loader.py, using the layout file already in the
      repository at data/AHRF_2019-2020/DOC/AHRF2019-2020.sas.
      Supplementary 2018-2020 variables (HPSA, physician counts).

  data/AHRF_2020-2021_SAS/2020-2021_AHRFDUA.doc
      32 KB. HRSA data use agreement for the 2020-2021 AHRF release.
      Not read by the dashboard.

Already in the repository (not in this archive)
-----------------------------------------------
  data/covid_confirmed_usafacts.csv
  data/covid_deaths_usafacts.csv
  data/covid_county_population_usafacts.csv
  data/ahrf2023.csv
  data/geojson-counties-fips.json
  data/state_political.csv
  data/AHRF_2019-2020/DOC/  (AHRF 2019-2020 layout file and documentation)

Without this archive
--------------------
The dashboard still runs. Vaccination features are hidden and AHRF data
comes from data/ahrf2023.csv only.

Sources
-------
CDC:  https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh
HRSA: https://data.hrsa.gov/data/download

The AHRF files are distributed under the HRSA data use agreement included
in this archive. See data/README.md in the repository for the full list
of data files.
