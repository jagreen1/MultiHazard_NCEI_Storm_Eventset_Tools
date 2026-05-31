# NCEI_Storm_Multihazard_Eventset
This repository contains Python scripts that allows the user to bulk download and clean the NOAA NCEI's Storm Events Database, and then generate single-hazard and multi-hazard eventset. Eventsets can be customized by defining time period (years), the multi-hazard timelag overlap (days), the hazard/peril event types (see database documentation), and various event impact threshold filters (deaths, injuries, crop damage, building damage).

The download and cleaning capabilities can access the most up to date files from NOAA NCEI. However, the multi-hazard event set generation script is currently only designed to create event sets up until 2024, and will encounter errors if containing events ater 2024. Future work will modify the event set generation script to automatically limit input parameters to only allow years included in the user provided data.

This is the first publically avaliable Multi-hazard Eventset with impact/loss data specific to the United States. Future work can adapt these scripts for the SHELDUS database led by Melanie Gall at ASU, however their database require commercial access.
The formatting of the multi-hazard pair event output has been designed in a manner similar to the MYRIAD-HESA multi-hazard event set developed by Judith Claassen.
https://github.com/judithclaassen/MYRIAD-HESA/

## Scripts


1) Clean_NCEI_Storm_Database.py
- bulk download NOAA NCEI Storm Event Database
- preprocess/clean NCEI Storm Event Database

2) Generate_NCEI_Storm_Multihazard_Eventset.py
- create customizable event sets for 1) all hazards (AE), 2) single-hazards (SH), 3) multi-hazard pairs (MHP), and 4) unique multi-hazard events (MH)


The input database files necessary to run these scripts can be downloaded via HTML/FTP on the NCEI website, or automatically using Clean_NCEI_Storm_Database.py.
- https://www.ncdc.noaa.gov/stormevents/ftp.jsp
- https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
- ftp://ftp.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/





## Citation
Please cite the datasets and preprocessing script if used in any publications:
- **Green, J. (2026) NCEI Storm Multihazard Eventset. [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.20285674.**
- **Green, J. (2026) MultiHazard_NCEI_Storm_Eventset_Tools. Github. https://github.com/jagreen1/MultiHazard_NCEI_Storm_Eventset_Tools.**
