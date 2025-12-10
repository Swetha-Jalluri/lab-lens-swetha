# Lab Lens DVC Commands Reference

================================================================================
MACOS COMMANDS
================================================================================

## SETUP
cd /Users/Admin/Desktop/lab-lens
source venv/bin/activate
pip install dvc

## CHECK STATUS
dvc status

## RUN PIPELINE
dvc repro

## TRACK DATA FILES
dvc add data-pipeline/data/raw/mimic_discharge_labs.csv

## COMMIT TO GIT
git add dvc.lock .dvc .gitignore
git add data-pipeline/data/raw/mimic_discharge_labs.csv.dvc
git commit -m "Update DVC pipeline"

## VIEW PIPELINE DAG
dvc dag

## SETUP REMOTE STORAGE (Optional)
dvc remote add -d storage s3://bucket-name/path
dvc push

## PULL DATA FROM REMOTE
dvc pull

## FORCE RERUN
dvc repro --force

## CLEAN CACHE
dvc gc


================================================================================
WINDOWS COMMANDS (PowerShell)
================================================================================

## SETUP
cd C:\Users\YourUsername\Desktop\lab-lens
.\venv\Scripts\Activate
pip install dvc

## CHECK STATUS
dvc status

## RUN PIPELINE
dvc repro

## TRACK DATA FILES
dvc add data-pipeline/data/raw/mimic_discharge_labs.csv

## COMMIT TO GIT
git add dvc.lock .dvc .gitignore
git add data-pipeline/data/raw/mimic_discharge_labs.csv.dvc
git commit -m "Update DVC pipeline"

## VIEW PIPELINE DAG
dvc dag

## SETUP REMOTE STORAGE (Optional)
dvc remote add -d storage s3://bucket-name/path
dvc push

## PULL DATA FROM REMOTE
dvc pull

## FORCE RERUN
dvc repro --force

## CLEAN CACHE
dvc gc


================================================================================
WINDOWS COMMANDS (CMD)
================================================================================

## SETUP
cd C:\Users\YourUsername\Desktop\lab-lens
venv\Scripts\activate
pip install dvc

## CHECK STATUS
dvc status

## RUN PIPELINE
dvc repro

## TRACK DATA FILES
dvc add data-pipeline/data/raw/mimic_discharge_labs.csv

## COMMIT TO GIT
git add dvc.lock .dvc .gitignore
git add data-pipeline/data/raw/mimic_discharge_labs.csv.dvc
git commit -m "Update DVC pipeline"

## VIEW PIPELINE DAG
dvc dag

## SETUP REMOTE STORAGE (Optional)
dvc remote add -d storage s3://bucket-name/path
dvc push

## PULL DATA FROM REMOTE
dvc pull

## FORCE RERUN
dvc repro --force

## CLEAN CACHE
dvc gc


================================================================================
QUICK REFERENCE
================================================================================

CHECK:           dvc status
RUN:             dvc repro
FORCE RUN:       dvc repro --force
TRACK DATA:      dvc add <file>
VIEW DAG:        dvc dag
PUSH:            dvc push
PULL:            dvc pull
CLEAN:           dvc gc


================================================================================
COMMON WORKFLOWS
================================================================================

## FIRST TIME RUN
1. cd /Users/Admin/Desktop/lab-lens  (or Windows path)
2. source venv/bin/activate  (or venv\Scripts\activate on Windows)
3. pip install dvc
4. dvc status
5. dvc repro

## DAILY USE
1. source venv/bin/activate
2. dvc repro
3. git add dvc.lock
4. git commit -m "Update pipeline"

## IF ERRORS
dvc repro --force