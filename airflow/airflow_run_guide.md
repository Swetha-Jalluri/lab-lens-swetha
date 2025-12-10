# Lab Lens Airflow - Essential Commands

================================================================================
MACOS
================================================================================

## START
cd /Users/Admin/Desktop/lab-lens
docker compose up -d
sleep 60
open http://localhost:8080

## LOGIN
Username: admin
Password: admin

## LOGS
docker compose logs -f airflow-scheduler

## STATUS
docker compose ps

## STOP
docker compose down

## RESTART
docker compose restart

## CLEAN RESTART (if errors)
docker compose down -v
docker compose up -d


================================================================================
WINDOWS (PowerShell)
================================================================================

## START
cd C:\Users\YourUsername\Desktop\lab-lens
docker compose up -d
Start-Sleep -Seconds 60
Start-Process "http://localhost:8080"

## LOGIN
Username: admin
Password: admin

## LOGS
docker compose logs -f airflow-scheduler

## STATUS
docker compose ps

## STOP
docker compose down

## RESTART
docker compose restart

## CLEAN RESTART (if errors)
docker compose down -v
docker compose up -d


================================================================================
IN AIRFLOW UI
================================================================================

1. Find "lab_lens_mimic_pipeline"
2. Toggle switch to ON (blue)
3. Click Play button ▶
4. Select "Trigger DAG"


================================================================================
IF ERRORS
================================================================================

docker compose down -v
docker compose up -d
sleep 60