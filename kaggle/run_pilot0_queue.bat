@echo off
cd /d "C:\Users\abdulhamid batayhi\Desktop\ai-column-structure"
set KAGGLE_CLI=C:\Users\abdulhamid batayhi\AppData\Local\Programs\Python\Python313\Scripts\kaggle.exe
rem the phase-0 jobs file is generated here, from the current pilot revision (T8 RS-4); none is kept in git
"C:\Users\abdulhamid batayhi\Desktop\ai-mof-cof-dynamics\.venv\Scripts\python.exe" kaggle\pilot_jobs.py 0 kaggle\jobs_pilot0.json || exit /b 1
"C:\Users\abdulhamid batayhi\Desktop\ai-mof-cof-dynamics\.venv\Scripts\python.exe" -u kaggle\run_queue.py kaggle\jobs_pilot0.json >> kaggle\queue_pilot0.log 2>&1
