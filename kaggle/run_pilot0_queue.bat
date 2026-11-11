@echo off
cd /d "C:\Users\abdulhamid batayhi\Desktop\ai-column-structure"
set KAGGLE_CLI=C:\Users\abdulhamid batayhi\AppData\Local\Programs\Python\Python313\Scripts\kaggle.exe
"C:\Users\abdulhamid batayhi\Desktop\ai-mof-cof-dynamics\.venv\Scripts\python.exe" -u kaggle\run_queue.py kaggle\jobs_pilot0.json >> kaggle\queue_pilot0.log 2>&1
