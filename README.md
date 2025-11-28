adaptive-scheduler
==================

Minimal scaffold for an adaptive scheduling service (FastAPI) with simple monitoring, scheduling logic, logging and a load-test client.

Quick start
-----------

- Create a virtualenv and install requirements:

```powershell
python -m venv env; .\env\Scripts\Activate.ps1; pip install -r requirements.txt
```

- Run the API server:

```powershell
uvicorn app.server:app --reload --port 8000
```

- Run a quick load test (from project root):

```powershell
python -m load_test.client --url http://localhost:8000/schedule --threads 4 --requests 100
```

Files created
-------------

- `app/` - FastAPI app and core modules
- `load_test/` - simple client to generate load
- `analysis/` - small analysis & training stubs
- `metrics_log.csv` - runtime metrics log (CSV)
- `requirements.txt` - Python dependencies

License: MIT
