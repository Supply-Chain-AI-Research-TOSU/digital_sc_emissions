#!/bin/bash
source ~/projects/digital_sc_emissions/venv/bin/activate

cd ~/projects/digital_sc_emissions/backend
uvicorn main:app --host 0.0.0.0 --port 8010 --workers 1
