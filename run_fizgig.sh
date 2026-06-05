#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python lora_trainer_gui.py &
disown
