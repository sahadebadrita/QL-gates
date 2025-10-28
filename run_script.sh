#!/bin/bash

# Navigate to the directory containing your Python script
#cd /path/to/your/project/directory

# Activate the virtual environment
source graphs/bin/activate  # For Linux/macOS
# venv\Scripts\activate   # For Windows (if using a .bat file or similar)

# Run your Python script
python checkhadamard.py > h.txt

# Deactivate the virtual environment (optional, but good practice)
deactivate
