#!/bin/bash

echo "extracting video frames..."
bash extract_frame.sh
python extractFaceCrops.py

echo "extracting video audio..."
bash extract_wave.sh
