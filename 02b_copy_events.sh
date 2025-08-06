#!/bin/bash

# Script to copy events to the correct directory structure for Delphes processing
# Based on the structure expected by 03a_read_delphes.py

set -e  # Exit on any error

echo "Setting up directory structure for Delphes processing..."

# Create base directory
FINAL_EVENTS_DIR="/vols/cms/us322/02b_final_events_14"
SOURCE_DIR="/vols/cms/us322/02_event_generation_14"

mkdir -p "$FINAL_EVENTS_DIR"

echo "Creating directory structure..."

# 1. Copy signal_sm events (both run_01 and run_01_decayed_1)
echo "Copying signal_sm events..."
mkdir -p "$FINAL_EVENTS_DIR/signal_sm/batch_0"
cp -r "$SOURCE_DIR/mg_processes/signal_sm/Events/run_01" "$FINAL_EVENTS_DIR/signal_sm/batch_0/"
cp -r "$SOURCE_DIR/mg_processes/signal_sm/Events/run_01_decayed_1" "$FINAL_EVENTS_DIR/signal_sm/batch_0/"

# 2. Copy background_0 events  
echo "Copying background_0 events..."
mkdir -p "$FINAL_EVENTS_DIR/background_0/batch_0"
cp -r "$SOURCE_DIR/mg_processes_2/background_0/Events/run_01" "$FINAL_EVENTS_DIR/background_0/batch_0/"

# 3. Copy signal_supp events (9 BSM points: morphing basis vectors 1-9, but likely 0-8)
echo "Copying signal_supp events..."
mkdir -p "$FINAL_EVENTS_DIR/signal_supp"

# Check which morphing basis vectors exist and copy them (both run_01 and run_01_decayed_1)
for i in {0..9}; do
    if [ -d "$SOURCE_DIR/mg_processes/signal_supp/morphing_basis_vector_$i" ]; then
        echo "Copying morphing_basis_vector_$i..."
        mkdir -p "$FINAL_EVENTS_DIR/signal_supp/mb_vector_$i/batch_0"
        cp -r "$SOURCE_DIR/mg_processes/signal_supp/morphing_basis_vector_$i/Events/run_01" "$FINAL_EVENTS_DIR/signal_supp/mb_vector_$i/batch_0/"
        cp -r "$SOURCE_DIR/mg_processes/signal_supp/morphing_basis_vector_$i/Events/run_01_decayed_1" "$FINAL_EVENTS_DIR/signal_supp/mb_vector_$i/batch_0/"
    else
        echo "Warning: morphing_basis_vector_$i not found, skipping..."
    fi
done

echo "Directory structure created successfully!"