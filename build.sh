#!/bin/bash

# Compilation and Testing Script for IMU Feature Extraction
# Simple alternative to Makefiles

echo "=========================================="
echo "IMU Feature Extraction - Build & Test"
echo "=========================================="
echo ""

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Platform: macOS"
    COMPILER="clang -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp"
else
    echo "Platform: Linux"
    COMPILER="gcc -fopenmp"
fi

# Compile the program
echo "Compiling..."
$COMPILER -O2 -Wall -o imu_feature_extraction imu_feature_extraction.c -lm

if [ $? -eq 0 ]; then
    echo "✓ Compilation successful!"
    echo ""
else
    echo "✗ Compilation failed!"
    exit 1
fi

# Ask if user wants to run tests
if [ "$1" == "test" ]; then
    echo "=========================================="
    echo "Running tests with different thread counts"
    echo "=========================================="
    echo ""
    
    for threads in 1 2 4 8; do
        echo "=========================================="
        echo "Testing with $threads thread(s)"
        echo "=========================================="
        OMP_NUM_THREADS=$threads ./imu_feature_extraction 100000
        echo ""
    done
elif [ "$1" == "run" ]; then
    echo "Running with default configuration (100000 samples)..."
    echo ""
    ./imu_feature_extraction 100000
else
    echo "Build complete! Usage:"
    echo "  ./imu_feature_extraction [samples]           - Basic mode"
    echo "  ./imu_feature_extraction [samples] [window]  - Custom config"
    echo "  Examples:"
    echo "    ./imu_feature_extraction 100000"
    echo "    ./imu_feature_extraction 200000 250"
    echo "    OMP_NUM_THREADS=4 ./imu_feature_extraction 100000"
    echo ""
    echo "Or use this script:"
    echo "  ./build.sh run    - Compile and run"
    echo "  ./build.sh test   - Compile and test with 1,2,4,8 threads"
fi
