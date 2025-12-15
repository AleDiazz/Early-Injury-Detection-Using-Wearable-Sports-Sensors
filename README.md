# Early-Injury-Detection-Using-Wearable-Sports-Sensors
High-Performance Computing Final Report: Parallel Feature Extraction with OpenMP for Early Injury Detection Using Wearable Sports Sensors

# Parallel Feature Extraction with OpenMP

A C program that simulates wearable IMU (Inertial Measurement Unit) sensor data and performs feature extraction both sequentially and in parallel using OpenMP.

## Overview

This program demonstrates parallel computing techniques applied to sensor data processing, specifically:

- **Synthetic IMU data generation**: Simulates accelerometer and gyroscope data from wearable sensors
- **Feature extraction**: Computes multiple features (RMS, variance, peak impact, asymmetry, frequency)
- **Parallel processing**: Uses OpenMP to parallelize feature extraction across time windows
- **Performance comparison**: Measures and compares sequential vs parallel execution times

## Features Extracted

1. **RMS (Root Mean Square)**: Overall signal magnitude
2. **Variance**: Signal variability
3. **Peak Impact**: Maximum acceleration (for impact detection)
4. **Asymmetry**: Difference between left and right sensors
5. **Frequency**: Estimated dominant frequency via zero-crossing analysis

## Usage

# Basic mode - specify number of samples
./imu_feature_extraction 100000

# Advanced mode - specify samples and window size
./imu_feature_extraction 200000 250

# Control thread count with environment variable
OMP_NUM_THREADS=4 ./imu_feature_extraction 100000

# Test with different configurations
./imu_feature_extraction 50000 100
./imu_feature_extraction 100000 200
./imu_feature_extraction 200000 250
OMP_NUM_THREADS=8 ./imu_feature_extraction 100000

## Configuration

You can configure the workload via command-line arguments:

# Syntax
./imu_feature_extraction [total_samples] [window_size]

# Default values if not specified:
#   total_samples = 100000
#   window_size = 200

Other constants are defined in the source code:

#define SAMPLING_RATE 100       // Hz
#define NUM_AXES 3              // X, Y, Z axes
#define NUM_SENSORS 2           // Left and right sensors

### Validation

The program automatically validates that sequential and parallel results match within a numerical tolerance (1e-6), ensuring correctness of the parallel implementation.

The program is compiled with:
- `-fopenmp`: Enable OpenMP support
- `-O2`: Optimization level 2 for better performance
- `-lm`: Link math library for mathematical functions
