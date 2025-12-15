/*
 * IMU Feature Extraction with OpenMP Parallelization
 * 
 * BASIC MODE (Validation & Speedup):
 * ./imu_feature_extraction [samples]
 * Example: ./imu_feature_extraction 100000
 * 
 * ADVANCED MODE (Custom Configuration):
 * ./imu_feature_extraction [samples] [window_size]
 * Examples:
 *   ./imu_feature_extraction 50000 100
 *   ./imu_feature_extraction 200000 250
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#define SAMPLING_RATE 100
#define NUM_AXES 3
#define NUM_SENSORS 2
#define TOLERANCE 1e-6

float *accel;
float *gyro;

double get_walltime() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}

void generate_sensor_data(int total_samples) {
    int sensor, axis, i;
    float t, freq, base_signal, noise, spike, sensor_offset;
    
    srand(42);
    
    for (sensor = 0; sensor < NUM_SENSORS; sensor++) {
        for (axis = 0; axis < NUM_AXES; axis++) {
            for (i = 0; i < total_samples; i++) {
                t = (float)i / SAMPLING_RATE;
                freq = 1.0 + axis * 0.5 + sensor * 0.3;
                base_signal = 2.0 * sin(2.0 * M_PI * freq * t);
                noise = ((float)rand() / RAND_MAX - 0.5) * 0.5;
                
                spike = 0.0;
                if (axis == 2 && (rand() % 500) == 0) {
                    spike = 10.0 * ((float)rand() / RAND_MAX);
                }
                
                sensor_offset = sensor * 0.2;
                
                accel[sensor * NUM_AXES * total_samples + axis * total_samples + i] = 
                    base_signal + noise + spike + sensor_offset;
                gyro[sensor * NUM_AXES * total_samples + axis * total_samples + i] = 
                    base_signal * 0.5 + noise;
            }
        }
    }
}

float compute_rms(int window_idx, int window_size, int total_samples) {
    int start = window_idx * window_size;
    int end = start + window_size;
    int sensor, axis, i;
    float sum_squares = 0.0;
    int count = 0;
    
    for (sensor = 0; sensor < NUM_SENSORS; sensor++) {
        for (axis = 0; axis < NUM_AXES; axis++) {
            for (i = start; i < end; i++) {
                float val = accel[sensor * NUM_AXES * total_samples + axis * total_samples + i];
                sum_squares += val * val;
                count++;
            }
        }
    }
    
    return sqrt(sum_squares / count);
}

float compute_variance(int window_idx, int window_size, int total_samples) {
    int start = window_idx * window_size;
    int end = start + window_size;
    int sensor, axis, i;
    float sum = 0.0;
    int count = 0;
    
    // Calculate mean
    for (sensor = 0; sensor < NUM_SENSORS; sensor++) {
        for (axis = 0; axis < NUM_AXES; axis++) {
            for (i = start; i < end; i++) {
                sum += accel[sensor * NUM_AXES * total_samples + axis * total_samples + i];
                count++;
            }
        }
    }
    
    float mean = sum / count;
    float sum_squared_diff = 0.0;
    
    // Calculate variance
    for (sensor = 0; sensor < NUM_SENSORS; sensor++) {
        for (axis = 0; axis < NUM_AXES; axis++) {
            for (i = start; i < end; i++) {
                float diff = accel[sensor * NUM_AXES * total_samples + axis * total_samples + i] - mean;
                sum_squared_diff += diff * diff;
            }
        }
    }
    
    return sum_squared_diff / count;
}

float compute_peak_impact(int window_idx, int window_size, int total_samples) {
    int start = window_idx * window_size;
    int end = start + window_size;
    int sensor, axis, i;
    float max_val = 0.0;
    
    for (sensor = 0; sensor < NUM_SENSORS; sensor++) {
        for (axis = 0; axis < NUM_AXES; axis++) {
            for (i = start; i < end; i++) {
                float abs_val = fabs(accel[sensor * NUM_AXES * total_samples + axis * total_samples + i]);
                if (abs_val > max_val) {
                    max_val = abs_val;
                }
            }
        }
    }
    
    return max_val;
}

float compute_asymmetry(int window_idx, int window_size, int total_samples) {
    int start = window_idx * window_size;
    int end = start + window_size;
    int sensor, axis, i;
    float rms[NUM_SENSORS] = {0.0};
    
    // Calculate RMS for each sensor
    for (sensor = 0; sensor < NUM_SENSORS; sensor++) {
        float sum_squares = 0.0;
        int count = 0;
        
        for (axis = 0; axis < NUM_AXES; axis++) {
            for (i = start; i < end; i++) {
                float val = accel[sensor * NUM_AXES * total_samples + axis * total_samples + i];
                sum_squares += val * val;
                count++;
            }
        }
        
        rms[sensor] = sqrt(sum_squares / count);
    }
    
    return fabs(rms[0] - rms[1]);
}

float compute_frequency_feature(int window_idx, int window_size, int total_samples) {
    int start = window_idx * window_size;
    int end = start + window_size;
    int i;
    int zero_crossings = 0;
    
    // Count zero crossings
    for (i = start; i < end - 1; i++) {
        float current = accel[i];
        float next = accel[i + 1];
        
        if ((current >= 0 && next < 0) || (current < 0 && next >= 0)) {
            zero_crossings++;
        }
    }
    
    float duration = (float)window_size / SAMPLING_RATE;
    float frequency = (float)zero_crossings / (2.0 * duration);
    
    return frequency;
}

void feature_extraction_sequential(int num_windows, int window_size, int total_samples,
                                   float *rms, float *variance, float *peak, 
                                   float *asymmetry, float *frequency) {
    int w;
    
    for (w = 0; w < num_windows; w++) {
        rms[w] = compute_rms(w, window_size, total_samples);
        variance[w] = compute_variance(w, window_size, total_samples);
        peak[w] = compute_peak_impact(w, window_size, total_samples);
        asymmetry[w] = compute_asymmetry(w, window_size, total_samples);
        frequency[w] = compute_frequency_feature(w, window_size, total_samples);
    }
}

void feature_extraction_parallel(int num_windows, int window_size, int total_samples,
                                float *rms, float *variance, float *peak, 
                                float *asymmetry, float *frequency) {
    int w;
    
    #pragma omp parallel for private(w) schedule(dynamic)
    for (w = 0; w < num_windows; w++) {
        rms[w] = compute_rms(w, window_size, total_samples);
        variance[w] = compute_variance(w, window_size, total_samples);
        peak[w] = compute_peak_impact(w, window_size, total_samples);
        asymmetry[w] = compute_asymmetry(w, window_size, total_samples);
        frequency[w] = compute_frequency_feature(w, window_size, total_samples);
    }
}

int main(int argc, char *argv[]) {
    int i;
    int total_samples = 100000;
    int window_size = 200;
    float *rms1, *variance1, *peak1, *asymmetry1, *frequency1;
    float *rms2, *variance2, *peak2, *asymmetry2, *frequency2;
    double time1, time2;
    
    // Parse command line arguments
    if (argc > 1) {
        total_samples = atoi(argv[1]);
    }
    if (argc > 2) {
        window_size = atoi(argv[2]);
    }
    
    int num_windows = total_samples / window_size;
    
    // Allocate memory for sensor data
    accel = (float *)malloc(NUM_SENSORS * NUM_AXES * total_samples * sizeof(float));
    gyro = (float *)malloc(NUM_SENSORS * NUM_AXES * total_samples * sizeof(float));
    
    // Allocate memory for sequential results
    rms1 = (float *)malloc(num_windows * sizeof(float));
    variance1 = (float *)malloc(num_windows * sizeof(float));
    peak1 = (float *)malloc(num_windows * sizeof(float));
    asymmetry1 = (float *)malloc(num_windows * sizeof(float));
    frequency1 = (float *)malloc(num_windows * sizeof(float));
    
    // Allocate memory for parallel results
    rms2 = (float *)malloc(num_windows * sizeof(float));
    variance2 = (float *)malloc(num_windows * sizeof(float));
    peak2 = (float *)malloc(num_windows * sizeof(float));
    asymmetry2 = (float *)malloc(num_windows * sizeof(float));
    frequency2 = (float *)malloc(num_windows * sizeof(float));

    // Generate sensor data
    generate_sensor_data(total_samples);
    
    // Sequential version
    time1 = get_walltime();
    feature_extraction_sequential(num_windows, window_size, total_samples,
                                  rms1, variance1, peak1, asymmetry1, frequency1);
    time1 = get_walltime() - time1;
    
    // Parallel version
    time2 = get_walltime();
    feature_extraction_parallel(num_windows, window_size, total_samples,
                               rms2, variance2, peak2, asymmetry2, frequency2);
    time2 = get_walltime() - time2;

    // Validation - compare sequential and parallel results
    int correct = 1;
    double max_diff = 0.0;
    
    for (i = 0; i < num_windows; i++) {
        double diff;
        
        diff = fabs(rms1[i] - rms2[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
        
        diff = fabs(variance1[i] - variance2[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
        
        diff = fabs(peak1[i] - peak2[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
        
        diff = fabs(asymmetry1[i] - asymmetry2[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
        
        diff = fabs(frequency1[i] - frequency2[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
        
        if (max_diff > TOLERANCE) {
            correct = 0;
            break;
        }
    }

    // Print results
    printf("=== IMU Feature Extraction Results ===\n");
    printf("Total samples: %d\n", total_samples);
    printf("Window size: %d\n", window_size);
    printf("Number of windows: %d\n", num_windows);
    printf("Threads: %d\n", omp_get_max_threads());
    printf("\n");
    printf("Sequential time: %.6f s\n", time1);
    printf("Parallel time: %.6f s\n", time2);
    printf("Speedup: %.2fx\n", time1 / time2);
    printf("\n");
    printf("Max difference: %.2e\n", max_diff);
    printf("Validation: %s\n", correct ? "PASSED" : "FAILED");
    printf("\n");
    printf("Sample features (window 0):\n");
    printf("  RMS: %.6f, Variance: %.6f, Peak: %.6f\n", 
           rms1[0], variance1[0], peak1[0]);
    printf("  Asymmetry: %.6f, Frequency: %.2f Hz\n", 
           asymmetry1[0], frequency1[0]);
    
    // Free allocated memory
    free(accel);
    free(gyro);
    free(rms1);
    free(variance1);
    free(peak1);
    free(asymmetry1);
    free(frequency1);
    free(rms2);
    free(variance2);
    free(peak2);
    free(asymmetry2);
    free(frequency2);
    
    return 0;
}
