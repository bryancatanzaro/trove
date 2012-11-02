#pragma once

struct cuda_timer {
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    void start() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        cudaEventRecord(start_event, 0);
    }

    float stop() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        float time = 0;
        cudaEventElapsedTime(&time, start_event, stop_event);
        return time;
    }
};
