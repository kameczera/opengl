#ifndef OBJECTS_CUH
#define OBJECTS_CUH

#include <cuda_runtime.h>
#include <cmath>

#include "vec3.cuh"

class circle {
    private:
    
    public:
        vec3 center;
        int radius;
        __host__ __device__ circle(int radius, point3 center) : radius(radius), center(center) { }
        __device__ bool hit_circle(int x, int y, int z) {
            int dx = x - center.x();
            int dy = y - center.y();
            int dz = z - center.z();
            int dist2 = dx*dx + dy*dy + dz*dz;
            return dist2 <= radius * radius;
        }
};

#endif