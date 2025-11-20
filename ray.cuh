#ifndef RAY_CUH
#define RAY_CUH

#include "vec3.cuh"

class ray {
    public:
        point3 pos;
        vec3 dir;
        __host__ __device__ ray() : pos(0,0,0), dir(0,0,0) { }
        __host__ __device__ ray(point3 pos, vec3 dir) : pos(pos), dir(dir) { }
        __host__ __device__ void walk() {
            pos += dir;
        }
};


#endif