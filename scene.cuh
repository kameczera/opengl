#ifndef SCENE_CUH
#define SCENE_CUH

#include <cuda_runtime.h>
#include "vec3.cuh"
#include "ray.cuh"
#include "objects.cuh"

inline circle* create_circles() {
    point3 center(200, 100, 1);
    circle* circles = (circle*) malloc(sizeof(circle));
    circles[0] = circle(30, center);
    return circles;
}

inline ray* create_rays() {
    const int N = 3;
    ray* rays = (ray*) malloc(sizeof(ray) * N);
    for(int i = 0; i < N; i++) {
        point3 pos(5 + 50 * i, 0, 0);
        vec3 dir(1, 0, 0);
        rays[i] = ray(pos, dir);
    }
    return rays;
}

class scene {
    public:
        circle* d_circles;
        int circles_len;
        ray* d_rays;
        int rays_len;
        vec3 cam;

        __host__ scene() : d_circles(nullptr), circles_len(0), d_rays(nullptr), rays_len(0), cam(0,0,0) {
            circle* h_circles = create_circles();
            circles_len = 1;
            cudaMalloc(&d_circles, sizeof(circle) * circles_len);
            cudaMemcpy(d_circles, h_circles, sizeof(circle) * circles_len, cudaMemcpyHostToDevice);

            ray* h_rays = create_rays();
            rays_len = 3;
            cudaMalloc(&d_rays, sizeof(ray) * rays_len);
            cudaMemcpy(d_rays, h_rays, sizeof(ray) * rays_len, cudaMemcpyHostToDevice);

            free(h_circles);
            free(h_rays);
        }

        __host__ ~scene() {
            if (d_circles) cudaFree(d_circles);
            if (d_rays) cudaFree(d_rays);
        }
};

#endif