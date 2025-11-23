#ifndef RENDER_CUH
#define RENDER_CUH

#include "objects.cuh"
#include "vec3.cuh"
#include "ray.cuh"

using namespace std;

const float MY_PI = 3.1415; 
// static float time = 0.0f;

extern GLuint pbo;
extern GLuint tex;
extern struct cudaGraphicsResource* cuda_pbo_resource;
extern int WIDTH;
extern int HEIGHT;

circle* create_circles() {
    vec3 center(200, 100, 1);
    circle* circles = new circle(30, center);
    return circles;
}

ray* create_rays() {
    vec3 dir(1,0,0);
    ray* rays = (ray*) malloc(sizeof(ray) * 3);
    for(int i = 0; i < 3; i++) {
        point3 pos = point3(1, 50 + i * 50, 0);
        rays[i] = ray(pos, dir);
    }
    return rays;
}

class scene {
    public:
        circle* d_circles;
        int circles_len;
        vec3 cam;

        scene() : d_circles(nullptr), circles_len(0), cam(0, 0, 0) {
            circles_len = 1;
            circle* h_circles = create_circles();
            cudaMalloc(&d_circles, sizeof(circle) * circles_len);
            cudaMemcpy(d_circles, h_circles, sizeof(circle) * circles_len, cudaMemcpyHostToDevice);
 
            delete h_circles;
        }
};

extern scene* g_scene;

__device__ uchar4 pack_color(int row, int col, int width, int height) {

    float fx = col / float(width);
    float fy = row / float(height);

    uchar4 color;

    color.x = (unsigned char)(fx * 255.0f);
    color.y = (unsigned char)(fy * 255.0f);
    color.z = (unsigned char)( (fx+fy)*0.5f * 255);
    color.w = 255;

    return color;
}

__global__ void renderGradient(uchar4* pixels, int width, int height, scene* scene_ptr) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;
    int idx = row * width + col;
    for(int i = 0; i < scene_ptr->circles_len; i++) {
        if(scene_ptr->d_circles[i].hit_circle(col + scene_ptr->cam.x(), row + scene_ptr->cam.y(), scene_ptr->cam.z())) {
            pixels[idx] = pack_color(row, col, width, height);
        } else {
            pixels[idx] = pack_color(0, 0, width, height);
        }
    }
}

void runCuda(int width, int height, scene* scene_ptr) {
    uchar4* dptr;
    size_t size;
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &size, cuda_pbo_resource);
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    renderGradient<<<grid, block>>>(dptr, width, height, scene_ptr);
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void display() {
    // time += .01f; 

    runCuda(WIDTH, HEIGHT, g_scene);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(-1, -1);
    glTexCoord2f(1, 0); glVertex2f(1, -1);
    glTexCoord2f(1, 1); glVertex2f(1, 1);
    glTexCoord2f(0, 1); glVertex2f(-1, 1);
    glEnd();
    glDisable(GL_TEXTURE_2D);

    glutSwapBuffers();
    glutPostRedisplay();
}

#endif
