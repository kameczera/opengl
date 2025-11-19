#ifndef RENDER_CUH
#define RENDER_CUH

#include "objects.cuh"
#include "vec3.cuh"

using namespace std;

const float MY_PI = 3.1415; 

extern GLuint pbo;
extern GLuint tex;
extern struct cudaGraphicsResource* cuda_pbo_resource;

struct scene_data {
    circle* d_circles;
    int circles_len;
    vec3 cam;
};

static scene_data scene;

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

__global__ void renderGradient(uchar4* pixels, circle* circles, int circles_len, int width, int height, int cam_x, int cam_y) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;
    int idx = row * width + col;
    for(int i = 0; i < circles_len; i++) {
        if(circles[i].hit_circle(col + cam_x, row + cam_y, 0)) {
            pixels[idx] = pack_color(row, col, width, height);
        } else {
            pixels[idx] = pack_color(0, 0, width, height);
        }
    }
}

void runCuda(int width, int height) {
    uchar4* dptr;
    size_t size;
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &size, cuda_pbo_resource);

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    renderGradient<<<grid, block>>>(dptr, scene.d_circles, scene.circles_len, width, height, scene.cam.x(), scene.cam.y());
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void display() {
    static float time = 0.0f;
    time += .01f;
    double aspect_ratio = 16.0 / 9.0;
    int width = 400;
    int height = int(width / aspect_ratio);
    height = (height < 1) ? 1 : height;

    runCuda(width, height);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);

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