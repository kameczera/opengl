#ifndef RENDER_CUH
#define RENDER_CUH

#include "objects.cuh"
#include "vec3.cuh"
#include "scene.cuh"

using namespace std;

const float MY_PI = 3.1415; 

extern GLuint pbo;
extern GLuint tex;
extern struct cudaGraphicsResource* cuda_pbo_resource;
static int WIDTH;
static int HEIGHT;
extern __managed__ scene* g_scene;
static float tempo = 0.0f;

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 27:
        case 'q':
        case 'Q':
            exit(0);
            break;
        case ' ': g_scene->cam.translate_y(1); break;
        case 'x': g_scene->cam.translate_y(-1); break;
        case 'w': g_scene->cam.translate_x(1); break;
        case 's': g_scene->cam.translate_x(-1); break;
        case 'd': g_scene->cam.translate_z(1); break;
        case 'a': g_scene->cam.translate_z(-1); break;
        default: break;
    }
    glutPostRedisplay();
}

// void passiveMotionCallback(int x, int y) {
//     printf("Passive motion at (%d, %d)\n", x, y);
// }

void createPBO() {
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * 4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

void createTexture() {
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

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

__global__ void renderGradient(uchar4* pixels, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;
    int idx = row * width + col;
    for(int i = 0; i < g_scene->circles_len; i++) {
        if(g_scene->d_circles[i].hit_circle(col + g_scene->cam.x(), row + g_scene->cam.y(), g_scene->cam.z())) {
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
    renderGradient<<<grid, block>>>(dptr, width, height);
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void display() {
    tempo += .01f;
    // update_rays();
    runCuda(WIDTH, HEIGHT);

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