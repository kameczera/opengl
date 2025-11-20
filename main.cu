#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>

#include "render.cuh"
#include "vec3.cuh"
#include "scene.cuh"

GLuint pbo = 0;
GLuint tex = 0;
struct cudaGraphicsResource* cuda_pbo_resource;
__managed__ scene* g_scene = nullptr;

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("CUDA + OpenGL Gradient");

    glewInit();
    createPBO();
    createTexture();

    glutKeyboardFunc(keyboard);
    // glutPassiveMotionFunc(passiveMotionCallback);
    // allocate managed scene and construct it in-place
    cudaMallocManaged(&g_scene, sizeof(scene));
    new (g_scene) scene();
    glutDisplayFunc(display);
    glutMainLoop();

    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);

    // explicitly destroy and free the managed scene
    if (g_scene) {
        g_scene->~scene();
        cudaFree(g_scene);
    }

    return 0;
}
