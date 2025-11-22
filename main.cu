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
static int HEIGHT = 225;
static int WIDTH = 400;
scene* g_scene;

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 27:
        case 'q':
        case 'Q':
            exit(0);
            break;
        case 'w': g_scene->cam.translate_y(5); break;
        case 's': g_scene->cam.translate_y(-5); break;
        case 'a': g_scene->cam.translate_x(-5); break;
        case 'd': g_scene->cam.translate_x(5); break;
        case ' ': g_scene->cam.translate_z(5); break;
        case 'x': g_scene->cam.translate_z(-5); break;
        default: break;
    }
    glutPostRedisplay();
}

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

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("CUDA + OpenGL Gradient");

    glewInit();
    createPBO();
    createTexture();
    cudaMallocManaged (&g_scene, sizeof(scene));
    new(g_scene) scene();


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
