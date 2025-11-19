#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>

#include "render.cuh"
#include "vec3.cuh"

GLuint pbo = 0;
GLuint tex = 0;
struct cudaGraphicsResource* cuda_pbo_resource;
static int HEIGHT = 225;
static int WIDTH = 400;

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 27:
        case 'q':
        case 'Q':
            exit(0);
            break;
        case 'w': scene.cam.translate_y(5); break;
        case 's': scene.cam.translate_y(-5); break;
        case 'a': scene.cam.translate_x(-5); break;
        case 'd': scene.cam.translate_x(5); break;
        case ' ': scene.cam.translate_z(5); break;
        case 'x': scene.cam.translate_z(-5); break;
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

circle* create_circles() {
    vec3 center(200, 100, 1);
    circle* circles = new circle(30, center);
    return circles;
}

scene_data init_scene() {
    scene_data scene;
    scene.circles_len = 1;

    circle* h_circles = create_circles();

    cudaMalloc(&scene.d_circles, sizeof(circle));
    cudaMemcpy(scene.d_circles, h_circles, sizeof(circle), cudaMemcpyHostToDevice);

    vec3 cam(0, 0, 0);
    scene.cam = cam;

    delete h_circles;

    return scene;
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("CUDA + OpenGL Gradient");

    glewInit();
    createPBO();
    createTexture();

    scene = init_scene();

    glutKeyboardFunc(keyboard);
    // glutSpecialFunc(specialKeys);

    glutDisplayFunc(display);
    glutMainLoop();

    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);

    return 0;
}
