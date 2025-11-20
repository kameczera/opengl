#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

class vec3 {
  public:
    int e[3];

    __host__ __device__ vec3() : e{0,0,0} {}
    __host__ __device__ vec3(int e0, int e1, int e2) : e{e0, e1, e2} {}

    __host__ __device__ int x() const { return e[0]; }
    __host__ __device__ int y() const { return e[1]; }
    __host__ __device__ int z() const { return e[2]; }

    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ int operator[](int i) const { return e[i]; }
    __host__ __device__ int& operator[](int i) { return e[i]; }

    __host__ __device__ vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ vec3& operator*=(int t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ vec3& operator/=(int t) {
        return *this *= 1/t;
    }

    __host__ __device__ int length() const {
        return (int) sqrtf((float) length_squared());
    }

    __host__ __device__ int length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    __host__ __device__ void translate_x(int x) {
        e[0] += x;
    }

    __host__ __device__ void translate_y(int y) {
        e[1] += y;
    }

    __host__ __device__ void translate_z(int z) {
        e[2] += z;
    }
};

using point3 = vec3;

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(int t, const vec3& v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, int t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3& v, int t) {
    return (1/t) * v;
}

__host__ __device__ inline int dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

#endif