#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include <array>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    MESH,
    TRIANGLE
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Bounds3 {
    glm::vec3 pMin, pMax;
    glm::vec3 centroid;

    Bounds3()
    {
        float minNum = FLT_MIN;
        float maxNum = FLT_MAX;
        pMin = glm::vec3(minNum);
        pMax = glm::vec3(maxNum);
    }

    Bounds3(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3) {
        pMin = glm::vec3(fmin(p1.x, p2.x), fmin(p1.y, p2.y), fmin(p1.z, p2.z));
        pMax = glm::vec3(fmax(p1.x, p2.x), fmax(p1.y, p2.y), fmax(p1.z, p2.z));

        pMin = glm::vec3(fmin(pMin.x, p3.x), fmin(pMin.y, p3.y), fmin(pMin.z, p3.z));
        pMax = glm::vec3(fmax(pMax.x, p3.x), fmax(pMax.y, p3.y), fmax(pMax.z, p3.z));

        centroid = (p1 + p2 + p3) / 3.f;
    }

    glm::vec3 Diagonal() const { return pMax - pMin; }

    int maxExtent() const
    {
        auto d = Diagonal();
        if (d.x > d.y && d.x > d.z) { return  0; }
        else if (d.y > d.x) { return 1; }
        else { return 2; }
    }

    glm::vec3 Centroid() { return 0.5f * pMin + 0.5f * pMax; }

    bool IntersectP(const Ray& ray) const
    {
        glm::vec3 invDir = glm::vec3(1 / ray.direction.x, 1 / ray.direction.y, 1 / ray.direction.z);

        auto temp1 = pMax - ray.origin;
        auto temp2 = pMin - ray.origin;
        glm::vec3 ttop = temp1 * invDir;
        glm::vec3 tbot = temp2 * invDir;

        auto tmin = glm::vec3(std::min(ttop.x, tbot.x), std::min(ttop.y, tbot.y), std::min(ttop.z, tbot.z));
        auto tmax = glm::vec3(std::max(ttop.x, tbot.x), std::max(ttop.y, tbot.y), std::max(ttop.z, tbot.z));

        float t0 = std::max(tmin.x, (std::max)(tmin.y, tmin.z));
        float t1 = std::min(tmax.x, (std::min)(tmax.y, tmax.z));
        return t0 <= t1 && t1 >= 0;
    }

    bool Inside(const glm::vec3& p, const Bounds3& b)
    {
        return (p.x >= b.pMin.x && p.x <= b.pMax.x
            && p.y >= b.pMin.y && p.y <= b.pMax.y
            && p.z >= b.pMin.z && p.z <= b.pMax.z);
    }
};
 

struct Geom {
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    glm::vec3 pos[3];
    glm::vec3 normal[3];
    glm::vec2 uv[3];
    bool isObj{ false };

    const char* textureName;
    unsigned char* img;
    int texture_width;
    int texture_height;
    int channels;

    Bounds3 bbox;
    int obj_start_offset;
    int obj_end;

    glm::vec3 endPos;
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective{0};
    float hasRefractive{0};
    float indexOfRefraction{0};
    float emittance{0};
    float microfacet{0};
    float roughness{0};
    float metalness{ 0 };

    const char* textureName;
    unsigned char* img;
    int texture_width;
    int texture_height;
    int channels;
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float lensRadius;
    float focalDistance;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  glm::vec2 uv;
};


struct Triangle {
    glm::vec3 pos[3];
    glm::vec3 normal[3];
    glm::vec2 uv[3];
    int materialId;

    Triangle() {}
};

static Bounds3 Union(const Bounds3& b1, const Bounds3& b2) {
    Bounds3 ret;
    ret.pMin = glm::vec3(
        (std::min)(b1.pMin.x, b2.pMin.x),
        (std::min)(b1.pMin.y, b2.pMin.y),
        (std::min)(b1.pMin.z, b2.pMin.z));
    ret.pMax = glm::vec3(
        (std::max)(b1.pMax.x, b2.pMax.x),
        (std::max)(b1.pMax.y, b2.pMax.y),
        (std::max)(b1.pMax.z, b2.pMax.z));
    return ret;
}

inline Bounds3 Union(const Bounds3& b1, const glm::vec3& p) {
    Bounds3 ret;
    ret.pMin = glm::vec3((std::min)(b1.pMin.x, p.x),
        (std::min)(b1.pMin.y, p.y),
        (std::min)(b1.pMin.z, p.z));
    ret.pMax = glm::vec3((std::max)(b1.pMax.x, p.x),
        (std::max)(b1.pMax.y, p.y),
        (std::max)(b1.pMax.z, p.z));
    return ret;
}

struct Obj {
    Bounds3 box;
    Geom* data;
};

struct BVHBuildNode {
    Bounds3 bounds;
    BVHBuildNode* left;
    BVHBuildNode* right;
    Triangle* m_tri;
    int tri_index;
    int split_axis;

    int firstPrimOffset = 0, nPrimitives = 0;

    BVHBuildNode() {
        bounds = Bounds3();
        left = nullptr; right = nullptr;
        tri_index = -1;
        m_tri = nullptr;
    }
};

struct BVHGPUNode {
    Bounds3 bounds;
    Triangle* m_tri;
    int split_axis;
    int offset_to_second_child;
};
