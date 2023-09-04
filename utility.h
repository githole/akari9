#pragma once

#include <algorithm>
#include <optional>
#include <vector>
#include <chrono>

#include "vec3.h"

#define PI 3.141592654f

namespace utility
{

constexpr float kLarge = 1e+32f;
using Color = Float3;

template<typename T>
T clampValue(const T& x, const T& a, const T& b)
{
    if (x < a)
        return a;
    if (x > b)
        return b;
    return x;
}

struct Ray
{
    Float3 org;
    Float3 dir;
};

struct Image
{
    std::vector<float> body_;
    int width_{};
    int height_{};

    Image() = default;

    Image(int w, int h) : width_(w), height_(h)
    {
        body_.resize(w * h * 3);
    }

    bool isValid() const
    {
        return !body_.empty();
    }

    size_t clampedIndex(int x, int y) const
    {
        if (x <= 0)
            x = 0;
        if (x >= width_)
            x = width_ - 1;
        if (y <= 0)
            y = 0;
        if (y >= height_)
            y = height_ - 1;
        return ((size_t)x + (size_t)y * width_) * 3;
    }

    Color load(int x, int y) const
    {
        const auto index{ clampedIndex(x, y) };
        return {
            body_[index + 0],
            body_[index + 1],
            body_[index + 2],
        };
    }

    void store(int x, int y, const Color& color)
    {
        const auto index{ clampedIndex(x, y) };
        body_[index + 0] = color[0];
        body_[index + 1] = color[1];
        body_[index + 2] = color[2];
    }

    void accum(int x, int y, const Color& color)
    {
        const auto index{ clampedIndex(x, y) };
        body_[index + 0] += color[0];
        body_[index + 1] += color[1];
        body_[index + 2] += color[2];
    }
};


bool writeHDRImage(const char* filename, const Image& image);

int writePNGImage(char const* filename, int w, int h, int comp, const void* data, int stride_in_bytes);

int writeJPEGImage(char const* filename, int w, int h, int comp, const void* data, int quality);


Image loadHDRImage(const char* filename);

inline float easeOutQuad(float x)
{
    return 1 - (1 - x) * (1 - x);
}

inline float easeInExpo(float x)
{
    return x == 0 ? 0 : pow(2, 10 * x - 10);
}

inline float easeOutQuint(float x)
{
    return 1 - pow(1 - x, 5);
}

inline float easeOutExpo(float x)
{
    return x == 1 ? 1 : 1 - pow(2, -10 * x);
}

inline float easeInOutCubic(float x)
{
    return x < 0.5 ? 4 * x * x * x : 1 - pow(-2 * x + 2, 3) / 2;
}

template<typename T>
struct Image3DT
{
    std::vector<T> body_;
    uint32_t X_, Y_, Z_;

    void load_from_file(const char* filename)
    {
        FILE* fp{ fopen(filename, "rb") };
        if (!fp)
        {
            return;
        }

        fread(&X_, sizeof(uint32_t), 1, fp);
        fread(&Y_, sizeof(uint32_t), 1, fp);
        fread(&Z_, sizeof(uint32_t), 1, fp);
        body_.resize(X_ * Y_ * Z_);
        fread(body_.data(), sizeof(T), body_.size(), fp);
        fclose(fp);
    }

    void init(uint32_t X, uint32_t Y, uint32_t Z)
    {
        body_.resize(X * Y * Z);
        X_ = X;
        Y_ = Y;
        Z_ = Z;
    }

    void setZero()
    {
        std::fill(body_.begin(), body_.end(), T(0.0f));
    }
#if 0
    T majorant() const
    {
        T maxv(-1);

        for (auto v : body_)
        {
            maxv = std::max(maxv, v);
        }

        return maxv;
    }
#endif
    size_t clampedIndex(int x, int y, int z) const
    {
        if (x <= 0)
            x = 0;
        if (x >= X_)
            x = X_ - 1;
        if (y <= 0)
            y = 0;
        if (y >= Y_)
            y = Y_ - 1;
        if (z <= 0)
            z = 0;
        if (z >= Z_)
            z = Z_ - 1;
        return (size_t)x + (size_t)y * X_ + (size_t)z * X_ * Y_;
    }

    void store_clamped(int x, int y, int z, T v)
    {
        const auto index{ clampedIndex(x, y, z) };
        body_[index] = v;
    }

    T load_clamped(int x, int y, int z) const
    {
        const auto index{ clampedIndex(x, y, z) };
        return body_[index];
    }

    T load(int x, int y, int z) const
    {
        if (x < 0 || y < 0 || z < 0 || x >= X_ || y >= Y_ || z >= Z_)
        {
            return {};
        }
        return load_clamped(x, y, z);
    }

    void local_minmax(int x, int y, int z, T& minv, T& maxv) const
    {
        minv = kLarge;
        maxv = -kLarge;

        const float vx1 = load(x + 1, y, z);
        const float vx2 = load(x - 1, y, z);
        const float vy1 = load(x, y + 1, z);
        const float vy2 = load(x, y - 1, z);
        const float vz1 = load(x, y, z + 1);
        const float vz2 = load(x, y, z - 1);
        const float Z = load(x, y, z);

        minv = std::min({ Z, vx1, vx2, vy1, vy2, vz1, vz2 });
        maxv = std::max({ Z, vx1, vx2, vy1, vy2, vz1, vz2 });
    }

    T load_trilinear(float u, float v, float w) const
    {
        if (u < 0 || v < 0 || w < 0 ||
            u >= 1 || v >= 1 || w >= 1)
        {
            return {};
        }

        const float fu = u * X_;
        const float fv = v * Y_;
        const float fw = w * Z_;

        const int iu = (int)fu;
        const float wu = fu - iu;

        const int iv = (int)fv;
        const float wv = fv - iv;

        const int iw = (int)fw;
        const float ww = fw - iw;

        T sum = {};
        for (int i = 0; i < 8; ++i)
        {
            const int u0 = i & 1;
            const int v0 = (i & 2) >> 1;
            const int w0 = (i & 4) >> 2;

            sum +=
                (u0 ? wu : (1 - wu)) *
                (v0 ? wv : (1 - wv)) *
                (w0 ? ww : (1 - ww)) *
                load(iu + u0, iv + v0, iw + w0);
        }
        return sum;
    }
};

using Image3D = Image3DT<float>;
using Color3D = Image3DT<utility::Color>;

struct Hitpoint
{
    float distance{ kLarge };
    Float3 position; // world space
    Float3 normal; // world space
};

struct AABB
{
    Float3 bounds[2];
    Float3 center;

    AABB(const Float3& vmin, const Float3& vmax)
    {
        bounds[0] = vmin;
        bounds[1] = vmax;
        center = (bounds[0] + bounds[1]) * 0.5f;
    }

    AABB()
    {
        for (int i = 0; i < 3; ++i)
        {
            bounds[0][i] = kLarge;
            bounds[1][i] = -kLarge;
        }
    }

    void merge(const AABB& aabb)
    {
        for (int i = 0; i < 3; ++i)
        {
            bounds[0][i] = std::min(bounds[0][i], aabb.bounds[0][i]);
            bounds[1][i] = std::max(bounds[1][i], aabb.bounds[1][i]);
        }
    }

    std::optional<Hitpoint> intersect(const Ray& ray) const
    {
        auto tmphp = intersect_(ray);

        if (tmphp && tmphp->distance >= 0)
        {
            return tmphp;
        }

        return {};
    }

    std::optional<Hitpoint> intersect_(const Ray& ray) const
    {
        float tmin, tmax, tymin, tymax, tzmin, tzmax;


        Float3 invdir(1.0f / ray.dir[0], 1.0f / ray.dir[1], 1.0f / ray.dir[2]);
        int sign[3];
        sign[0] = (invdir[0] < 0);
        sign[1] = (invdir[1] < 0);
        sign[2] = (invdir[2] < 0);

        tmin = (bounds[sign[0]][0] - ray.org[0]) * invdir[0];
        tmax = (bounds[1 - sign[0]][0] - ray.org[0]) * invdir[0];
        tymin = (bounds[sign[1]][1] - ray.org[1]) * invdir[1];
        tymax = (bounds[1 - sign[1]][1] - ray.org[1]) * invdir[1];

        if ((tmin > tymax) || (tymin > tmax))
            return {};

        int axis = 0;

        if (tymin > tmin)
        {
            axis = 1;
            tmin = tymin;
        }
        if (tymax < tmax)
            tmax = tymax;

        tzmin = (bounds[sign[2]][2] - ray.org[2]) * invdir[2];
        tzmax = (bounds[1 - sign[2]][2] - ray.org[2]) * invdir[2];

        if ((tmin > tzmax) || (tzmin > tmax))
            return {};

        if (tzmin > tmin)
        {
            axis = 2;
            tmin = tzmin;
        }
        if (tzmax < tmax)
            tmax = tzmax;

        Hitpoint hitpoint;
        hitpoint.distance = tmin > 0 ? tmin : tmax;
        hitpoint.position = ray.org + hitpoint.distance * ray.dir;

        Float3 normal(0.0f, 0.0f, 0.0f);
        normal[axis] = 1.0f;
        if (center[axis] > hitpoint.position[axis])
        {
            normal *= -1.0f;
        }
        hitpoint.normal = normal;

        return hitpoint;
    }
};



namespace random {

    inline uint32_t rotr(uint32_t x, int shift) {
        return (x >> shift) | (x << (32 - shift));
    }

    inline uint64_t rotr(uint64_t x, int shift) {
        return (x >> shift) | (x << (64 - shift));
    }

    struct splitmix64 {
        uint64_t x;

        splitmix64(uint64_t a = 0) : x(a) {}

        uint64_t next() {
            uint64_t z = (x += 0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
            z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
            return z ^ (z >> 31);
        }
    };

    // PCG(64/32)
    // http://www.pcg-random.org/download.html
    // initial_inc from official library
    struct PCG_64_32 {
        uint64_t state;
        uint64_t inc;

        PCG_64_32(uint64_t initial_state = 0x853c49e6748fea9bULL,
            uint64_t initial_inc = 0xda3e39cb94b95bdbULL)
            : state(initial_state), inc(initial_inc) {}

        void set_seed(uint64_t seed) {
            splitmix64 s(seed);
            state = s.next();
        }

        using return_type = uint32_t;
        return_type next() {
            auto oldstate = state;
            state = oldstate * 6364136223846793005ULL + (inc | 1);
            uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
            uint32_t rot = oldstate >> 59u;

            return rotr(xorshifted, rot);
        }

        // [0, 1)
        float next01() {
            return (float)(((double)next()) /
                ((double)std::numeric_limits<uint32_t>::max() + 1));
        }


        float next(float minV, float maxV)
        {
            return next01() * (maxV - minV) + minV;
        }
    };

} // namespace random

inline Float3 sample_uniform_sphere_surface(float u, float v) {
    const float tz = u * 2 - 1;
    const float phi = v * PI * 2;
    const float k = sqrt(1.0 - tz * tz);
    const float tx = k * cos(phi);
    const float ty = k * sin(phi);
    return Float3(tx, ty, tz);
}

template <typename Vec3>
inline void createOrthoNormalBasis(const Vec3& normal, Vec3* tangent, Vec3* binormal) {
    if (abs(normal[0]) > abs(normal[1]))
    {
        (*tangent) = cross(Vec3(0, 1, 0), normal);
        (*tangent) = normalize(*tangent);
    }
    else
    {
        (*tangent) = cross(Vec3(1, 0, 0), normal);
        (*tangent) = normalize(*tangent);
    }
    (*binormal) = cross(normal, *tangent);
    (*binormal) = normalize(*binormal);
}

inline float remap(float x, float a, float b)
{
    return x * (b - a) + a;
}

struct Timer
{
    const char* label;
    std::chrono::steady_clock::time_point start_time;

    Timer(const char* l) : label(l) 
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

    ~Timer()
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "* " << label << ":" << (duration / 1000.0f) << " [sec]" << std::endl;

    }
};

}