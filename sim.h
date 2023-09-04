#pragma once

#include <array>

#include "utility.h"

#define USE_MACCORMACK 1
#define USE_BFECC 0

#define USE_REFLECTION 1

namespace fluid
{
using Volume = utility::Image3D;


struct Simulation
{
    Volume density, density0;
    Volume density_sub, density_sub0;
    Volume u, u0;
    Volume v, v0;
    Volume w, w0;
    Volume tmp, tmp2, p, div;

    Volume curlx, curly, curlz, curl;
    Volume u1, v1, w1;

    utility::Image tex;

    std::array<utility::Float3, 8> vel_dir;

    int X, Y, Z;

    template<typename Rng>
    void init(int x, int y, int z, Rng& rng)
    {
        X = x;
        Y = y;
        Z = z;
        density.init(X, Y, Z);
        density.setZero();
        curlx = curly = curlz = curl =
            tmp = tmp2 = p = div = u = u0 = v = v0 = w = w0 = density0 = density;
        density_sub = density_sub0 = density;
        u1 = v1 = w1 = u;

        for (auto& dir : vel_dir)
        {
            const float u = rng.next01();
            const float v = rng.next01();
            dir = utility::sample_uniform_sphere_surface(u, v);
            // std::cout << dir << std::endl;
        }
    }

    float diff = 1e-6f;
    float vc_epsilon = 10.0f;
    bool use_vc = false;

    std::array<utility::Float3, 16> dir_arr;
};

void set_bnd(int b, Volume& v);
void sim_onestep(Simulation& sim, int frame_number, float dt);

void sim(int X, int Y, int Z, const char* output_dir, int max_frame_number, float dt);
} // namespace::fluid
