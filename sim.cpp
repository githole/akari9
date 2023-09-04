#define  _CRT_SECURE_NO_WARNINGS

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <array>
#include <iostream>

#include "utility.h"
#include "vec3.h"
#include "sim.h"

static bool g_bnd_override = true;

namespace fluid
{


void set_bnd(int b, Volume& v)
{
    if (b == -1)
    {
#pragma omp parallel for schedule(dynamic, 1)
        for (int iy = 1; iy < v.Y_ - 1; ++iy)
        {
            for (int iz = 1; iz < v.Z_ - 1; ++iz)
            {
                v.store_clamped(0, iy, iz, 0);
                v.store_clamped(v.X_ - 1, iy, iz, 0);
            }
        }

#pragma omp parallel for schedule(dynamic, 1)
        for (int ix = 1; ix < v.X_ - 1; ++ix)
        {
            for (int iz = 1; iz < v.Z_ - 1; ++iz)
            {
                v.store_clamped(ix, 0, iz, 0);
                v.store_clamped(ix, v.Y_ - 1, iz, 0);
            }
        }

#pragma omp parallel for schedule(dynamic, 1)
        for (int ix = 1; ix < v.X_ - 1; ++ix)
        {
            for (int iy = 1; iy < v.Y_ - 1; ++iy)
            {
                v.store_clamped(ix, iy, 0, 0);
                v.store_clamped(ix, iy, v.Z_ - 1, 0);
            }
        }
    }
    else
    {
        if (g_bnd_override)
        {
            b = 0;
        }

#pragma omp parallel for schedule(dynamic, 1)
        for (int iy = 1; iy < v.Y_ - 1; ++iy)
        {
            for (int iz = 1; iz < v.Z_ - 1; ++iz)
            {
                v.store_clamped(0, iy, iz,
                    b == 1 ? -v.load_clamped(1, iy, iz) : v.load_clamped(1, iy, iz));
                v.store_clamped(v.X_ - 1, iy, iz,
                    b == 1 ? -v.load_clamped(v.X_ - 2, iy, iz) : v.load_clamped(v.X_ - 2, iy, iz));
            }
        }

#pragma omp parallel for schedule(dynamic, 1)
        for (int ix = 1; ix < v.X_ - 1; ++ix)
        {
            for (int iz = 1; iz < v.Z_ - 1; ++iz)
            {
                v.store_clamped(ix, 0, iz,
                    b == 2 ? -v.load_clamped(ix, 1, iz) : v.load_clamped(ix, 1, iz));
                v.store_clamped(ix, v.Y_ - 1, iz,
                    b == 2 ? -v.load_clamped(ix, v.Y_ - 2, iz) : v.load_clamped(ix, v.Y_ - 2, iz));
            }
        }

#pragma omp parallel for schedule(dynamic, 1)
        for (int ix = 1; ix < v.X_ - 1; ++ix)
        {
            for (int iy = 1; iy < v.Y_ - 1; ++iy)
            {
                v.store_clamped(ix, iy, 0,
                    b == 3 ? -v.load_clamped(ix, iy, 1) : v.load_clamped(ix, iy, 1));
                v.store_clamped(ix, iy, v.Z_ - 1,
                    b == 3 ? -v.load_clamped(ix, iy, v.Z_ - 2) : v.load_clamped(ix, iy, v.Z_ - 2));
            }
        }
    }

    // 隅っこ
    for (int i = 0; i < 8; ++i)
    {
        const int x = (i & 2) ? 0 : v.X_ - 1;
        const int y = (i & 4) ? 0 : v.Y_ - 1;
        const int z = (i & 8) ? 0 : v.Z_ - 1;

        double sum = 0;
        int cnt = 0;
        for (int ox : {-1, 1})
        {
            for (int oy : {-1, 1})
            {
                for (int oz : {-1, 1})
                {
                    const int fx = x + ox;
                    const int fy = y + oy;
                    const int fz = z + oz;

                    if (fx < 0 || v.X_ <= fx ||
                        fy < 0 || v.Y_ <= fy ||
                        fz < 0 || v.Z_ <= fz)
                    {
                        continue;
                    }

                    cnt++;
                    sum += v.load_clamped(fx, fy, fz);
                }
            }
        }
        if (cnt > 0)
        {
            v.store_clamped(x, y, z, sum / cnt);
        }
    }
}
} // namespace::fluid


namespace
{
    using namespace fluid;
void vorticity_confinement(
    Volume& u, Volume& v, Volume& w,
    Volume& curlx, Volume& curly, Volume& curlz, Volume& curl,
    float dt, float epsilon
)
{
    const int X = u.X_;
    const int Y = u.Y_;
    const int Z = u.Z_;

    // 1. Vorticityの計算
#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 1; iz < Z - 1; ++iz)
    {
        for (int iy = 1; iy < Y - 1; ++iy)
        {
            for (int ix = 1; ix < X - 1; ++ix)
            {
                curlz.store_clamped(ix, iy, iz,
                    (u.load_clamped(ix, iy + 1, iz) - u.load_clamped(ix, iy - 1, iz) - v.load_clamped(ix + 1, iy, iz) + v.load_clamped(ix - 1, iy, iz)) * 0.5f);
                curly.store_clamped(ix, iy, iz,
                    (u.load_clamped(ix, iy, iz + 1) - u.load_clamped(ix, iy, iz - 1) - w.load_clamped(ix + 1, iy, iz) + w.load_clamped(ix - 1, iy, iz)) * 0.5f);
                curlx.store_clamped(ix, iy, iz,
                    (v.load_clamped(ix, iy, iz + 1) - v.load_clamped(ix, iy, iz - 1) - w.load_clamped(ix, iy + 1, iz) + w.load_clamped(ix, iy - 1, iz)) * 0.5f);

                curl.store_clamped(ix, iy, iz,
                    sqrt(
                        curlx.load_clamped(ix, iy, iz) * curlx.load_clamped(ix, iy, iz) +
                        curly.load_clamped(ix, iy, iz) * curly.load_clamped(ix, iy, iz) +
                        curlz.load_clamped(ix, iy, iz) * curlz.load_clamped(ix, iy, iz)
                    ));
            }
        }
    }
    set_bnd(0, curl);
    /*
    // 不要
    set_bnd(0, curlx);
    set_bnd(0, curly);
    set_bnd(0, curlz);
    */

    // 2. Vorticityの勾配とVorticity Confinement Forceの計算、3. 速度の更新

#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 1; iz < Z - 1; ++iz)
    {
        for (int iy = 1; iy < Y - 1; ++iy)
        {
            for (int ix = 1; ix < X - 1; ++ix)
            {
                double dx = (curl.load_clamped(ix + 1, iy, iz) - curl.load_clamped(ix - 1, iy, iz)) * 0.5;
                double dy = (curl.load_clamped(ix, iy + 1, iz) - curl.load_clamped(ix, iy - 1, iz)) * 0.5;
                double dz = (curl.load_clamped(ix, iy, iz + 1) - curl.load_clamped(ix, iy, iz - 1)) * 0.5;

                // Calculate the length of the gradient
                double length = sqrt(dx * dx + dy * dy + dz * dz) + 1.0e-10;  // avoid division by zero

                // Normalize the gradient
                dx /= length;
                dy /= length;
                dz /= length;

                // Vorticity Confinement Force calculation
                double fvcx = epsilon * (dy * curlz.load_clamped(ix, iy, iz) - dz * curly.load_clamped(ix, iy, iz));
                double fvcy = epsilon * (dz * curlx.load_clamped(ix, iy, iz) - dx * curlz.load_clamped(ix, iy, iz));
                double fvcz = epsilon * (dx * curly.load_clamped(ix, iy, iz) - dy * curlx.load_clamped(ix, iy, iz));

                // Update the velocities with the confinement force
                u.store_clamped(ix, iy, iz, u.load_clamped(ix, iy, iz) + dt * fvcx);
                v.store_clamped(ix, iy, iz, v.load_clamped(ix, iy, iz) + dt * fvcy);
                w.store_clamped(ix, iy, iz, w.load_clamped(ix, iy, iz) + dt * fvcz);
            }
        }
    }

    set_bnd(1, u);
    set_bnd(2, v);
    set_bnd(3, w);
}

void diffuse(int b, Volume& v, Volume& tmp, const Volume& v0, float diff, float dt)
{
    const int X = v.X_;
    const int Y = v.Y_;
    const int Z = v.Z_;

    float a = dt * diff *  X * Y * Z;

    // ヤコビ法
    constexpr int kIteration = 10;
    v.setZero();
    for (int k = 0; k < kIteration; k++)
    {
#pragma omp parallel for schedule(dynamic, 1)
        for (int iz = 1; iz < Z - 1; ++iz)
        {
            for (int iy = 1; iy < Y - 1; ++iy)
            {
                for (int ix = 1; ix < X - 1; ++ix)
                {
                    tmp.store_clamped(ix, iy, iz,
                        (v0.load_clamped(ix, iy, iz) + a * (
                            v.load_clamped(ix + 1, iy, iz) + v.load_clamped(ix - 1, iy, iz) +
                            v.load_clamped(ix, iy + 1, iz) + v.load_clamped(ix, iy - 1, iz) +
                            v.load_clamped(ix, iy, iz + 1) + v.load_clamped(ix, iy, iz - 1)
                            ) / (1 + 6 * a)));
                }
            }
        }
        set_bnd(b, tmp);
        std::swap(v, tmp);
    }
}


void advect(int b, Volume& new_d, const Volume& prev_d, const Volume& u, const Volume& v, const Volume& w, Volume&tmp, Volume& tmp2, float dt)
{
    const int X = new_d.X_;
    const int Y = new_d.Y_;
    const int Z = new_d.Z_;

#if !USE_MACCORMACK // MacCormackかどうか

#if !USE_BFECC

#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 1; iz < Z - 1; ++iz)
    {
        for (int iy = 1; iy < Y - 1; ++iy)
        {
            for (int ix = 1; ix < X - 1; ++ix)
            {
                float x0 = ix - dt * X * u.load_clamped(ix, iy, iz);
                float y0 = iy - dt * Y * v.load_clamped(ix, iy, iz);
                float z0 = iz - dt * Z * w.load_clamped(ix, iy, iz);

                float u0 = x0 / X;
                float v0 = y0 / Y;
                float w0 = z0 / Z;

                const float v = prev_d.load_trilinear(u0, v0, w0);
                new_d.store_clamped(ix, iy, iz, v);
            }
        }
    }

    set_bnd(b, new_d);

#else
#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 1; iz < Z - 1; ++iz)
    {
        for (int iy = 1; iy < Y - 1; ++iy)
        {
            for (int ix = 1; ix < X - 1; ++ix)
            {
                float x0 = ix - dt * X * u.load_clamped(ix, iy, iz);
                float y0 = iy - dt * Y * v.load_clamped(ix, iy, iz);
                float z0 = iz - dt * Z * w.load_clamped(ix, iy, iz);

                float u0 = x0 / X;
                float v0 = y0 / Y;
                float w0 = z0 / Z;

                const float v = prev_d.load_trilinear(u0, v0, w0);
                tmp.store_clamped(ix, iy, iz, v);
            }
        }
    }
    set_bnd(b, tmp);

#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 1; iz < Z - 1; ++iz)
    {
        for (int iy = 1; iy < Y - 1; ++iy)
        {
            for (int ix = 1; ix < X - 1; ++ix)
            {
                float x0 = ix + dt * X * u.load_clamped(ix, iy, iz);
                float y0 = iy + dt * Y * v.load_clamped(ix, iy, iz);
                float z0 = iz + dt * Z * w.load_clamped(ix, iy, iz);

                float u0 = x0 / X;
                float v0 = y0 / Y;
                float w0 = z0 / Z;

                const float v = tmp.load_trilinear(u0, v0, w0);
                tmp2.store_clamped(ix, iy, iz, v);
            }
        }
    }
    set_bnd(b, tmp2);


#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 1; iz < Z - 1; ++iz)
    {
        for (int iy = 1; iy < Y - 1; ++iy)
        {
            for (int ix = 1; ix < X - 1; ++ix)
            {
                float local_minv = -utility::kLarge, local_maxv = utility::kLarge;
                prev_d.local_minmax(ix, iy, iz, local_minv, local_maxv);

                tmp.store_clamped(ix, iy, iz,
                    utility::clampValue(
                    1.5f * prev_d.load_clamped(ix, iy, iz) - 0.5f * tmp2.load_clamped(ix, iy, iz),
                    local_minv, local_maxv));
            }
        }
    }
    set_bnd(b, tmp);

#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 1; iz < Z - 1; ++iz)
    {
        for (int iy = 1; iy < Y - 1; ++iy)
        {
            for (int ix = 1; ix < X - 1; ++ix)
            {
                float x0 = ix - dt * X * u.load_clamped(ix, iy, iz);
                float y0 = iy - dt * Y * v.load_clamped(ix, iy, iz);
                float z0 = iz - dt * Z * w.load_clamped(ix, iy, iz);

                float u0 = x0 / X;
                float v0 = y0 / Y;
                float w0 = z0 / Z;

                const float v = tmp.load_trilinear(u0, v0, w0);
                new_d.store_clamped(ix, iy, iz, v);
            }
        }
    }
    set_bnd(b, new_d);

#endif

#else
#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 1; iz < Z - 1; ++iz)
    {
        for (int iy = 1; iy < Y - 1; ++iy)
        {
            for (int ix = 1; ix < X - 1; ++ix)
            {
                float x0 = ix - dt * X * u.load_clamped(ix, iy, iz);
                float y0 = iy - dt * Y * v.load_clamped(ix, iy, iz);
                float z0 = iz - dt * Z * w.load_clamped(ix, iy, iz);

                float u0 = x0 / X;
                float v0 = y0 / Y;
                float w0 = z0 / Z;

                const float v = prev_d.load_trilinear(u0, v0, w0);
                tmp.store_clamped(ix, iy, iz, v);
            }
        }
    }
    set_bnd(b, tmp);

    const auto maccormack_relaxation = 1.0f;
#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 1; iz < Z - 1; ++iz)
    {
        for (int iy = 1; iy < Y - 1; ++iy)
        {
            for (int ix = 1; ix < X - 1; ++ix)
            {
                float x1 = ix + dt * X * u.load_clamped(ix, iy, iz);
                float y1 = iy + dt * Y * v.load_clamped(ix, iy, iz);
                float z1 = iz + dt * Z * w.load_clamped(ix, iy, iz);

                float u1 = x1 / X;
                float v1 = y1 / Y;
                float w1 = z1 / Z;

                float local_minv = -utility::kLarge, local_maxv = utility::kLarge;
                tmp.local_minmax(ix, iy, iz, local_minv, local_maxv);

                const float corrected_val = tmp.load_trilinear(u1, v1, w1);

                new_d.store_clamped(ix, iy, iz, 
                    utility::clampValue(
                        tmp.load(ix, iy, iz) + maccormack_relaxation * 0.5f * (prev_d.load(ix, iy, iz) - corrected_val),
                        local_minv, local_maxv));
                
                /*
                new_d.store_clamped(ix, iy, iz,
                    tmp.load(ix, iy, iz) + 0.5f * (prev_d.load(ix, iy, iz) - corrected_val));

                new_d.store_clamped(ix, iy, iz,
                    maccormack_relaxation * new_d.load_clamped(ix, iy, iz) +
                    (1.0f - maccormack_relaxation) * prev_d.load_clamped(ix, iy, iz));
                */

            }
        }
    }
    set_bnd(b, new_d);


#endif
}

void project(
    Volume& u, Volume& v, Volume& w,
    Volume& tmp, Volume& p, Volume& div
)
{
    const int X = u.X_;
    const int Y = u.Y_;
    const int Z = u.Z_;

    // div 決定
#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 1; iz < Z - 1; ++iz)
    {
        for (int iy = 1; iy < Y - 1; ++iy)
        {
            for (int ix = 1; ix < X - 1; ++ix)
            {
                div.store_clamped(ix, iy, iz,
                    -1.0f / 3.0f * 
                    ((u.load_clamped(ix + 1, iy, iz) - u.load_clamped(ix - 1, iy, iz)) / X +
                     (v.load_clamped(ix, iy + 1, iz) - v.load_clamped(ix, iy - 1, iz)) / Y +
                     (w.load_clamped(ix, iy, iz + 1) - w.load_clamped(ix, iy, iz - 1)) / Z)
                    );
                p.store_clamped(ix, iy, iz, 0);
            }
        }
    }

    set_bnd(0, div);
    set_bnd(0, p);

    // ヤコビ法
    constexpr int kIteration = 40;
    tmp = p;
    for (int k = 0; k < kIteration; k++)
    {
#pragma omp parallel for schedule(dynamic, 1)
        for (int iz = 1; iz < Z - 1; ++iz)
        {
            for (int iy = 1; iy < Y - 1; ++iy)
            {
                for (int ix = 1; ix < X - 1; ++ix)
                {
                    tmp.store_clamped(ix, iy, iz,
                        (div.load_clamped(ix, iy, iz)
                        + p.load_clamped(ix + 1, iy, iz)
                        + p.load_clamped(ix - 1, iy, iz)
                        + p.load_clamped(ix, iy + 1, iz)
                        + p.load_clamped(ix, iy - 1, iz)
                        + p.load_clamped(ix, iy, iz + 1)
                        + p.load_clamped(ix, iy, iz - 1)
                        ) / 6.0f
                    );
                }
            }
        }
        set_bnd(0, tmp);
        std::swap(p, tmp);
    }

    // 速度更新
#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 1; iz < Z - 1; ++iz)
    {
        for (int iy = 1; iy < Y - 1; ++iy)
        {
            for (int ix = 1; ix < X - 1; ++ix)
            {
                u.store_clamped(ix, iy, iz,
                    u.load_clamped(ix, iy, iz) - 0.5f * X * (p.load_clamped(ix + 1, iy, iz) - p.load_clamped(ix - 1, iy, iz)));

                v.store_clamped(ix, iy, iz,
                    v.load_clamped(ix, iy, iz) - 0.5f * Y * (p.load_clamped(ix, iy + 1, iz) - p.load_clamped(ix, iy - 1, iz)));

                w.store_clamped(ix, iy, iz,
                    w.load_clamped(ix, iy, iz) - 0.5f * Z * (p.load_clamped(ix, iy, iz + 1) - p.load_clamped(ix, iy, iz - 1)));
            }
        }
    }

    set_bnd(1, u);
    set_bnd(2, v);
    set_bnd(3, w);
}

void dens_step(
    Volume& density, Volume& density0,
    const Volume& u, const Volume& v, const Volume& w,
    Volume& tmp, Volume& tmp2,
    float dt, float diff
)
{
#if 0
    diffuse(0, density, tmp, density0, diff, dt);
    std::swap(density, density0);
#endif
    advect(0, density, density0, u, v, w, tmp, tmp2, dt);
}

void reflection(int b, const Volume& u_1_2, Volume& u_tilda_1_2)
{
    const int N = u_tilda_1_2.body_.size();
    for (int i = 0; i < N; ++i)
    {
        u_tilda_1_2.body_[i] = 2 * u_1_2.body_[i] - u_tilda_1_2.body_[i];
    }
    set_bnd(b, u_tilda_1_2);
}

void vel_step(
    Simulation& sim,
    Volume& u, Volume& v, Volume& w,
    Volume& u0, Volume& v0, Volume& w0,
    Volume&tmp, Volume& p, Volume& div,
    float dt, float diff)
{
#if 0
    diffuse(1, u, tmp, u0, diff, dt);
    diffuse(2, v, tmp, v0, diff, dt);
    diffuse(3, w, tmp, w0, diff, dt);
    project(u, v, w, tmp, p, div);

    std::swap(u0, u);
    std::swap(v0, v);
    std::swap(w0, w);
#endif

    if (sim.use_vc)
    {
        vorticity_confinement(u0, v0, w0, sim.curlx, sim.curly, sim.curlz, sim.curl, dt, sim.vc_epsilon);
    }

#if !USE_REFLECTION // reflectionつかうかどうか
    advect(1, u, u0, u0, v0, w0, tmp, sim.tmp2, dt);
    advect(2, v, v0, u0, v0, w0, tmp, sim.tmp2, dt);
    advect(3, w, w0, u0, v0, w0, tmp, sim.tmp2, dt);
    project(u, v, w, tmp, p, div);
#else
    // reflection
    advect(1, sim.u1, u0, u0, v0, w0, tmp, sim.tmp2, 0.5f * dt);
    advect(2, sim.v1, v0, u0, v0, w0, tmp, sim.tmp2, 0.5f * dt);
    advect(3, sim.w1, w0, u0, v0, w0, tmp, sim.tmp2, 0.5f * dt);
    // u1: u_tilda^{1/2}

    u0 = sim.u1;
    v0 = sim.v1;
    w0 = sim.w1;
    project(u0, v0, w0, tmp, p, div);
    // u0: u^{1/2}

    reflection(1, u0, sim.u1);
    reflection(2, v0, sim.v1);
    reflection(3, w0, sim.w1);
    // u1: u_hat^{1/2}

    advect(1, u, sim.u1, u0, v0, w0, tmp, sim.tmp2, 0.5f * dt);
    advect(2, v, sim.v1, u0, v0, w0, tmp, sim.tmp2, 0.5f * dt);
    advect(3, w, sim.w1, u0, v0, w0, tmp, sim.tmp2, 0.5f * dt);
    // u: u_tilda^{1}

    project(u, v, w, tmp, p, div);

#endif
}

void
add_density(
    const Simulation& sim,
    Volume& v,
    int frame_number)
{

    const int X = v.X_;
    const int Y = v.Y_;
    const int Z = v.Z_;

    const int cx = X / 2;
    const int cy = Y / 2;
    const int cz = Z / 2;

#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 0; iz < Z; ++iz)
    {
        for (int iy = 0; iy < Y; ++iy)
        {
            for (int ix = 0; ix < X; ++ix)
            {
                const utility::Float3 center(cx, cy, cz);
                const utility::Float3 pos(ix, iy, iz);

                const float dx = (pos[0] - center[0]);
                const float dz = (pos[2] - center[2]);
                const float l = sqrt(dx * dx + dz * dz);

                if (2 <= iy && iy <= 5 && l <= 50)
                {
                    v.store_clamped(ix, iy, iz, 1.0f);
                }

            }
        }
    }

    set_bnd(0, v);
}

void add_velocity(
    const Simulation& sim,
    Volume& u, Volume& v, Volume& w,
    int frame_number
)
{
    const int X = v.X_;
    const int Y = v.Y_;
    const int Z = v.Z_;

    const int cx = X / 2;
    const int cy = Y / 2;
    const int cz = Z / 2;

    utility::random::PCG_64_32 rng;
    rng.set_seed(frame_number);

    // rng.set_seed(0);
#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 0; iz < Z; ++iz)
    {
        for (int iy = 0; iy < Y; ++iy)
        {
            for (int ix = 0; ix < X; ++ix)
            {
                const int index = ix + iy * X + iz * X * Y;

                float tu = (ix + 0.5f) / X;
                float tv = (1.0f - (iy + 0.5f) / Y);

                tu = utility::remap(tu, -0.2f, 1.2f);
                tv = utility::remap(tv, -0.2f, 1.2f) - 0.3f;

                const int iix = tu * sim.tex.width_;
                const int iiy = tv * sim.tex.height_;

                const float tex_value = sim.tex.load(iix, iiy)[0];

                if ((64 - 8) <= iz && iz <= (64 + 8))
                {
                    if (tex_value > 0)
                    {
                        u.store_clamped(ix, iy, iz, rng.next(-5, 5));
                        v.store_clamped(ix, iy, iz, rng.next(-5, 5));
                        w.store_clamped(ix, iy, iz, rng.next(-5, 5));
                    }
                }
            }
        }
    }

    set_bnd(1, u);
    set_bnd(2, v);
    set_bnd(3, w);
}


void setup(Simulation& sim, utility::random::PCG_64_32& rng)
{
    const auto X = sim.X;
    const auto Y = sim.Y;
    const auto Z = sim.Z;

    sim.tex = utility::loadHDRImage("./etc/tex.hdr");
#if 0
    if (!sim.tex.isValid())
    {
        sim.tex = utility::loadHDRImage("C:/Code/VSProjects/akari9/x64/Release/tex.hdr");
    }
#endif

#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 0; iz < Z; ++iz)
    {
        for (int iy = 0; iy < Y; ++iy)
        {
            for (int ix = 0; ix < X; ++ix)
            {
                const int index = ix + iy * X + iz * X * Y;


                constexpr int cx = 64;
                constexpr int cy = 64;
                constexpr int cz = 64;
                constexpr int radius = 50;

                const utility::Float3 center(cx, cy, cz);
                const utility::Float3 pos(ix, iy, iz);

                const float l = length(center - pos);
#if 0
                //if (l <= radius)
                {
                    sim.density0.store_clamped(ix, iy, iz, 1.0f);

                    /*
                    sim.u0.store_clamped(ix, iy, iz, rng.next(-5, 5));
                    sim.v0.store_clamped(ix, iy, iz, rng.next(-5, 5));
                    sim.w0.store_clamped(ix, iy, iz, rng.next(-5, 5));
                    */
                }
#endif

                float tu = (ix + 0.5f) / X;
                float tv = (1.0f - (iy + 0.5f) / Y);

                tu = utility::remap(tu, -0.2f, 1.2f);
                tv = utility::remap(tv, -0.2f, 1.2f) - 0.3f;

                const int iix = tu * sim.tex.width_;
                const int iiy = tv * sim.tex.height_;

                const float tex_value = sim.tex.load(iix, iiy)[0];

                if ((64 - 8) <= iz && iz <= (64 + 8))
                {
                    sim.density0.store_clamped(ix, iy, iz, 5 * tex_value);
                    if (tex_value > 0)
                    {
                        sim.u0.store_clamped(ix, iy, iz, rng.next(-5, 5));
                        sim.v0.store_clamped(ix, iy, iz, rng.next(-5, 5));
                        sim.w0.store_clamped(ix, iy, iz, rng.next(-5, 5));
                    }
                }
            }
        }
    }
}

void dump(const char* filename, const Volume& v)
{
    const int X = v.X_;
    const int Y = v.Y_;
    const int Z = v.Z_;

    utility::Image image(X, Y);


    for (int ix = 0; ix < X; ++ix)
    {
        for (int iy = 0; iy < Y; ++iy)
        {
            const int iz = Z / 2;
            const float value = v.load_clamped(ix, iy, iz);

            const int index = ix + iy * X;

            if (value > 0)
                image.store(ix, iy, utility::Color(value, 0, 0));
            else
                image.store(ix, iy, utility::Color(0, 0, -value));
        }
    }

    utility::writeHDRImage(filename, image);
}


void
add_density2(
    const Simulation& sim,
    Volume& v,
    int frame_number)
{

    const int X = v.X_;
    const int Y = v.Y_;
    const int Z = v.Z_;

    const int cx = X / 2;
    const int cy = Y / 2;
    const int cz = Z / 2;

#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 0; iz < Z; ++iz)
    {
        for (int iy = 0; iy < Y; ++iy)
        {
            for (int ix = 0; ix < X; ++ix)
            {
            }
        }
    }

    set_bnd(0, v);
}

void add_velocity2(
    const Simulation& sim,
    Volume& u, Volume& v, Volume& w,
    int frame_number
)
{
    const int X = v.X_;
    const int Y = v.Y_;
    const int Z = v.Z_;

    const int cx = X / 2;
    const int cy = Y / 2;
    const int cz = Z / 2;

    utility::random::PCG_64_32 rng;
    // rng.set_seed(0);

    rng.set_seed(frame_number);
#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 0; iz < Z; ++iz)
    {
        for (int iy = 0; iy < Y; ++iy)
        {
            for (int ix = 0; ix < X; ++ix)
            {
                constexpr int cx = 64;
                constexpr int cy = 64;
                constexpr int cz = 64;
                constexpr int radius = 12;

                const utility::Float3 pos(ix, iy, iz);

                if (length(utility::Float3(0, 62, 64) - pos) <= radius)
                {
                    u.store_clamped(ix, iy, iz, 3.5f);
                    v.store_clamped(ix, iy, iz, rng.next(-0.3, 0.3));
                    w.store_clamped(ix, iy, iz, rng.next(-0.3, 0.3));
                }

                if (length(utility::Float3(128, 64, 64) - pos) <= radius)
                {
                    u.store_clamped(ix, iy, iz, -3.0f);
                    v.store_clamped(ix, iy, iz, rng.next(-0.3, 0.3));
                    w.store_clamped(ix, iy, iz, rng.next(-0.3, 0.3));
                }

                /*
                if (length(utility::Float3(64, 64, 64) - pos) <= 5)
                {
                    u.store_clamped(ix, iy, iz, rng.next(-1, 1));
                    v.store_clamped(ix, iy, iz, rng.next(-1, 1));
                    w.store_clamped(ix, iy, iz, rng.next(-1, 1));
                }
                */
            }
        }
    }

    set_bnd(1, u);
    set_bnd(2, v);
    set_bnd(3, w);
}


void setup2(Simulation& sim, utility::random::PCG_64_32& rng)
{
    const auto X = sim.X;
    const auto Y = sim.Y;
    const auto Z = sim.Z;

    sim.tex = utility::loadHDRImage("./etc/tex.hdr");

#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 0; iz < Z; ++iz)
    {
        for (int iy = 0; iy < Y; ++iy)
        {
            for (int ix = 0; ix < X; ++ix)
            {
                constexpr int cx = 64;
                constexpr int cy = 64;
                constexpr int cz = 64;
                constexpr int radius = 12;

                const utility::Float3 pos(ix, iy, iz);

                if (length(utility::Float3(0, 62, 64) - pos) <= radius)
                {
                    sim.density0.store_clamped(ix, iy, iz, 2.0f);
                }

                if (length(utility::Float3(128, 64, 64) - pos) <= radius)
                {
                    sim.density_sub0.store_clamped(ix, iy, iz, 2.0f);
                }
            }
        }
    }
}




void
add_density3(
    const Simulation& sim,
    Volume& v,
    int frame_number)
{

    const int X = v.X_;
    const int Y = v.Y_;
    const int Z = v.Z_;

    const int cx = X / 2;
    const int cy = Y / 2;
    const int cz = Z / 2;

#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 0; iz < Z; ++iz)
    {
        for (int iy = 0; iy < Y; ++iy)
        {
            for (int ix = 0; ix < X; ++ix)
            {
            }
        }
    }

    set_bnd(0, v);
}

void add_velocity3(
    const Simulation& sim,
    Volume& u, Volume& v, Volume& w,
    int frame_number,
    float dt
)
{
    const int X = v.X_;
    const int Y = v.Y_;
    const int Z = v.Z_;

    const int cx = X / 2;
    const int cy = Y / 2;
    const int cz = Z / 2;

    utility::random::PCG_64_32 rng;
    //rng.set_seed(0);

    rng.set_seed(frame_number);
#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 0; iz < Z; ++iz)
    {
        for (int iy = 0; iy < Y; ++iy)
        {
            for (int ix = 0; ix < X; ++ix)
            {
                /*
                constexpr int cx = 64;
                constexpr int cy = 64;
                constexpr int cz = 64;
                constexpr int radius = 24;

                const utility::Float3 pos(ix, iy, iz);

                auto vec = utility::Float3(0, 127, 0) - pos;

                if (length(utility::Float3(32, 127, 32) - pos) <= radius)
                {
                    u.store_clamped(ix, iy, iz, rng.next(-5, 5));
                    v.store_clamped(ix, iy, iz, rng.next(-5, 5));
                    w.store_clamped(ix, iy, iz, rng.next(-5, 5));
                }


                if (length(utility::Float3(64, 32, 16) - pos) <= radius)
                {
                    u.store_clamped(ix, iy, iz, rng.next(-5, 5));
                    v.store_clamped(ix, iy, iz, rng.next(-5, 5));
                    w.store_clamped(ix, iy, iz, rng.next(-5, 5));
                }
                */


                const utility::Float3 pos(ix, iy, iz);
                auto vec = pos - utility::Float3(64, 64, 64);

                utility::Float3 vel;

                vel[0] = 0.25f * rng.next(-1, 1);
                vel[1] = 0.25f * rng.next(-1, 1);
                vel[2] = 0.25f * rng.next(-1, 1);

                for (auto& dir : sim.dir_arr)
                {
                    const float cdot = dot(normalize(vec), dir);
                    if (cdot >= 0.9f)
                    {
                        auto s = (cdot - 0.9f) / 0.1f;
                        s = pow(s, 2);


                        vel[0] += s * (dir[0] * rng.next(3, 6) + 0.02 * rng.next(-1, 1));
                        vel[1] += s * (dir[1] * rng.next(3, 6) + 0.02 * rng.next(-1, 1));
                        vel[2] += s * (dir[2] * rng.next(3, 6) + 0.02 * rng.next(-1, 1));
                    }
                }

                if (210 <= frame_number && frame_number <= 220)
                {
                    u.store_clamped(ix, iy, iz, u.load_clamped(ix, iy, iz) + dt * vel[0]);
                    v.store_clamped(ix, iy, iz, v.load_clamped(ix, iy, iz) + dt * vel[1]);
                    w.store_clamped(ix, iy, iz, w.load_clamped(ix, iy, iz) + dt * vel[2]);
                }
            }
        }
    }

    set_bnd(1, u);
    set_bnd(2, v);
    set_bnd(3, w);
}


void setup3(Simulation& sim, utility::random::PCG_64_32& rng)
{
    const auto X = sim.X;
    const auto Y = sim.Y;
    const auto Z = sim.Z;

    /*
    for (auto& dir : sim.dir_arr)
    {
        dir = utility::sample_uniform_sphere_surface(rng.next01(), rng.next01());
    }
    */

    for (int iu = 0; iu < 4; ++iu)
    {
        for (int iv = 0; iv < 4; ++iv)
        {
            sim.dir_arr[iu + iv * 4] = utility::sample_uniform_sphere_surface(
                (iu + 0.5f + rng.next(-0.3, 0.3)) / 4.0f,
                (iv + 0.5f + rng.next(-0.3, 0.3)) / 4.0f);
        }
    }

    
#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 0; iz < Z; ++iz)
    {
        for (int iy = 0; iy < Y; ++iy)
        {
            for (int ix = 0; ix < X; ++ix)
            {
                constexpr int cx = 64;
                constexpr int cy = 64;
                constexpr int cz = 64;
                constexpr int radius = 48;

                const utility::Float3 pos(ix, iy, iz);

                /*
                if (length(utility::Float3(32, 90, 32) - pos) <= 24)
                {
                    sim.density0.store_clamped(ix, iy, iz, 2.0f);
                    sim.u0.store_clamped(ix, iy, iz, rng.next(-5, 5));
                    sim.v0.store_clamped(ix, iy, iz, rng.next(-5, 5));
                    sim.w0.store_clamped(ix, iy, iz, rng.next(-5, 5));
                }
                */

                if (length(utility::Float3(64, 64, 64) - pos) <= 32)
                {
                    sim.density_sub0.store_clamped(ix, iy, iz, 2.0f);
                    sim.u0.store_clamped(ix, iy, iz, rng.next(-5, 5));
                    sim.v0.store_clamped(ix, iy, iz, rng.next(-5, 5));
                    sim.w0.store_clamped(ix, iy, iz, rng.next(-5, 5));
                }


            }
        }
    }
}


} // namespace


namespace fluid
{

void sim_onestep(Simulation& sim, int frame_number, float dt)
{
    printf("sim\n");

    if (0 <= frame_number && frame_number <= 119)
    {
        g_bnd_override = true;

        if (frame_number == 0)
        {
            utility::random::PCG_64_32 rng;
            setup(sim, rng);
        }

        add_velocity(sim, sim.u0, sim.v0, sim.w0, frame_number);
        vel_step(sim, sim.u, sim.v, sim.w, sim.u0, sim.v0, sim.w0, sim.tmp, sim.p, sim.div, dt, sim.diff);

        add_density(sim, sim.density0, frame_number);
        dens_step(sim.density, sim.density0, sim.u, sim.v, sim.w, sim.tmp, sim.tmp2, dt, sim.diff);
    }
    else if (120 <= frame_number && frame_number <= 209)
    {
        g_bnd_override = true;

        if (frame_number == 120)
        {
            utility::random::PCG_64_32 rng;
            setup2(sim, rng);
        }

        add_velocity2(sim, sim.u0, sim.v0, sim.w0, frame_number);
        vel_step(sim, sim.u, sim.v, sim.w, sim.u0, sim.v0, sim.w0, sim.tmp, sim.p, sim.div, dt, sim.diff);

        add_density2(sim, sim.density0, frame_number);
        dens_step(sim.density, sim.density0, sim.u, sim.v, sim.w, sim.tmp, sim.tmp2, dt, sim.diff);
        dens_step(sim.density_sub, sim.density_sub0, sim.u, sim.v, sim.w, sim.tmp, sim.tmp2, dt, sim.diff);
    }
    else if (210 <= frame_number && frame_number <= 299)
    {
        g_bnd_override = false;

        if (frame_number == 210)
        {
            utility::random::PCG_64_32 rng;
            setup3(sim, rng);
        }

        add_velocity3(sim, sim.u0, sim.v0, sim.w0, frame_number, dt);
        vel_step(sim, sim.u, sim.v, sim.w, sim.u0, sim.v0, sim.w0, sim.tmp, sim.p, sim.div, dt, sim.diff);

        add_density3(sim, sim.density0, frame_number);
        dens_step(sim.density, sim.density0, sim.u, sim.v, sim.w, sim.tmp, sim.tmp2, dt, sim.diff);
        dens_step(sim.density_sub, sim.density_sub0, sim.u, sim.v, sim.w, sim.tmp, sim.tmp2, dt, sim.diff);
    }
    std::swap(sim.density, sim.density0);
    std::swap(sim.density_sub, sim.density_sub0);
    std::swap(sim.u, sim.u0);
    std::swap(sim.v, sim.v0);
    std::swap(sim.w, sim.w0);
}



void sim(int X, int Y, int Z, const char* output_dir, int max_frame_number, float dt)
{
}

} // namespace::fluid