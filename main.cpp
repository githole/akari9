#define  _CRT_SECURE_NO_WARNINGS

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <iostream>
#include <chrono>

#include "utility.h"
#include "vec3.h"
#include "sim.h"

struct BaseParameter
{
    int width{ 1024 };
    int height{ 1024 };
    int super_sample_count{ 1 };

    uint32_t fluid_X{ 128 };
    uint32_t fluid_Y{ 128 };
    uint32_t fluid_Z{ 128 };

    float movie_time{ 10.0f };
    int max_frame_count{ 240 };

    uint32_t cache_size{ 60 };
} g_param;

namespace renderer
{

struct Camera
{
    utility::Float3 org;
    utility::Float3 dir;
    utility::Float3 up;

    // [meter]
    float distance_to_film;
    float film_width;
    float film_height; 
};

// Ç±ÇÃu, vÇÕ[-1, 1]
utility::Ray
generateCameraRay(const Camera& camera, float u, float v)
{
    const auto side = normalize(cross(camera.up, camera.dir));
    const auto up = normalize(cross(side, camera.dir));

    const auto p_on_film =
        camera.org + camera.distance_to_film * camera.dir +
        side * u * camera.film_width / 2.0f +
        up * v * camera.film_height / 2.0f;

    const auto dir = normalize(p_on_film - camera.org);

    return { camera.org, dir };
}



struct Scene
{
    utility::AABB scene_aabb;
    utility::Image3D vol0;
    utility::Image3D vol1;
//    std::array<float, 2> vol_w{ 1.0f, 0.0f };
    bool enable_multi_color{ false };
    utility::Float3 color0{ 1.0f, 1.0f, 1.0f };
    utility::Float3 color1{ 1.0f, 1.0f, 1.0f };

    std::array<float, 2> vol_w0{ 1.0f, 0.3f };
    std::array<float, 2> vol_w1{ 0.2f, 1.0f };

    utility::Color bg{ 0.2f, 0.2f, 0.2f };

    float sample_scattering_coeff(int color_index, const utility::Float3& ws_pos) const
    {
        const utility::Float3 volume_org = scene_aabb.bounds[0];
        const utility::Float3 volume_size = scene_aabb.bounds[1] - scene_aabb.bounds[0];
        utility::Float3 v;
        for (int i = 0; i < 3; ++i)
        {
            v[i] = (ws_pos[i] - volume_org[i]) / volume_size[i];
        }

        auto vol_w = color_index == 0 ? vol_w0 : vol_w1;
        const float v0 = vol_w[0] ? vol_w[0] * vol0.load_trilinear(v[0], v[1], v[2]) : 0.0f;
        const float v1 = vol_w[1] ? vol_w[1] * vol1.load_trilinear(v[0], v[1], v[2]) : 0.0f;
        return 30.0f * (v0 + v1);
    }
#if 0
    float majorant(float w) const
    {
        return 30.f * (w * (w == 0 ? 0 : vol0.majorant()) + (1 - w) * (w == 1 ? 0 : vol1.majorant()));
    }
#endif

    utility::Float3 to_sun_dir;
    utility::Float3 sun_intensity;
} g_scene;

utility::Float3
worldspace_to_volumespace(const Scene& scene, const utility::Float3& ws_pos)
{
    const utility::Float3 volume_org = scene.scene_aabb.bounds[0];
    const utility::Float3 volume_size = scene.scene_aabb.bounds[1] - g_scene.scene_aabb.bounds[0];
    utility::Float3 v;
    for (int i = 0; i < 3; ++i)
    {
        v[i] = (ws_pos[i] - volume_org[i]) / volume_size[i];
    }
    return v;
}

struct Cache
{
    utility::Image3D transmittance_to_light0;
    utility::Image3D transmittance_to_light1;
};

void setup_data(int frame_number, const fluid::Volume& vol0, const fluid::Volume& vol1)
{
    g_scene.vol0 = vol0;
    g_scene.vol1 = vol1;

    g_scene.scene_aabb = utility::AABB(
        utility::Float3(-0.25f, -0.25f, -0.25f),
        utility::Float3(0.25f,  0.25f, 0.25f));

//    g_scene.to_sun_dir = normalize(utility::Float3(0, 1, 0));
    g_scene.to_sun_dir = normalize(utility::Float3(3, 6, 2));
    g_scene.sun_intensity = 50.0f * utility::Float3(1, 1, 1);
}

void
upscale(Scene& scene, int frame_number)
{
    printf("upscale\n");
    const int X = scene.vol0.X_;
    const int Y = scene.vol0.Y_;
    const int Z = scene.vol0.Z_;

    utility::Image3D vol0, vol1;

    vol0.init(2 * X, 2 * Y, 2 * Z);
    vol1.init(2 * X, 2 * Y, 2 * Z);

    utility::random::PCG_64_32 rng;

#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 0; iz < 2 * Z; ++iz)
    {
        for (int iy = 0; iy < 2 * Y; ++iy)
        {
            for (int ix = 0; ix < 2 * X; ++ix)
            {
#if 0
                float v0 = 0;
                float v1 = 0;

                for (int sx = -1; sx <= 1; ++sx)
                {
                    for (int sy = -1; sy <= 1; ++sy)
                    {
                        for (int sz = -1; sz <= 1; ++sz)
                        {
                            const float u = (ix + sx + 0.5f) / (2 * X);
                            const float v = (iy + sy + 0.5f) / (2 * Y);
                            const float w = (iz + sz + 0.5f) / (2 * Z);

                            float scale = 1;

                            if (ix <= 64)
                            {
                                scale = ix / 64.0f;
                                scale *= scale;
                            }
                            if (ix >= (2 * X) - 64)
                            {
                                scale = ((2 * X) - ix) / 64.0f;
                                scale *= scale;
                            }

                            v0 += scale * scene.vol0.load_trilinear(u, v, w);
                            v1 += scale * scene.vol1.load_trilinear(u, v, w);
                        }
                    }
                }
                vol0.store_clamped(ix, iy, iz, v0 / 27);
                vol1.store_clamped(ix, iy, iz, v1 / 27);
#endif

#if 1
                const float u = (ix + 0.5f) / (2 * X);
                const float v = (iy + 0.5f) / (2 * Y);
                const float w = (iz + 0.5f) / (2 * Z);

                float scale = 1;

                if (120 <= frame_number && frame_number <= 209)
                {

                    if (ix <= 64)
                    {
                        scale = ix / 64.0f;
                        scale *= scale;
                    }
                    if (ix >= (2 * X) - 64)
                    {
                        scale = ((2 * X) - ix) / 64.0f;
                        scale *= scale;
                    }
                }

                vol0.store_clamped(ix, iy, iz, scale * scene.vol0.load_trilinear(u, v, w));
                vol1.store_clamped(ix, iy, iz, scale * scene.vol1.load_trilinear(u, v, w));
#endif
            }
        }
    }

    scene.vol0 = vol0;
    scene.vol1 = vol1;
    printf("end upscale\n");
}

void
setup_cache(const Scene& scene, Cache& cache)
{
    printf("setup_cache\n");
    auto& transmittance_to_light0 = cache.transmittance_to_light0;
    auto& transmittance_to_light1 = cache.transmittance_to_light1;
    transmittance_to_light0.init(g_param.cache_size, g_param.cache_size, g_param.cache_size);
    transmittance_to_light1.init(g_param.cache_size, g_param.cache_size, g_param.cache_size);

    const utility::Float3 scene_org = scene.scene_aabb.bounds[0];
    const utility::Float3 scene_size = scene.scene_aabb.bounds[1] - g_scene.scene_aabb.bounds[0];

    const int kIntegrationStep = g_param.cache_size;

#pragma omp parallel for schedule(dynamic, 1)
    for (int iz = 0; iz < transmittance_to_light0.Z_; ++iz)
    {
        for (int iy = 0; iy < transmittance_to_light0.Y_; ++iy)
        {
            for (int ix = 0; ix < transmittance_to_light0.X_; ++ix)
            {
                const float u = (ix + 0.5f) / transmittance_to_light0.X_;
                const float v = (iy + 0.5f) / transmittance_to_light0.Y_;
                const float w = (iz + 0.5f) / transmittance_to_light0.Z_;

                const float ws_x = u * scene_size[0] + scene_org[0];
                const float ws_y = v * scene_size[1] + scene_org[1];
                const float ws_z = w * scene_size[2] + scene_org[2];

                const auto start_pos = utility::Float3(ws_x, ws_y, ws_z);

                auto hp = scene.scene_aabb.intersect(utility::Ray{ start_pos, scene.to_sun_dir });
                if (hp)
                {
                    const auto end_pos = hp->position;
                    const auto thickness = length(end_pos - start_pos);
                    const float delta_distance = thickness / kIntegrationStep;

                    // ë‰å`åˆéÆÇ≈êœï™ÇµÇƒtransmittanceéZèo
                    auto current_pos = start_pos;
                    float prev_scattering_coeff0 = 0;
                    float prev_scattering_coeff1 = 0;

                    double sum0 = 0, sum1 = 0;
                    for (int k = 0; k < kIntegrationStep; ++k)
                    {
                        const float scattering_coeff0 = g_scene.sample_scattering_coeff(0, current_pos);
                        const float scattering_coeff1 = g_scene.sample_scattering_coeff(1, current_pos);

                        if (k >= 1)
                        {
                            sum0 += (prev_scattering_coeff0 + scattering_coeff0) / 2.0f * delta_distance;
                            sum1 += (prev_scattering_coeff1 + scattering_coeff1) / 2.0f * delta_distance;
                        }

                        prev_scattering_coeff0 = scattering_coeff0;
                        prev_scattering_coeff1 = scattering_coeff1;
                        current_pos += scene.to_sun_dir * delta_distance;
                    }

                    transmittance_to_light0.store_clamped(ix, iy, iz, exp(-sum0));
                    transmittance_to_light1.store_clamped(ix, iy, iz, exp(-sum1));
                }
            }
        }
    }

#if 1
    // èÙÇ›çûÇ›Å`


    const int X = transmittance_to_light0.X_;
    const int Y = transmittance_to_light0.Y_;
    const int Z = transmittance_to_light0.Z_;

    constexpr float diff = 0.1f;

    const float a = diff * X * Y * Z;

    // ÉÑÉRÉrñ@

    auto apply = [&](auto& transmittance_to_light) {
        constexpr int kIteration = 1;

        utility::Image3D v, tmp;
        v.init(X, Y, Z);
        tmp = v;

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
                            (transmittance_to_light.load_clamped(ix, iy, iz) + a * (
                                v.load_clamped(ix + 1, iy, iz) + v.load_clamped(ix - 1, iy, iz) +
                                v.load_clamped(ix, iy + 1, iz) + v.load_clamped(ix, iy - 1, iz) +
                                v.load_clamped(ix, iy, iz + 1) + v.load_clamped(ix, iy, iz - 1)
                                ) / (1 + 6 * a)));
                    }
                }
            }
            fluid::set_bnd(-1, tmp);
            std::swap(v, tmp);
        }
        transmittance_to_light = v;
    };

    apply(transmittance_to_light0);
    apply(transmittance_to_light1);

#endif

    printf("end setup_cache\n");
}

#if 0
template <typename Rng>
float delta_tracking(
    const Scene& scene, float majorant, float bound,
    const utility::Float3& org, const utility::Float3& dir,
    Rng& rng)
{
    float t = 0;
    do {
        t -= log(1.0 - rng.next01()) / majorant;
    } while (scene.sample_scattering_coeff(org + t * dir) / majorant < rng.next01() && t < bound);
    return t;
}

template<typename Rng>
float estimate_transmittance_ratio(
    const Scene& scene, float majorant, float bound,
    const utility::Float3& org, const utility::Float3& dir,
    Rng& rng)
{
    float t = 0;
    float T = 1;

    for (;;) {
        t -= log(1.0 - rng.next01()) / majorant;
        if (t > bound)
            break;
        T *= 1 - scene.sample_scattering_coeff(org + t * dir) / majorant;
    }

    return T;
}
#endif

utility::Color
radiance_pt(const Scene& scene, const Cache& cache, const utility::Ray& ray, utility::random::PCG_64_32& rng)
{
#if 0
    auto hp = scene.scene_aabb.intersect(ray);
    if (!hp)
    {
        return{};
    }
    auto current_pos = hp->position + 0.0001f * ray.dir;
    auto current_dir = ray.dir;

    float contribution = 0;

    float total_phase = 1;
    constexpr int kMaxBounce = 1;
    const float majorant = scene.majorant(1.0f);

    constexpr float kAlbedo = 1.0f;
    const float kLightPower = scene.sun_intensity[0];
    constexpr float phase = 1.0f / (4.0f * PI);

    for (int bounce = 0; bounce < kMaxBounce; ++bounce)
    {
        auto hp2 = scene.scene_aabb.intersect(utility::Ray{ current_pos, current_dir });
        if (!hp2)
        {
            return contribution * utility::Color(1, 1, 1);
        }
        auto bound = hp2->distance;
        float next_t = delta_tracking(scene, majorant, bound, current_pos, current_dir, rng);
        if (next_t >= bound)
        {
            return contribution * utility::Color(1, 1, 1);
        }

        // éUóê or ãzé˚
        if (rng.next01() > kAlbedo)
        {
            return contribution * utility::Color(1, 1, 1);
        }

        // lighting
        const auto event_pos = current_pos + next_t * current_dir;
        hp2 = scene.scene_aabb.intersect(utility::Ray{ event_pos, scene.to_sun_dir });
        if (!hp2)
        {
            return contribution * utility::Color(1, 1, 1);
        }
         bound = hp2->distance;
        const float tr = estimate_transmittance_ratio(scene, majorant, bound, event_pos, scene.to_sun_dir, rng);
        contribution += total_phase * phase * tr * kLightPower;

        utility::Float3 next_dir;
        float pdf_omega;

        utility::Float3 dir_tangent, dir_binormal;
        utility::createOrthoNormalBasis(current_dir, &dir_tangent, &dir_binormal);
        
        next_dir = utility::sample_uniform_sphere_surface(rng.next01(), rng.next01());
        pdf_omega = phase;
        // phase_function->sample(dir, dir_tangent, dir_binormal, next_dir, sampler, pdf_omega);

        total_phase *= phase / pdf_omega;
        current_pos = event_pos;
        current_dir = next_dir;
    }

    return contribution * utility::Color(1, 1, 1);
#endif
    return {};
}

utility::Color
radiance(const Scene& scene, int color_index, const Cache& cache, const utility::Ray& ray)
{
    auto hp = scene.scene_aabb.intersect(ray);
    if (!hp)
    {
        return scene.bg;
    }

    const auto start_pos = hp->position;

    auto hp2 = scene.scene_aabb.intersect(utility::Ray{ start_pos + 0.0001f * ray.dir, ray.dir });
    if (!hp2)
    {
        return scene.bg;
    }
    const auto end_pos = hp2->position;
    const auto thickness = length(end_pos - start_pos);

    constexpr int kStep = 64;
    const float delta_distance = thickness / kStep;
    utility::Float3 current_pos = start_pos;
    utility::Color contribution(0, 0, 0);

    double transmittance_sum = 0;
    float scattering_coeff_prev;
    float integrand_prev = 0;

    double sum = 0;
    constexpr float phase = 1.0f / (4 * PI);

    for (int step = 0; step < kStep; ++step)
    {
        const float volume_density = g_scene.sample_scattering_coeff(color_index, current_pos);
        const float scattering_coeff = volume_density;

        if (step >= 1)
        {
            transmittance_sum += (scattering_coeff_prev + scattering_coeff) / 2.0f * delta_distance;
        }

        const float transmittance = exp(-transmittance_sum);
        const auto volume_uv = worldspace_to_volumespace(scene, current_pos);
        const float transmittance_to_light =
            (color_index == 0 ? cache.transmittance_to_light0 : cache.transmittance_to_light1).load_trilinear(volume_uv[0], volume_uv[1], volume_uv[2]);
        const float L = phase * scene.sun_intensity[0] * transmittance_to_light;
        const float integrand = transmittance * scattering_coeff * L;

        if (step >= 1)
        {
            sum += (integrand_prev + integrand) / 2.0f * delta_distance;
        }

        integrand_prev = integrand;
        scattering_coeff_prev = scattering_coeff;
        current_pos += ray.dir * delta_distance;
    }
    contribution = utility::Float3(1, 1, 1) * (float)sum + (float)exp(-transmittance_sum) * scene.bg;

    return contribution;
}


void render(const char* output_dir, float current_time , int frame_number)
{
    utility::Timer _("render");

    printf("frame_number: %d\n", frame_number);

    const auto aspect = (float)g_param.width / g_param.height;

    // setup camera

    Camera camera;
    //camera.org = utility::Float3(10 - 2 * currentTime / 10.0f, 7 - currentTime / 10.0f, 10 - 2 * currentTime / 10.0f);
    //camera.dir = normalize(utility::Float3(0, 2 - currentTime / 10.0f, 0) - camera.org);
    //camera.up = utility::Float3(0, 1, 0);
    //camera.distanceToFilm = 1.0f;
    //camera.filmHeight = 0.5f;
    //camera.filmWidth = camera.filmHeight * aspect;


    // scene2

    /*
    camera.org = utility::Float3(2, 1, 1.5);
    camera.dir = normalize(-camera.org);
    camera.up = utility::Float3(0, 1, 0);

    const auto aspect = (float)g_param.width / g_param.height;
    camera.distance_to_film = 1.0f;
    camera.film_height = 0.2f;
    camera.film_width = camera.film_height * aspect;
    */


    // éŒÇﬂè„
    /*
    camera.org = utility::Float3(5, 1, 3);
    camera.dir = normalize(-camera.org);
    camera.up = utility::Float3(0, 1, 0);

    const auto aspect = (float)g_param.width / g_param.height;
    camera.distance_to_film = 1.0f;
    camera.film_height = 0.2f;
    camera.film_width = camera.film_height * aspect;
    */

    if (0 <= frame_number && frame_number <= 119)
    {
        // scene1

        if (frame_number <= 60)
        {
            camera.org = utility::Float3(0.0, -0.1, 2.5 + 0.5f * utility::easeInExpo((60.0f - frame_number)/60.0f));
        }
        else
        {
            camera.org = utility::Float3(0.0, -0.1, 2.5);
        }

        camera.dir = normalize(utility::Float3(0, -0.15, 0) - camera.org);
        camera.up = utility::Float3(0, 1, 0);

        camera.distance_to_film = 1.0f;
        camera.film_height = 0.2f;
        camera.film_width = camera.film_height * aspect;
    }
    else if (120 <= frame_number && frame_number  <= 209)
    {
        /*
        camera.org = utility::Float3(0, 0, 8);
        camera.dir = utility::Float3(0, 0, -1);
        camera.up = utility::Float3(0, 1, 0);

        const auto aspect = (float)g_param.width / g_param.height;
        camera.distance_to_film = 2.0f;
        camera.film_height = 0.2f;
        camera.film_width = camera.film_height * aspect;
        */
        const int f = frame_number - 120;

        // âÒì]ÉAÉjÉÅ
        float a;
        float y;
        if (30 <= f && f < 90)
        {
            float T = ((f - 30) / 60.0f);
            a = 0.0f + 2.8f * utility::easeInOutCubic(T);
            y = 1.0f * utility::easeInOutCubic(T);
        }
        else
        {
            a = 0.0f;
            y = 0.0f;
        }
        camera.org = utility::Float3(2 * cos(a), y, 2 * sin(a));

        camera.dir = normalize(-camera.org);
        camera.up = utility::Float3(0, 1, 0);

        const auto aspect = (float)g_param.width / g_param.height;
        camera.distance_to_film = 1.0f;
        camera.film_height = 0.2f;
        camera.film_width = camera.film_height * aspect;
    }
    else if (210 <= frame_number && frame_number <= 299)
    {
        camera.org = utility::Float3(0, 0, 4.8);
        camera.dir = utility::Float3(0, 0, -1);
        camera.up = utility::Float3(0, 1, 0);

        const auto aspect = (float)g_param.width / g_param.height;
        camera.distance_to_film = 2.0f;
        camera.film_height = 0.2f;
        camera.film_width = camera.film_height * aspect;
    }
    
    /*
    camera.org = utility::Float3(0, 0, 5);
    camera.dir = utility::Float3(0, 0, -1);
    camera.up = utility::Float3(0, 1, 0);

    const auto aspect = (float)g_param.width / g_param.height;
    camera.distance_to_film = 2.0f;
    camera.film_height = 0.2f;
    camera.film_width = camera.film_height * aspect;
    */

    // render
    utility::Image image(g_param.width, g_param.height);
    const int ss_count = g_param.super_sample_count;

    Cache cache;
    setup_cache(g_scene, cache);

    upscale(g_scene, frame_number);

#pragma omp parallel for schedule(dynamic, 1)
    for (int iy = 0; iy < g_param.height; ++iy)
    {
//        printf("{%d}", iy);
        for (int ix = 0; ix < g_param.width; ++ix)
        {
            utility::random::PCG_64_32 rng;

            rng.set_seed(ix + iy * g_param.width);

            for (int sx = 0; sx < ss_count; ++sx)
            {
                for (int sy = 0; sy < ss_count; ++sy)
                {
                    const float u = -(ix + (sx + 0.5f) / ss_count) / image.width_ * 2 + 1;
                    const float v = (iy + (sy + 0.5f) / ss_count) / image.height_ * 2 - 1;
                    const auto ray = generateCameraRay(camera, u, v);

                    if (g_scene.enable_multi_color)
                    {
                        const auto c0 = radiance(g_scene, 0, cache, ray)[0];
                        const auto c1 = radiance(g_scene, 1, cache, ray)[0];

                        const auto col = c0 * g_scene.color0 + c1 * g_scene.color1;
                        image.accum(ix, iy, col * (1.0f / (ss_count * ss_count)));
                    }
                    else
                    {

                        const auto c = radiance(g_scene, 0, cache, ray);
                        //const auto c = radiance_pt(g_scene, cache, ray, rng);
                        image.accum(ix, iy, c * (1.0f / (ss_count * ss_count)));
                    }
                }
            }
        }
    }

    // utility::writeHDRImage("./dump.hdr", image);

    // hdr -> sdr

    {
        utility::Timer _("hdr2sdr");

        constexpr int comp = 3;
        std::vector<uint8_t> ldr_image(image.width_ * image.height_ * comp);

        float screen_scale = 0.5;

        /*

        if (current_time < 0.1f)
        {
            screen_scale = 0;
        }

        if (current_time < 0.6f)
        {
            screen_scale = utility::easeOutQuad((current_time - 0.1f) / 0.5f);
        }

        if (current_time > 9.5f)
        {
            screen_scale = 1.0f - utility::easeOutExpo((current_time - 9.5f) / 0.5f);
        }
        */

        if (210 <= frame_number)
        {
            screen_scale = 0.6f;
        }

        if (280 <= frame_number && frame_number <= 299)
        {
            screen_scale *= (1.0f - ((frame_number - 280) / 19.0f));
        }

        if (0 <= frame_number && frame_number <= 19)
        {
            screen_scale *= (frame_number) / 19.0f;
        }

        if (frame_number <= 119)
        {
#pragma omp parallel for schedule(dynamic, 1)
            for (int iy = 0; iy < image.height_; ++iy)
            {
                for (int ix = 0; ix < image.width_; ++ix)
                {
                    const auto col = image.load(ix, iy);

                    const uint8_t r{ (uint8_t)utility::clampValue(screen_scale * pow(col[0], 1 / 2.2f) * 255, 0.0f, 255.0f) };
                    const uint8_t g{ (uint8_t)utility::clampValue(screen_scale * pow(col[1], 1 / 2.2f) * 255, 0.0f, 255.0f) };
                    const uint8_t b{ (uint8_t)utility::clampValue(screen_scale * pow(col[2], 1 / 2.2f) * 255, 0.0f, 255.0f) };

                    const auto idx{ (ix + iy * image.width_) * comp };

                    ldr_image[idx + 0] = r;
                    ldr_image[idx + 1] = g;
                    ldr_image[idx + 2] = b;
                }
            }
        }
        else
        {
#pragma omp parallel for schedule(dynamic, 1)
            for (int iy = 0; iy < image.height_; ++iy)
            {
                for (int ix = 0; ix < image.width_; ++ix)
                {
                    const auto col = image.load(ix, iy);

                    const uint8_t r{ (uint8_t)utility::clampValue(screen_scale * pow(col[0], 1 / 2.3f) * 255, 0.0f, 255.0f) };
                    const uint8_t g{ (uint8_t)utility::clampValue(screen_scale * pow(col[1], 1 / 2.3f) * 255, 0.0f, 255.0f) };
                    const uint8_t b{ (uint8_t)utility::clampValue(screen_scale * pow(col[2], 1 / 2.3f) * 255, 0.0f, 255.0f) };

                    const auto idx{ (ix + iy * image.width_) * comp };

                    ldr_image[idx + 0] = r;
                    ldr_image[idx + 1] = g;
                    ldr_image[idx + 2] = b;
                }
            }
        }

        char buf[256];
        sprintf(buf, "%s%03d.jpg", output_dir, frame_number);
//        utility::writePNGImage(buf, image.width_, image.height_, comp, ldr_image.data(), image.width_ * sizeof(uint8_t) * comp);
        utility::writeJPEGImage(buf, image.width_, image.height_, comp, ldr_image.data(), 100);
        printf("output: %s\n", buf);
    }
}

}

int main(int argc, char** argv)
{
    if (argc <= 2)
    {
        return -1;
    }

    const int begin_frame = atoi(argv[1]);
    const int end_frame = atoi(argv[2]);
    const char* output_dir = argc >= 4 ? argv[3] : "./";
    const int mode = argc >= 5 ? atoi(argv[4])  : 0;
    const bool rough = argc >= 6 ? atoi(argv[5]) : 0;
    printf("begin:%d, end:%d\n", begin_frame, end_frame);

    if (rough)
    {
        g_param.width /= 2;
        g_param.height /= 2;
    }

    constexpr int maxFrameNumber = 240;
    constexpr float dt = 1.0f / 30.0f; // 30fps

    auto start_time = std::chrono::high_resolution_clock::now();

    if (mode == 0)
    {
        printf("render mode\n");

        fluid::Simulation simulation;

        utility::random::PCG_64_32 rng;

        simulation.init(g_param.fluid_X, g_param.fluid_Y, g_param.fluid_Z, rng);

        for (int frame_number = begin_frame; frame_number <= end_frame; ++frame_number)
        {
            const float current_time = (float)frame_number / (g_param.max_frame_count - 1) * g_param.movie_time;

            if (0 <= frame_number && frame_number <= 119)
            {
                renderer::g_scene.enable_multi_color = false;
                renderer::g_scene.bg = utility::Color(0.1f, 0.1f, 0.1f);
                renderer::g_scene.color0 = utility::Color(1.0f, 1.0f, 1.0f);
                renderer::g_scene.color1 = utility::Color(1.0f, 1.0f, 1.0f);
                simulation.use_vc = false;

                if (frame_number < 60)
                {
                    utility::Timer _("sim onestep");
                    fluid::sim_onestep(simulation, frame_number, dt * 0.01f);
                }

                if (60 <= frame_number)
                {
                    utility::Timer _("sim onestep");
                    fluid::sim_onestep(simulation, frame_number, dt);
                }
                renderer::setup_data(frame_number, simulation.density, simulation.density_sub);
                renderer::render(output_dir, current_time, frame_number);
            }
            else if (120 <= frame_number && frame_number <= 209)
            {
                if (frame_number == 120)
                {
                    simulation.init(g_param.fluid_X, g_param.fluid_Y, g_param.fluid_Z, rng);
                }

                renderer::g_scene.enable_multi_color = true;
                renderer::g_scene.bg = utility::Color(0.02f, 0.02f, 0.02f);
                renderer::g_scene.color0 = utility::Color(0.95f, 0.7f, 0.7f);
                renderer::g_scene.color1 = utility::Color(0.05f, 0.1f, 0.95f);
                renderer::g_scene.vol_w0 = { 1.0f, 0.3f };
                renderer::g_scene.vol_w1 = { 0.2f, 1.0f };
                simulation.use_vc = false;

                if (frame_number <= 150)
                {
                    fluid::sim_onestep(simulation, frame_number, dt);
                }
                renderer::setup_data(frame_number, simulation.density, simulation.density_sub);
                renderer::render(output_dir, current_time, frame_number);
            }
            else if (210 <= frame_number && frame_number <= 299)
            {
                if (frame_number == 210)
                {
                    simulation.init(g_param.fluid_X, g_param.fluid_Y, g_param.fluid_Z, rng);
                }

                renderer::g_scene.enable_multi_color = true;
                renderer::g_scene.bg = utility::Color(0.02f, 0.02f, 0.02f);
                renderer::g_scene.color0 = utility::Color(0.7f, 0.2f, 0.25f);
                renderer::g_scene.color1 = utility::Color(0.2f, 0.7f, 0.3f);
                renderer::g_scene.vol_w0 = { 0.9f, 0.1f };
                renderer::g_scene.vol_w1 = { 0.1f, 0.9f };
                simulation.use_vc = true;

                if (frame_number == 210)
                {
                    fluid::sim_onestep(simulation, frame_number, dt);
                }

                fluid::sim_onestep(simulation, frame_number, dt);
                renderer::setup_data(frame_number, simulation.density, simulation.density_sub);
                renderer::render(output_dir, current_time, frame_number);
            }

        }
    }
    else if (mode == 1)
    {
        printf("sim mode\n");
        fluid::sim(g_param.fluid_X, g_param.fluid_Y, g_param.fluid_Z, output_dir, maxFrameNumber, dt);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "time: " << (duration/1000.0f) << " [sec]" << std::endl;

    printf("END\n");
    return 0;
}