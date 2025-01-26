#pragma once
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>
#include "utils.h"
#include "vec_math.h"

// helper functions
// hemisphere uniform sampling



__forceinline__ __device__ float3 fresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(fminf(fmaxf(1.0 - cosTheta, 0.000001), 1.0), 5.0);
} 

__forceinline__ __device__ float _RadicalInverse_VdC(uint32_t bits) {
    // http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
    // efficient VanDerCorpus calculation.

    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

__forceinline__ __device__ float2 Hammersley(uint32_t i, uint32_t N) {
    return make_float2(float(i) / float(N), _RadicalInverse_VdC(i));
}

// `Xi` indicates a 2D uniform sampling results
__forceinline__ __device__ float3 ImportanceSampleGGX(const float2 Xi, const float3 normal,
                                                      const float roughness) {
    const float a = roughness * roughness;

    const float phi = 2.0f * M_PIf * Xi.x;
    const float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    const float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    // from spherical coordinates to cartesian coordinates - halfway vector
    float3 H = make_float3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);

    // if (cosTheta < 0)
    //     printf("H.z: %f\n", cosTheta);

    // from tangent-space H vector to world-space sample vector
    float3 up = abs(normal.z) < 0.999 ? make_float3(0.0, 0.0, 1.0) : make_float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);

    float3 sampleVec = tangent * H.x + bitangent * H.y + normal * H.z;
    return normalize(sampleVec);
}

__forceinline__ __device__ float DistributionGGX(float3 N, float3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NoH = max(dot(N, H), 0.0);
    float NoH2 = NoH * NoH;

    float nom = a2;
    float denom = (NoH2 * (a2 - 1.0) + 1.0);
    denom = M_PIf * denom * denom;

    return nom / denom;
}

// ----------------------------------------------------------------------------
__forceinline__ __device__ float GeometrySchlickGGX(const float n_d_v, const float roughness) {
    // note that we use a different k for IBL
    float a = roughness;
    float k = (a * a) / 2.0;

    float nom = n_d_v;
    float denom = n_d_v * (1.0 - k) + k;

    return nom / denom;
}

__forceinline__ __device__ float GeometrySmith(const float3 N, const float3 V, const float3 L,
                                               const float roughness) {
    float n_d_v = max(dot(N, V), 0.0);
    float n_d_l = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(n_d_v, roughness);
    float ggx1 = GeometrySchlickGGX(n_d_l, roughness);

    return ggx1 * ggx2;
}

__forceinline__ __device__ float eval_sh(const float3 N, const float3 V, const float3 L,
    const float roughness) {
    float n_d_v = max(dot(N, V), 0.0);
    float n_d_l = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(n_d_v, roughness);
    float ggx1 = GeometrySchlickGGX(n_d_l, roughness);

    return ggx1 * ggx2;
}

__forceinline__ __device__ float3 get_position(
	const int x,
	const int y,
	const float cx,
	const float cy,
	const float fx,
	const float fy,
	const float depth
) {
	float3 dir = make_float3(
		(float(x) - cx) / fx,
		(float(y) - cy) / fy,
		1.0f
	);
	return dir * depth;
}

__forceinline__ __device__ int2 get_coord(
	const float cx,
	const float cy,
	const float fx,
	const float fy,
	const float3 pos
) {
	float3 dir = make_float3(
		pos.x / (pos.z+0.0000001f),
		pos.y / (pos.z+0.0000001f),
		1.0f
	);

	int2 xy = {roundf(dir.x * fx + cx), roundf(dir.y * fy + cy)};
	return xy;
}

__forceinline__ __device__ float smoothstep(float edge0, float edge1, float x) {
    x = (x - edge0) / (edge1 - edge0);
    x = fmaxf(0.0f, fminf(1.0f, x));
    return x * x * (3.0f - 2.0f * x);
}