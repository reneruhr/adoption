#pragma once
#include "common.h"

constexpr f32 pi_int_max_inv = pi * 0x1p-31f;
constexpr f32 pi2_uint_max_inv= tau * 0x1p-32f;


#ifdef __APPLE__
#include <arm_neon.h>

struct vec2x4
{
	float32x4_t x;
	float32x4_t y;
};


// returns true if at least one element is not zero
inline bool is_not_zero(vec2x4 p)
{
	const float32x4_t zero = vdupq_n_f32(0.);	
	const uint32x4_t  one  = vdupq_n_u32(~0);	
	uint32x4_t greater = vcgtq_f32(p.x, zero);
	uint32x4_t less    = vcltq_f32(p.x, zero);
	uint32x4_t not_zero = vorrq_u32(greater, less);

	greater = vcgtq_f32(p.y, zero);
	less    = vcltq_f32(p.y, zero);

	not_zero = vorrq_u32(not_zero, vorrq_u32(greater, less));
	
	return !vaddvq_u32( veorq_u32(not_zero, one));
}

inline vec2x4 reflect_x4(vec2x4 v)
{
	return {vnegq_f32(v.x), vnegq_f32(v.y)};
}

#else

#include <immintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>

struct vec2x4
{
	__m128 x;
	__m128 y;
};

struct vec2x8
{
	__m256 x;
	__m256 y;
};

struct vec2x16
{
	__m512 x;
	__m512 y;
};

struct vec3x8
{
	__m256 x;
	__m256 y;
	__m256 z;
};

inline bool is_not_zero(vec2x4 p)
{
	return !_mm_test_all_zeros(_mm_castps_si128(p.x), _mm_set1_epi32(~0)) || !_mm_test_all_zeros(_mm_castps_si128(p.y), _mm_set1_epi32(~0));
}

inline vec2x4 reflect_x4(vec2x4 p)
{
	return { _mm_sub_ps(_mm_setzero_ps(), p.x), _mm_sub_ps(_mm_setzero_ps(), p.y) };
}


// can be replaced using _mm512_kor/and if need this
inline bool is_not_zero(vec2x16 p)
{
	return !_mm512_test_epi32_mask(_mm512_castps_si512(p.x), _mm512_setzero_si512()) || !_mm512_test_epi32_mask(_mm512_castps_si512(p.y), _mm512_setzero_si512());
}

inline vec2x16 reflect_x4(vec2x16 p)
{
	return { _mm512_sub_ps(_mm512_setzero_ps(), p.x), _mm512_sub_ps(_mm512_setzero_ps(), p.y) };
}
const __m256 one_x8  = _mm256_set1_ps(1.f);
#endif
