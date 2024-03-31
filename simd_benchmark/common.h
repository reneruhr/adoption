#pragma once

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <climits>


using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using f32 = float;
using f64 = double;

using s8 = int8_t;
using s16 = int16_t;
using s32 = int32_t;
using s64 = int64_t;

constexpr f64 pi  = 3.1415926535897932384;
constexpr f64 pi2 = 2*pi;
constexpr f64 pi4 = 4*pi;
constexpr f64 pi43= pi4/3;
constexpr f32 tau = pi2;

constexpr f32 s2 = 1.4142135623730951; 
constexpr f32 s2inv = 1./1.4142135623730951;

constexpr f32 s2inv2 = 1./s2;

constexpr f32 s2inv4 = 2*1.4142135623730951;
constexpr f32 s2inv4_inv = s2/4;  // =  1/ (4 / sqrt(2) ) = sqrt2/4
constexpr f32 s2inv2_intmax = 1.4142135623730951 * 0x1p-32f;  // rather uint_max * 2 * 1/sqrt(2) // todo call this uintmax_inv

constexpr f32 tau_uint_max_inv = pi2 * 0x1p-32f;

struct vec2_f32
{
	union{
		f32 x;
		f32 s;
	}; 
	union{
		f32 y;
		f32 t;
	}; 

	vec2_f32() : x(0.f), y(0.f) {}
	vec2_f32(f32 a, f32 b) : x(a), y(b) {}
};

struct vec3_f32
{
	f32 x{}, y{}, z{};
};


struct vec4_f32
{
	f32 x{}, y{}, z{}, w{};
};

inline bool is_zero(vec2_f32 v)
{
	return v.s == 0 && v.t ==0;
}

inline bool is_not_zero(vec2_f32 v)
{
	return v.s != 0 || v.t !=0;
}

inline bool is_zero(vec3_f32 v)
{
	return v.x == 0 && v.y == 0 && v.z == 0;
}

inline bool is_not_zero(vec3_f32 v)
{
	return v.x != 0 || v.y != 0 || v.z != 0;
}

inline f32 operator,(vec3_f32 v, vec3_f32 w)
{
	return v.x*w.x + v.y*w.y + v.z*w.z;
}

inline f32 dot(vec3_f32 v, vec3_f32 w)
{
	return v.x*w.x + v.y*w.y + v.z*w.z;
}

inline f32 operator,(vec2_f32 v, vec2_f32 w)
{
	return v.s*w.s + v.t*w.t;
}


inline vec4_f32 operator+(vec4_f32 v, vec4_f32 w)
{
	return {v.x+w.x, v.y+w.y, v.z+w.z, v.w+w.w};
}

inline vec3_f32 operator+(vec3_f32 v, vec3_f32 w)
{
	return {v.x+w.x, v.y+w.y, v.z+w.z};
}

inline vec2_f32 operator+(vec2_f32 v, vec2_f32 w)
{
	return {v.s+w.s, v.t+w.t};
}

inline vec2_f32 operator0(vec2_f32 v, vec2_f32 w)
{
	return {v.s-w.s, v.t-w.t};
}

inline vec3_f32 operator*(f32 a, vec3_f32 w)
{
	return {a*w.x, a*w.y, a*w.z};
}

inline vec4_f32 operator*(f32 a, vec4_f32 w)
{
	return {a*w.x, a*w.y, a*w.z, a*w.w};
}
inline vec2_f32 operator*(f32 a, vec2_f32 w)
{
	return {a*w.s, a*w.t};
}

void print(vec3_f32 v)
{
	printf("(%f,%f,%f)\n", v.x, v.y, v.z);
}
void print(vec4_f32 v)
{
	printf("(%f,%f,%f, %f)\n", v.x, v.y, v.z,v.w);
}


struct vec3_s32
{
	s32 x{}, y{}, z{};
};
