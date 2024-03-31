#include "common.h"



inline f64 discrepancy(u32 count, u32 n_samples, f64 area){
	return std::abs((f64)count/(f64)n_samples - area);
}

inline bool in_halfspace(vec3_f32 n, const vec3_f32& v)
{
	return (n,v)>0;
}

// spherical sector
struct cone
{
	f32 h;
	vec3_f32 n;
};

inline const char* label(cone)
{
	return "Cone test";
}

inline bool is_inside(cone cone, vec3_f32 v)
{
	f32 h = 1.f-(cone.n,v);
	return h < cone.h;
}

inline f64 area(cone cone)
{
	return pi2*cone.h;
}

inline f64 ambient_vol(cone)
{
	return pi4;
}

inline void print(cone cone)
{
	f64 A = area(cone);
	printf("Cone n=(%.1f,%.1f,%.1f) h=%.1f of area=%.3f, relative area=%.3f\n",
			cone.n.x, cone.n.y, cone.n.z, cone.h, A, A/ambient_vol(cone));
}



struct annulus 
{
	f32 r0,r1;
};

inline const char* label(annulus)
{
	return "Annulus test";
}

inline bool is_inside(annulus a, vec2_f32 p)
{
	f32 r2 = (p, p);
	return  (r2 > a.r0 * a.r0) && (r2 < a.r1 * a.r1);
}

inline f64 area(annulus a)
{
	return  pi * (a.r1 * a.r1 - a.r0 * a.r0);
}

inline f64 ambient_vol(annulus)
{
	return pi;
}

inline void print(annulus a)
{
	f64 S = area(a);
	printf("Annulus %.1f r < %.1f of area=%.3f, relative area=%.3f.\n", a.r0, a.r1, S, S/ambient_vol(a));
}

// Sector inside a circle of 2*theta degrees centered around a unit vector u, restricted to the radii between r0 and r1
struct wedge
{
	f32 r0,r1;
	f32 theta, cos_theta;
	vec2_f32 u;
};

inline const char* label(wedge)
{
	return "Wedge test";
}


inline bool is_inside(wedge wedge, vec2_f32 p)
{
	
	f32 r = std::sqrt((p,p));
	
	if(r <= wedge.r0 || r >= wedge.r1) return false;

	if(wedge.cos_theta*r >=  (wedge.u,p) ) return false;

	return true;
}
inline f64 area(wedge wedge)
{
	return (f64)wedge.theta * ((f64)wedge.r1*wedge.r1-wedge.r0*wedge.r0);
}

inline f64 ambient_vol(wedge)
{
	return pi;
}

inline void print(wedge wedge)
{
	f64 S = area(wedge);
	printf("Wedge u=(%.1f,%.1f) theta=%.1f  %.1f < r < %.1f of area=%.3f, relative area=%.3f.\n", 
			wedge.u.s, wedge.u.t, wedge.theta, wedge.r0,wedge.r1, S, S/ambient_vol(wedge));
}


struct wedge_2 {
	f32 r0_2,r1_2;
	f32 theta, cos_theta_2;
	vec2_f32 u;
};

inline const char* label(wedge_2)
{
	return "Fast Wedge test";
}

wedge_2 make_wedge_2(wedge wedge) 
{
	return { wedge.r0 * wedge.r0, wedge.r1 * wedge.r1, wedge.theta, wedge.cos_theta * wedge.cos_theta, wedge.u };
}

inline bool is_inside(wedge_2 wedge, vec2_f32 p)
{
	f32 r_2 = (p, p);

	if ( (r_2 <= wedge.r0_2) || (r_2 >= wedge.r1_2) ) return false;

	f32 uxp = (wedge.u, p);
	if ((wedge.cos_theta_2 * r_2 >= uxp*uxp ) || (uxp < 0.) ) return false;

	return true;
}

inline f64 area(wedge_2 wedge)
{
	return (f64)wedge.theta * ((f64)wedge.r1_2-wedge.r0_2);
}

inline f64 ambient_vol(wedge_2)
{
	return pi;
}

//Set to detect if disk adoption is biased. 
struct adoption_set 
{
	f32 edge = 1./std::sqrt(2);
};

inline bool is_inside(adoption_set oc, vec2_f32 p)
{
	if(p.s>=oc.edge) return true;

	vec2_f32 q = {p.s+2*oc.edge, p.t};

	if(q.s>=oc.edge && (q,q)<=1) return true; 
	
	return false;
}

inline f64 area(adoption_set set)
{
	return 0.5*(pi-std::pow(2.f*set.edge ,2));
}

inline f64 ambient_vol(adoption_set)
{
	return pi;
}

inline void print(adoption_set set)
{
	f64 S = area(set);
	printf("Adoption Set of area=%.3f, relative area=%.3f.\n", 
			S, S/ambient_vol(set));
}


//3d spherical cone intersected with one of same degrees but smaller radius
struct shell
{
	f32 r0,r1;
	f32 cos_theta;
	vec3_f32 n;
};

inline bool is_inside(shell shell, vec3_f32 v)
{
	
	f32 r = std::sqrt((v,v));
	
	if(shell.r0 >= r || shell.r1 <= r) return false;
	
	if((shell.n,v) <= shell.cos_theta*r) return false;

	return true;
}


inline f64 area(shell shell)
{
	return pi2/3*(std::pow(shell.r1,3) - std::pow(shell.r0,3)) * (1.-shell.cos_theta);
}

inline f64 ambient_vol(shell)
{
	return pi43;
}

inline void print(shell shell)
{
	f64 V = area(shell);
	printf("Shell=(%.1f,%.1f,%.1f) costheta=%.1f  %.1f < r < %.1f of volume=%.3f, relative volume=%.3f.\n"
			, shell.n.x, shell.n.y, shell.n.z, shell.cos_theta, shell.r0, shell.r1, V, V/ambient_vol(shell));
}


#ifdef __APPLE__
#include <arm_neon.h>

struct wedge_simd
{
	float32x4_t r0_2;
	float32x4_t r1_2;
	float32x4_t cos_theta_2;
	float32x4_t u[2];
	
	wedge_simd(wedge wedge)
	{
		r0_2 = vdupq_n_f32(wedge.r0*wedge.r0);
		r1_2 = vdupq_n_f32(wedge.r1*wedge.r1);
		cos_theta_2 =  vdupq_n_f32(wedge.cos_theta*wedge.cos_theta);
		u[0] =  vdupq_n_f32(wedge.u.x);
		u[1] =  vdupq_n_f32(wedge.u.y);
	}

};

inline wedge_simd make_simd(wedge wedge) {
	return wedge_simd(wedge);
}

//todo: norm as fma
inline uint32x4_t is_inside_simd(wedge_simd wedge, float32x4_t p[2])
{

	float32x4_t r_2 = vaddq_f32( vmulq_f32(p[0],p[0]), vmulq_f32(p[1],p[1]) );

	float32x4_t r_costheta_2 = vmulq_f32(wedge.cos_theta_2,r_2); 

	float32x4_t uxp = vaddq_f32( vmulq_f32(wedge.u[0],p[0]), vmulq_f32(wedge.u[1],p[1]) ); 
		
	uint32x4_t res  = vandq_u32(vcltq_f32(wedge.r0_2, r_2),  vcgtq_f32(wedge.r1_2, r_2));
	uint32x4_t res2 = vandq_u32(vcgtq_f32(vmulq_f32(uxp, uxp), r_costheta_2), vcgeq_f32(uxp, vdupq_n_f32(0.f)) );
	
	uint32x4_t count = vandq_u32(vandq_u32(res, res2), vdupq_n_u32(1));

	return count;
}

#else
#include <immintrin.h>

struct wedge_simd
{
	__m128 r0_2;
	__m128 r1_2;
	__m128 cos_theta_2;
	__m128 u[2];
	
	wedge_simd(wedge wedge)
	{
		r0_2 = _mm_set1_ps(wedge.r0*wedge.r0);
		r1_2 = _mm_set1_ps(wedge.r1*wedge.r1);
		cos_theta_2 = _mm_set1_ps(wedge.cos_theta*wedge.cos_theta);
		u[0] = _mm_set1_ps(wedge.u.x);
		u[1] = _mm_set1_ps(wedge.u.y);
	}

};

inline wedge_simd make_simd(wedge wedge) {
	return wedge_simd(wedge);
}

inline __m128i is_inside_simd(wedge_simd wedge, __m128 p[2])
{

	__m128 r_2 = _mm_add_ps( _mm_mul_ps(p[0],p[0]), _mm_mul_ps(p[1],p[1]) );

	__m128 r_costheta_2 = _mm_mul_ps(wedge.cos_theta_2,r_2); 

	__m128 uxp = _mm_fmadd_ps( wedge.u[0], p[0], _mm_mul_ps(wedge.u[1],p[1]) ); 
		
	__m128 res  = _mm_and_ps(_mm_cmplt_ps(wedge.r0_2, r_2),					   _mm_cmpgt_ps(wedge.r1_2, r_2)       );
	__m128 res2 = _mm_and_ps(_mm_cmpgt_ps(_mm_mul_ps(uxp, uxp), r_costheta_2), _mm_cmpnlt_ps(uxp, _mm_setzero_ps()) );
	
	__m128i count = _mm_and_si128(_mm_castps_si128(_mm_and_ps(res, res2)), _mm_set1_epi32(1));

	return count;
}

struct wedge_simd_x8
{
	__m256 r0_2;
	__m256 r1_2;
	__m256 cos_theta_2;
	__m256 u[2];
	
	wedge_simd_x8(wedge wedge)
	{
		r0_2 = _mm256_set1_ps(wedge.r0*wedge.r0);
		r1_2 = _mm256_set1_ps(wedge.r1*wedge.r1);
		cos_theta_2 = _mm256_set1_ps(wedge.cos_theta*wedge.cos_theta);
		u[0] = _mm256_set1_ps(wedge.u.x);
		u[1] = _mm256_set1_ps(wedge.u.y);
	}

};

inline wedge_simd_x8 make_simd_x8(wedge wedge) {
	return wedge_simd_x8(wedge);
}

inline __mmask8 is_inside_simd(wedge_simd_x8 wedge, __m256 p[2])
{

	__m256 r_2 = _mm256_add_ps( _mm256_mul_ps(p[0],p[0]), _mm256_mul_ps(p[1],p[1]) );

	__m256 r_costheta_2 = _mm256_mul_ps(wedge.cos_theta_2,r_2); 

	__m256 uxp = _mm256_fmadd_ps( wedge.u[0], p[0], _mm256_mul_ps(wedge.u[1],p[1]) ); 
		
	__mmask8 res  =   _mm256_cmp_ps_mask(wedge.r0_2, r_2, _CMP_LE_OQ) & 
					   _mm256_cmp_ps_mask(wedge.r1_2, r_2, _CMP_GE_OQ) &
					   _mm256_cmp_ps_mask(uxp, _mm256_setzero_ps(), _CMP_GE_OQ) & 
					   _mm256_cmp_ps_mask(_mm256_mul_ps(uxp, uxp), r_costheta_2, _CMP_GE_OQ);
	
	return res;
}

struct wedge_simd_x16
{
	__m512 r0_2;
	__m512 r1_2;
	__m512 cos_theta_2;
	__m512 u[2];
	
	wedge_simd_x16(wedge wedge)
	{
		r0_2 = _mm512_set1_ps(wedge.r0*wedge.r0);
		r1_2 = _mm512_set1_ps(wedge.r1*wedge.r1);
		cos_theta_2 = _mm512_set1_ps(wedge.cos_theta*wedge.cos_theta);
		u[0] = _mm512_set1_ps(wedge.u.x);
		u[1] = _mm512_set1_ps(wedge.u.y);
	}

};

inline wedge_simd_x16 make_simd_x16(wedge wedge) {
	return wedge_simd_x16(wedge);
}

inline __mmask16 is_inside_simd(wedge_simd_x16 wedge, __m512 p[2])
{

	__m512 r_2 = _mm512_add_ps( _mm512_mul_ps(p[0],p[0]), _mm512_mul_ps(p[1],p[1]) );

	__m512 r_costheta_2 = _mm512_mul_ps(wedge.cos_theta_2,r_2); 

	__m512 uxp = _mm512_fmadd_ps( wedge.u[0], p[0], _mm512_mul_ps(wedge.u[1],p[1]) ); 
		
	__mmask16 res  =   _mm512_cmp_ps_mask(wedge.r0_2, r_2, _CMP_LE_OQ) & 
					   _mm512_cmp_ps_mask(wedge.r1_2, r_2, _CMP_GE_OQ) &
					   _mm512_cmp_ps_mask(uxp, _mm512_setzero_ps(), _CMP_GE_OQ) & 
					   _mm512_cmp_ps_mask(_mm512_mul_ps(uxp, uxp), r_costheta_2, _CMP_GE_OQ);
	
	return res;
}


struct cone_x8
{
	__m256 h;
	__m256 n[3];

	cone_x8(cone cone)
	{
		h    = _mm256_set1_ps(cone.h  );
		n[0] = _mm256_set1_ps(cone.n.x);
		n[1] = _mm256_set1_ps(cone.n.y);
		n[2] = _mm256_set1_ps(cone.n.z);
	}
};

inline cone_x8 make_simd_x8(cone cone) {
	return cone_x8(cone);
}


inline __mmask8 is_inside_simd(cone_x8 cone, __m256 v[3])
{
	__m256 nv = _mm256_fmadd_ps(cone.n[0], v[0], _mm256_fmadd_ps(cone.n[1], v[1], _mm256_mul_ps(cone.n[2], v[2])));
	__m256 h = _mm256_sub_ps(_mm256_set1_ps(1.f), nv);
	return _mm256_cmp_ps_mask(h, cone.h, _CMP_LT_OQ);
}
#endif






