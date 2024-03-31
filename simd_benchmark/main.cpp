// cl /O2 /arch:AVX512 /arch:AVX2 /std:c++20 main.cpp
#include "common.h"
#include "common_simd.h"
#include "profiler.cpp"
#include "discrepancy.cpp"
#include "rng_integer.h"
#include "rng.h"

#ifndef MAX
#define MAX(x,y) (x) > (y) ? (x) : (y)
#endif

#ifndef MIN 
#define MIN(x,y) (x) < (y) ? (x) : (y)
#endif

struct result
{
	f64 area_{};
	f64 discrepancy_{};
	u64 time_{};
	u64 data_{};
	f32 copy_sum_{};
	const char* label;

	// Discrepancy test results
	template <class TestShape>
	result(u32 sum, u32 n_samples, const TestShape& set, u64 start, u64 end, const char* label) :
		area_((f64)sum/(f64)n_samples),	
		discrepancy_(discrepancy(sum, n_samples, area(set)/ambient_vol(set))),
		time_(end-start),
		data_(n_samples),
		label(label)
	{}

	// Copy test results
	result(f32 sum, u32 n_samples, u64 start, u64 end, const char* label) :
		time_(end-start),
		data_{n_samples},
		copy_sum_{sum},
		label(label)
	{}

	result() {}

};



struct avg_result
{
	u32 n_results{};

	f64 discrepancy_min{std::numeric_limits<f64>::max()};
	f64 discrepancy_max{ 0 };
	f64 discrepancy_avg{};
	u64 time_min{std::numeric_limits<u64>::max()};
	u64 time_max{0};
	u64 time_sum{};
	u64 data{};
	const char* label_sampler;
	const char* label_test;
	const char* comment;

	avg_result(result* results, u32 n_results, u64 data, const char* label_sampler, const char* label_test, const char* comment = "")
		: n_results(n_results), data(data), label_sampler(label_sampler), label_test(label_test), comment(comment)
	{
		
		for (u32 i{ 0 }; i < n_results; i++) 
		{
			discrepancy_min = fmin(results[i].discrepancy_, discrepancy_min);
			discrepancy_max = fmax(results[i].discrepancy_, discrepancy_max);
			discrepancy_avg += results[i].discrepancy_;

			time_min = MIN(results[i].time_, time_min);
			time_max = MAX(results[i].time_, time_max);
			time_sum += results[i].time_;
		}
		
		discrepancy_avg /= n_results;

	}

	avg_result() {};
};

void print(const result& result)
{
	f64 time_s = (f64)result.time_/cpu_frequency();
	printf("Results for %s\n", result.label);
	printf("\tTime taken: %f ms (%llu cycles) \n", time_s*1e3, result.time_);
	if(result.area_ || result.discrepancy_){
		printf("\tThroughput: %.0f Million intersections/s \n", result.data_/time_s*1.e-6);
		printf("\tRelative Intersections: %.6f", result.area_);
		printf("\tDiscrepancy: %.6f", result.discrepancy_);
		printf("\tRelative Error: %.6f\n", result.discrepancy_/result.area_);
	}else if(result.copy_sum_){
		printf("\tThroughput: %.3f Gb/s \n", result.data_/time_s*1.e-9);
		printf("\tCopy sum: %.3f\n", result.copy_sum_);

	}
}


void print(const avg_result& result)
{
	f64 time_min_s = (f64)result.time_min/cpu_frequency();
	f64 time_max_s = (f64)result.time_max/cpu_frequency();
	f64 time_avg_s = (f64)result.time_sum/cpu_frequency()/result.n_results;
	
	printf("Results for %s on %s", result.label_sampler, result.label_test);
	if (strcmp(result.comment,"")) printf(" (%s)\n", result.comment);
	else					  printf("\n");  

	printf("\tMin Time taken: %0.3f ms (%llu cycles) \n", time_min_s*1e3, result.time_min);
	printf("\tMax Time taken: %0.3f ms (%llu cycles) \n", time_max_s*1e3, result.time_max);
	printf("\tAvg Time taken: %0.3f ms (%llu cycles) \n", time_avg_s*1e3, (u64)((f64)result.time_sum/result.n_results));

	printf("\tMin Discrepancy: %.6f\n", result.discrepancy_min);
	printf("\tMax Discrepancy: %.6f\n", result.discrepancy_max);
	printf("\tAvg Discrepancy: %.6f\n", result.discrepancy_avg);

	printf("\tAvg Throughput: %.0f Million intersections/second \n", result.data/time_avg_s*1.e-6);
#ifdef __APPLE__

	printf("\tAvg Cost : %.2f ns / intersection \n", (f64)ticks_to_ns(result.time_sum) / result.n_results / result.data);
#else
	printf("\tAvg Cost : %.2f cycles / intersection \n", (f64)result.time_sum / result.n_results / result.data);
#endif
}


template <class Set, class Sampler>
result discrepancy_test_x8(Set set, Sampler &&sampler, u32 n_samples)
{
	u32 count = n_samples / 8;
	
	u64 start_time = cpu_timer();

	__m256i sum = _mm256_setzero_si256();

	auto set_simd = make_simd_x8(set);

	const __m256i one = _mm256_set1_epi32(1);

	while(count--)
	{
		alignas(32) auto v = sampler();
		__mmask8 dirac = is_inside_simd(set_simd, reinterpret_cast<__m256*>(&v));
		sum = _mm256_mask_add_epi32(sum, dirac, sum, one);
	}

	sum = _mm256_hadd_epi32(sum, sum);
	sum = _mm256_hadd_epi32(sum, sum);

	__m128i sum_a = _mm256_extracti128_si256(sum, 0);
	__m128i sum_b = _mm256_extracti128_si256(sum, 1);

	u32 sum_s = _mm_cvtsi128_si32(_mm_add_epi32(sum_a, sum_b));

	u64 end_time = cpu_timer();

	return result(sum_s, n_samples, set, start_time, end_time, sampler.label);
}

template <class Set, class Sampler>
result discrepancy_test_x16(Set set, Sampler &&sampler, u32 n_samples)
{
	u32 count = n_samples / 16;
	
	u64 start_time = cpu_timer();

	__m512i sum = _mm512_setzero_si512();

	auto set_simd = make_simd_x16(set);

	const __m512i one = _mm512_set1_epi32(1);

	while(count--)
	{
		alignas(64) auto v = sampler();
		__mmask16 dirac = is_inside_simd(set_simd, reinterpret_cast<__m512*>(&v));
		sum = _mm512_mask_add_epi32(sum, dirac, sum, one);
	}

	u32 sum_s = _mm512_reduce_add_epi32(sum);

	u64 end_time = cpu_timer();

	return result(sum_s, n_samples, set, start_time, end_time, sampler.label);
}

struct disk_adoption_x8_state
{
	__m256  cache[2];
	__m256i seed;
};

const __m256 s2inv4_inv_x8	  = _mm256_set1_ps( s2inv4_inv);
const __m256 s2_x8			  = _mm256_set1_ps( s2);
const __m256 zeros_x8		  = _mm256_setzero_ps();

const __m256 s2inv2_uint_max_x8  = _mm256_set1_ps(s2inv2_intmax);
const __m256 s2inv_x8          = _mm256_set1_ps(s2inv);

inline __m256 to_centered_s2_interval(__m256i x)
{
	const __m256 f = _mm256_cvtepu32_ps(x);

	return _mm256_fmsub_ps(s2inv2_uint_max_x8, f, s2inv_x8);
}

const __m256 uint_max2_x8  = _mm256_set1_ps(0x1p-31);

inline __m256 to_centered_2_interval(__m256i x)
{
	const __m256 f = _mm256_cvtepu32_ps(x);

	return _mm256_fmsub_ps(uint_max2_x8, f, one_x8);
}

inline __m256i xorshift(__m256i& x)
{
	
	__m256i t1 = _mm256_slli_epi32(x, 13);
			 x = _mm256_xor_si256(x, t1);

	__m256i t2 = _mm256_srli_epi32(x, 17);
	         x = _mm256_xor_si256(x, t2);

	__m256i t3 = _mm256_slli_epi32(x, 5);
		     x = _mm256_xor_si256(x, t3);

	return x;
}

inline __m512i xorshift(__m512i& x)
{
	
	const __m512i t1 = _mm512_slli_epi32(x, 13);
			 x = _mm512_xor_si512(x, t1);

	const __m512i t2 = _mm512_srli_epi32(x, 17);
	         x = _mm512_xor_si512(x, t2);

	const __m512i t3 = _mm512_slli_epi32(x, 5);
		     x = _mm512_xor_si512(x, t3);

	return x;
}

inline vec2x8 disk_adoption_x8(disk_adoption_x8_state& state)
{

	__m256 x = to_centered_s2_interval(xorshift(state.seed));
	__m256 y = to_centered_s2_interval(xorshift(state.seed));

	__m256 t = _mm256_mul_ps(s2inv4_inv_x8, 
					 _mm256_fmadd_ps(x, x, 
			         _mm256_fmadd_ps(y, y, _mm256_set1_ps(1.f))));

	const __m256 h0 = _mm256_cmp_ps(x, t, _CMP_GT_OQ);
	const __m256 h2 = _mm256_cmp_ps(y, t, _CMP_GT_OQ);

	t = _mm256_sub_ps(zeros_x8, t);

	const __m256 h1 = _mm256_cmp_ps(x, t, _CMP_LT_OQ);
	const __m256 h3 = _mm256_cmp_ps(y, t, _CMP_LT_OQ);

	const __m256 mask0 = _mm256_and_ps(s2_x8, h0);
	const __m256 mask2 = _mm256_and_ps(s2_x8, h2);
	const __m256 mask1 = _mm256_and_ps(s2_x8, h1);
	const __m256 mask3 = _mm256_and_ps(s2_x8, h3);

	__m256 cache_x = _mm256_add_ps(mask1,
					 _mm256_sub_ps(x, mask0));	

 	__m256 cache_y = _mm256_add_ps(mask3,
				     _mm256_sub_ps(y, mask2));	

	__m256 cache_new_mask = _mm256_or_ps(_mm256_or_ps(h0, h2), _mm256_or_ps(h1, h3));
						
	__m256 cache_old_mask = _mm256_cmp_ps(_mm256_setzero_ps(), state.cache[0], _CMP_NEQ_OQ);

	x = _mm256_or_ps    (state.cache[0],
		_mm256_andnot_ps(cache_old_mask, x));

	y = _mm256_or_ps	 (state.cache[1],
		_mm256_andnot_ps(cache_old_mask, y));

	state.cache[0] = _mm256_andnot_ps(cache_old_mask,
					 _mm256_and_ps(cache_new_mask, cache_x));
	state.cache[1] = _mm256_andnot_ps(cache_old_mask,
					 _mm256_and_ps(cache_new_mask, cache_y));
	
	return {x,y};
}


inline vec2x8 disk_rejection_simd(__m256i& seed)
{	
	__m256 x = _mm256_setzero_ps();
	__m256 y = _mm256_setzero_ps();
	__m256 mask = _mm256_setzero_ps(); 
	do{ 
		__m256 s = to_centered_2_interval(xorshift(seed));
		__m256 t = to_centered_2_interval(xorshift(seed));
		__m256 norm = _mm256_fmadd_ps(s,s, _mm256_mul_ps(t,t));
		__m256 valid = _mm256_cmp_ps(one_x8, norm, _CMP_GE_OQ);
		valid = _mm256_andnot_ps(mask, valid);	
		mask  = _mm256_or_ps(mask, valid);

		s = _mm256_and_ps(valid, s);
		t = _mm256_and_ps(valid, t);

		x = _mm256_or_ps(s, x);
		y = _mm256_or_ps(t, y);
	  }
	while  (!_mm256_test_all_ones(_mm256_castps_si256(mask)));
  	
  	return {x,y};
}


inline vec2x8 disk_polar_simd(__m256i& seed)
{	
	__m256 phi =  _mm256_mul_ps(_mm256_set1_ps(tau_uint_max_inv), _mm256_cvtepu32_ps(xorshift(seed))) ;
	__m256 r   =  _mm256_mul_ps(_mm256_set1_ps(uint_max_inv    ), _mm256_cvtepu32_ps(xorshift(seed)));

	r = _mm256_sqrt_ps(r);
	__m256 cos;
	__m256 sin = _mm256_sincos_ps(&cos, phi);
	__m256 x = _mm256_mul_ps(sin, r);
	__m256 y = _mm256_mul_ps(cos, r);

  	return {x,y};
}


struct polar_sampler_x8
{
	__m256i seed;

	polar_sampler_x8(){
		xorwow_sampler sampler;
		seed = _mm256_set_epi32(sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler());
	}

	const char* label = "Polar method x8 X86";

	vec2x8 operator()() {
		return disk_polar_simd(seed);
	}
};
struct rejection_sampler_x8
{
	__m256i seed;

	rejection_sampler_x8(){
		xorwow_sampler sampler;
		seed = _mm256_set_epi32(sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler());
	}

	const char* label = "Rejection method x8 X86";

	vec2x8 operator()() {
		return disk_rejection_simd(seed);
	}
};

struct adoption_sampler_x8
{
	alignas(32) disk_adoption_x8_state state{};

	adoption_sampler_x8(){
		xorwow_sampler sampler;
		state.seed = _mm256_set_epi32(sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler());
		state.cache[0] = _mm256_setzero_ps();
		state.cache[1] = _mm256_setzero_ps();
	}

	const char* label = "Adoption method x8";

	vec2x8 operator()() {
		return disk_adoption_x8(state);
	}
};

using seed512 = __m512i;

inline vec2x16 disk_polar_sincos_simd(seed512& a)
{	
	__m512 phi =  _mm512_mul_ps(_mm512_set1_ps(tau_uint_max_inv), _mm512_cvtepu32_ps(xorshift(a))) ;
	__m512 r   =  _mm512_mul_ps(_mm512_set1_ps(uint_max_inv    ), _mm512_cvtepu32_ps(xorshift(a)));

	r = _mm512_sqrt_ps(r);
	__m512 cos;
	__m512 sin = _mm512_sincos_ps(&cos, phi);
	__m512 x = _mm512_mul_ps(sin, r);
	__m512 y = _mm512_mul_ps(cos, r);

  	return {x,y};
}

const __m512 s2inv2_uint_max_x16  = _mm512_set1_ps(s2inv2_intmax);
const __m512 s2inv_x16          = _mm512_set1_ps(s2inv);

inline __m512 to_centered_s2_interval(__m512i x)
{
	const __m512 f = _mm512_cvtepu32_ps(x);

	return _mm512_fmsub_ps(s2inv2_uint_max_x16, f, s2inv_x16);
}

const __m512 uint_max2_x16  = _mm512_set1_ps(0x1p-31);
const __m512 one_x16 = _mm512_set1_ps(1.f);

inline __m512 to_centered_2_interval(__m512i x)
{
	const __m512 f = _mm512_cvtepu32_ps(x);

	return _mm512_fmsub_ps(uint_max2_x16, f, one_x16);
}

// Todo: move bit arithmetic to simd register
inline vec2x16 disk_rejection_simd(seed512& a)
{	
	const __m512  one   = _mm512_set1_ps(1.);
	__m512 x = _mm512_setzero_ps();
	__m512 y = _mm512_setzero_ps();
	__mmask16 mask{ 0 };
	do{ 
		const __m512 s = to_centered_2_interval(xorshift(a));
		const __m512 t = to_centered_2_interval(xorshift(a));
		const __m512 norm  = _mm512_fmadd_ps(t,t,_mm512_mul_ps(s,s));
		__mmask16 valid = _mm512_cmp_ps_mask(one, norm, _CMP_GE_OQ);
		valid &= ~mask;
		mask  |=  valid;
		x = _mm512_mask_blend_ps(valid, x, s);
		y = _mm512_mask_blend_ps(valid, y, t);
	  }
	while(mask!=__mmask16(~0));
  	
  	return {x,y};
}


struct polar_sampler_x16
{
	__m512i seed;

	polar_sampler_x16(){
		xorwow_sampler sampler;
		seed  = _mm512_set_epi32(sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler());
	}

	const char* label = "Polar method x16 X86";

	vec2x16 operator()() {
		return disk_polar_sincos_simd(seed);
	}
};

struct rejection_sampler_x16
{
	__m512i seed;

	rejection_sampler_x16(){
		xorwow_sampler sampler;
		seed  = _mm512_set_epi32(sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler());
	}

	const char* label = "Rejection method 2d SIMD X86  AVX512";

	vec2x16 operator()() {
		return disk_rejection_simd(seed);
	}
};

struct disk_adoption_x16_state
{
	__m512  cache[2];
	__m512i seed;
};
const __m512 s2_x16			  = _mm512_set1_ps( s2);
const __m512 s2inv4_inv_x16	  = _mm512_set1_ps( s2inv4_inv);
const __m512 zeros = _mm512_setzero_ps();

inline vec2x16 disk_adoption_x16(disk_adoption_x16_state& state)
{
	__m512 x = to_centered_s2_interval(xorshift(state.seed));
	__m512 y = to_centered_s2_interval(xorshift(state.seed));

	__m512 t = _mm512_mul_ps(s2inv4_inv_x16, 
					 _mm512_fmadd_ps(x, x, 
			         _mm512_fmadd_ps(y, y, _mm512_set1_ps(1.f))));

	const __mmask16 h0 = _mm512_cmp_ps_mask(x, t, _CMP_GT_OQ);
	const __mmask16 h2 = _mm512_cmp_ps_mask(y, t, _CMP_GT_OQ);

	t = _mm512_sub_ps(zeros, t);

	const __mmask16 h1 = _mm512_cmp_ps_mask(x, t, _CMP_LT_OQ);
	const __mmask16 h3 = _mm512_cmp_ps_mask(y, t, _CMP_LT_OQ);


	__m512 cache_x = _mm512_mask_sub_ps(x,       h0,       x, s2_x16);
		   cache_x = _mm512_mask_add_ps(cache_x, h1,       x, s2_x16);

	__m512 cache_y = _mm512_mask_sub_ps(y,       h2,       y, s2_x16);
		   cache_y = _mm512_mask_add_ps(cache_y, h3,       y, s2_x16);

	const __mmask16 cache_set  = _mm512_cmp_ps_mask(state.cache[0], zeros, _CMP_NEQ_OQ);
	
    x = _mm512_mask_blend_ps(cache_set, x, state.cache[0]);
    y = _mm512_mask_blend_ps(cache_set, y, state.cache[1]);
	
	const __mmask16 new_cache = _mm512_kandn(cache_set, _mm512_kor(_mm512_kor(h0,h2), _mm512_kor(h1,h3)));

	state.cache[0] = _mm512_mask_blend_ps(new_cache ,zeros, cache_x);
	state.cache[1] = _mm512_mask_blend_ps(new_cache, zeros, cache_y);

	return {x,y};
}

struct adoption_sampler_x16
{
	alignas(64) disk_adoption_x16_state state{};

	adoption_sampler_x16(){
		xorwow_sampler sampler;
		state.seed = _mm512_set_epi32(sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler());
		state.cache[0] = _mm512_setzero_ps();
		state.cache[1] = _mm512_setzero_ps();
	}

	const char* label = "Adoption method x16";

	vec2x16 operator()() {
		return disk_adoption_x16(state);
	}
};



template <class Sampler>
vec2_f32 make_unit_vec(Sampler& sampler)
{
	vec2_f32 u{ disk_rejection(sampler) };
	f32      r{ 0 };
	while ((r = (u, u)) == 0)
		u = disk_rejection(sampler);

	u.x /= std::sqrt(r);
	u.y /= std::sqrt(r);
	return u;
}



const u32 n_wedges{12};

wedge* make_wedges()
{
	xorwow_sampler  sampler;
	wedge* wedges = new wedge[n_wedges];	

	float theta = pi / 2;
	wedges[0] = { .r0 = 0.0f, .r1 = 1.f, .theta = theta, .cos_theta = (f32)std::cos(theta), .u=make_unit_vec(sampler)};
	wedges[1] = { .r0 = 0.0f, .r1 = 1.f, .theta = theta, .cos_theta = (f32)std::cos(theta), .u=make_unit_vec(sampler)};

	theta /= 2;
	wedges[2] = { .r0 = 0.0f, .r1 = 1.f, .theta = theta, .cos_theta = (f32)std::cos(theta), .u=make_unit_vec(sampler)};
	wedges[3] = { .r0 = 0.0f, .r1 = 1.f, .theta = theta, .cos_theta = (f32)std::cos(theta), .u=make_unit_vec(sampler)};

	theta /= 2;
	wedges[4] = { .r0 = 0.0f, .r1 = 1.f, .theta = theta, .cos_theta = (f32)std::cos(theta), .u=make_unit_vec(sampler)};
	wedges[5] = { .r0 = 0.0f, .r1 = 1.f, .theta = theta, .cos_theta = (f32)std::cos(theta), .u=make_unit_vec(sampler)};

	theta = pi / 2;
	wedges[6] = { .r0 = 0.5f, .r1 = 1.f, .theta = theta, .cos_theta = (f32)std::cos(theta), .u=make_unit_vec(sampler)};

	theta /= 2;
	wedges[7] = { .r0 = 0.5f, .r1 = 1.f, .theta = theta, .cos_theta = (f32)std::cos(theta), .u=make_unit_vec(sampler)};

	theta /= 2;
	wedges[8] = { .r0 = 0.5f, .r1 = 1.f, .theta = theta, .cos_theta = (f32)std::cos(theta), .u=make_unit_vec(sampler)};

	theta = pi / 2;
	wedges[9] = { .r0 = 0.0f, .r1 = .5f, .theta = theta, .cos_theta = (f32)std::cos(theta), .u=make_unit_vec(sampler)};

	theta /= 2;
	wedges[10] = { .r0 = 0.0f, .r1 = .5f, .theta = theta, .cos_theta = (f32)std::cos(theta), .u=make_unit_vec(sampler)};

	theta /= 2;
	wedges[11] = { .r0 = 0.0f, .r1 = .5f, .theta = theta, .cos_theta = (f32)std::cos(theta), .u=make_unit_vec(sampler)};
	
	return wedges;
}

template<class Set, class Sampler>
void test_suite_x8(Set* sets, u32 n_sets, Sampler sampler, u32 n_samples) 
{
	result* results= new result[n_sets];

	for (u32 i = 0; i < n_sets; i++) 
	{
		results[i] = discrepancy_test_x8(sets[i], sampler, n_samples);
	}
	avg_result avg{ results, n_sets, n_samples, sampler.label , label(sets[0])};
	print(avg);

	delete[] results;
}

template<class Sampler>
void test_suite_disk_x8(Sampler sampler, u32 n_samples) 
{
	wedge* wedges = make_wedges();

	test_suite_x8(wedges, n_wedges, sampler, n_samples);

	delete[] wedges;
}

template<class Set, class Sampler>
void test_suite_x16(Set* sets, u32 n_sets, Sampler sampler, u32 n_samples) 
{
	result* results= new result[n_sets];

	for (u32 i = 0; i < n_sets; i++) 
	{
		results[i] = discrepancy_test_x16(sets[i], sampler, n_samples);
	}
	avg_result avg{ results, n_sets, n_samples, sampler.label , label(sets[0])};
	print(avg);

	delete[] results;
}

template<class Sampler>
void test_suite_disk_x16(Sampler sampler, u32 n_samples) 
{
	wedge* wedges = make_wedges();

	test_suite_x16(wedges, n_wedges, sampler, n_samples);

	delete[] wedges;
}

int main(int argc, char** args)
{

	u32 n_samples = 16*1'000'000;
	if(argc>1)
		n_samples = atoi(args[1]);

	xorwow_sampler sampler;	


	const u32 n_begin = 1024*1024;
	const u32 n_max   =  10*1024*1024;
	const u32 mult = 2;

	printf("Discrepancy Test on Disk:\n\n");
	for (u32 n = n_begin; n < n_max; n *= mult) 
    {
		printf("Using %u samples.\n\n", n);

		printf("\nx8.\n");
		test_suite_disk_x8(polar_sampler_x8{}, n);
		test_suite_disk_x8(rejection_sampler_x8{}, n);
		test_suite_disk_x8(adoption_sampler_x8{}, n);

		printf("\nx16.\n");
		test_suite_disk_x16(polar_sampler_x16{}, n);
		test_suite_disk_x16(rejection_sampler_x16{}, n);
		test_suite_disk_x16(adoption_sampler_x16{}, n);

	}
}