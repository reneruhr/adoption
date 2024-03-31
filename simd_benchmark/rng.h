#pragma once
#include "common.h"
#include "rng_integer.h"


template <class Sampler>
inline vec2_f32 disk_polar(Sampler& sampler)
{	
	const f32 z   =   uint_max_inv * sampler();
	const f32 phi =   tau_uint_max_inv * sampler();
	const f32 r = std::sqrt(z);
	return {r*std::cos(phi), r*std::sin(phi)};
}

template <class Sampler>
inline vec2_f32 disk_polar_f(Sampler& sampler)
{	
	const f32 z   =   uint_max_inv * sampler();
	const f32 phi =   tau_uint_max_inv * sampler();
	const f32 r = std::sqrt(z);
	return {r*cosf(phi), r*sinf(phi)};
}

template <class Sampler>
inline vec2_f32 disk_rejection(Sampler& sampler)
{	
	vec2_f32 p;
	do{ 
		p.s = 1.f-int_max2*sampler();
		p.t = 1.f-int_max2*sampler();
	  }     
	while  ( (p,p) >= 1.f);
  	
  	return p;
}

template <class Sampler>
inline vec2_f32 disk_rejection2(Sampler& sampler)
{	
	vec2_f32 p;
	do{ 
		p.s = 1.f-int_max2*sampler();
		p.t = 1.f-int_max2*sampler(1);
	  }     
	while  ( (p,p) >= 1.f);
  	
  	return p;
}

template <class Sampler>
vec3_f32 ball_rejection(Sampler& sampler)
{	
	vec3_f32 p;
	do{ 
		p.x = 1.f-int_max2*sampler();
		p.y = 1.f-int_max2*sampler();
		p.z = 1.f-int_max2*sampler();
	  }     
	while  ( (p,p) >= 1.f);
  	
  	return p;
}


inline vec3_f32 disk_to_sphere(vec2_f32 p)
{	
	const f32 s = (p,p);
	const f32 t = 2*std::sqrt(std::fmax(0.f,1.f-s));
	const f32 z = 1.f-2*s;
  	return {p.x*t,p.y*t, z};
}


// Marsarglia Choosing a point on the surface of a sphere 1972 Annals of Statistic
template <class Sampler>
vec3_f32 sphere_rejection(Sampler& sampler)
{	
	f32 x,y,s;
	do{ x = 1.f-int_max2*sampler();
		y = 1.f-int_max2*sampler();
	  }     while  ((s=x*x+y*y) > 1.f);
  	
	const f32 t = std::sqrt(std::fmax(0.f,1.f-s));
	const f32 z = 1.f-2*s;
  	return {2*x*t,2*y*t, z};
}


template <class Sampler>
inline vec3_f32 sphere_polar(Sampler& sampler)
{	
	const f32 z   = 1.f - int_max2 * sampler();
	const f32 phi =   tau_uint_max_inv * sampler();
	const f32 r = std::sqrt(std::fmax(0.f,1.f-z*z));
	return {r*std::cos(phi), r*std::sin(phi), z};
}
