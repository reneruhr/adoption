#pragma once
#include <cmath>
#include <cstdint>

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using f32 = float;
using f64 = double;

constexpr f32 uint_max_inv =  0x1p-32f;
constexpr f32  int_max2    =  0x1p-31f; 

inline u32 lcg(u32& seed)
{
	return seed = 1664525 * seed + 10139042233;
}

// PCG rng O'Neil (www.pcg-random.org)
constexpr u64 pcg32_init[2]   {0x853c49e6748fea9b, 0xda3e39cb94b95bdb};
constexpr u64 pcg32_multiplier{0x5851f42d4c957f2d};

struct pcg_state
{
	u64 seed{0};
	u64 c{0}; 
};

inline u32 pcg32(pcg_state& state)
{
	u32 a = ((state.seed >> 18) ^ state.seed) >> 27;
	u32 b = state.seed >> 59;
	state.seed = state.seed * pcg32_multiplier + state.c;
#pragma warning(push)
#pragma warning(disable : 4146) //silly microsoft compiler warning
	return (a >> b) | (a >> ((-b) & 31));
#pragma warning(pop)
}

inline pcg_state init_pcg32(u64 seed=pcg32_init[0], u64 stream_id=pcg32_init[1])
{
	pcg_state state;
  	state.c = {(stream_id << 1) | 1};
	pcg32(state);
	state.seed+=seed;
	pcg32(state);
	return state;
}


// Marsaglia Xorshift RNG's 2003    Journal of Statistical Software

struct xorshift_state
{
	u32 seed{2463534242};
};

inline u32 xorshift(xorshift_state& state)
{
	u32& x = state.seed;
	x^= x <<13;
	x^= x >>17;
	x^= x << 5;
	return x;
}

inline u32 xorshift(u32& x)
{
	x^= x <<13;
	x^= x >>17;
	x^= x <<5;
	return x;
}

inline u32 xorshift2(u32& x)
{
	x^= x <<5;
	x^= x >>9;
	x^= x <<7;
	return x;
}


inline int xorshift_signed(int& x)
{
	x^= x <<13;
	x^= static_cast<int>(static_cast<unsigned int>(x) >> 17);
	x^= x <<5;
	return x;
}

inline int xorshift2_signed(int& x)
{
	x^= x <<5;
	x^= static_cast<int>(static_cast<unsigned int>(x) >> 9);
	x^= x <<7;
	return x;
}

inline f32 xorshift_f32(xorshift_state& state)
{
	u32& x = state.seed;
	x^= x <<13;
	x^= x >>17;
	x^= x << 5;
	return x*0x1p-32f;
}

struct xorwow_state
{
	u32 s[5] = {123456789, 362436069,521288629,88675123,5783321};
	u32 c{2463534242};
};

inline u32 xorwow(xorwow_state& state)
{
	auto&s = state.s;
	u32 x = s[0]^(s[0] >>2);
	s[0] = s[1];
	s[1] = s[2];
	s[2] = s[3];
	s[3] = s[4];
	s[4] = (s[4]^(s[4]<<4)) ^ (x^(x<<1));
	state.c+=362437;
	return s[4]+state.c;
}

struct xorshift64_state
{
	u64 seed{88172645463325252};
};

inline u64 xorshift(xorshift64_state& state)
{
	u64& x = state.seed;
	x^= x <<13;
	x^= x >> 7;
	x^= x <<17;
	return x;
}

inline f32 rand_f32(u32 u)
{
	return u*0x1p-32f;
}

inline f64 rand_f64(u64 u)
{
	return u*0x1p-64f;
}

struct xorshift_sampler
{
	xorshift_state state;
	u32 operator()(){ return xorshift(state); }
};

struct xorshift2_sampler
{
	u32 seed[2] {2463534242,123456789};
	u32 operator()(){ return xorshift (seed[0]); }
	u32 operator()(int){ return xorshift2(seed[1]); }
};

struct xorwow_sampler
{
	xorwow_state state;
	u32 operator()(){ return xorwow(state); }
};

struct xorshift_sampler_f32
{
	xorshift_state state;
	f32 operator()(){ return rand_f32(xorshift(state)); }
};

inline u8 xorshift_bounded(xorshift_state& state, u32 bound)
{
#pragma warning(push)
#pragma warning(disable : 4146)
	u32 threshold = -bound %bound;
#pragma warning(pop)
	for(;;){
		u32 u = xorshift(state);
		if(u >= threshold)
			return u%bound;
	}
};


// Slightly biased but ok.
struct bernoulli_walk
{
	u32 n;
	u32 seed{2463534242};
	const char *label = "Bernoulli walk";

	bernoulli_walk(u32 n) : n(n) {}

	u32 operator()()
	{
		return xorshift(seed) % n;	
	}
};

struct bitwise_walk
{
	u32 mask; 
	u32 seed{2463534242};
	const char *label = "Bitwise walk";

	bitwise_walk(u32 num_bits)
	{
		mask = (1 << num_bits)-1;
	}
	
	u32 operator()()
	{
		return xorshift(seed) & mask;	
	}
};

struct bitwise_walk_lcg
{
	u8 mask; 
	u8 shift;
	u32 seed{2463534242};
	const char *label = "Bitwise walk lcg";

	bitwise_walk_lcg(u32 num_bits)
	{
		mask = (1 << num_bits)-1;
		shift = 32 - num_bits;
	}
	
	u32 operator()()
	{
		return (lcg(seed) >> shift) & mask;	
	}
};


// Backtrack if choice sequence a,b satisfies b=a+(n/2) mod n
// Enforce non-backtracking by storing last choice, go to its inverse
// located n/2 slots further up
// and sampling among the n-1 following entries
struct no_backtracking_walk 
{
	const u32 n; 
	u32 seed{2463534242};
	u32 last{0};
	const char *label = "No backtracking walk";

	no_backtracking_walk(u32 n) : n(n) {}
	
	u32 operator()()
	{
		last = (last+(n/2+1)+(xorshift(seed) % (n-1) )) %n;	
		return last;
	}
};

struct no_backtracking_walk5
{
	u32 seed{2463534242};
	u32 last{0};
	const char *label = "No backtracking walk";

	u32 operator()()
	{
		last = (last+4+(xorshift(seed) % 5 )) %6;	
		return last;
	}
};

struct bitwise_cached_walk 
{
	u32 seed{2463534242};
	u8 k{0};
	u8 bits;
	const char *label = "Bitwise cached walk";

	bitwise_cached_walk(u8 num_bits)
	{
		bits = num_bits;
	}
	
	u32 operator()()
	{
		if(k== 32/bits)
		{
			k=1;
			return xorshift(seed) & ((1 << bits) - 1);	
		}
		else{
			return (seed >> (k++)*bits) & ((1 << bits) - 1);
		}
	}
};


struct bitwise_cached_walk_lcg
{
	u32 seed{2463534242};
	u8 k{0};
	u8 bits;
	const char *label = "Bitwise cached walk lcg";

	bitwise_cached_walk_lcg(u8 num_bits)
	{
		bits = num_bits;
	}
	
	u32 operator()()
	{
		if(k== 32/bits)
		{
			k=1;
			return lcg(seed) & ((1 << bits) - 1);	
		}
		else{
			return (seed >> (k++)*bits) & ((1 << bits) - 1);
		}
	}
};


struct bitwise_cached_walk_lcg_const
{
	u32 seed{2463534242};
	u8 k{0};
	u8 bits;
	u8 mask;
	const char *label = "Bitwise cached walk lcg less comp";

	bitwise_cached_walk_lcg_const(u8 num_bits)
	{
		bits = num_bits;
		mask = (1 << bits) - 1;
	}
	
	u32 operator()()
	{
		if(k== 32/bits)
		{
			k=1;
			return lcg(seed) & mask;	
		}
		else{
			return (seed >> (k++) * bits) & mask;
		}
	}
};

struct walk_123
{
	u32 n{0};
	const char *label = "123 walk";

	u32 operator()()
	{
		return  (n&4) ?  3 - (n++&3) : n++&3;
	}
};
