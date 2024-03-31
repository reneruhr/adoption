// pbrt 4th. Section 8.6
f32 radical_inverse(u32 b, u32 a)
{
	f32 inv_base = 1.f / b;
	f32 inv_base_mult = 1;
	u32 c{};
	while (a)
	{
		u32 next = a / b;
		u32 digit = a - next * b;
		c = c * b + digit;
		inv_base_mult *= inv_base;
		a = next;
	}
	return std::min(c * inv_base_mult,1.f);

}

#include "sobolmatrices.cpp"
// pbrt 4th. Section 8.7
u32 multiply_generator(u32 a, u32 start) 
{
    u32 v = 0;
    for (u32 i = start; a != 0; ++i, a >>= 1)
        if (a & 1)
            v ^= sobol_matrices32[i];
    return v;
}

// pbrt 4th. Section B.2.7
u32 reverse_bits(u32 n) 
{
    n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
    n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
    return n;
}

points grid(u32 m, u32 n, f32 a, f32 b, f32 *vertices)
{
	u32 n_vertices = m * n;
	for (u32 j{}; j < n; j++)
		for (u32 i{}; i < m; i++)
		{
			vertices[2 * (i + j * m)    ] = a * i / m + a/m/2;
			vertices[2 * (i + j * m) + 1] = b * j / n + b/n/2;
		}
	return points{vertices, n_vertices};
}

points halton(u32 m, u32 n, f32 a, f32 b, f32 *vertices)
{
	u32 n_vertices = m * n;
	for (u32 j{}; j < n; j++)
		for (u32 i{}; i < m; i++)
		{
			u32 u = i + j * m;
			vertices[2 * u    ] = a * radical_inverse(2, u);
			vertices[2 * u + 1] = b * radical_inverse(3, u);
		}
	return points{vertices, n_vertices};
}

points hammersley(u32 m, u32 n, f32 a, f32 b, f32 *vertices)
{
	u32 n_vertices = m * n;
	for (u32 j{}; j < n; j++)
		for (u32 i{}; i < m; i++)
		{
			u32 u = i + j * m;
			vertices[2 * u    ] = a * u / n_vertices;
			vertices[2 * u + 1] = b * radical_inverse(2, u);
		}
	return points{vertices, n_vertices};
}

points sobol(u32 m, u32 n, f32 a, f32 b, f32 *vertices)
{
	u32 n_vertices = m * n;
	u32 start{1};
	for (u32 j{}; j < n; j++)
		for (u32 i{}; i < m; i++)
		{
			u32 u = i + j * m;
			vertices[2 * u    ] = a * 0x1p-32f * multiply_generator(u, start      * SobolMatrixSize);
			vertices[2 * u + 1] = b * 0x1p-32f * multiply_generator(u, (start + 1) * SobolMatrixSize);
		}
	return points{vertices, n_vertices};
}

points jitter(u32 m, u32 n, f32 a, f32 b, f32 *vertices)
{
	u32 n_vertices = m * n;
	
	for (u32 j{}; j < n; j++)
		for (u32 i{}; i < m; i++)
		{
			auto ran = pcg2d(vec2_u32{ i,j });
			
			vertices[2 * (i + j * m)    ] = a * i / m + a/m*ran.x*0x1p-32f;
			vertices[2 * (i + j * m) + 1] = b * j / n + b/n*ran.y*0x1p-32f;
		}
	return points{vertices, n_vertices};
}

points random(u32 m, u32 n, f32 a, f32 b, f32 *vertices)
{
	u32 n_vertices = m * n;
	
	for (u32 j{}; j < n; j++)
		for (u32 i{}; i < m; i++)
		{
			auto ran = pcg2d(vec2_u32{ i,j });
			
			vertices[2 * (i + j * m)    ] = a*ran.x*0x1p-32f;
			vertices[2 * (i + j * m) + 1] = b*ran.y*0x1p-32f;
		}
	return points{vertices, n_vertices};
}

points jitter(u32 m, u32 n, f32 a, f32 b, u32 ran, f32 *vertices)
{
	u32 n_vertices = m * n;
	
	for (u32 j{}; j < n; j++) for (u32 i{}; i < m; i++)
		{
			ran = pcg(ran);
			vertices[2 * (i + j * m)    ] = a * i / m + a/m*ran*0x1p-32f;
			ran = pcg(ran);
			vertices[2 * (i + j * m) + 1] = b * j / n + b/n*ran*0x1p-32f;
		}
	return points{vertices, n_vertices};
}

points random(u32 m, u32 n, f32 a, f32 b, u32 ran, f32 *vertices)
{
	u32 n_vertices = m * n;
	
	for (u32 j{}; j < n; j++) for (u32 i{}; i < m; i++)
		{
			ran = pcg(ran);
			vertices[2 * (i + j * m)    ] = a*ran*0x1p-32f;
			ran = pcg(ran);
			vertices[2 * (i + j * m) + 1] = b*ran*0x1p-32f;
		}
	return points{vertices, n_vertices};
}
