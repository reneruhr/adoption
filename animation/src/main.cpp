#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cassert>
#include <cmath>

#include <utility>
#include <numbers>
#include <filesystem>
#include <string>
#include <span>
#include <algorithm>
#include <vector>
#include <future>

using u8  = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using f32 = float;
using f64 = double;

#include "quaternions.h"
#include "common.h"
#include "vec_math.h"
#include "random_int.h"

#include "gl_backend.cpp"
#include "samplers.cpp"
#include "warps.cpp"
#include "statistics.cpp"
#include "memory_allocator.cpp"
#include "primitives_2d.cpp"
#include "fontrendering.cpp"

const GLuint WIDTH = 1024, HEIGHT = 1024;
const f32 ratio = (f32)HEIGHT / WIDTH;
vec2_f32 cam_size{ (f32)WIDTH, (f32)HEIGHT};
vec2_f32 scroll_offset = vec2_f32(0);
vec2_f32 awds_offset = vec2_f32(0);
f32 scroll_speed = 300.f;
f32 awds_speed = 100.f;

enum mode { pause, running } mode;
u32 second{};


arena arena0(1*gigabyte);  
arena arena1(1*gigabyte);  

// events 
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void scroll_callback(GLFWwindow* window, double x, double y);
struct key_event { int key; int action; };
const u32 n_queue{ 1024 };
struct events
{
	u32 begin{};
	u32 end{};
	key_event queue[n_queue];
} events{};
void register_event(key_event event);
bool has_events();
key_event poll_event();
void event_loop();

struct per_frame_data
{
	quat q{};
	vec3_f32 translation{0.};
	f32  scale{1};
	vec4_f32 color{1.};
};

per_frame_data interpolate(per_frame_data a, per_frame_data b, f32 t)
{
	return { .q = slerp(a.q, b.q, t), .translation = lerp(a.translation, b.translation, t), .scale = std::lerp(a.scale, b.scale, t), .color = lerp(a.color, b.color, t) };
}


struct per_frame_data2
{
	u32 first;
	u32 count;
};

struct scene
{
	u32 n_seconds;
	u32 n_meshes;
	u32 *mesh_table;
	per_frame_data *data;
	opengl_context context;

	per_frame_data2 *data2;
	font_rendering fonts;

	f32 rate;
};

mesh unify(mesh *meshes, u32 *mesh_table, u32 n_meshes, arena& arena)
{
	mesh mesh{};
	u32* offsets_ind = alloc_n<u32>(arena, n_meshes);
	u32* offsets_ver = alloc_n<u32>(arena, n_meshes);
	for (u32 u{}; u<n_meshes; u++)
	{
		offsets_ind[u] = mesh.n_indices;
		offsets_ver[u] = mesh.n_vertices;
		mesh.n_vertices += meshes[u].n_vertices;
		mesh.n_indices  += meshes[u].n_indices;
	}

	mesh.vertices 	   = alloc_n<f32>(arena, 2 * mesh.n_vertices);
	mesh.colors   	   = alloc_n<f32>(arena, 4 * mesh.n_vertices);
	mesh.mesh_indices  = alloc_n<u32>(arena, 1 * mesh.n_vertices);
	mesh.indices  	   = alloc_n<u32>(arena, 1 * mesh.n_indices);

	for (u32 u{}; u<n_meshes; u++)
	{
		memcpy(mesh.vertices 	 + offsets_ver[u]*2, meshes[u].vertices, meshes[u].n_vertices * 2 * sizeof(f32));
		memcpy(mesh.colors   	 + offsets_ver[u]*4, meshes[u].colors,   meshes[u].n_vertices * 4 * sizeof(f32));
		memcpy(mesh.indices  	 + offsets_ind[u],   meshes[u].indices,  meshes[u].n_indices  * 1 * sizeof(u32));
		for (u32 m{}; m< meshes[u].n_indices; m++)
			mesh.indices[m + offsets_ind[u]]     += offsets_ver[u];
		for (u32 m{}; m< meshes[u].n_vertices; m++)
			mesh.mesh_indices[m + offsets_ver[u]] = mesh_table[u];
	}

	return mesh;
}

mat4_f32 cur_cam()
{
	vec2_f32 cam_offs = scroll_offset+awds_offset;
	mat4_f32 p = orthographic(cam_offs.x,cam_offs.x+cam_size.x, cam_offs.y,cam_offs.y+cam_size.y,-2,2);
	mat4_f32 v = scale(1.f);
	mat4_f32 camera = mul(p, v);
	return camera;
}




scene init_scene()
{
	font_rendering fonts;
	fonts.init_fonts(arena0);

	f32 scale = 200.f;
		
	auto disk0		= disk(1, 300, arena0);
	auto square     = rectangle(std::sqrt(2.),std::sqrtf(2.), arena0);
	auto segment    = circular_segment(1, pi_f/2, 100, arena0);
	auto point		= disk(0.03, 8, arena0);
	const u32 n_disk    = 5;
	const u32 n_square  = 1;
	const u32 n_segment = 4;
	const u32 n_point   = 20;
	const u32 n_point2  = 20;

	
	const u32 n_seconds = 82;
	const u32 n_meshes = n_disk + n_square + n_segment + n_point + n_point2;
	const u32 n_per_frame_data = n_seconds * n_meshes;

	
	u32* mesh_table = alloc_n<u32>(arena0, n_meshes);
	for (u32 u{}; u < n_meshes; u++) mesh_table[u] = u;

	const u32 id_disk    = 0;
	const u32 id_square  = n_disk;
	const u32 id_segment = n_disk+n_square;
	const u32 id_point  = n_disk+n_square+n_segment;
	const u32 id_point2  = n_disk+n_square+n_segment+n_point;

	per_frame_data *keyframes= alloc_n<per_frame_data>(arena0, n_per_frame_data);
	u32* dummy= alloc_n<u32>(arena0, 100000);
	per_frame_data2 *font_times= alloc_n<per_frame_data2>(arena0, n_seconds);
	
	mat4_f32 camera = cur_cam();
	vec3_f32 center(cam_size.x / 2, cam_size.y / 2, 0.f);
	vec3_f32 on_top(0, 0, 0.01);

	for (u32 u{ 0 }; u < n_seconds; u++) font_times[u] = { .first= 0, .count = 0 };

	auto add_text = [font_times, &fonts, center](const char* text, u32 s0, u32 s1, u32 line = 0)
		{
			u32 first = fonts.context.count;
			if(line) first = font_times[s0].first;
			fonts.add_text(text, center.x-500, center.y + 300-line*48, 0.8, 0.8, 0, arena0);
			u32 count = fonts.context.count - first;
			for (u32 u{ s0 }; u < s0 + s1; u++)
			{
				font_times[u] = per_frame_data2{ .first = first, .count = count };
			}
		};
	
	vec3_f32 pt_color = 0.3*rgb(57, 61, 27);
	vec3_f32 segment_colors[4] = { rgb(234, 152, 83), rgb(195, 93, 227), rgb(35, 250, 35), rgb(255, 11, 18) };
	vec3_f32 disk_translations[4] = {vec3_f32(std::sqrt(2),0,0) ,vec3_f32(0,std::sqrt(2),0) , vec3_f32(-std::sqrt(2),0,0) ,vec3_f32(0,-std::sqrt(2),0) };

	auto in = [](u32 u0, u32 u1, u32 u) { return  (u0 <= u) && ( u <= u1 ); };
	auto mod_alpha = [](vec4_f32& v, f32 a) { v.w = a; return v; };
	auto keyframe = [keyframes, mesh_table, n_meshes, n_seconds](u32 second, u32 meshid) -> per_frame_data&
		{
			assert((second < n_seconds) && "Too far in future");
			assert((mesh_table[meshid] < n_meshes) && "Mesh not defined");
			return keyframes[second * n_meshes + mesh_table[meshid]]; 
		};

	auto fade_in = [keyframe, mod_alpha, n_seconds](u32 meshid, u32 start, u32 end, f32 f0 =0., f32 f1=1.)
		{
			for (u32 u{ start }; u <= end; u++)
				mod_alpha(keyframe(u, meshid).color, std::lerp(f0, f1, ((f32)(u-start))/(end-start)));
			for (u32 u{ end + 1 }; u < n_seconds; u++)
				keyframe(u, meshid) = keyframe(end, meshid);
		};

	auto set_rgb = [keyframe](u32 meshid, u32 start, u32 end, vec3_f32 rgb)
		{
			for (u32 u{ start }; u <= end; u++) keyframe(u, meshid).color = vec4_f32(rgb, keyframe(u, meshid).color.w);
		};
	auto set_translation = [keyframe](u32 meshid, u32 start, u32 end, vec3_f32 t)
		{
			for (u32 u{ start }; u <= end; u++) keyframe(u, meshid).translation = t;
		};
	auto translate = [keyframe](u32 meshid, u32 start, u32 end, vec3_f32 translation)
		{
			for (u32 u{ start }; u <= end; u++) keyframe(u, meshid).translation =   keyframe(u, meshid).translation + translation;
		};
	auto dilate= [keyframe](u32 meshid, u32 start, u32 end, f32 scale)
		{
			for (u32 u{ start }; u <= end; u++) keyframe(u, meshid).scale=   keyframe(u, meshid).scale*scale;
		};
	auto rotate = [keyframe](u32 meshid, u32 start, u32 end, quat q)
		{
			for (u32 u{ start }; u <= end; u++) keyframe(u, meshid).q = q*keyframe(u, meshid).q;
		};
	auto random_vec2 = [seed = vec2_u32(3,1)](f32 a, f32 b) mutable
		{
			seed = pcg2d(seed);
			return vec2_f32(a*0x1p-32f * seed.x, b*0x1p-32f * seed.y);
		};
	auto random_vec2_in_segment = [seed = vec2_u32(3,1), disk_translations](f32 a, f32 b, u32 s) mutable
		{
			vec2_f32 v;
			vec2_f32 t = { disk_translations[s].x, disk_translations[s].y };
			do
			{
				seed = pcg2d(seed);
				v = std::sqrt(2)/2*vec2_f32(0x1p-31f * seed.x-1.,  0x1p-31f * seed.y-1.);

			} while ( dot(v-t,v-t)>=1  );
			return vec2_f32{a * v.x, b * v.y};
		};
	auto random_vec2_outside_segment = [seed = vec2_u32(3,1), disk_translations](f32 a, f32 b) mutable
		{
			vec2_f32 v;
			bool searching= false;
			do
			{
				seed = pcg2d(seed);
				v = std::sqrt(2)/2*vec2_f32(0x1p-31f * seed.x-1.,  0x1p-31f * seed.y-1.);
				searching = false;
				for (u32 s{}; s<4; s++)
				{
					vec2_f32 t = { disk_translations[s].x, disk_translations[s].y };
					if (dot(v - t, v - t) < 1) searching = true;
				}
			} while (searching);
			return vec2_f32{a * v.x, b * v.y};
		};



	u32 s_disk_appear = 0;
	add_text("Adoption Sampling.", s_disk_appear, 4);
	add_text("The disk we wish to sample.", s_disk_appear+4, 4);

	for (u32 u{}; u < n_seconds; u++) for (u32 m{}; m < n_meshes; m++) keyframe(u, m) = { quat{}, center, scale, vec4_f32(1,1,1,0) };
	set_rgb(id_disk, 0, n_seconds-1, rgb(229, 234, 231));
	fade_in(id_disk, s_disk_appear, s_disk_appear+1);
	

	u32 s_square_appear = s_disk_appear+2;
	set_rgb(id_square, 0, n_seconds-1, rgb(229, 234, 231)-rgb(20,20,20));
	translate(id_square, 0, n_seconds-1, vec3_f32(-std::sqrt(2) / 2 * scale, -std::sqrt(2) / 2 * scale, 0) + on_top);
	fade_in(id_square, s_square_appear, s_square_appear+1);

	add_text("Inscribed square.", s_square_appear+2, 4);
	add_text("We only sample from the square.", s_square_appear+6, 4);

	u32 s_point_appear = s_square_appear+2;
	u32 pts_per_s = 4;
	for (u32 p{}; p<n_point; p++)
	{
	set_rgb(id_point+p, 0, n_seconds-1, pt_color);
	translate(id_point+p, 0, n_seconds-1, 2*on_top + vec3_f32(  scale*std::sqrt(2)*(random_vec2(1, 1)+vec2_f32(-0.5,-0.5)), 0));
	fade_in(id_point+p, s_point_appear+p/pts_per_s, s_point_appear+p/pts_per_s+1);
	fade_in(id_point+p, s_point_appear+n_point/pts_per_s+4, s_point_appear+n_point/pts_per_s+5, 1.,0.);
	}


	u32 s_segment_appear = s_point_appear + n_point/pts_per_s + 1;
	u32 s_segment_disappear = s_point_appear + n_point/pts_per_s + 1 + n_segment;
	u32 s_other_disk_appear = s_segment_disappear+4;

	add_text("We miss these segments.", s_segment_appear, 4);
	add_text("Translated copies of them can be found in the square.", s_segment_appear+4, 4);
	add_text("By intersecting", s_segment_appear+8, 4);
	add_text("with", s_segment_appear+8, 4, 1);
	add_text("translated disks.", s_segment_appear+8, 4, 2);

	for (u32 s{}; s<n_segment; s++)
	{
	set_rgb(id_segment+s, 0, n_seconds-1, segment_colors[s]);
	rotate(id_segment + s, 0, n_seconds-1, { std::cos(-pi_f / 8 + pi_f / 4 * s),0,0, std::sin(-pi_f / 8 + pi_f / 4 * s) });
	translate(id_segment + s, 0, n_seconds-1, 2 * on_top);
	fade_in(id_segment+s, s_segment_appear+s, s_segment_appear+s+1);

	//translate(id_segment+s, s_segment_disappear+2, s_segment_disappear+3, -scale*0.5*disk_translations[s]);
	translate(id_segment+s, s_segment_disappear+2, n_seconds-1, -scale*disk_translations[s]);

	set_rgb(id_disk+1+s, 0, n_seconds-1, segment_colors[(s+2)%4]);
	fade_in(id_disk+1+s, s_other_disk_appear, s_other_disk_appear+1,0,0.5);
	translate(id_disk+1+s, 0, n_seconds-1, scale * disk_translations[s]);
	translate(id_disk+1+s, s_other_disk_appear, n_seconds-1, 1 * on_top);
	}


	u32 s_point_reappear = s_other_disk_appear + 4;
	add_text("We sample in this translate and 'adopt' a copy.", s_point_reappear+1, 4);
	add_text("The adopted sample now samples the missing parts.", s_point_reappear+5, 4);

	for (u32 s{ 0 }; s < n_segment; s++)
	{
	if (s != 2)
	{
		fade_in(id_segment + s, s_point_reappear, s_point_reappear + 1, 1, 0);
		translate(id_segment + s, s_point_reappear+1, n_seconds-1, -3*on_top);
	}
	if (s != 0) 
	{
		fade_in(id_disk + 1 + s, s_point_reappear, s_point_reappear + 1, 0.5, 0);
		translate(id_disk + 1 + s, s_point_reappear, n_seconds-1, -3*on_top);
	}
	}

	for (u32 p{}; p<n_point; p++)
	{
	set_translation(id_point+p, s_point_reappear-1, n_seconds-1, center+4*on_top + vec3_f32(  scale*random_vec2_in_segment(1, 1, 0), 0));
	fade_in(id_point+p, s_point_reappear+p/pts_per_s, s_point_reappear+p/pts_per_s+1);
	}

	u32 s_move_segment = s_point_reappear + n_point / pts_per_s + 1;

	for (u32 p{}; p<n_point; p++)
	{
	translate(id_point + p, s_move_segment, n_seconds-1, scale*disk_translations[2]);
	}
	translate(id_segment+2, s_move_segment, n_seconds-1, scale*disk_translations[2]);


	u32 s_clear= s_move_segment+4;
	for (u32 p{}; p<n_point; p++)
	{
	translate(id_point+p, s_clear, n_seconds-1, -4*on_top);
	fade_in(id_point+p, s_clear-2, s_clear-1, 1,0);
	}
	translate(id_segment+2, s_clear, n_seconds-1, -4*on_top);
	translate(id_disk+1, s_clear, n_seconds-1, -4*on_top);
	fade_in(id_segment+2, s_clear-1, s_clear, 1,0);
	fade_in(id_disk+1, s_clear-1, s_clear, 0.5,0);
	translate(id_segment+2, s_clear+1, n_seconds-1, -scale*disk_translations[2]);


	u32 s_final = s_clear+1;

	vec2_f32 pts_locations[n_point2]
	{
		random_vec2_outside_segment(1,1),
		random_vec2_in_segment(1,1,0),
		random_vec2_outside_segment(1,1),
		random_vec2_in_segment(1,1,1),
		random_vec2_outside_segment(1,1),
		random_vec2_in_segment(1,1,2),
		random_vec2_outside_segment(1,1),
		random_vec2_in_segment(1,1,3),
		random_vec2_outside_segment(1,1),
		random_vec2_outside_segment(1,1),
		random_vec2_in_segment(1,1,0),
		random_vec2_in_segment(1,1,2),
		random_vec2_outside_segment(1,1),
		random_vec2_in_segment(1,1,1),
		random_vec2_outside_segment(1,1),
		random_vec2_in_segment(1,1,2),
		random_vec2_outside_segment(1,1),
		random_vec2_in_segment(1,1,3),
		random_vec2_outside_segment(1,1),
		random_vec2_in_segment(1,1,0)
	};

	add_text("The algorithm in action.", s_final, 5);

	u32 delay{};
	for (u32 p{}; p<n_point; p++)
	{
		auto v = pts_locations[p];
		u32 s{};
		for (; s<4; s++)
		{
			vec2_f32 t = { disk_translations[s].x, disk_translations[s].y };
			if (dot(v - t, v - t) <= 1)
				break;
		}
		
		f32 pt_large = 3.;
		set_translation(id_point+p, s_final, n_seconds-1, center+vec3_f32(scale*v.x,scale*v.y,0));
		translate(id_point+p, s_final+p+delay, n_seconds-1, 7*on_top);
		dilate(id_point+p, s_final+p+delay, s_final+p+1+delay, pt_large);
		fade_in(id_point+p, s_final+p+delay, s_final+p+1+delay, 0,1);
		dilate(id_point+p, s_final+p+1+delay, n_seconds-1, 1./pt_large);
		if(s < 4)
		{
			delay++;
			s = (s + 2) % 4;
			translate(id_segment+s, s_final+p-1+delay, n_seconds-1, 4*on_top);
			fade_in(id_segment+s, s_final+p+delay, s_final+p+1+delay, 0,1);
			dilate(id_point2+p, s_final+p+delay, s_final+p+1+delay, pt_large);

			set_rgb(id_point2+p, s_final-2, n_seconds-1, segment_colors[s]);
			set_translation(id_point2+p, s_final-1, n_seconds-1, center+vec3_f32(scale*v.x,scale*v.y,0)+scale*disk_translations[s]);
			translate(id_point2+p, s_final+p+delay, n_seconds-1, 7*on_top);
			fade_in(id_point2+p, s_final+p+delay, s_final+p+1+delay, 0,1);



			translate(id_segment+s, s_final+p+2+delay, n_seconds-1, -4*on_top);
			fade_in(id_segment+s, s_final+p+2+delay, s_final+p+3+delay, 1,0);
			dilate(id_point2+p, s_final+p+1+delay, n_seconds-1, 1./pt_large);
			delay++;
		}
	}
	u32 s_final_image = s_final + delay + n_point+3;

	add_text("The final samples.", s_final_image, 5);
	for (u32 p{}; p<n_point2; p++)
	{
		set_rgb(id_point2+p, s_final_image, n_seconds-1, pt_color);
	}
	translate(id_square, s_final_image,n_seconds-1, -2*on_top);
	fade_in(id_square, s_final_image,s_final_image,1,0);

	mesh meshes[] = { disk0, disk0, disk0, disk0, disk0, square, segment, segment, segment, segment, 
			          point, point, point, point, point, point, point, point, point, point, 
			          point, point, point, point, point, point, point, point, point, point, 
			          point, point, point, point, point, point, point, point, point, point, 
					  point, point, point, point, point, point, point, point, point, point };

	mesh buffer = unify(meshes, mesh_table, n_meshes, arena0);

	storage_info storages_temp[] =
	{
		{.binding = 0, .size = buffer.n_vertices * sizeof(f32)*2, .data = reinterpret_cast<u32*>(buffer.vertices)},
		{.binding = 1, .size = buffer.n_vertices * sizeof(f32)*4, .data = reinterpret_cast<u32*>(buffer.colors)},
		{.binding = 2, .size = buffer.n_vertices * sizeof(u32)*1, .data = reinterpret_cast<u32*>(buffer.mesh_indices)},
		{.binding = 3, .size = n_meshes * sizeof(per_frame_data) ,.data = reinterpret_cast<u32*>(keyframes)},
	};
	u32 n_storages = sizeof(storages_temp) / sizeof(storage_info);

	storage_info* storages = alloc_n<storage_info>(arena0, n_storages);
	memcpy(storages, storages_temp, sizeof(storages_temp));

	opengl_context context_mesh
	{
		.program		= compile_shaders( R"(
												#version 460 core

												vec3 quat_mult(vec4 q, vec3 v)
												{
												   return v + 2*cross(q.yzw, (q.x*v + cross(q.yzw, v)));
												}

												layout(std140, binding = 0) uniform camera{
													mat4 cam;
												};

												struct per_frame_data 
												{
													vec4 quat;
													vec3 translation;
													float scale;
													vec4 color;
												}; 

												layout(std430, binding = 0) restrict readonly buffer vertices {
													vec2 in_vertices[];
												};

												layout(std430, binding = 1) restrict readonly buffer colors {
													vec4 in_colors[];
												};

												layout(std430, binding = 2) restrict readonly buffer meshids {
													unsigned int in_meshid[];
												};

												layout(std430, binding = 3) restrict readonly buffer per_frame_datas{
													per_frame_data pfd[];
												};


												layout (location=0) out vec4 out_color;

												void main() 
												{
													unsigned int id = in_meshid[gl_VertexID];
													vec3 pos = quat_mult(pfd[id].quat, vec3(pfd[id].scale * in_vertices[gl_VertexID], 0)) + pfd[id].translation;
													gl_Position = cam*vec4(pos, 1.0);
													out_color = pfd[id].color * in_colors[gl_VertexID];	
												})", 

												R"(
												#version 460 core

												layout (location=0) in vec4  out_color;
												layout (location=0) out vec4 color;
												void main(void)
												{
													color = out_color;
												    float gamma_inverse = 1./2.2;
													color.rgb = pow(color.rgb, vec3(gamma_inverse));
												}

												)"),
		.mode	= GL_TRIANGLES,
		.first	= 0,
		.count  = buffer.n_indices,
		.draw_mode		= opengl_context::draw_mode::elements,
		.vao	= create_vao(),
		.ebo      = create_buffer(sizeof(u32)*buffer.n_indices),
		.uniform  = create_buffer(sizeof(mat4_f32), GL_DYNAMIC_DRAW),
		.n_storages = n_storages,
		.storages   =   storages
	};

	for (u32 u{}; u < n_storages; u++)
	{
		auto& storage = context_mesh.storages[u];
		storage.buffer = buffer_storage(storage.size, storage.data, GL_DYNAMIC_STORAGE_BIT);
		bind_storage_buffer(storage.buffer, storage.binding);
	}


	map_buffer(buffer.indices, sizeof(u32)*buffer.n_indices, context_mesh.ebo);
	map_buffer(camera.m,   sizeof(mat4_f32), context_mesh.uniform);
	bind_uniform_block(context_mesh.uniform, 0);
	bind_ebo(&context_mesh);

	return scene
			{
				.n_seconds = n_seconds,
				.n_meshes = n_meshes,
				.mesh_table = mesh_table,
				.data= keyframes,
				.context =context_mesh,
				.data2= font_times,
				.fonts = fonts,
				.rate = 1.
			};
}


int main(void)
{
	auto window = create_window(WIDTH, HEIGHT, "Disk sampling");

	glfwSetKeyCallback(window, key_callback);
	glfwSetScrollCallback(window, scroll_callback);

	glViewport(0, 0, WIDTH, HEIGHT);
	glDisable(GL_PROGRAM_POINT_SIZE);
	glPointSize(4.);
	glLineWidth(3.);
	const GLfloat grey[] = { 0.1f, 0.1f, 0.1f, 1.0f };
	const GLfloat bg_blog[] = { 246. / 255, 241. / 255, 241. / 255,1.f };
	const float far_value = 1.0f;
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  


	u32 fps = 140;
	u32 frame{};

	auto scene = init_scene();

	auto advance_frame = [&frame, fps=(u32)(fps/scene.rate), duration = scene.n_seconds]() mutable
		{
			if (++frame == fps)
			{
				frame = 0;
				if (++second == duration)
					second = 0;
			}
		};
	mode = pause;

    while (!glfwWindowShouldClose(window)) 
	{
		// Input 
		event_loop();

        glClearBufferfv(GL_COLOR, 0, grey);
		glClearBufferfv(GL_DEPTH, 0, &far_value);

		char title[256];
		sprintf(title, "%s, %u, %u", "Disk Sampling Animation", second, frame);
		glfwSetWindowTitle(window, title);

		
		auto cam = cur_cam();

		per_frame_data *interpolated_frames= alloc_n<per_frame_data>(arena1, scene.n_meshes);
		for (u32 u{}; u<scene.n_meshes; u++)
		{
			per_frame_data a = scene.data[second * scene.n_meshes + u];
			per_frame_data b = scene.data[std::min(scene.n_seconds - 1, second + 1) * scene.n_meshes + u];
			interpolated_frames[u] = interpolate(a, b, (f32)frame / (fps/scene.rate-1));
		}



		for (u32 u{}; u < scene.context.n_storages; u++)
		{
			auto& storage = scene.context.storages[u];
			bind_storage_buffer(storage.buffer, storage.binding);
		}

		update_buffer_storage(scene.context.storages[3].buffer,  sizeof(per_frame_data)*scene.n_meshes, interpolated_frames);
		bind_uniform_block(scene.context.uniform, 0);
		map_buffer(cam.m,   sizeof(mat4_f32), scene.context.uniform);
		draw(&scene.context);

		for (u32 u{}; u < scene.fonts.context.n_storages; u++)
		{
			auto& storage = scene.fonts.context.storages[u];
			bind_storage_buffer(storage.buffer, storage.binding);
		}

		cam = mul(cam, translation(vec3_f32(0, 0, 1)));
		bind_uniform_block(scene.fonts.context.uniform, 0);
		map_buffer(cam.m,   sizeof(mat4_f32), scene.fonts.context.uniform);
		glBindTextures(0, 1, &scene.fonts.context.tex);
		draw_indexed(&scene.fonts.context, scene.data2[second].first, scene.data2[second].count);
		
		glfwSwapBuffers(window);


		if(mode==running) advance_frame();

		arena1.clear();
	}

    glfwTerminate();


    return 0;
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
	else if (action == GLFW_RELEASE)
		register_event({ key, action });
}

void scroll_callback(GLFWwindow* window, double x, double y)
{
	scroll_offset.x += scroll_speed*x;
	scroll_offset.y += scroll_speed*y;
}


void register_event(key_event event)
{
	events.queue[events.end] = event;
	events.end = (events.end + 1u) % n_queue;
}

bool has_events()
{
	return events.begin != events.end;
}

void event_loop()
{
	glfwPollEvents();

	while (has_events())
	{
		auto event = poll_event();
		if (event.key == GLFW_KEY_R)
		{
		}
		else if (event.key == GLFW_KEY_T)
		{
		}
		else if (event.key == GLFW_KEY_ENTER)
		{
		}
		else if (event.key == GLFW_KEY_UP)
		{
		}
		else if (event.key == GLFW_KEY_DOWN)
		{
		}
		else if (event.key == GLFW_KEY_RIGHT)
		{
			second++;
		}
		else if (event.key == GLFW_KEY_LEFT)
		{
			if(second) second--;
		}
		else if (event.key == GLFW_KEY_SPACE)
		{
			if (mode == running) mode = pause;
			else                 mode = running;
		}
		else if (event.key == GLFW_KEY_A)
		{
			awds_offset.x -= awds_speed;
		}
		else if (event.key == GLFW_KEY_W)
		{
			awds_offset.y += awds_speed;
		}
		else if (event.key == GLFW_KEY_S)
		{
			awds_offset.y -= awds_speed;
		}
		else if (event.key == GLFW_KEY_D)
		{
			awds_offset.x += awds_speed;
		}
	}
}

key_event poll_event()
{
	assert(has_events() && "polled for events but no events");
	key_event event = events.queue[events.begin];
	events.begin = (events.begin + 1u) % n_queue;
	return event;
}
