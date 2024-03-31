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
#include <numeric>
#include <thread>

using u8  = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using f32 = float;
using f64 = double;

#include "quaternions.h"
#include "common.h"
#include "vec_math.h"
#include "random_int.h"

#include "memory_allocator.cpp"
#include "gl_backend.cpp"
#include "samplers.cpp"
#include "warps.cpp"
#include "statistics.cpp"
#include "colors.cpp"
#include "primitives_2d.cpp"
#include "fontrendering.cpp"
#include "canvas.cpp"
#include "test_functions.cpp"

const GLuint WIDTH = 4*1024, HEIGHT = 3*1024;
const f32 ratio = (f32)HEIGHT / WIDTH;
vec2_f32 cam_size{ (f32)WIDTH, (f32)HEIGHT};
vec2_f32 scroll_offset = vec2_f32(0);
vec2_f32 awds_offset = vec2_f32(0);
f32 scroll_speed = 300.f;
f32 awds_speed = 100.f;

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

struct scene
{
	u32 n_seconds;
	u32 n_meshes;
	u32 *mesh_table;
	per_frame_data *data;
	opengl_context context;

	font_rendering fonts;

	f32 rate;
};

mat4_f32 cur_cam()
{
	vec2_f32 cam_offs = scroll_offset+awds_offset;
	mat4_f32 p = orthographic(cam_offs.x,cam_offs.x+cam_size.x, cam_offs.y,cam_offs.y+cam_size.y,-2,2);
	mat4_f32 v = scale(1.f);
	mat4_f32 camera = mul(p, v);
	return camera;
}


using disk_func = f32(*)(f32, f32);
using warp_func = void(*)(f32* out, f32* in, u32& n_out, u32 n_in, f32 a, f32 b, f32 R);
using pts_func = points(*)(u32 m, u32 n, f32 a, f32 b, f32* vertices);
using ran_pts_func = points(*)(u32 m, u32 n, f32 a, f32 b, u32 ran, f32* vertices);
using scramble_func = u32(*)(u32 u, u32 seed);


//Error plot
const u32 m0 = 8;
const u32 m1 = 80;
const u32 m = m1 - m0 + 1;
// Average version
const u32 n_ran = 100;


struct plotvals 
{
	u32 n;
	f32 *u;
	f32 *d;
};

pts_func     samplers[] = { grid, halton, hammersley, sobol};
ran_pts_func random_samplers[] = { jitter, random,sobol_scrambled, halton_scrambled, hammersley_scrambled};//  , halton_permute, hammersley_permute, sobol_permute, halton_fast_owen, hammersley_fast_owen, sobol_fast_owen };
warp_func  	 warpers[] = { polar_warp, concentric_warp, rejection_warp, adoption_warp };
disk_func 	 testers[] = { bilinear, checkerboard, spiral, head, drum, square2_in_disk, gaussian, quatered_disk, concentric };
f32       	 testres[] = { -1, -1, spiral_integral, -1, -1, square_integral, gaussian_integral, quatered_disk_integral, concentric_integral };
f32       	 mult[] = { 1.f, 1.f, std::sqrt(4 / pi_f), std::sqrt(2.f - pi_f / 2) };

const char* warpers_labels[] = { "polar", "concentric", "rejection", "adoption", "n^-1/2", "n^-0.75" };
const char* samplers_labels[] = { "grid", "jitter", "halton", "hammersley", "sobol", "random" };
const char* random_samplers_labels[] = { "jitter", "random", "sobol", "halton", "hammersley" };//"halton_permute", "hammersley_permute", "sobol_permute", "halton_fast_owen", "hammersley_fast_owen", "sobol_fast_owen" };
const char* testers_labels[] = { "bilinear", "checker", "spiral", "head", "drum", "square", "gaussian", "quatered", "concentric" };

const u32 n_s = sizeof(samplers) / sizeof(pts_func);
const u32 n_rs = sizeof(random_samplers) / sizeof(ran_pts_func);
const u32 n_w = sizeof(warpers) / sizeof(warp_func);
const u32 n_t = sizeof(testers) / sizeof(disk_func);

plotvals plots[(n_rs)*n_w*n_t];
plotvals plots_min[(n_rs)*n_w*n_t];





f32 abs_err(points p, f32 mean, disk_func f)
{
	f64 m{};
	for (u32 u{}; u < p.n_vertices; u++)
	{
		f64 z = f(p.vertices[2 * u], p.vertices[2 * u + 1]);
		m += z;
	}
	return (f32)std::abs(m / p.n_vertices-mean);
}

void calcate_points(plotvals *vals,plotvals *vals_min, arena& arena, warp_func warp, ran_pts_func f, disk_func test, f32 res)
{
	u32 n{ m1 * m1 * 2 };
	vals->u = alloc_n<f32>(arena, m);
	vals->d = alloc_n<f32>(arena, m);
	for (u32 k{ m0 }; k <= m1; k++) vals->u[k - m0] = k;
	vals_min->u = alloc_n<f32>(arena, m);
	vals_min->d = alloc_n<f32>(arena, m);
	for (u32 k{ m0 }; k <= m1; k++) vals_min->u[k - m0] = k;

	points out{ .vertices = alloc_n<f32>(arena, 2*n), .n_vertices = n };
	f32* vertices = alloc_n<f32>(arena, 2*n);
	f32 min_err = 1./(1<<20);

	f64* a = alloc_n<f64>(arena, n_ran);

	for (u32 i{ m0 }; i <= m1; i++)
	{
		u32 u = vals->u[i - m0];
		u64 uu{};
		for (u32 j{}; j < n_ran; j++)
		{
			points pts = f(u, u, 2.f, 2.f, pcg(1 + j), vertices);
			warp(out.vertices, pts.vertices, out.n_vertices, pts.n_vertices, 2, 2, 1);
			a[j] = abs_err(out, res, test);
			uu += out.n_vertices;
			out.n_vertices = n;
		}
		vals->u[i - m0] = (f64)uu / n_ran;
		auto sts = mean_statistics(a, n_ran);
		vals->d[i - m0] = std::max((f32)sts.mean, min_err);
		vals_min->u[i - m0] = (f64)uu / n_ran;
		vals_min->d[i - m0] = std::max((f32)sts.min, min_err);
	}
}

f32 slope(const f32* x, const f32* y, size_t size) 
{
    f32 x_sum = std::accumulate(x, x + size, 0.0f);
    f32 y_sum = std::accumulate(y, y + size, 0.0f);
    f32 xy_sum = std::inner_product(x, x + size, y, 0.0f);
    f32 xx_sum = std::inner_product(x, x + size, x, 0.0f);

    f32 b = (size * xy_sum - x_sum * y_sum) / (size * xx_sum - x_sum * x_sum);
    return b;
}

f32 log_slope(const f32* x, const f32* y, size_t size)
{
	f32* xl = alloc_n<f32>(arena0, size);
	f32* yl = alloc_n<f32>(arena0, size);
	for (u32 u{}; u<size; u++)
	{
		xl[u] = std::log(x[u] > 0 ? x[u] : 1e-20);
		yl[u] = std::log(y[u] > 0 ? y[u] : 1e-20);
	}
	return slope(xl, yl, size);
}

mesh untexture(mesh mesh)
{
	mesh.uv = nullptr;
	return mesh;
}

void setup_plots(graph_canvas& canvas, font_rendering& fonts, u32 t, u32 s)
{
		plotvals asymptotic_line[] = {
			{.n = m, .u = alloc_n<f32>(arena0, m), .d = alloc_n<f32>(arena0, m) },
			{.n = m, .u = alloc_n<f32>(arena0, m), .d = alloc_n<f32>(arena0, m) }
			};

		for (u32 k{ m0 }; k <= m1; k++)
		{
			asymptotic_line[0].u[k - m0] = k*k;
			asymptotic_line[0].d[k - m0] = 1. / k;
			asymptotic_line[1].u[k - m0] = k*k;
			asymptotic_line[1].d[k - m0] = std::pow(k * k, -0.75);
		}

		f32 canvas_offset = (s*n_t+t) * canvas.size.y * 1.2;
		canvas.add_background(0, -canvas_offset, arena0);
		canvas.add_axis(0,-canvas_offset, arena0);
		vec2_f32 relative_legend_pos = 0.7 * canvas.size + vec2_f32(-25,+210);
		vec2_f32 legend_pos = relative_legend_pos + canvas.pos + vec2_f32(0, -canvas_offset);
		canvas.add_legend(0.7 * canvas.size.x, 0.7 * canvas.size.y -canvas_offset, 0.4*canvas.size, arena0);


		std::string label2 = random_samplers_labels[s];
		label2 += std::string(" ") + testers_labels[t];
		fonts.add_text(label2.c_str(), legend_pos.x-30, legend_pos.y+50, 0.8, 0.8, black, arena0);

		f32 min_x{ 1.e+10 }, min_y{ 1.e+10 }, max_x{}, max_y{};

		for (u32 w{}; w <= n_w+1; w++)
		{
			f32* x, * y;
			u32 u = w + n_w*(s + t * n_rs);
			if(w<n_w){
				x = plots[u].u;
				y = plots[u].d;
			}else
			{
				x = asymptotic_line[w-n_w].u;
				y = asymptotic_line[w-n_w].d;
			}
			min_x = std::min(min_x, *std::min_element(x, x + m));
			min_y = std::min(min_y, *std::min_element(y, y + m));
			max_x = std::max(max_x, *std::max_element(x, x + m));
			max_y = std::max(max_y, *std::max_element(y, y + m));
		}

		
		if(min_x == 0) min_x = 1.e-15f;
		if(min_y == 0) min_y = 1.e-15f;

		f32 x_range = std::log(max_x/min_x);
		f32 y_range = std::log(max_y/min_y);

		for (u32 w{}; w < n_w+2; w++)
		{
			u32 u = w + n_w*(s + t * n_rs);
			u32 col_id = w;
			f32 alpha;
			if (w < n_w)
			{
				canvas.add_graph(plots[u].u, plots[u].d, m, graph_canvas::loglog, vec2_f32(x_range, y_range), vec2_f32(min_x, min_y), vec2_f32(0, -canvas_offset), col_id, arena0);
				alpha = log_slope(plots[u].u, plots[u].d, m);
			}
			else
			{
				col_id = black;
				f32 y_offset = (w - n_w) == 0 ? 200 : 100;
				canvas.add_graph(asymptotic_line[w-n_w].u, asymptotic_line[w-n_w].d, m, graph_canvas::loglog, vec2_f32(x_range, y_range), vec2_f32(min_x, min_y), vec2_f32(0, -canvas_offset-y_offset), col_id, arena0);
			}



			std::string label = warpers_labels[w];
			if (w < n_w)
			{
				char alpha_label[15];
				snprintf(alpha_label, sizeof(alpha_label), " (n^ %.2f)", alpha);
				label += alpha_label;
			}
			f32 line[4] = { relative_legend_pos.x - 50, relative_legend_pos.y+10 - (f32)50 * w - canvas_offset , relative_legend_pos.x-4, relative_legend_pos.y+10 - (f32)50 * w - canvas_offset, };
			vec4_f32 line_color[2];
			std::fill(line_color, line_color + 2, colors[col_id]);
			canvas.add_line_mesh({ .vertices = line, .n_vertices = 2, .colors = &line_color[0].x }, 0.6, arena0);
			fonts.add_text(label.c_str(), legend_pos.x, legend_pos.y - (f32)50 * w, 0.6, 0.6, black, arena0);
		}

		{
			u32 n_x_tags = 10;
			u32 n_y_tags = 10;

			f32* x_tags = alloc_n<f32>(arena0, n_x_tags + 2);
			f32* y_tags = alloc_n<f32>(arena0, n_y_tags + 2);

			u32 uu = n_w * (s + t * n_rs);
			//f32* x = plots[uu].u;
			//f32* y = plots[uu].d;
			f32* x= alloc_n<f32>(arena0, n_x_tags+1);
			f32* y= alloc_n<f32>(arena0, n_y_tags+1);
			for(u32 u{1}; u<n_x_tags+1; u++)
			{
				x[u] = (f32)(1 << (u+2));
				y[u] = std::clamp(1.f / x[u], min_y, max_y);
				x[u] *= x[u];
				x[u] = std::clamp(x[u], min_x, max_x);
			}

			for (u32 u{ 1 }; u < n_x_tags + 1; u++)
			{
				//u32 k = m * u / (n_x_tags+1);
				//x_tags[u] = (x[k] > 0 ? std::log(x[k] / min_x) / x_range * canvas.size.x : 0);
				x_tags[u] = std::log(x[u] / min_x) / x_range * canvas.size.x;
				char label[50];
				snprintf(label, sizeof(label), "%.0f", x[u]);
				//snprintf(label, sizeof(label), "%.0f", x[k]);
				fonts.add_text(label, canvas.pos.x+x_tags[u], canvas.pos.y-30-canvas_offset, .4, .4, black, arena0);
			}
			for (u32 u{ 1 }; u < n_y_tags + 1; u++)
			{
				//u32 k = m * u / (n_y_tags+1);
				//y_tags[u] = (y[k] > 0 ? std::log(y[k] / min_y) / y_range * canvas.size.y : 0);
				y_tags[u] = std::log(y[u] / min_y) / y_range * canvas.size.y;
				char label[10];
				snprintf(label, sizeof(label), "%.1e", y[u]);
				fonts.add_text(label, canvas.pos.x-80, canvas.pos.y+y_tags[u]- canvas_offset, .4, .4, black, arena0);
			}
			x_tags[0] = 0;
			y_tags[0] = 0;
			x_tags[n_x_tags + 1] = canvas.size.x;
			y_tags[n_y_tags + 1] = canvas.size.y;
			canvas.add_tags(x_tags, y_tags, n_x_tags+2, n_y_tags+2, vec2_f32(0, -canvas_offset), arena0);
		}



}

void setup_sampled_disk_square(graph_canvas& canvas, u32 sampler, u32 u)
{
	u32 s = sampler;
	f32 x_offset = 700;
	u32 n = 2 * u * u;

	vec2_f32 square_pos = { 100, 3000 + s*1000 };
	f32 scale = 500;
	canvas.add_shape(untexture(rectangle(1, 1, arena0)), vec3_f32(square_pos, 0.5), scale, arena0);
	points out{ .vertices = alloc_n<f32>(arena0, 2 * n), .n_vertices = n };
	f32* vertices = alloc_n<f32>(arena0, 2* u * u);
	auto f = samplers[s];
	points in = f(u, u, 1, 1, vertices);
	canvas.add_points(in, vec3_f32(square_pos, 0.6), scale, arena0);

	for (u32 w{}; w < n_w; w++) 
	{
		f32 a = std::sqrt(pi_f) * mult[w];
		vec2_f32 disk_pos = { 1000+x_offset*w, 3000 +  scale / 2 + s*1000 };
		auto warp = warpers[w];
		warp(out.vertices, in.vertices, out.n_vertices, in.n_vertices, 1, 1, 1/a);
		canvas.add_shape(untexture(disk(1/a, 100, arena0)), vec3_f32(disk_pos, 0.5), scale, arena0);
		canvas.add_points(out, vec3_f32(disk_pos, 0.6), scale, arena0);
		out.n_vertices = n;
	}
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
	const float far_value = 1.0f;
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  

	head_img = load_image("data/head_image.png");
	drum_img = load_image("data/drum_image.png");

	testres[0] = bilinear_integral();
	testres[1] = checkerboard_integral();
	testres[3] = head_integral();
	testres[4] = drum_integral();
	//for (u32 t{}; t < n_t; t++) testres[t] = function_integral(testers[t]);


	std::thread threads[n_rs * n_t * n_w];
	arena arena_t[n_rs * n_t * n_w];
	for (u32 s{}; s < n_rs; s++) for (u32 t{}; t < n_t; t++) for (u32 w{}; w < n_w; w++)
	{
		u32 u = w + n_w*(s + t * n_rs);
		arena_t[u] = subarena(arena0, 1 * megabyte);
		threads[u] = std::thread(calcate_points, plots + u, plots_min+u, std::ref(arena_t[u]), warpers[w], random_samplers[s], testers[t], testres[t]);
	}
	for (u32 s{}; s < n_rs; s++) for (u32 t{}; t < n_t; t++) for (u32 w{}; w < n_w; w++)
	{
		u32 u = w + n_w*(s + t * n_rs);
		threads[u].join();
	}

	mesh graph{ .vertices = alloc_n<f32>(arena0, 2 * (m - 1) * 2), .n_vertices = (m - 1) * 2, .colors = alloc_n<f32>(arena0, 4 * (m - 1) * 2) };
	auto center = 0.5 * cam_size;
	auto canvas = graph_canvas(vec2_f32(1024, 1024), center, arena0);
	
	font_rendering fonts;
	fonts.init_fonts(arena0);

	for (u32 s{}; s < n_rs; s++) for (u32 t{}; t<n_t; t++)
	{
		setup_plots(canvas, fonts, t, s);
	}

	for (u32 t{}; t < n_t; t++)
	{
		auto mesh = disk(300., 300, arena0);
		mesh.texture = draw(testers[t], 1024, arena0);
		canvas.add_shape(mesh, vec3_f32(500, 2000.-700.*t,0.8) ,1., arena0);
		fonts.add_text(testers_labels[t], 100, 2000.-700.*t, 0.8, 0.8, white, arena0);
	}

	for (u32 s{}; s < n_w; s++)
	{
		setup_sampled_disk_square(canvas, s, 32);
	}

    while (!glfwWindowShouldClose(window)) 
	{
		event_loop();

        glClearBufferfv(GL_COLOR, 0, &colors[grey].x);
		glClearBufferfv(GL_DEPTH, 0, &far_value);

		auto cam = cur_cam();
	
		canvas.draw_canvas(cam);

		cam = mul(cam, translation(vec3_f32(0, 0, 1)));
		bind_uniform_block(fonts.context.uniform, 0);
		map_buffer(cam.m,   sizeof(mat4_f32),fonts.context.uniform);
		glBindTextures(0, 1, &fonts.context.tex);
		for (u32 u{}; u < fonts.context.n_storages; u++) 
			bind_storage_buffer(fonts.context.storages[u].buffer, fonts.context.storages[u].binding);
		draw(&fonts.context);

		glfwSwapBuffers(window);

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
		}
		else if (event.key == GLFW_KEY_LEFT)
		{
		}
		else if (event.key == GLFW_KEY_SPACE)
		{
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
		else if (event.key == GLFW_KEY_PAGE_UP)
		{
			awds_offset.y += 100*awds_speed;
		}
		else if (event.key == GLFW_KEY_PAGE_DOWN)
		{
			awds_offset.y -= 100*awds_speed;
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
