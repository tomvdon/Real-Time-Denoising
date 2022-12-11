#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#include "device_launch_parameters.h"
#include <thrust/partition.h>

#define DIRECT 0
#define CACHE_FIRST_BOUNCE 0
#define SORT_MATERIAL 1
#define COMPACTION 1
#define DEPTH_OF_FIELD 0
#define ANTI_ALIASING 0
#define BOUNDING_BOX 0

#define MAX_INTERSECT_DIST 10000.f

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

bool denoise_on = true;

void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static glm::vec3* dev_dn_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

//GBuffer
static GBufferPixel* dev_gBuffer = NULL;

//BVH
static GPUBVHNode* dev_bvh_nodes = NULL;
static Tri* dev_tris = NULL;

// TODO: static variables for device memory, any extra info you need, etc
//for caching first bounce
#if CACHE_FIRST_BOUNCE
static ShadeableIntersection* dev_firstBounce = NULL;
static PathSegment* dev_first_paths = NULL;
#endif
//for tiny_obj
//static Object* dev_objects = NULL;
static Geom* dev_tinyobj = NULL;
static float* dev_denoise = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_dn_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_dn_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// TODO: initialize any extra device memeory you need
#if CACHE_FIRST_BOUNCE
	cudaMalloc(&dev_firstBounce, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_firstBounce, 0, pixelcount * sizeof(ShadeableIntersection));
	cudaMalloc(&dev_first_paths, pixelcount * sizeof(PathSegment));
#endif
	cudaMalloc(&dev_tinyobj, scene->Obj_geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_tinyobj, scene->Obj_geoms.data(), scene->Obj_geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	//GBuffer
	cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

	//BVH
	cudaMalloc(&dev_tris, scene->num_tris * sizeof(Tri));
	cudaMemcpy(dev_tris, scene->mesh_tris_sorted.data(), scene->num_tris * sizeof(Tri), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_bvh_nodes, scene->bvh_nodes_gpu.size() * sizeof(GPUBVHNode));
	cudaMemcpy(dev_bvh_nodes, scene->bvh_nodes_gpu.data(), scene->bvh_nodes_gpu.size() * sizeof(GPUBVHNode), cudaMemcpyHostToDevice);

	// Denoise image buffer
	cudaMalloc(&dev_denoise, pixelcount * 3 * sizeof(float));
	cudaMemset(dev_denoise, 0, pixelcount * 3 * sizeof(float));

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	cudaFree(dev_dn_image);
	// TODO: clean up any extra device memory you created
#if CACHE_FIRST_BOUNCE
	cudaFree(dev_firstBounce);
	cudaFree(dev_first_paths);
#endif
	cudaFree(dev_tinyobj);

	//GBuffer
	cudaFree(dev_gBuffer);

	//BVH
	cudaFree(dev_tris);
	cudaFree(dev_bvh_nodes);

	//cuDNN
	cudaFree(dev_denoise);

	checkCUDAError("pathtraceFree");
}

__host__ __device__
glm::vec2 ConcentricSampleDisk(const glm::vec2& u)
{
	glm::vec2 uOffset = 2.f * u - glm::vec2(1, 1);

	if (uOffset.x == 0 && uOffset.y == 0)
		return glm::vec2(0, 0);

	float theta, r;
	double pi = 3.14159265359;
	if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
		r = uOffset.x;
		theta = pi / 4 * (uOffset.y / uOffset.x);
	}
	else {
		r = uOffset.y;
		theta = pi / 2 - pi / 4 * (uOffset.x / uOffset.y);
	}
	return r * glm::vec2(std::cos(theta), std::sin(theta));
}

__global__ void generateGBuffer(
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	GBufferPixel* gBuffer) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{

		gBuffer[idx].t = shadeableIntersections[idx].t;
		if (gBuffer[idx].t != -1.0f)
		{
			gBuffer[idx].normal = shadeableIntersections[idx].surfaceNormal;
			gBuffer[idx].position = getPointOnRay(pathSegments[idx].ray, shadeableIntersections[idx].t);
		}
		else
		{
			gBuffer[idx].normal = glm::vec3(0.0f);
			gBuffer[idx].position = glm::vec3(0.0f);
		}

	}
}


/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;


	float jitter_x = 0.f, jitter_y = 0.f;
#if ANTI_ALIASING
	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u(-0.5, 0.5);
		jitter_x = u(rng);
		jitter_y = u(rng);

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x + jitter_x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + jitter_y - (float)cam.resolution.y * 0.5f)
		);
#else
	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#endif

#if DEPTH_OF_FIELD
		//adapted from pbrt
		if (cam.lensRadius > 0) {
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
			thrust::uniform_real_distribution<float> u101(0, 1);
			thrust::uniform_real_distribution<float> u201(0, 1);
			glm::vec2 rand(u101(rng), u201(rng));
			glm::vec2 pLens = cam.lensRadius * ConcentricSampleDisk(rand);
			float ft = cam.focalDistance / -segment.ray.direction.z;
			glm::vec3 pFocus = ft * segment.ray.direction;

			segment.ray.origin += glm::vec3(pLens.x, pLens.y, 0);
			segment.ray.direction = glm::normalize(pFocus - glm::vec3(pLens.x, pLens.y, 0));
		}
#endif
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
	}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth,
	int num_paths,
	PathSegment * pathSegments,
	Geom * geoms,
	int geoms_size,
	Geom * triangles,
	int triangle_size,
	ShadeableIntersection * intersections,
	int iter
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		glm::vec2 uv = glm::vec2(-1, -1);
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		glm::vec2 tmp_uv = glm::vec2(-1, -1);

		int materialId;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);

			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?
			else if (geom.type == TRIANGLE) {
				t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside);
			}
			else if (geom.type == MESH) {
#if BOUNDING_BOX
				t = meshIntersectionTest(geom, pathSegment.ray, triangles, triangle_size, true,
					tmp_intersect, tmp_normal, tmp_uv, outside);
#else 
				t = meshIntersectionTest(geom, pathSegment.ray, triangles, triangle_size, false,
					tmp_intersect, tmp_normal, tmp_uv, outside);
#endif 

			}
			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;


				uv = tmp_uv;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].uv = uv;

		}
	}
}

//addapted from NicholasMoon
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, Tri * tris
	, int tris_size
	, ShadeableIntersection * intersections
	, GPUBVHNode * bvh_nodes
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{

		Ray r = pathSegments[path_index].ray;

		ShadeableIntersection isect;
		isect.t = MAX_INTERSECT_DIST;

		float t;
		glm::vec3 tmp_intersect;
		glm::vec2 tmp_uv = glm::vec2(-1, -1);
		glm::vec3 tmp_normal;
		bool outside = true;

		int obj_ID = -1;

		glm::vec3 intersect_point;
		glm::vec3 normal;
		glm::vec2 uv = glm::vec2(-1, -1);
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		int matId = 1;
		if (tris_size != 0)
		{
			int stack_pointer = 0;
			int cur_node_index = 0;
			int node_stack[128];
			GPUBVHNode cur_node;
			glm::vec3 P;
			glm::vec3 s;
			float t1;
			float t2;
			float tmin;
			float tmax;
			
			while (true)
			{
				cur_node = bvh_nodes[cur_node_index];
				auto invDir = 1.f / r.direction;
				// (ray-aabb test node)
				t1 = (cur_node.AABB_min.x - r.origin.x) * invDir.x;
				t2 = (cur_node.AABB_max.x - r.origin.x) * invDir.x;
				tmin = glm::min(t1, t2);
				tmax = glm::max(t1, t2);
				t1 = (cur_node.AABB_min.y - r.origin.y) * invDir.y;
				t2 = (cur_node.AABB_max.y - r.origin.y) * invDir.y;
				tmin = glm::max(tmin, glm::min(t1, t2));
				tmax = glm::min(tmax, glm::max(t1, t2));
				t1 = (cur_node.AABB_min.z - r.origin.z) * invDir.z;
				t2 = (cur_node.AABB_max.z - r.origin.z) * invDir.z;
				tmin = glm::max(tmin, glm::min(t1, t2));
				tmax = glm::min(tmax, glm::max(t1, t2));
				if (tmax >= tmin) {
					// we intersected AABB
					if (cur_node.tri_index != -1) {
						// this is leaf node
						// triangle intersection test
						Tri tri = tris[cur_node.tri_index];

						//t = triangleIntersectionTest(tri, r, tmp_intersect, tmp_normal, tmp_uv, outside);


						t = glm::dot(tri.plane_normal, (tri.pos[0] - r.origin)) / glm::dot(tri.plane_normal, r.direction);
						if (t >= -0.0001f) {
							P = r.origin + t * r.direction;
							// barycentric coords
							s = glm::vec3(glm::length(glm::cross(P - tri.pos[1], P - tri.pos[2])),
								glm::length(glm::cross(P - tri.pos[2], P - tri.pos[0])),
								glm::length(glm::cross(P - tri.pos[0], P - tri.pos[1]))) / tri.S;

							if (s.x >= -0.0001f && s.x <= 1.0001f && s.y >= -0.0001f && s.y <= 1.0001f &&
								s.z >= -0.0001f && s.z <= 1.0001f && (s.x + s.y + s.z <= 1.0001f) && (s.x + s.y + s.z >= -0.0001f) && t_min > t) {
								t_min = t;
								hit_geom_index = cur_node.tri_index;
								normal = glm::normalize(s.x * tri.normal[0] + s.y * tri.normal[1] + s.z * tri.normal[2]);
								matId = tri.mat_ID;
							}
						}

						// if last node in tree, we are done
						if (stack_pointer == 0) {
							break;
						}
						// otherwise need to check rest of the things in the stack
						stack_pointer--;
						cur_node_index = node_stack[stack_pointer];
					}
					else {
						node_stack[stack_pointer] = cur_node.offset_to_second_child;
						stack_pointer++;
						cur_node_index++;
					}
				}
				else {
					// didn't intersect AABB, remove from stack
					if (stack_pointer == 0) {
						break;
					}
					stack_pointer--;
					cur_node_index = node_stack[stack_pointer];
				}
			}
		}

		for (int i = 0; i < geoms_size; ++i)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside);

			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?
			//else if (geom.type == TRIANGLE) {
			//	t = triangleIntersectionTest(geom, r, tmp_intersect, tmp_normal, tmp_uv, outside);
			//}

			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
				uv = tmp_uv;
			}

		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			if (hit_geom_index >= geoms_size)
				intersections[path_index].materialId = matId;
			else
				intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].uv = uv;

		}
	}
}




// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance);
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
				pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
				pathSegments[idx].color *= u01(rng); // apply some noise because why not
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
		}
	}
}

//__global__ void kernSimpleShade(
//	int iter, 
//	int num_paths, 
//	int depth,
//	ShadeableIntersection* shadeableIntersections, 
//	PathSegment* pathSegments,         
//	Material* materials)
//{
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	if (idx < num_paths)
//	{
//		ShadeableIntersection &intersection = shadeableIntersections[idx];
//		PathSegment &ps = pathSegments[idx];
//		if (ps.remainingBounces <= 0) return;
//
//		if (intersection.t > 0.0f) { // if the intersection exists...
//			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
//			thrust::uniform_real_distribution<float> u01(0, 1);
//
//			Material &material = materials[intersection.materialId];
//			glm::vec3 materialColor = material.color;
//
//			if (material.emittance > 0.0f) {
//				ps.remainingBounces = 0;
//				ps.color *= (materialColor * material.emittance);
//			}
//			
//			else{
//				glm::vec3 intersect = getPointOnRay(ps.ray, intersection.t);
//				scatterRay(ps, intersection, material, rng);
//				//scatterRay(ps, intersect, intersection.surfaceNormal, material, rng);
//				ps.remainingBounces--;
//			}
//		}
//		else {
//			ps.remainingBounces = 0;
//			ps.color = glm::vec3(0);
//		}
//	}
//}

__global__ void kernSimpleShade(
	int iter,
	int num_paths,
	int depth,
	ShadeableIntersection * shadeableIntersections,
	PathSegment * pathSegments,
	Material * materials,
	glm::vec3 camPos)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection& intersection = shadeableIntersections[idx];
		PathSegment& ps = pathSegments[idx];
		if (ps.remainingBounces <= 0) return;

		if (intersection.t > 0.0f) { // if the intersection exists...
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material& material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			if (material.emittance > 0.0f) {
				ps.remainingBounces = 0;
				ps.color *= (materialColor * material.emittance);
			}

			else {
				if (pathSegments->remainingBounces == 1) {
#if DIRECT
					//hardcode 2 lights, and randomly get 0/1
					thrust::uniform_real_distribution<float> u02(0, 2);

					thrust::uniform_real_distribution<float> u1(0, 3);
					//thrust::uniform_real_distribution<float> u2(0, 3);
					glm::vec3 lightPos = glm::vec3(0, 10, 0);
					int whichlight = int(u02(rng));
					if (whichlight == 0) {
						float x = u1(rng);
						float z = u1(rng);
						lightPos.x = x;
						lightPos.z = z;
					}
					else {
						float y = u1(rng);
						float z = u1(rng);
						lightPos.x = 0;
						lightPos.y = y;
						lightPos.z = z;
					}

					glm::vec3 intersect = getPointOnRay(ps.ray, intersection.t);
					glm::vec3 normal = intersection.surfaceNormal;
					glm::vec2 uv = intersection.uv;
					Ray r;
					r.origin = intersect;
					r.direction = glm::normalize(lightPos - intersect);
					ps.ray = r;
					ps.color *= material.color;
					ps.remainingBounces--;
#else
					glm::vec3 intersect = getPointOnRay(ps.ray, intersection.t);
					scatterRay(ps, intersection, material, rng, camPos);
					//scatterRay(ps, intersect, intersection.surfaceNormal, material, rng);
					ps.remainingBounces--;
#endif 
				}
				else {
					glm::vec3 intersect = getPointOnRay(ps.ray, intersection.t);
					scatterRay(ps, intersection, material, rng, camPos);
					//scatterRay(ps, intersect, intersection.surfaceNormal, material, rng);
					ps.remainingBounces--;
				}
				//glm::vec3 intersect = getPointOnRay(ps.ray, intersection.t);
				//scatterRay(ps, intersection, material, rng, camPos);
				////scatterRay(ps, intersect, intersection.surfaceNormal, material, rng);
				//ps.remainingBounces--;
			}
		}
		else {
			ps.remainingBounces = 0;
			ps.color = glm::vec3(0);
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

//comparators
struct isZero {
	__host__ __device__
		bool operator()(const PathSegment& ps) {
		return ps.remainingBounces;
	}
};

struct compareMaterial {
	__host__ __device__
		bool operator()(const ShadeableIntersection& i1, const ShadeableIntersection& i2) {
		return i1.materialId < i2.materialId;
	}
};

__global__ void vecToBuff(int numPixels, glm::vec3* image, float* denoise, glm::ivec2 resolution, int iter) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::vec3 color;
		color.x = glm::clamp((pix.x / iter), 0.f, 1.f);
		color.y = glm::clamp((pix.y / iter), 0.f, 1.f);
		color.z = glm::clamp((pix.z / iter), 0.f, 1.f);

		// Each thread writes one pixel location in the texture (textel)
		denoise[index] = color.x;
		denoise[index + numPixels] = color.y;
		denoise[index + 2 * numPixels] = color.z;
	}
}

__global__ void buffToVec(int numPixels, glm::vec3* image, float* denoise, glm::ivec2 resolution, int iter) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);

		glm::vec3 color = glm::vec3(0,0,0);
		color.x = glm::clamp(denoise[index], 0.f, 1.f);
		color.y = glm::clamp(denoise[index + numPixels], 0.f, 1.f);
		color.z = glm::clamp(denoise[index + 2 * numPixels], 0.f, 1.f);

		// Each thread writes one pixel location in the texture (textel)
		image[index] = color;
	}
}

void dnCNN(cudnnHandle_t handle, std::vector<layer>& model) {
	auto setup_start = chrono::high_resolution_clock::now();

	const Camera& cam = hst_scene->state.camera;

	tensor input = tensor(1, 3, cam.resolution.y, cam.resolution.x);
	// Use copy of image so that we can subtract noise from original image
	float* dev_input;
	cudaMalloc(&dev_input, sizeof(float) * 64 * cam.resolution.y * cam.resolution.x);
	cudaMemcpy(dev_input, dev_denoise, sizeof(float) * 3 * cam.resolution.y * cam.resolution.x, cudaMemcpyDeviceToDevice);
	input.dev = dev_input;
	//input.host = h_input;

	tensor output = tensor(1, 64, cam.resolution.y, cam.resolution.x);
	float* dev_output;
	cudaMalloc(&dev_output, sizeof(float) * 64 * cam.resolution.y * cam.resolution.x);
	output.dev = dev_output;

	const float alpha = 1.f, beta = 0.f;
	cudnnActivationDescriptor_t activation;
	checkCUDNN(cudnnCreateActivationDescriptor(&activation));
	checkCUDNN(cudnnSetActivationDescriptor(activation,
		CUDNN_ACTIVATION_RELU,
		CUDNN_PROPAGATE_NAN,
		0.0));
	auto setup_end = chrono::high_resolution_clock::now();
	auto setup_duration = std::chrono::duration_cast<std::chrono::microseconds>(setup_end - setup_start);
	std::cout << "setup: " << setup_duration.count() << std::endl;

	auto nn_start = chrono::high_resolution_clock::now();
	for (int i = 0; i < 20; ++i) {
		setup_start = chrono::high_resolution_clock::now();
		// Load filter
		int in_chan = 64;
		int out_chan = 64;
		if (i == 0) {
			in_chan = 3;
		}
		if (i == 19) {
			out_chan = 3;
		}
		// Conv forward
		convolutionalForward(handle, model[i], input, output);
		setup_end = chrono::high_resolution_clock::now();
		setup_duration = std::chrono::duration_cast<std::chrono::microseconds>(setup_end - setup_start);
		std::cout << "conv " << i << " :" << setup_duration.count() << std::endl;

		setup_start = chrono::high_resolution_clock::now();
		// Add bias, done in place
		addBias(handle, model[i], output, false);
		setup_end = chrono::high_resolution_clock::now();
		setup_duration = std::chrono::duration_cast<std::chrono::microseconds>(setup_end - setup_start);
		std::cout << "bias " << i << " :" << setup_duration.count() << std::endl;
		// ReLU
		// TODO closer look at documentation, it says HW-packed is faster. Does this mean NHWC is faster than NCHW?
		// Can be done in place
		if (i != 19) {
			setup_start = chrono::high_resolution_clock::now();
			checkCUDNN(cudnnActivationForward(handle,
				activation,
				&alpha,
				model[i].output_desc,
				output.dev,
				&beta,
				model[i].output_desc,
				input.dev));
			setup_end = chrono::high_resolution_clock::now();
			setup_duration = std::chrono::duration_cast<std::chrono::microseconds>(setup_end - setup_start);
			std::cout << "relu " << i << " :" << setup_duration.count() << std::endl;
		}
		// Free input stuff, set input to output
		//cudaFree(input.dev);
		//free(input.host);
		input.n = output.n;
		input.c = output.c;
		input.h = output.h;
		input.w = output.w;
		//input.dev = output.dev;
		//input.host = output.host;
		setup_end = chrono::high_resolution_clock::now();
		setup_duration = std::chrono::duration_cast<std::chrono::microseconds>(setup_end - setup_start);
		std::cout << "layer " << i << " :" << setup_duration.count() << std::endl;
	}

	//tensor image = tensor(1, 3, cam.resolution.y, cam.resolution.x);
	//image.dev = dev_denoise;
	const float sub_alpha = -1.f, sub_beta = 1.f;
	checkCUDNN(cudnnAddTensor(handle, &sub_alpha, model[19].output_desc, output.dev, 
				&sub_beta, model[0].input_desc, dev_denoise));
	//addBias(handle, image, output, true);
	auto nn_end = chrono::high_resolution_clock::now();
	auto nn_duration = std::chrono::duration_cast<std::chrono::microseconds>(nn_end - nn_start);
	std::cout << "NN DUR: " << nn_duration.count() << std::endl;

	setup_start = chrono::high_resolution_clock::now();
	cudnnDestroyActivationDescriptor(activation);
	cudaFree(input.dev);
	cudaFree(output.dev);
	//free(input.host);
	setup_end = chrono::high_resolution_clock::now();
	setup_duration = std::chrono::duration_cast<std::chrono::microseconds>(setup_end - setup_start);
	std::cout << "clean up: "<< setup_duration.count() << std::endl;
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
 
void pathtrace(uchar4* pbo, cudnnHandle_t handle, std::vector<layer>& model, int frame, int iter) {
	auto pt_start = chrono::high_resolution_clock::now();
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;
	long long intersections = 0;
	long long shading_time = 0;
	long long material_time = 0;
	long long compaction = 0;
	long long ray_time = 0;

	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing
#if CACHE_FIRST_BOUNCE
	if (iter == 1) {
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_first_paths);
		checkCUDAError("generate camera ray");
	}
	cudaMemcpy(dev_paths, dev_first_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);

#else
	auto ray_start = chrono::high_resolution_clock::now();

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
	cudaDeviceSynchronize();
	auto ray_end = chrono::high_resolution_clock::now();
	auto ray_duration = std::chrono::duration_cast<std::chrono::microseconds>(ray_end - ray_start);

	ray_time += ray_duration.count();
#endif
	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete && (!denoise_on || iter < 5000)) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		//create blocks
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		// tracing
#if CACHE_FIRST_BOUNCE
		//if first intersection in iteration 1, compute intersection to dev_firstBounce
		//and then copy dev_firstBounce to dev_intersections
		if (iter == 1 && depth == 0) {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth,
				num_paths,
				dev_paths,
				dev_geoms,
				hst_scene->geoms.size(),
				dev_tinyobj,
				hst_scene->Obj_geoms.size(),
				dev_firstBounce
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
			cudaMemcpy(dev_intersections, dev_firstBounce, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		//if not first iteration but first bounce
		//just copy to dev_intersections
		else if (iter != 1 && depth == 0) {
			cudaMemcpy(dev_intersections, dev_firstBounce, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth,
				num_paths,
				dev_paths,
				dev_geoms,
				hst_scene->geoms.size(),
				dev_tinyobj,
				hst_scene->Obj_geoms.size(),
				dev_intersections,
				iter
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
		}
#else 
		//computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
		//	depth,
		//	num_paths,
		//	dev_paths,
		//	dev_geoms,
		//	hst_scene->geoms.size(),
		//	dev_tinyobj,
		//	hst_scene->Obj_geoms.size(),
		//	dev_intersections,
		//	iter
		//	);
		auto inter_start = chrono::high_resolution_clock::now();
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_tris
			, hst_scene->num_tris
			, dev_intersections
			, dev_bvh_nodes
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		auto inter_end = chrono::high_resolution_clock::now();
		auto inter_duration = std::chrono::duration_cast<std::chrono::microseconds>(inter_end - inter_start);
		intersections += inter_duration.count();

#endif

		if (depth == 0 && iter == 1)
		{
			generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBuffer);
		}

		depth++;

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

		/*shadeFakeMaterial <<<numblocksPathSegmentTracing, blockSize1d >>> (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
		);*/
		auto pos = cam.position;
		auto shade_start = chrono::high_resolution_clock::now();

		kernSimpleShade << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			depth,
			dev_intersections,
			dev_paths,
			dev_materials,
			pos
			);
		cudaDeviceSynchronize();
		auto shade_end = chrono::high_resolution_clock::now();
		auto shade_duration = std::chrono::duration_cast<std::chrono::microseconds>(shade_end - shade_start);
		shading_time += shade_duration.count();

		//stream compaction
		auto compaction_start = chrono::high_resolution_clock::now();

#if COMPACTION
		//referring to the first element of the second partition
		dev_path_end = thrust::stable_partition(thrust::device, dev_paths, dev_path_end, isZero());
		num_paths = dev_path_end - dev_paths;
#endif
		cudaDeviceSynchronize();
		auto compaction_end = chrono::high_resolution_clock::now();
		auto compaction_duration = std::chrono::duration_cast<std::chrono::microseconds>(compaction_end - compaction_start);
		compaction += compaction_duration.count();

		auto material_start = chrono::high_resolution_clock::now();

#if SORT_MATERIAL
		//sort dev_intersectoins and dev_paths based on materialId
		thrust::stable_sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compareMaterial());
#endif
		cudaDeviceSynchronize();
		auto material_end = chrono::high_resolution_clock::now();
		auto material_duration = std::chrono::duration_cast<std::chrono::microseconds>(material_end - material_start);
		std::cout << "Material: " << material_time << std::endl;

		material_time += material_duration.count();

		if (depth >= traceDepth || num_paths == 0)
			iterationComplete = true; // TODO: should be based off stream compaction results.


		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	auto pt_end = chrono::high_resolution_clock::now();
	auto pt_duration = std::chrono::duration_cast<std::chrono::microseconds>(pt_end - pt_start);
	if (true)
	{
		std::cout << "PathTracing " << pt_duration.count() << std::endl;
		std::cout << "Ray Generation: " << ray_time << std::endl;
		std::cout << "Intersection: " << intersections << std::endl;
		std::cout << "Compaction: " << compaction << std::endl;
		std::cout << "Material: " << material_time << std::endl;
		std::cout << "Shading: " << shading_time << std::endl;
	}
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	if (!denoise_on || iter < 5000) {
		// Assemble this iteration and apply it to the image
		//finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);
		// Send results to OpenGL buffer for rendering
		//sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
	}
		
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);
	if (denoise_on) {
		auto dn_start = chrono::high_resolution_clock::now();
		cudaMemcpy(dev_dn_image, dev_image,
			pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
		std::cout << "Denoising" << std::endl;

		// Denoise
		//TODO the way opencv and likely cudnn works is mirrored from the PBO, does this affect denoising?
		auto dncnn_start = chrono::high_resolution_clock::now();
		vecToBuff << <blocksPerGrid2d, blockSize2d >> > (pixelcount, dev_dn_image, dev_denoise, cam.resolution, iter);
		auto dncnn_end = chrono::high_resolution_clock::now();
		auto dncnn_duration = std::chrono::duration_cast<std::chrono::microseconds>(dncnn_end - dncnn_start);
		std::cout << "PBO to model: " << dncnn_duration.count() << std::endl;
		dncnn_start = chrono::high_resolution_clock::now();
		dnCNN(handle, model);
		dncnn_end = chrono::high_resolution_clock::now();
		dncnn_duration = std::chrono::duration_cast<std::chrono::microseconds>(dncnn_end - dncnn_start);
		std::cout << "Model Time: " << dncnn_duration.count() << std::endl;
		dncnn_start = chrono::high_resolution_clock::now();
		buffToVec << <blocksPerGrid2d, blockSize2d >> > (pixelcount, dev_dn_image, dev_denoise, cam.resolution, iter);
		sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, 1, dev_dn_image);
		dncnn_end = chrono::high_resolution_clock::now();
		dncnn_duration = std::chrono::duration_cast<std::chrono::microseconds>(dncnn_end - dncnn_start);
		std::cout << "Model to PBO: " << dncnn_duration.count() << std::endl;
		auto dn_end = chrono::high_resolution_clock::now();
		dncnn_duration = std::chrono::duration_cast<std::chrono::microseconds>(dn_end - dn_start);
		std::cout << "Full denoise: " << dncnn_duration.count() << std::endl;
	}

	///////////////////////////////////////////////////////////////////////////
	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_dn_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
