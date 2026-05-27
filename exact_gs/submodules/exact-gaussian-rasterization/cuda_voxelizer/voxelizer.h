

#ifndef CUDA_VOXELIZER_H_INCLUDED
#define CUDA_VOXELIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaVoxelizer
{
    class Voxelizer
	{
	public:

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P,
			const int nVoxel_x, int nVoxel_y, int nVoxel_z,
			const float sVoxel_x, float sVoxel_y, float sVoxel_z,
			const float center_x, float center_y, float center_z,
			const float* means3D,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const bool prefiltered,
			float* out_volume,
			int* radii_x = nullptr,
			int* radii_y = nullptr,
			int* radii_z = nullptr,
			bool debug = false);

		static void backward(
			const int P, int R, 
			const int nVoxel_x, int nVoxel_y, int nVoxel_z,
			const float sVoxel_x, float sVoxel_y, float sVoxel_z,
			const float center_x, float center_y, float center_z,
			const float* means3D,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const int* radii_x, const int* radii_y, const int* radii_z,
			char* geom_buffer,
			char* binning_buffer,
			char* img_buffer,
			const float* dL_dpix,
			float* dL_dmean3D_norm,
			float* dL_dconic3D,
			float* dL_dopacity,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dscale,
			float* dL_drot,
			bool debug);
	};

};

#endif