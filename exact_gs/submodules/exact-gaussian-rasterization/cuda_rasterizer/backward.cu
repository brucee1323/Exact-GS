/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * Modified by Guangpu Yang
 * 
 * 
 */
#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.  
	glm::vec4 q = rot;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;   

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);   
 
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);  
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);    
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;   
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };
}


template<int C>
__global__ void preprocessCUDA_new(
	int P,
	const float3* means,
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}



// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const float* __restrict__ means3D, 
	const float* __restrict__ cov3Ds,  
	const float* __restrict__ viewmatrix, 
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ mus,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	glm::vec3* __restrict__ dL_dmeans, 
	float* __restrict__ dL_dcov3D,   
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dmu,
	const  float dDetector,   
    const  float  DSO,  
    const  float   DSD
	)
{	
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };  
	const uint32_t pix_id = W * pix.y + pix.x;  
	float2 pixf = { (float)pix.x, (float)pix.y};
	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x]; 
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	bool done = !inside;
	int toDo = range.y - range.x;
	float3 pixG = { (pixf.x-(float)W/2 + 0.5) * dDetector, (pixf.y-(float)H/2 + 0.5) * dDetector , DSD}; 
	transformNorm(pixG);  
	glm::mat3 Rot2 = glm::mat3(
		1.0-pixG.x*pixG.x/(1.0+pixG.z), -pixG.y*pixG.x/(1.0+pixG.z),  -pixG.x,
		-pixG.y*pixG.x/(1.0+pixG.z),    1-pixG.y*pixG.y/(1.0+pixG.z),  -pixG.y,
		pixG.x,  pixG.y,  1.0-(pixG.x*pixG.x+pixG.y*pixG.y)/(1.0+pixG.z) );   
	glm::mat3 Rot1 = glm::mat3(
			viewmatrix[0], viewmatrix[4], viewmatrix[8],
			viewmatrix[1], viewmatrix[5], viewmatrix[9],
			viewmatrix[2], viewmatrix[6], viewmatrix[10]);   
	glm::mat3 Rot = Rot1 * Rot2; 

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_mu[BLOCK_SIZE];
	__shared__  float collected_cov3D[BLOCK_SIZE*6];  
	__shared__ float3 collected_mean3D[BLOCK_SIZE];  

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;   
	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	// // Gradient of pixel coordinate w.r.t. normalized 
	// // screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;   
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();   //BLOCK_SIZE=16*16   
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_mu[block.thread_rank()] = mus[coll_id];
			for  (int ii = 0;  ii < 6; ii++)
			{
				collected_cov3D[block.thread_rank()*6+ii]	= cov3Ds [coll_id * 6 +ii];
			}

			collected_mean3D[block.thread_rank()] ={ means3D[3 * coll_id], means3D[3 * coll_id + 1],  means3D[3 * coll_id + 2]    };
		}
		block.sync();
		
		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;   //for all gaussian at a tile
			if (contributor >= last_contributor)  //for speed up
				continue;

			const float4 con_o = collected_conic_opacity[j];
			float* cov3D = collected_cov3D+ j * 6;  
			float3  mean3D = collected_mean3D[j];  
			

			glm::mat3 Vrk = glm::mat3(
				cov3D[0], cov3D[1], cov3D[2],
				cov3D[1], cov3D[3], cov3D[4],
				cov3D[2], cov3D[4], cov3D[5]);  

			glm::mat3 cov_pix = glm::transpose(Rot) * glm::transpose(Vrk) * Rot;  

			glm::vec3 Position = glm::vec3(mean3D.x,mean3D.y,mean3D.z ); 
			
			glm::vec3 p_pix = glm::transpose(Rot) * Position + glm::transpose(Rot2) * glm::vec3(0,0,DSO );    
			
			float det3 = glm::determinant(cov_pix);
			
			float det2=cov_pix[0][0] * cov_pix[1][1]-cov_pix[0][1] * cov_pix[0][1];

			float opa_square = 2 * M_PI * det3 / det2;

			if (opa_square <= 0.0f)
				continue;

			float opa_mu =  sqrt(2 * M_PI * det3 / det2);
			
			float det_inv = 1.f / det2;

			float3 conic = { cov_pix[1][1] * det_inv, -cov_pix[0][1] * det_inv, cov_pix[0][0] * det_inv};
						
			float power = -0.5f * (conic .x * p_pix.x * p_pix.x + conic .z * p_pix.y * p_pix.y) - conic.y * p_pix.x * p_pix.y;
		
			if (power > 0.0f)
				continue;
			
			const float G = exp(power);

			const float alpha = con_o.w * opa_mu * G;
			
			if (alpha < 0.00000001f)
				continue;
	
			// Backward
			float dL_dalpha = 0.0f;

			const int global_id = collected_id[j];

			for (int ch = 0; ch < C; ch++)
			{
				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += 1.f * dL_dchannel;
			}

			const float dL_dG = con_o.w * opa_mu * dL_dalpha;
	
			const float dL_dopa_mu= con_o.w * G * dL_dalpha;

			const float dG_dp_pix_x = G * (-conic.x * p_pix.x - conic.y * p_pix.y )  ;   

			const float dG_dp_pix_y = G * (-conic.z * p_pix.y - conic.y * p_pix.x )  ;  

			glm::vec3 dG_dmean3D = Rot * glm::vec3(dG_dp_pix_x,dG_dp_pix_y,0.0f);
			
			glm::vec3 dG_p_c = Rot2 * glm::vec3(dG_dp_pix_x,dG_dp_pix_y,0.0f);

			float3 dL_dconic = { -0.5f * G * p_pix.x * p_pix.x * dL_dG, -1.0f * G * p_pix.x * p_pix.y * dL_dG ,-0.5f * G * p_pix.y * p_pix.y * dL_dG };
		
			float hata = cov_pix[0][0];

			float hatb = cov_pix[0][1];

			float hatc = cov_pix[0][2];

			float hatd = cov_pix[1][1];

			float hate = cov_pix[1][2];

			float hatf = cov_pix[2][2];

			float dL_dhata = 0, dL_dhatb = 0, dL_dhatc = 0, dL_dhatd = 0, dL_dhate = 0, dL_dhatf = 0;

			float denom2inv = 1.0f /(det2 * det2) ;  

			float pi_mu= M_PI / opa_mu ;

			float circ_diamond = det3 / det2;

			if (denom2inv != 0.0f && opa_mu != 0.0f)
			{
				// exp(*) part gradient
				dL_dhata = denom2inv * (-hatd * hatd * dL_dconic.x + hatb * hatd * dL_dconic.y + (det2 - hata * hatd) * dL_dconic.z);  
				dL_dhatd = denom2inv * (-hata * hata * dL_dconic.z + hata * hatb * dL_dconic.y + (det2 - hata * hatd) * dL_dconic.x);  
				dL_dhatb = denom2inv * (2 * hatb * hatd * dL_dconic.x - (det2+ 2 * hatb * hatb) * dL_dconic.y + 2 * hata * hatb * dL_dconic.z);
				dL_dhata += pi_mu * ((hatd * hatf - hate * hate) /det2 -  hatd * circ_diamond /det2) * dL_dopa_mu;
				dL_dhatb += pi_mu * ((2 * hatc * hate - 2 * hatf * hatb) /det2 + 2 * hatb * circ_diamond /det2) *dL_dopa_mu;
				dL_dhatc += pi_mu * ((2 * hatb * hate - 2 * hatd * hatc) /det2) *dL_dopa_mu;
				dL_dhatd += pi_mu * ((hata * hatf - hatc * hatc) /det2 -  hata *circ_diamond /det2) *dL_dopa_mu;
				dL_dhate += pi_mu * ((2 * hatb * hatc - 2 * hata * hate) /det2) *dL_dopa_mu;
				dL_dhatf += pi_mu * ((hata * hatd - hatb * hatb) /det2) * dL_dopa_mu;

				// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry
				atomicAdd(&dL_dcov3D[6*global_id+0],Rot[0][0]*Rot[0][0]*dL_dhata + Rot[0][0]*Rot[1][0]*dL_dhatb + Rot[0][0]*Rot[2][0]*dL_dhatc + Rot[1][0]*Rot[1][0]*dL_dhatd + Rot[1][0]*Rot[2][0]*dL_dhate + Rot[2][0]*Rot[2][0]*dL_dhatf);

				atomicAdd(&dL_dcov3D[6*global_id+3],Rot[0][1]*Rot[0][1]*dL_dhata + Rot[0][1]*Rot[1][1]*dL_dhatb + Rot[0][1]*Rot[2][1]*dL_dhatc + Rot[1][1]*Rot[1][1]*dL_dhatd + Rot[1][1]*Rot[2][1]*dL_dhate + Rot[2][1]*Rot[2][1]*dL_dhatf);

				atomicAdd(&dL_dcov3D[6*global_id+5],Rot[0][2]*Rot[0][2]*dL_dhata + Rot[0][2]*Rot[1][2]*dL_dhatb + Rot[0][2]*Rot[2][2]*dL_dhatc + Rot[1][2]*Rot[1][2]*dL_dhatd + Rot[1][2]*Rot[2][2]*dL_dhate + Rot[2][2]*Rot[2][2]*dL_dhatf);
				
				atomicAdd(&dL_dcov3D[6*global_id+1],2*Rot[0][0]*Rot[0][1]*dL_dhata + (Rot[0][1]*Rot[1][0]+Rot[0][0]*Rot[1][1])*dL_dhatb + (Rot[0][1]*Rot[2][0]+Rot[0][0]*Rot[2][1])*dL_dhatc + 2*Rot[1][0]*Rot[1][1]*dL_dhatd + (Rot[1][1]*Rot[2][0]+Rot[1][0]*Rot[2][1])*dL_dhate + 2*Rot[2][0]*Rot[2][1]*dL_dhatf);
				
				atomicAdd(&dL_dcov3D[6*global_id+2],2*Rot[0][0]*Rot[0][2]*dL_dhata + (Rot[0][2]*Rot[1][0]+Rot[0][0]*Rot[1][2])*dL_dhatb + (Rot[0][2]*Rot[2][0]+Rot[0][0]*Rot[2][2])*dL_dhatc + 2*Rot[1][0]*Rot[1][2]*dL_dhatd + (Rot[1][2]*Rot[2][0]+Rot[1][0]*Rot[2][2])*dL_dhate + 2*Rot[2][0]*Rot[2][2]*dL_dhatf);
				
				atomicAdd(&dL_dcov3D[6*global_id+4],2*Rot[0][1]*Rot[0][2]*dL_dhata + (Rot[0][2]*Rot[1][1]+Rot[0][1]*Rot[1][2])*dL_dhatb + (Rot[0][2]*Rot[2][1]+Rot[0][1]*Rot[2][2])*dL_dhatc + 2*Rot[1][1]*Rot[1][2]*dL_dhatd + (Rot[1][2]*Rot[2][1]+Rot[1][1]*Rot[2][2])*dL_dhate + 2*Rot[2][1]*Rot[2][2]*dL_dhatf);

			}
			
			atomicAdd(&dL_dmeans[global_id].x,dG_dmean3D.x*dL_dG);
			atomicAdd(&dL_dmeans[global_id].y,dG_dmean3D.y*dL_dG);
			atomicAdd(&dL_dmeans[global_id].z,dG_dmean3D.z*dL_dG);
			atomicAdd(&(dL_dopacity[global_id]), opa_mu * G * dL_dalpha);
			atomicAdd(&(dL_dmu[global_id]),  dL_dopa_mu);
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG *dG_p_c.x*dDetector*p_pix.z/DSD* ddelx_dx ); 
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG *dG_p_c.y*dDetector*p_pix.z/DSD* ddely_dy ); 
		}
	}

	
}


void BACKWARD::preprocess(
	int P,
	const float3* means3D,
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	const float* dL_dmu,
	glm::vec3* dL_dmean3D,
	float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	const int mode)
{
	preprocessCUDA_new<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P,
		(float3*)means3D,
		radii,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcov3D,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const float* means3D, 
	const float* cov3Ds, 
	const float* viewmatrix, 
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float4* conic_opacity,
	const float* mus,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	glm::vec3* dL_dmean3D,  
	float* dL_dcov3D,  
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dmu,
	const float dDetector,   
    const float  DSO,  
    const float  DSD 
	)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		means3D, 
		cov3Ds,  
		viewmatrix, 
		ranges,
		point_list,
		W, H,
		means2D,
		conic_opacity,
		mus,
		n_contrib,
		dL_dpixels,
		(glm::vec3*)dL_dmean3D,  
		dL_dcov3D,  
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dmu,
		dDetector,   
        DSO,  
        DSD
		);
}

