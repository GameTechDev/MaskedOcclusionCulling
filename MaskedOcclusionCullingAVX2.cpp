/*
 * Copyright (c) 2016, Intel Corporation
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of Intel Corporation nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include <new.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include "MaskedOcclusionCulling.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compiler specific functions: currently only MSC and Intel compiler should work.
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef _MSC_VER

	#include <intrin.h>

	#define FORCE_INLINE __forceinline

	static FORCE_INLINE unsigned long find_clear_lsb(unsigned int *mask)
	{
		unsigned long idx;
		_BitScanForward(&idx, *mask);
		*mask &= *mask - 1;
		return idx;
	}

	#ifndef __AVX2__
		#error For best performance, MaskedOcclusionCullingAVX2.cpp should be compiled with /arch:AVX2
	#endif
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// AVX specific defines and constants
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SIMD_LANES             8
#define TILE_HEIGHT_SHIFT      3

#define SIMD_LANE_IDX _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)

#define SIMD_SUB_TILE_COL_OFFSET _mm256_setr_epi32(0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3, 0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3)
#define SIMD_SUB_TILE_ROW_OFFSET _mm256_setr_epi32(0, 0, 0, 0, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT)
#define SIMD_SUB_TILE_COL_OFFSET_F _mm256_setr_ps(0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3, 0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3)
#define SIMD_SUB_TILE_ROW_OFFSET_F _mm256_setr_ps(0, 0, 0, 0, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT)

#define SIMD_SHUFFLE_SCANLINE_TO_SUBTILES _mm256_setr_epi8( 0x0, 0x4, 0x8, 0xC,	0x1, 0x5, 0x9, 0xD,	0x2, 0x6, 0xA, 0xE,	0x3, 0x7, 0xB, 0xF,	0x0, 0x4, 0x8, 0xC,	0x1, 0x5, 0x9, 0xD,	0x2, 0x6, 0xA, 0xE,	0x3, 0x7, 0xB, 0xF)

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// AVX specific typedefs and functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef __m256 __mw;
typedef __m256i __mwi;

#define mw_f32 m256_f32
#define mw_i32 m256i_i32

#define _mmw_set1_ps _mm256_set1_ps
#define _mmw_setzero_ps _mm256_setzero_ps
#define _mmw_andnot_ps _mm256_andnot_ps
#define _mmw_fmadd_ps _mm256_fmadd_ps
#define _mmw_fmsub_ps _mm256_fmsub_ps
#define _mmw_min_ps _mm256_min_ps
#define _mmw_max_ps _mm256_max_ps
#define _mmw_movemask_ps _mm256_movemask_ps
#define _mmw_blendv_ps _mm256_blendv_ps
#define _mmw_cmpge_ps(a,b) _mm256_cmp_ps(a, b, _CMP_GE_OQ)
#define _mmw_cmpgt_ps(a,b) _mm256_cmp_ps(a, b, _CMP_GT_OQ)
#define _mmw_cmpeq_ps(a,b) _mm256_cmp_ps(a, b, _CMP_EQ_OQ)
#define _mmw_floor_ps(x) _mm256_round_ps(x, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)
#define _mmw_ceil_ps(x) _mm256_round_ps(x, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)
#define _mmw_shuffle_ps _mm256_shuffle_ps
#define _mmw_insertf32x4_ps _mm256_insertf128_ps

#define _mmw_set1_epi32 _mm256_set1_epi32
#define _mmw_setzero_epi32 _mm256_setzero_si256
#define _mmw_andnot_epi32 _mm256_andnot_si256
#define _mmw_min_epi32 _mm256_min_epi32
#define _mmw_max_epi32 _mm256_max_epi32
#define _mmw_subs_epu16 _mm256_subs_epu16
#define _mmw_mullo_epi32 _mm256_mullo_epi32
#define _mmw_cmpeq_epi32 _mm256_cmpeq_epi32
#define _mmw_testz_epi32 _mm256_testz_si256
#define _mmw_cmpgt_epi32 _mm256_cmpgt_epi32
#define _mmw_srai_epi32 _mm256_srai_epi32
#define _mmw_srli_epi32 _mm256_srli_epi32
#define _mmw_slli_epi32 _mm256_slli_epi32
#define _mmw_sllv_ones(x) _mm256_sllv_epi32(SIMD_BITS_ONE, x)
#define _mmw_transpose_epi8(x) _mm256_shuffle_epi8(x, SIMD_SHUFFLE_SCANLINE_TO_SUBTILES)

#define _mmw_cvtps_epi32 _mm256_cvtps_epi32
#define _mmw_cvttps_epi32 _mm256_cvttps_epi32
#define _mmw_cvtepi32_ps _mm256_cvtepi32_ps

#define _mmx_dp4_ps(a, b) _mm_dp_ps(a, b, 0xFF)
#define _mmx_fmadd_ps _mm_fmadd_ps
#define _mmx_max_epi32 _mm_max_epi32
#define _mmx_min_epi32 _mm_min_epi32

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Specialized AVX input assembly function for general vertex gather 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef MaskedOcclusionCulling::VertexLayout VertexLayout;

static FORCE_INLINE void GatherVertices(__m256 *vtxX, __m256 *vtxY, __m256 *vtxW, const float *inVtx, const unsigned int *inTrisPtr, int numLanes, const VertexLayout &vtxLayout)
{
	assert(numLanes >= 1);

	const __m256i SIMD_TRI_IDX_OFFSET = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
	static const __m256i SIMD_LANE_MASK[9] = {
		_mm256_setr_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
		_mm256_setr_epi32(~0,  0,  0,  0,  0,  0,  0,  0),
		_mm256_setr_epi32(~0, ~0,  0,  0,  0,  0,  0,  0),
		_mm256_setr_epi32(~0, ~0, ~0,  0,  0,  0,  0,  0),
		_mm256_setr_epi32(~0, ~0, ~0, ~0,  0,  0,  0,  0),
		_mm256_setr_epi32(~0, ~0, ~0, ~0, ~0,  0,  0,  0),
		_mm256_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0,  0,  0),
		_mm256_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0,  0),
		_mm256_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0)
	};

	// Compute per-lane index list offset that guards against out of bounds memory accesses
	__m256i safeTriIdxOffset = _mm256_and_si256(SIMD_TRI_IDX_OFFSET, SIMD_LANE_MASK[numLanes]);

	// Fetch triangle indices. 
	__m256i vtxIdx[3];
	vtxIdx[0] = _mmw_mullo_epi32(_mm256_i32gather_epi32((const int*)inTrisPtr + 0, safeTriIdxOffset, 4), _mmw_set1_epi32(vtxLayout.mStride));
	vtxIdx[1] = _mmw_mullo_epi32(_mm256_i32gather_epi32((const int*)inTrisPtr + 1, safeTriIdxOffset, 4), _mmw_set1_epi32(vtxLayout.mStride));
	vtxIdx[2] = _mmw_mullo_epi32(_mm256_i32gather_epi32((const int*)inTrisPtr + 2, safeTriIdxOffset, 4), _mmw_set1_epi32(vtxLayout.mStride));

	char *vPtr = (char *)inVtx;

	// Fetch triangle vertices
	for (int i = 0; i < 3; i++)
	{
		vtxX[i] = _mm256_i32gather_ps((float *)vPtr, vtxIdx[i], 1);
		vtxY[i] = _mm256_i32gather_ps((float *)(vPtr + vtxLayout.mOffsetY), vtxIdx[i], 1);
		vtxW[i] = _mm256_i32gather_ps((float *)(vPtr + vtxLayout.mOffsetW), vtxIdx[i], 1);
	}
}

namespace MaskedOcclusionCullingAVX2
{

	static MaskedOcclusionCulling::Implementation gInstructionSet = MaskedOcclusionCulling::AVX2;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Include common algorithm implementation (general, SIMD independent code)
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	#include "MaskedOcclusionCullingCommon.inl"

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Utility function to create a new object using the allocator callbacks
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	typedef MaskedOcclusionCulling::pfnAlignedAlloc            pfnAlignedAlloc;
	typedef MaskedOcclusionCulling::pfnAlignedFree             pfnAlignedFree;

	MaskedOcclusionCulling *CreateMaskedOcclusionCulling(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree)
	{
		MaskedOcclusionCullingPrivate *object = (MaskedOcclusionCullingPrivate *)memAlloc(32, sizeof(MaskedOcclusionCullingPrivate));
		new (object) MaskedOcclusionCullingPrivate(memAlloc, memFree);
		return object;
	}
};
