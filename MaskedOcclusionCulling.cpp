/*
 * Copyright 2017 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http ://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */
#include <new.h>
#include <stdlib.h>
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

	static void *aligned_alloc(size_t alignment, size_t size)
	{
		return _aligned_malloc(size, alignment);
	}
	
	static void aligned_free(void *ptr)
	{
		_aligned_free(ptr);
	}
	
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility functions (not directly related to the algorithm/rasterizer)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void MaskedOcclusionCulling::TransformVertices(const float *mtx, const float *inVtx, float *xfVtx, unsigned int nVtx, const VertexLayout &vtxLayout)
{
	// This function pretty slow, about 10-20% slower than if the vertices are stored in aligned SOA form.
	if (nVtx == 0)
		return;

	// Load matrix and swizzle out the z component. For post-multiplication (OGL), the matrix is assumed to be column 
	// major, with one column per SSE register. For pre-multiplication (DX), the matrix is assumed to be row major.
	__m128 mtxCol0 = _mm_loadu_ps(mtx);
	__m128 mtxCol1 = _mm_loadu_ps(mtx + 4);
	__m128 mtxCol2 = _mm_loadu_ps(mtx + 8);
	__m128 mtxCol3 = _mm_loadu_ps(mtx + 12);

	int stride = vtxLayout.mStride;
	const char *vPtr = (const char *)inVtx;
	float *outPtr = xfVtx;

	// Iterate through all vertices and transform
	for (unsigned int vtx = 0; vtx < nVtx; ++vtx)
	{
		__m128 xVal = _mm_load1_ps((float*)(vPtr));
		__m128 yVal = _mm_load1_ps((float*)(vPtr + vtxLayout.mOffsetY));
		__m128 zVal = _mm_load1_ps((float*)(vPtr + vtxLayout.mOffsetZ));

		__m128 xform = _mm_add_ps(_mm_mul_ps(mtxCol0, xVal), _mm_add_ps(_mm_mul_ps(mtxCol1, yVal), _mm_add_ps(_mm_mul_ps(mtxCol2, zVal), mtxCol3)));
		_mm_storeu_ps(outPtr, xform);
		vPtr += stride;
		outPtr += 4;
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Typedefs
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef MaskedOcclusionCulling::pfnAlignedAlloc pfnAlignedAlloc;
typedef MaskedOcclusionCulling::pfnAlignedFree  pfnAlignedFree;
typedef MaskedOcclusionCulling::VertexLayout    VertexLayout;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Common SSE2/SSE4.1 defines
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SIMD_LANES             4
#define TILE_HEIGHT_SHIFT      2

#define SIMD_LANE_IDX _mm_setr_epi32(0, 1, 2, 3)

#define SIMD_SUB_TILE_COL_OFFSET _mm_setr_epi32(0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3)
#define SIMD_SUB_TILE_ROW_OFFSET _mm_setzero_si128()
#define SIMD_SUB_TILE_COL_OFFSET_F _mm_setr_ps(0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3)
#define SIMD_SUB_TILE_ROW_OFFSET_F _mm_setzero_ps()

#define SIMD_LANE_YCOORD_I _mm_setr_epi32(128, 384, 640, 896)
#define SIMD_LANE_YCOORD_F _mm_setr_ps(128.0f, 384.0f, 640.0f, 896.0f)

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Common SSE2/SSE4.1 functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef __m128 __mw;
typedef __m128i __mwi;

#define mw_f32 m128_f32
#define mw_i32 m128i_i32

#define _mmw_set1_ps                _mm_set1_ps
#define _mmw_setzero_ps             _mm_setzero_ps
#define _mmw_and_ps                 _mm_and_ps
#define _mmw_or_ps                  _mm_or_ps
#define _mmw_xor_ps                 _mm_xor_ps
#define _mmw_not_ps(a)              _mm_xor_ps((a), _mm_castsi128_ps(_mm_set1_epi32(~0)))
#define _mmw_andnot_ps              _mm_andnot_ps
#define _mmw_neg_ps(a)              _mm_xor_ps((a), _mm_set1_ps(-0.0f))
#define _mmw_abs_ps(a)              _mm_and_ps((a), _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)))
#define _mmw_add_ps                 _mm_add_ps
#define _mmw_sub_ps                 _mm_sub_ps
#define _mmw_mul_ps                 _mm_mul_ps
#define _mmw_div_ps                 _mm_div_ps
#define _mmw_min_ps                 _mm_min_ps
#define _mmw_max_ps                 _mm_max_ps
#define _mmw_movemask_ps            _mm_movemask_ps
#define _mmw_cmpge_ps(a,b)          _mm_cmpge_ps(a, b)
#define _mmw_cmpgt_ps(a,b)          _mm_cmpgt_ps(a, b)
#define _mmw_cmpeq_ps(a,b)          _mm_cmpeq_ps(a, b)
#define _mmw_fmadd_ps(a,b,c)        _mm_add_ps(_mm_mul_ps(a,b), c)
#define _mmw_fmsub_ps(a,b,c)        _mm_sub_ps(_mm_mul_ps(a,b), c)
#define _mmw_shuffle_ps             _mm_shuffle_ps
#define _mmw_insertf32x4_ps(a,b,c)  (b)
#define _mmw_cvtepi32_ps            _mm_cvtepi32_ps
#define _mmw_blendv_epi32(a,b,c)    simd_cast<__mwi>(_mmw_blendv_ps(simd_cast<__mw>(a), simd_cast<__mw>(b), simd_cast<__mw>(c)))

#define _mmw_set1_epi32             _mm_set1_epi32
#define _mmw_setzero_epi32          _mm_setzero_si128
#define _mmw_and_epi32              _mm_and_si128
#define _mmw_or_epi32               _mm_or_si128
#define _mmw_xor_epi32              _mm_xor_si128
#define _mmw_not_epi32(a)           _mm_xor_si128((a), _mm_set1_epi32(~0))
#define _mmw_andnot_epi32           _mm_andnot_si128
#define _mmw_neg_epi32(a)           _mm_sub_epi32(_mm_set1_epi32(0), (a))
#define _mmw_add_epi32              _mm_add_epi32
#define _mmw_sub_epi32              _mm_sub_epi32
#define _mmw_subs_epu16             _mm_subs_epu16
#define _mmw_cmpeq_epi32            _mm_cmpeq_epi32
#define _mmw_cmpgt_epi32            _mm_cmpgt_epi32
#define _mmw_srai_epi32             _mm_srai_epi32
#define _mmw_srli_epi32             _mm_srli_epi32
#define _mmw_slli_epi32             _mm_slli_epi32
#define _mmw_abs_epi32              _mm_abs_epi32
#define _mmw_cvtps_epi32            _mm_cvtps_epi32
#define _mmw_cvttps_epi32           _mm_cvttps_epi32

#define _mmx_fmadd_ps               _mmw_fmadd_ps
#define _mmx_max_epi32              _mmw_max_epi32
#define _mmx_min_epi32              _mmw_min_epi32

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SIMD casting functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, typename Y> FORCE_INLINE T simd_cast(Y A);
template<> FORCE_INLINE __m128  simd_cast<__m128>(float A) { return _mm_set1_ps(A); }
template<> FORCE_INLINE __m128  simd_cast<__m128>(__m128i A) { return _mm_castsi128_ps(A); }
template<> FORCE_INLINE __m128  simd_cast<__m128>(__m128 A) { return A; }
template<> FORCE_INLINE __m128i simd_cast<__m128i>(int A) { return _mm_set1_epi32(A); }
template<> FORCE_INLINE __m128i simd_cast<__m128i>(__m128 A) { return _mm_castps_si128(A); }
template<> FORCE_INLINE __m128i simd_cast<__m128i>(__m128i A) { return A; }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Specialized SSE input assembly function for general vertex gather 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FORCE_INLINE void GatherVertices(__m128 *vtxX, __m128 *vtxY, __m128 *vtxW, const float *inVtx, const unsigned int *inTrisPtr, int numLanes, const VertexLayout &vtxLayout)
{
	for (int lane = 0; lane < numLanes; lane++)
	{
		for (int i = 0; i < 3; i++)
		{
			char *vPtrX = (char *)inVtx + inTrisPtr[lane * 3 + i] * vtxLayout.mStride;
			char *vPtrY = vPtrX + vtxLayout.mOffsetY;
			char *vPtrW = vPtrX + vtxLayout.mOffsetW;

			vtxX[i].m128_f32[lane] = *((float*)vPtrX);
			vtxY[i].m128_f32[lane] = *((float*)vPtrY);
			vtxW[i].m128_f32[lane] = *((float*)vPtrW);
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SSE4.1 version
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace MaskedOcclusionCullingSSE41
{
	static FORCE_INLINE __m128i _mmw_mullo_epi32(const __m128i &a, const __m128i &b) { return _mm_mullo_epi32(a, b); }
	static FORCE_INLINE __m128i _mmw_min_epi32(const __m128i &a, const __m128i &b) { return _mm_min_epi32(a, b); }
	static FORCE_INLINE __m128i _mmw_max_epi32(const __m128i &a, const __m128i &b) { return _mm_max_epi32(a, b); }
	static FORCE_INLINE __m128 _mmw_blendv_ps(const __m128 &a, const __m128 &b, const __m128 &c) { return _mm_blendv_ps(a, b, c); }
	static FORCE_INLINE int _mmw_testz_epi32(const __m128i &a, const __m128i &b) { return _mm_testz_si128(a, b); }
	static FORCE_INLINE __m128 _mmx_dp4_ps(const __m128 &a, const __m128 &b) { return _mm_dp_ps(a, b, 0xFF); }
	static FORCE_INLINE __m128 _mmw_floor_ps(const __m128 &a) { return _mm_round_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC); }
	static FORCE_INLINE __m128 _mmw_ceil_ps(const __m128 &a) { return _mm_round_ps(a, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);	}
	static FORCE_INLINE __m128i _mmw_transpose_epi8(const __m128i &a)
	{
		const __m128i shuff = _mm_setr_epi8(0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF);
		return _mm_shuffle_epi8(a, shuff);
	}
	static FORCE_INLINE __m128i _mmw_sllv_ones(const __m128i &ishift)
	{
		__m128i shift = _mm_min_epi32(ishift, _mm_set1_epi32(32));

		// Uses lookup tables and _mm_shuffle_epi8 to perform _mm_sllv_epi32(~0, shift)
		const __m128i byteShiftLUT = _mm_setr_epi8(~0, ~0 << 1, ~0 << 2, ~0 << 3, ~0 << 4, ~0 << 5, ~0 << 6, ~0 << 7, 0, 0, 0, 0, 0, 0, 0, 0);
		const __m128i byteShiftOffset = _mm_setr_epi8(0, 8, 16, 24, 0, 8, 16, 24, 0, 8, 16, 24, 0, 8, 16, 24);
		const __m128i byteShiftShuffle = _mm_setr_epi8(0x0, 0x0, 0x0, 0x0, 0x4, 0x4, 0x4, 0x4, 0x8, 0x8, 0x8, 0x8, 0xC, 0xC, 0xC, 0xC);

		__m128i byteShift = _mm_shuffle_epi8(shift, byteShiftShuffle);
		byteShift = _mm_min_epi8(_mm_subs_epu8(byteShift, byteShiftOffset), _mm_set1_epi8(8));
		__m128i retMask = _mm_shuffle_epi8(byteShiftLUT, byteShift);

		return retMask;
	}

	static MaskedOcclusionCulling::Implementation gInstructionSet = MaskedOcclusionCulling::SSE41;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Include common algorithm implementation (general, SIMD independent code)
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	#include "MaskedOcclusionCullingCommon.inl"

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Utility function to create a new object using the allocator callbacks
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	static bool DetectSSE41()
	{
		static bool initialized = false;
		static bool SSE41Supported = false;

		int cpui[4];
		if (!initialized)
		{
			initialized = true;

			int nIds, nExIds;
			__cpuid(cpui, 0);
			nIds = cpui[0];
			__cpuid(cpui, 0x80000000);
			nExIds = cpui[0];

			if (nIds >= 1)
			{
				// Test SSE4.1 support
				__cpuidex(cpui, 1, 0);
				SSE41Supported = (cpui[2] & 0x080000) == 0x080000;
			}
		}
		return SSE41Supported;
	}

	MaskedOcclusionCulling *CreateMaskedOcclusionCulling(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree)
	{
		if (!DetectSSE41())
			return nullptr;

		MaskedOcclusionCullingPrivate *object = (MaskedOcclusionCullingPrivate *)memAlloc(32, sizeof(MaskedOcclusionCullingPrivate));
		new (object) MaskedOcclusionCullingPrivate(memAlloc, memFree);
		return object;
	}
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SSE2 version
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace MaskedOcclusionCullingSSE2
{
	static FORCE_INLINE __m128i _mmw_mullo_epi32(const __m128i &a, const __m128i &b) 
	{ 
		// Do products for even / odd lanes & merge the result
		__m128i even = _mm_and_si128(_mm_mul_epu32(a, b), _mm_setr_epi32(~0, 0, ~0, 0));
		__m128i odd = _mm_slli_epi64(_mm_mul_epu32(_mm_srli_epi64(a, 32), _mm_srli_epi64(b, 32)), 32);
		return _mm_or_si128(even, odd);
	}
	static FORCE_INLINE __m128i _mmw_min_epi32(const __m128i &a, const __m128i &b) 
	{ 
		__m128i cond = _mm_cmpgt_epi32(a, b);
		return _mm_or_si128(_mm_andnot_si128(cond, a), _mm_and_si128(cond, b));
	}
	static FORCE_INLINE __m128i _mmw_max_epi32(const __m128i &a, const __m128i &b) 
	{ 
		__m128i cond = _mm_cmpgt_epi32(b, a);
		return _mm_or_si128(_mm_andnot_si128(cond, a), _mm_and_si128(cond, b));
	}
	static FORCE_INLINE int _mmw_testz_epi32(const __m128i &a, const __m128i &b) 
	{ 
		return _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_and_si128(a, b), _mm_setzero_si128())) == 0xFFFF;
	}
	static FORCE_INLINE __m128 _mmw_blendv_ps(const __m128 &a, const __m128 &b, const __m128 &c)
	{	
		__m128 cond = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(c), 31));
		return _mm_or_ps(_mm_andnot_ps(cond, a), _mm_and_ps(cond, b));
	}
	static FORCE_INLINE __m128 _mmx_dp4_ps(const __m128 &a, const __m128 &b)
	{ 
		// Product and two shuffle/adds pairs (similar to hadd_ps)
		__m128 prod = _mm_mul_ps(a, b);
		__m128 dp = _mm_add_ps(prod, _mm_shuffle_ps(prod, prod, _MM_SHUFFLE(2, 3, 0, 1)));
		dp = _mm_add_ps(dp, _mm_shuffle_ps(dp, dp, _MM_SHUFFLE(0, 1, 2, 3)));
		return dp;
	}
	static FORCE_INLINE __m128 _mmw_floor_ps(const __m128 &a) 
	{ 
		int originalMode = _MM_GET_ROUNDING_MODE();
		_MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
		__m128 rounded = _mm_cvtepi32_ps(_mm_cvtps_epi32(a));
		_MM_SET_ROUNDING_MODE(originalMode);
		return rounded;
	}
	static FORCE_INLINE __m128 _mmw_ceil_ps(const __m128 &a) 
	{ 
		int originalMode = _MM_GET_ROUNDING_MODE();
		_MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
		__m128 rounded = _mm_cvtepi32_ps(_mm_cvtps_epi32(a));
		_MM_SET_ROUNDING_MODE(originalMode);
		return rounded;
	}
	static FORCE_INLINE __m128i _mmw_transpose_epi8(const __m128i &a)
	{
		// Perform transpose through two 16->8 bit pack and byte shifts
		__m128i res = a;
		const __m128i mask = _mm_setr_epi8(~0, 0, ~0, 0, ~0, 0, ~0, 0, ~0, 0, ~0, 0, ~0, 0, ~0, 0);
		res = _mm_packus_epi16(_mm_and_si128(res, mask), _mm_srli_epi16(res, 8));
		res = _mm_packus_epi16(_mm_and_si128(res, mask), _mm_srli_epi16(res, 8));
		return res;
	}
	static FORCE_INLINE __m128i _mmw_sllv_ones(const __m128i &ishift)
	{
		__m128i shift = _mmw_min_epi32(ishift, _mm_set1_epi32(32));
		
		// Uses scalar approach to perform _mm_sllv_epi32(~0, shift)
		static const unsigned int maskLUT[33] = {
			~0U << 0, ~0U << 1, ~0U << 2 ,  ~0U << 3, ~0U << 4, ~0U << 5, ~0U << 6 , ~0U << 7, ~0U << 8, ~0U << 9, ~0U << 10 , ~0U << 11, ~0U << 12, ~0U << 13, ~0U << 14 , ~0U << 15,
			~0U << 16, ~0U << 17, ~0U << 18 , ~0U << 19, ~0U << 20, ~0U << 21, ~0U << 22 , ~0U << 23, ~0U << 24, ~0U << 25, ~0U << 26 , ~0U << 27, ~0U << 28, ~0U << 29, ~0U << 30 , ~0U << 31,
			0U };

		__m128i retMask;
		retMask.m128i_u32[0] = maskLUT[shift.m128i_u32[0]];
		retMask.m128i_u32[1] = maskLUT[shift.m128i_u32[1]];
		retMask.m128i_u32[2] = maskLUT[shift.m128i_u32[2]];
		retMask.m128i_u32[3] = maskLUT[shift.m128i_u32[3]];
		return retMask;
	}

	static MaskedOcclusionCulling::Implementation gInstructionSet = MaskedOcclusionCulling::SSE2;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Include common algorithm implementation (general, SIMD independent code)
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	#include "MaskedOcclusionCullingCommon.inl"

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Utility function to create a new object using the allocator callbacks
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	MaskedOcclusionCulling *CreateMaskedOcclusionCulling(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree)
	{
		MaskedOcclusionCullingPrivate *object = (MaskedOcclusionCullingPrivate *)memAlloc(32, sizeof(MaskedOcclusionCullingPrivate));
		new (object) MaskedOcclusionCullingPrivate(memAlloc, memFree);
		return object;
	}
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Object construction and allocation
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace MaskedOcclusionCullingAVX512
{
	extern MaskedOcclusionCulling *CreateMaskedOcclusionCulling(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree);
}

namespace MaskedOcclusionCullingAVX2
{
	extern MaskedOcclusionCulling *CreateMaskedOcclusionCulling(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree);
}

MaskedOcclusionCulling *MaskedOcclusionCulling::Create()
{
	return Create(aligned_alloc, aligned_free);
}

MaskedOcclusionCulling *MaskedOcclusionCulling::Create(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree)
{
	MaskedOcclusionCulling *object = nullptr;

	// Return best supported version
	if (object == nullptr)
		object = MaskedOcclusionCullingAVX512::CreateMaskedOcclusionCulling(memAlloc, memFree);	// Use AVX512 version
	if (object == nullptr)
		object = MaskedOcclusionCullingAVX2::CreateMaskedOcclusionCulling(memAlloc, memFree);	// Use AVX2 version
	if (object == nullptr)
		object = MaskedOcclusionCullingSSE41::CreateMaskedOcclusionCulling(memAlloc, memFree);	// Use SSE4.1 version
	if (object == nullptr)
		object = MaskedOcclusionCullingSSE2::CreateMaskedOcclusionCulling(memAlloc, memFree);	// Use SSE2 (slow) version

	return object;
}

void MaskedOcclusionCulling::Destroy(MaskedOcclusionCulling *moc)
{
	pfnAlignedFree alignedFreeCallback = moc->mAlignedFreeCallback;
	moc->~MaskedOcclusionCulling();
	alignedFreeCallback(moc);
}
