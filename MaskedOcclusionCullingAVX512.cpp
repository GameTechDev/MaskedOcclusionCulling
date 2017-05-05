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
#include <string.h>
#include <assert.h>
#include <float.h>
#include "MaskedOcclusionCulling.h"

#if defined(__INTEL_COMPILER) // Make sure compiler features AVX-512 intrinsics. Visual Studio 2017 does not support integer AVX-512 intrinsics

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
		#error For best performance, MaskedOcclusionCullingAVX512.cpp should be compiled with /arch:AVX2
	#endif

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// AVX specific defines and constants
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SIMD_LANES             16
#define TILE_HEIGHT_SHIFT      4

#define SIMD_LANE_IDX _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

#define SIMD_SUB_TILE_COL_OFFSET _mm512_setr_epi32(0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3, 0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3, 0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3, 0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3)
#define SIMD_SUB_TILE_ROW_OFFSET _mm512_setr_epi32(0, 0, 0, 0, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT * 2, SUB_TILE_HEIGHT * 2, SUB_TILE_HEIGHT * 2, SUB_TILE_HEIGHT * 2, SUB_TILE_HEIGHT * 3, SUB_TILE_HEIGHT * 3, SUB_TILE_HEIGHT * 3, SUB_TILE_HEIGHT * 3)
#define SIMD_SUB_TILE_COL_OFFSET_F _mm512_setr_ps(0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3, 0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3, 0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3, 0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3)
#define SIMD_SUB_TILE_ROW_OFFSET_F _mm512_setr_ps(0, 0, 0, 0, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT, SUB_TILE_HEIGHT * 2, SUB_TILE_HEIGHT * 2, SUB_TILE_HEIGHT * 2, SUB_TILE_HEIGHT * 2, SUB_TILE_HEIGHT * 3, SUB_TILE_HEIGHT * 3, SUB_TILE_HEIGHT * 3, SUB_TILE_HEIGHT * 3)

#define SIMD_SHUFFLE_SCANLINE_TO_SUBTILES _mm512_set_epi8(0xF, 0xB, 0x7, 0x3, 0xE, 0xA, 0x6, 0x2, 0xD, 0x9, 0x5, 0x1, 0xC, 0x8, 0x4, 0x0, 0xF, 0xB, 0x7, 0x3, 0xE, 0xA, 0x6, 0x2, 0xD, 0x9, 0x5, 0x1, 0xC, 0x8, 0x4, 0x0, 0xF, 0xB, 0x7, 0x3, 0xE, 0xA, 0x6, 0x2, 0xD, 0x9, 0x5, 0x1, 0xC, 0x8, 0x4, 0x0, 0xF, 0xB, 0x7, 0x3, 0xE, 0xA, 0x6, 0x2, 0xD, 0x9, 0x5, 0x1, 0xC, 0x8, 0x4, 0x0)

#define SIMD_LANE_YCOORD_I _mm512_setr_epi32(128, 384, 640, 896, 1152, 1408, 1664, 1920, 2176, 2432, 2688, 2944, 3200, 3456, 3712, 3968)
#define SIMD_LANE_YCOORD_F _mm512_setr_ps(128.0f, 384.0f, 640.0f, 896.0f, 1152.0f, 1408.0f, 1664.0f, 1920.0f, 2176.0f, 2432.0f, 2688.0f, 2944.0f, 3200.0f, 3456.0f, 3712.0f, 3968.0f)

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// AVX specific typedefs and functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef __m512 __mw;
typedef __m512i __mwi;

#define mw_f32 __m512_f32
#define mw_i32 __m512i_i32

#define _mmw_set1_ps _mm512_set1_ps
#define _mmw_setzero_ps _mm512_setzero_ps
#define _mmw_andnot_ps _mm512_andnot_ps
#define _mmw_fmadd_ps _mm512_fmadd_ps
#define _mmw_fmsub_ps _mm512_fmsub_ps
#define _mmw_min_ps _mm512_min_ps
#define _mmw_max_ps _mm512_max_ps
#define _mmw_shuffle_ps _mm512_shuffle_ps

#define _mmw_set1_epi32 _mm512_set1_epi32
#define _mmw_setzero_epi32 _mm512_setzero_si512
#define _mmw_insertf32x4_ps _mm512_insertf32x4
#define _mmw_andnot_epi32 _mm512_andnot_si512
#define _mmw_min_epi32 _mm512_min_epi32
#define _mmw_max_epi32 _mm512_max_epi32
#define _mmw_subs_epu16 _mm512_subs_epu16
#define _mmw_mullo_epi32 _mm512_mullo_epi32
#define _mmw_srai_epi32 _mm512_srai_epi32
#define _mmw_srli_epi32 _mm512_srli_epi32
#define _mmw_slli_epi32 _mm512_slli_epi32
#define _mmw_sllv_ones(x) _mm512_sllv_epi32(SIMD_BITS_ONE, x)
#define _mmw_transpose_epi8(x) _mm512_shuffle_epi8(x, SIMD_SHUFFLE_SCANLINE_TO_SUBTILES)
#define _mmw_abs_epi32 _mm512_abs_epi32

#define _mmw_cvtps_epi32 _mm512_cvtps_epi32
#define _mmw_cvttps_epi32 _mm512_cvttps_epi32
#define _mmw_cvtepi32_ps _mm512_cvtepi32_ps

#define _mmx_dp4_ps(a, b) _mm_dp_ps(a, b, 0xFF)
#define _mmx_fmadd_ps _mm_fmadd_ps
#define _mmx_max_epi32 _mm_max_epi32
#define _mmx_min_epi32 _mm_min_epi32

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SIMD math operators
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, typename Y> FORCE_INLINE T simd_cast(Y A);
template<> FORCE_INLINE __m128  simd_cast<__m128>(float A) { return _mm_set1_ps(A); }
template<> FORCE_INLINE __m128  simd_cast<__m128>(__m128i A) { return _mm_castsi128_ps(A); }
template<> FORCE_INLINE __m128  simd_cast<__m128>(__m128 A) { return A; }
template<> FORCE_INLINE __m128i simd_cast<__m128i>(int A) { return _mm_set1_epi32(A); }
template<> FORCE_INLINE __m128i simd_cast<__m128i>(__m128 A) { return _mm_castps_si128(A); }
template<> FORCE_INLINE __m128i simd_cast<__m128i>(__m128i A) { return A; }
template<> FORCE_INLINE __m256  simd_cast<__m256>(float A) { return _mm256_set1_ps(A); }
template<> FORCE_INLINE __m256  simd_cast<__m256>(__m256i A) { return _mm256_castsi256_ps(A); }
template<> FORCE_INLINE __m256  simd_cast<__m256>(__m256 A) { return A; }
template<> FORCE_INLINE __m256i simd_cast<__m256i>(int A) { return _mm256_set1_epi32(A); }
template<> FORCE_INLINE __m256i simd_cast<__m256i>(__m256 A) { return _mm256_castps_si256(A); }
template<> FORCE_INLINE __m256i simd_cast<__m256i>(__m256i A) { return A; }
template<> FORCE_INLINE __m512  simd_cast<__m512>(float A) { return _mm512_set1_ps(A); }
template<> FORCE_INLINE __m512  simd_cast<__m512>(__m512i A) { return _mm512_castsi512_ps(A); }
template<> FORCE_INLINE __m512  simd_cast<__m512>(__m512 A) { return A; }
template<> FORCE_INLINE __m512i simd_cast<__m512i>(int A) { return _mm512_set1_epi32(A); }
template<> FORCE_INLINE __m512i simd_cast<__m512i>(__m512 A) { return _mm512_castps_si512(A); }
template<> FORCE_INLINE __m512i simd_cast<__m512i>(__m512i A) { return A; }

// Unary operators
static FORCE_INLINE __m128  operator-(const __m128  &A) { return _mm_xor_ps(A, _mm_set1_ps(-0.0f)); }
static FORCE_INLINE __m128i operator-(const __m128i &A) { return _mm_sub_epi32(_mm_set1_epi32(0), A); }
static FORCE_INLINE __m256  operator-(const __m256  &A) { return _mm256_xor_ps(A, _mm256_set1_ps(-0.0f)); }
static FORCE_INLINE __m256i operator-(const __m256i &A) { return _mm256_sub_epi32(_mm256_set1_epi32(0), A); }
static FORCE_INLINE __m512  operator-(const __m512  &A) { return _mm512_xor_ps(A, _mm512_set1_ps(-0.0f)); }
static FORCE_INLINE __m512i operator-(const __m512i &A) { return _mm512_sub_epi32(_mm512_set1_epi32(0), A); }
static FORCE_INLINE __m128  operator~(const __m128  &A) { return _mm_xor_ps(A, _mm_castsi128_ps(_mm_set1_epi32(~0))); }
static FORCE_INLINE __m128i operator~(const __m128i &A) { return _mm_xor_si128(A, _mm_set1_epi32(~0)); }
static FORCE_INLINE __m256  operator~(const __m256  &A) { return _mm256_xor_ps(A, _mm256_castsi256_ps(_mm256_set1_epi32(~0))); }
static FORCE_INLINE __m256i operator~(const __m256i &A) { return _mm256_xor_si256(A, _mm256_set1_epi32(~0)); }
static FORCE_INLINE __m512  operator~(const __m512  &A) { return _mm512_xor_ps(A, _mm512_castsi512_ps(_mm512_set1_epi32(~0))); }
static FORCE_INLINE __m512i operator~(const __m512i &A) { return _mm512_xor_si512(A, _mm512_set1_epi32(~0)); }
static FORCE_INLINE __m128 abs(const __m128 &a) { return _mm_and_ps(a, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF))); }
static FORCE_INLINE __m256 abs(const __m256 &a) { return _mm256_and_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF))); }
static FORCE_INLINE __m512 abs(const __m512 &a) { return _mm512_and_ps(a, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF))); }

// Binary operators
#define SIMD_BINARY_OP(SIMD_TYPE, BASE_TYPE, prefix, postfix, func, op) \
	static FORCE_INLINE SIMD_TYPE operator##op(const SIMD_TYPE &A, const SIMD_TYPE &B)		{ return _##prefix##_##func##_##postfix(A, B); } \
	static FORCE_INLINE SIMD_TYPE operator##op(const SIMD_TYPE &A, const BASE_TYPE B)		{ return _##prefix##_##func##_##postfix(A, simd_cast<SIMD_TYPE>(B)); } \
	static FORCE_INLINE SIMD_TYPE operator##op(const BASE_TYPE &A, const SIMD_TYPE &B)		{ return _##prefix##_##func##_##postfix(simd_cast<SIMD_TYPE>(A), B); } \
	static FORCE_INLINE SIMD_TYPE &operator##op##=(SIMD_TYPE &A, const SIMD_TYPE &B)		{ return (A = _##prefix##_##func##_##postfix(A, B)); } \
	static FORCE_INLINE SIMD_TYPE &operator##op##=(SIMD_TYPE &A, const BASE_TYPE B)			{ return (A = _##prefix##_##func##_##postfix(A, simd_cast<SIMD_TYPE>(B))); }

#define ALL_SIMD_BINARY_OP(type_suffix, base_type, postfix, func, op) \
	SIMD_BINARY_OP(__m128##type_suffix, base_type, mm, postfix, func, op) \
	SIMD_BINARY_OP(__m256##type_suffix, base_type, mm256, postfix, func, op) \
	SIMD_BINARY_OP(__m512##type_suffix, base_type, mm512, postfix, func, op)

ALL_SIMD_BINARY_OP(, float, ps, add, +)
ALL_SIMD_BINARY_OP(, float, ps, sub, -)
ALL_SIMD_BINARY_OP(, float, ps, mul, *)
ALL_SIMD_BINARY_OP(, float, ps, div, / )
ALL_SIMD_BINARY_OP(i, int, epi32, add, +)
ALL_SIMD_BINARY_OP(i, int, epi32, sub, -)
ALL_SIMD_BINARY_OP(, float, ps, and, &)
ALL_SIMD_BINARY_OP(, float, ps, or , | )
ALL_SIMD_BINARY_OP(, float, ps, xor, ^)
SIMD_BINARY_OP(__m128i, int, mm, si128, and, &)
SIMD_BINARY_OP(__m128i, int, mm, si128, or , | )
SIMD_BINARY_OP(__m128i, int, mm, si128, xor, ^)
SIMD_BINARY_OP(__m256i, int, mm256, si256, and, &)
SIMD_BINARY_OP(__m256i, int, mm256, si256, or , | )
SIMD_BINARY_OP(__m256i, int, mm256, si256, xor, ^)
SIMD_BINARY_OP(__m512i, int, mm512, si512, and, &)
SIMD_BINARY_OP(__m512i, int, mm512, si512, or , | )
SIMD_BINARY_OP(__m512i, int, mm512, si512, xor, ^)

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Specialized AVX input assembly function for general vertex gather 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef MaskedOcclusionCulling::VertexLayout VertexLayout;

static FORCE_INLINE void GatherVertices(__m512 *vtxX, __m512 *vtxY, __m512 *vtxW, const float *inVtx, const unsigned int *inTrisPtr, int numLanes, const VertexLayout &vtxLayout)
{
	assert(numLanes >= 1);

	const __m512i SIMD_TRI_IDX_OFFSET = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45);
	static const __m512i SIMD_LANE_MASK[17] = {
		_mm512_setr_epi32( 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0),
		_mm512_setr_epi32(~0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0),
		_mm512_setr_epi32(~0, ~0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0),
		_mm512_setr_epi32(~0, ~0, ~0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0),
		_mm512_setr_epi32(~0, ~0, ~0, ~0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0),
		_mm512_setr_epi32(~0, ~0, ~0, ~0, ~0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0),
		_mm512_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0),
		_mm512_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0,  0,  0,  0,  0,  0,  0,  0,  0,  0),
		_mm512_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,  0,  0,  0,  0,  0,  0,  0,  0),
		_mm512_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,  0,  0,  0,  0,  0,  0,  0),
		_mm512_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,  0,  0,  0,  0,  0,  0),
		_mm512_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,  0,  0,  0,  0,  0),
		_mm512_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,  0,  0,  0,  0),
		_mm512_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,  0,  0,  0),
		_mm512_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,  0,  0),
		_mm512_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,  0),
		_mm512_setr_epi32(~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0)
	};

	// Compute per-lane index list offset that guards against out of bounds memory accesses
	__m512i safeTriIdxOffset = _mm512_and_si512(SIMD_TRI_IDX_OFFSET, SIMD_LANE_MASK[numLanes]);

	// Fetch triangle indices. 
	__m512i vtxIdx[3];
	vtxIdx[0] = _mmw_mullo_epi32(_mm512_i32gather_epi32(safeTriIdxOffset, (const int*)inTrisPtr + 0, 4), _mmw_set1_epi32(vtxLayout.mStride));
	vtxIdx[1] = _mmw_mullo_epi32(_mm512_i32gather_epi32(safeTriIdxOffset, (const int*)inTrisPtr + 1, 4), _mmw_set1_epi32(vtxLayout.mStride));
	vtxIdx[2] = _mmw_mullo_epi32(_mm512_i32gather_epi32(safeTriIdxOffset, (const int*)inTrisPtr + 2, 4), _mmw_set1_epi32(vtxLayout.mStride));

	char *vPtr = (char *)inVtx;

	// Fetch triangle vertices
	for (int i = 0; i < 3; i++)
	{
		vtxX[i] = _mm512_i32gather_ps(vtxIdx[i], (float *)vPtr, 1);
		vtxY[i] = _mm512_i32gather_ps(vtxIdx[i], (float *)(vPtr + vtxLayout.mOffsetY), 1);
		vtxW[i] = _mm512_i32gather_ps(vtxIdx[i], (float *)(vPtr + vtxLayout.mOffsetW), 1);
	}
}

namespace MaskedOcclusionCullingAVX512
{
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Poorly implemented functions. TODO: fix common (maskedOcclusionCullingCommon.inl) code to improve perf
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	static FORCE_INLINE __m512 _mmw_floor_ps(__m512 x)
	{
		return _mm512_roundscale_ps(x, 1); // 1 = floor
	}

	static FORCE_INLINE __m512 _mmw_ceil_ps(__m512 x)
	{
		return _mm512_roundscale_ps(x, 2); // 2 = ceil
	}

	static FORCE_INLINE __m512i _mmw_cmpeq_epi32(__m512i a, __m512i b)
	{
		__mmask16 mask = _mm512_cmpeq_epi32_mask(a, b);
		return _mm512_mask_mov_epi32(_mm512_set1_epi32(0), mask, _mm512_set1_epi32(~0));
	}

	static FORCE_INLINE __m512i _mmw_cmpgt_epi32(__m512i a, __m512i b)
	{
		__mmask16 mask = _mm512_cmpgt_epi32_mask(a, b);
		return _mm512_mask_mov_epi32(_mm512_set1_epi32(0), mask, _mm512_set1_epi32(~0));
	}

	static FORCE_INLINE bool _mmw_testz_epi32(__m512i a, __m512i b)
	{
		__mmask16 mask = _mm512_cmpeq_epi32_mask(_mm512_and_si512(a, b), _mm512_set1_epi32(0));
		return mask == 0xFFFF;
	}

	static FORCE_INLINE __m512 _mmw_cmpge_ps(__m512 a, __m512 b)
	{
		__mmask16 mask = _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ);
		return _mm512_castsi512_ps(_mm512_mask_mov_epi32(_mm512_set1_epi32(0), mask, _mm512_set1_epi32(~0)));
	}

	static FORCE_INLINE __m512 _mmw_cmpgt_ps(__m512 a, __m512 b)
	{
		__mmask16 mask = _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);
		return _mm512_castsi512_ps(_mm512_mask_mov_epi32(_mm512_set1_epi32(0), mask, _mm512_set1_epi32(~0)));
	}

	static FORCE_INLINE __m512 _mmw_cmpeq_ps(__m512 a, __m512 b)
	{
		__mmask16 mask = _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ);
		return _mm512_castsi512_ps(_mm512_mask_mov_epi32(_mm512_set1_epi32(0), mask, _mm512_set1_epi32(~0)));
	}

	static FORCE_INLINE __mmask16 _mmw_movemask_ps(const __m512 &a)
	{
		__mmask16 mask = _mm512_cmp_epi32_mask(_mm512_and_si512(_mm512_castps_si512(a), _mm512_set1_epi32(0x80000000)), _mm512_set1_epi32(0), 4);	// a & 0x8000000 != 0
		return mask;
	}

	static FORCE_INLINE __m512 _mmw_blendv_ps(const __m512 &a, const __m512 &b, const __m512 &c)
	{
		__mmask16 mask = _mmw_movemask_ps(c);
		return _mm512_mask_mov_ps(a, mask, b);
	} 

	static FORCE_INLINE __m512i _mmw_blendv_epi32(const __m512i &a, const __m512i &b, const __m512i &c)
	{
		return _mm512_castps_si512(_mmw_blendv_ps(_mm512_castsi512_ps(a), _mm512_castsi512_ps(b), _mm512_castsi512_ps(c)));
	}

	static FORCE_INLINE __m512i _mmw_blendv_epi32(const __m512i &a, const __m512i &b, const __m512 &c) 
	{
		return _mm512_castps_si512(_mmw_blendv_ps(_mm512_castsi512_ps(a), _mm512_castsi512_ps(b), c));
	}

	static MaskedOcclusionCulling::Implementation gInstructionSet = MaskedOcclusionCulling::AVX512;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Include common algorithm implementation (general, SIMD independent code)
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	#include "MaskedOcclusionCullingCommon.inl"

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Utility function to create a new object using the allocator callbacks
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	typedef MaskedOcclusionCulling::pfnAlignedAlloc            pfnAlignedAlloc;
	typedef MaskedOcclusionCulling::pfnAlignedFree             pfnAlignedFree;

	static bool DetectAVX512()
	{
		static bool initialized = false;
		static bool AVX512Support = false;

		int cpui[4];
		if (!initialized)
		{
			initialized = true;
			AVX512Support = false;

			int nIds, nExIds;
			__cpuid(cpui, 0);
			nIds = cpui[0];
			__cpuid(cpui, 0x80000000);
			nExIds = cpui[0];

			if (nIds >= 7 && nExIds >= 0x80000001)
			{
				AVX512Support = true;

				// Check support for bit counter instructions (lzcnt)
				__cpuidex(cpui, 0x80000001, 0);
				if ((cpui[2] & 0x20) != 0x20)
					AVX512Support = false;

				// Check masks for misc instructions (FMA)
				static const unsigned int FMA_MOVBE_OSXSAVE_MASK = (1 << 12) | (1 << 22) | (1 << 27);
				__cpuidex(cpui, 1, 0);
				if ((cpui[2] & FMA_MOVBE_OSXSAVE_MASK) != FMA_MOVBE_OSXSAVE_MASK)
					AVX512Support = false;
				
				// Check XCR0 register to ensure that all registers are enabled (by OS)
				static const unsigned int XCR0_MASK = (1 << 7) | (1 << 6) | (1 << 5) | (1 << 2) | (1 << 1); // OPMASK | ZMM0-15 | ZMM16-31 | XMM | YMM
				if (AVX512Support && (_xgetbv(0) & XCR0_MASK) != XCR0_MASK)
					AVX512Support = false;

				// Detect AVX2 & AVX512 instruction sets
				static const unsigned int AVX2_FLAGS = (1 << 3) | (1 << 5) | (1 << 8); // BMI1 (bit manipulation) | BMI2 (bit manipulation)| AVX2
				static const unsigned int AVX512_FLAGS = AVX2_FLAGS | (1 << 16) | (1 << 17) | (1 << 28) | (1 << 30) | (1 << 31); // AVX512F | AVX512DQ | AVX512CD | AVX512BW | AVX512VL
				__cpuidex(cpui, 7, 0);
				if ((cpui[1] & AVX512_FLAGS) != AVX512_FLAGS)
					AVX512Support = false;
			}
		}
		return AVX512Support;
	}

	MaskedOcclusionCulling *CreateMaskedOcclusionCulling(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree)
	{
		if (!DetectAVX512())
			return nullptr;

		MaskedOcclusionCullingPrivate *object = (MaskedOcclusionCullingPrivate *)memAlloc(64, sizeof(MaskedOcclusionCullingPrivate));
		new (object) MaskedOcclusionCullingPrivate(memAlloc, memFree);
		return object;
	}
};

#else

namespace MaskedOcclusionCullingAVX512
{
	typedef MaskedOcclusionCulling::pfnAlignedAlloc            pfnAlignedAlloc;
	typedef MaskedOcclusionCulling::pfnAlignedFree             pfnAlignedFree;

	MaskedOcclusionCulling *CreateMaskedOcclusionCulling(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree)
	{
		return nullptr;
	}
};

#endif
