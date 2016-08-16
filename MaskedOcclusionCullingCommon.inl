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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Common SIMD math utility functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T> FORCE_INLINE T max(const T &a, const T &b) { return a > b ? a : b; }
template<typename T> FORCE_INLINE T min(const T &a, const T &b) { return a < b ? a : b; }

template<typename T, typename Y> FORCE_INLINE T simd_cast(Y A);
template<> FORCE_INLINE __m128  simd_cast<__m128>(float A) { return _mm_set1_ps(A); }
template<> FORCE_INLINE __m128  simd_cast<__m128>(__m128i A) { return _mm_castsi128_ps(A); }
template<> FORCE_INLINE __m128i simd_cast<__m128i>(int A) { return _mm_set1_epi32(A); }
template<> FORCE_INLINE __m128i simd_cast<__m128i>(__m128 A) { return _mm_castps_si128(A); }
template<> FORCE_INLINE __m256  simd_cast<__m256>(float A) { return _mm256_set1_ps(A); }
template<> FORCE_INLINE __m256  simd_cast<__m256>(__m256i A) { return _mm256_castsi256_ps(A); }
template<> FORCE_INLINE __m256i simd_cast<__m256i>(int A) { return _mm256_set1_epi32(A); }
template<> FORCE_INLINE __m256i simd_cast<__m256i>(__m256 A) { return _mm256_castps_si256(A); }

// Unary operators
static FORCE_INLINE __m128  operator-(const __m128  &A) { return _mm_xor_ps(A, _mm_set1_ps(-0.0f)); }
static FORCE_INLINE __m128i operator-(const __m128i &A) { return _mm_sub_epi32(_mm_set1_epi32(0), A); }
static FORCE_INLINE __m256  operator-(const __m256  &A) { return _mm256_xor_ps(A, _mm256_set1_ps(-0.0f)); }
static FORCE_INLINE __m256i operator-(const __m256i &A) { return _mm256_sub_epi32(_mm256_set1_epi32(0), A); }
static FORCE_INLINE __m128  operator~(const __m128  &A) { return _mm_xor_ps(A, _mm_castsi128_ps(_mm_set1_epi32(~0))); }
static FORCE_INLINE __m128i operator~(const __m128i &A) { return _mm_xor_si128(A, _mm_set1_epi32(~0)); }
static FORCE_INLINE __m256  operator~(const __m256  &A) { return _mm256_xor_ps(A, _mm256_castsi256_ps(_mm256_set1_epi32(~0))); }
static FORCE_INLINE __m256i operator~(const __m256i &A) { return _mm256_xor_si256(A, _mm256_set1_epi32(~0)); }
static FORCE_INLINE __m256 abs(const __m256 &a) { return _mm256_and_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF))); }
static FORCE_INLINE __m128 abs(const __m128 &a) { return _mm_and_ps(a, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF))); }

// Binary operators
#define SIMD_BINARY_OP(SIMD_TYPE, BASE_TYPE, prefix, postfix, func, op) \
	static FORCE_INLINE SIMD_TYPE operator##op(const SIMD_TYPE &A, const SIMD_TYPE &B)		{ return _##prefix##_##func##_##postfix(A, B); } \
	static FORCE_INLINE SIMD_TYPE operator##op(const SIMD_TYPE &A, const BASE_TYPE B)		{ return _##prefix##_##func##_##postfix(A, simd_cast<SIMD_TYPE>(B)); } \
	static FORCE_INLINE SIMD_TYPE operator##op(const BASE_TYPE &A, const SIMD_TYPE &B)		{ return _##prefix##_##func##_##postfix(simd_cast<SIMD_TYPE>(A), B); } \
	static FORCE_INLINE SIMD_TYPE &operator##op##=(SIMD_TYPE &A, const SIMD_TYPE &B)		{ return (A = _##prefix##_##func##_##postfix(A, B)); } \
	static FORCE_INLINE SIMD_TYPE &operator##op##=(SIMD_TYPE &A, const BASE_TYPE B)			{ return (A = _##prefix##_##func##_##postfix(A, simd_cast<SIMD_TYPE>(B))); }

#define ALL_SIMD_BINARY_OP(type_suffix, base_type, postfix, func, op) \
	SIMD_BINARY_OP(__m128##type_suffix, base_type, mm, postfix, func, op) \
	SIMD_BINARY_OP(__m256##type_suffix, base_type, mm256, postfix, func, op)

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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Common defines and constants
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SIMD_ALL_LANES_MASK    ((1 << SIMD_LANES) - 1)

// Tile dimensions are 32xN pixels. These values are not tweakable and the code must also be modified
// to support different tile sizes as it is tightly coupled with the SSE/AVX register size
#define TILE_WIDTH_SHIFT       5
#define TILE_WIDTH             (1 << TILE_WIDTH_SHIFT)
#define TILE_HEIGHT            (1 << TILE_HEIGHT_SHIFT)

// Sub-tiles (used for updating the masked HiZ buffer) are 8x4 tiles, so there are 4x2 sub-tiles in a tile
#define SUB_TILE_WIDTH          8
#define SUB_TILE_HEIGHT         4

// The number of fixed point bits used to represent edge slopes, this is enough for supporting 4k x 4k buffers.
// Note that too low precision may cause overshoots / false coverage during rasterization.
#define SLOPE_FP_BITS           16

// Maximum number of triangles that may be generated during clipping. We process SIMD_LANES triangles at a time and
// clip against 5 planes, so the max should be 5*8 = 40 (we immediately draw the first clipped triangle).
// This number must be a power of two.
#define MAX_CLIPPED             64
#define MAX_CLIPPED_WRAP        (MAX_CLIPPED - 1)

// Size of guard band in pixels. Clipping doesn't seem to be very expensive so we use a small guard band
// to improve rasterization performance. It's not recommended to set the guard band to zero, as this may
// cause leakage along the screen border due to precision/rounding.
#define GUARD_BAND_PIXEL_SIZE   1.0f

// We classify triangles as big if the bounding box is wider than this given threshold and use a tighter
// but slightly more expensive traversal algorithm. This improves performance greatly for sliver triangles
#define BIG_TRIANGLE            3

// Only gather statistics if enabled.
#if ENABLE_STATS != 0
	#define STATS_ADD(var, val)     (var) += (val)
#else
	#define STATS_ADD(var, val)
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SIMD common defines (constant values)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SIMD_BITS_ONE       _mmw_set1_epi32(~0)
#define SIMD_BITS_ZERO      _mmw_setzero_epi32()
#define SIMD_TILE_WIDTH     _mmw_set1_epi32(TILE_WIDTH)
#define SIMD_SLOPE_FP_SCALE _mmw_set1_ps(1 << SLOPE_FP_BITS)

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Private class containing the implementation
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class MaskedOcclusionCullingPrivate : public MaskedOcclusionCulling
{
public:
	struct ZTile
	{
		__mw		mZMin[2];
		__mwi		mMask;
	};

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Member variables
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	__mw			mHalfWidth;
	__mw			mHalfHeight;
	__mw			mCenterX;
	__mw			mCenterY;
	__m128			mCSFrustumPlanes[5];
	__m128			mIHalfSize;
	__m128			mICenter;
	__m128i			mIScreenSize;

	float			mNearDist;
	int				mWidth;
	int				mHeight;
	int				mTilesWidth;
	int				mTilesHeight;

	ZTile			*mMaskedHiZBuffer;
	ScissorRect		mFullscreenScissor;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Constructors and state handling
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	MaskedOcclusionCullingPrivate(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree) : mFullscreenScissor(0, 0, 0, 0)
	{
		mMaskedHiZBuffer = nullptr;
		mAlignedAllocCallback = memAlloc;
		mAlignedFreeCallback = memFree;

		mCSFrustumPlanes[0] = _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f);
		mCSFrustumPlanes[1] = _mm_setr_ps(1.0f, 0.0f, 1.0f, 0.0f);
		mCSFrustumPlanes[2] = _mm_setr_ps(-1.0f, 0.0f, 1.0f, 0.0f);
		mCSFrustumPlanes[3] = _mm_setr_ps(0.0f, 1.0f, 1.0f, 0.0f);
		mCSFrustumPlanes[4] = _mm_setr_ps(0.0f, -1.0f, 1.0f, 0.0f);

		memset(&mStats, 0, sizeof(OcclusionCullingStatistics));
	}

	~MaskedOcclusionCullingPrivate() override
	{
		if (mMaskedHiZBuffer != nullptr)
			mAlignedFreeCallback(mMaskedHiZBuffer);
		mMaskedHiZBuffer = nullptr;
	}

	void SetResolution(unsigned int width, unsigned int height) override
	{
		// Resolution must be a multiple of the subtile size
		assert(width % SUB_TILE_WIDTH == 0 && height % SUB_TILE_HEIGHT == 0);
		// Test if combination of resolution & SLOPE_FP_BITS bits may cause 32-bit overflow
		assert(7 * width < (1 << (31 - SLOPE_FP_BITS)));

		// Delete current masked hierarchical Z buffer
		if (mMaskedHiZBuffer != nullptr)
			mAlignedFreeCallback(mMaskedHiZBuffer);
		mMaskedHiZBuffer = nullptr;

		// Setup various resolution dependent constant values
		mWidth = (int)width;
		mHeight = (int)height;
		mTilesWidth = (int)(width + TILE_WIDTH - 1) >> TILE_WIDTH_SHIFT;
		mTilesHeight = (int)(height + TILE_HEIGHT - 1) >> TILE_HEIGHT_SHIFT;
		mCenterX = _mmw_set1_ps((float)mWidth  * 0.5f);
		mCenterY = _mmw_set1_ps((float)mHeight * 0.5f);
		mICenter = _mm_setr_ps((float)mWidth * 0.5f, (float)mWidth * 0.5f, (float)mHeight * 0.5f, (float)mHeight * 0.5f);
		mHalfWidth = _mmw_set1_ps((float)mWidth  * 0.5f);
		mHalfHeight = _mmw_set1_ps((float)mHeight * 0.5f);
		mIHalfSize = _mm_setr_ps((float)mWidth * 0.5f, (float)mWidth * 0.5f, (float)mHeight * 0.5f, (float)mHeight * 0.5f);
		mIScreenSize = _mm_setr_epi32(mWidth - 1, mWidth - 1, mHeight - 1, mHeight - 1);

		// Setup a full screen scissor rectangle
		mFullscreenScissor.mMinX = 0;
		mFullscreenScissor.mMinY = 0;
		mFullscreenScissor.mMaxX = mTilesWidth << TILE_WIDTH_SHIFT;
		mFullscreenScissor.mMaxY = mTilesHeight << TILE_HEIGHT_SHIFT;

		// Adjust clip planes to include a small guard band to avoid clipping leaks
		float guardBandWidth = (2.0f / (float)mWidth) * GUARD_BAND_PIXEL_SIZE;
		float guardBandHeight = (2.0f / (float)mHeight) * GUARD_BAND_PIXEL_SIZE;
		mCSFrustumPlanes[1] = _mm_setr_ps(1.0f - guardBandWidth, 0.0f, 1.0f, 0.0f);
		mCSFrustumPlanes[2] = _mm_setr_ps(-1.0f + guardBandWidth, 0.0f, 1.0f, 0.0f);
		mCSFrustumPlanes[3] = _mm_setr_ps(0.0f, 1.0f - guardBandHeight, 1.0f, 0.0f);
		mCSFrustumPlanes[4] = _mm_setr_ps(0.0f, -1.0f + guardBandHeight, 1.0f, 0.0f);

		// Allocate masked hierarchical Z buffer 
		mMaskedHiZBuffer = (ZTile *)mAlignedAllocCallback(32, sizeof(ZTile) * mTilesWidth * mTilesHeight);
	}

	void SetNearClipPlane(float nearDist) override
	{
		// Setup the near frustum plane
		mNearDist = nearDist;
		mCSFrustumPlanes[0] = _mm_setr_ps(0.0f, 0.0f, 1.0f, -nearDist);
	}

	void ClearBuffer() override
	{
		assert(mMaskedHiZBuffer != nullptr);

		// Iterate through all depth tiles and clear to default values
		for (int i = 0; i < mTilesWidth * mTilesHeight; i++)
		{
			mMaskedHiZBuffer[i].mMask = _mmw_setzero_epi32();

			// Clear z0 to beyond infinity to ensure we never merge with clear data
			mMaskedHiZBuffer[i].mZMin[0] = _mmw_set1_ps(-1.0f);
#if QUICK_MASK != 0
			// Clear z1 to nearest depth value as it is pushed back on each update
			mMaskedHiZBuffer[i].mZMin[1] = _mmw_set1_ps(FLT_MAX);
#else
			mMaskedHiZBuffer[i].mZMin[1] = _mmw_setzero_ps();
#endif
		}

#if ENABLE_STATS != 0
		memset(&mStats, 0, sizeof(OcclusionCullingStatistics));
#endif
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Polygon clipping functions
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	FORCE_INLINE int ClipPolygon(__m128 *outVtx, __m128 *inVtx, const __m128 &plane, int n) const
	{
		__m128 p0 = inVtx[n - 1];
		__m128 dist0 = _mmx_dp4_ps(p0, plane);

		// Loop over all polygon edges and compute intersection with clip plane (if any)
		int nout = 0;
		for (int k = 0; k < n; k++)
		{
			__m128 p1 = inVtx[k];
			__m128 dist1 = _mmx_dp4_ps(p1, plane);
			int dist0Neg = _mm_movemask_ps(dist0);
			if (!dist0Neg)	// dist0 > 0.0f
				outVtx[nout++] = p0;

			// Edge intersects the clip plane if dist0 and dist1 have opposing signs
			if (_mm_movemask_ps(dist0 ^ dist1))
			{
				// Always clip from the positive side to avoid T-junctions
				if (!dist0Neg)
				{
					__m128 t = dist0 / (dist0 - dist1);
					outVtx[nout++] = _mmx_fmadd_ps(p1 - p0, t, p0);
				}
				else
				{
					__m128 t = dist1 / (dist1 - dist0);
					outVtx[nout++] = _mmx_fmadd_ps(p0 - p1, t, p1);
				}
			}

			dist0 = dist1;
			p0 = p1;
		}
		return nout;
	}

	template<ClipPlanes CLIP_PLANE> void TestClipPlane(__mw *vtxX, __mw *vtxY, __mw *vtxW, unsigned int &straddleMask, unsigned int &triMask, ClipPlanes clipPlaneMask)
	{
		straddleMask = 0;
		// Skip masked clip planes
		if (!(clipPlaneMask & CLIP_PLANE))
			return;

		// Evaluate all 3 vertices against the frustum plane
		__mw planeDp[3];
		for (int i = 0; i < 3; ++i)
		{
			switch (CLIP_PLANE)
			{
			case ClipPlanes::CLIP_PLANE_LEFT:   planeDp[i] = vtxW[i] + vtxX[i]; break;
			case ClipPlanes::CLIP_PLANE_RIGHT:  planeDp[i] = vtxW[i] - vtxX[i]; break;
			case ClipPlanes::CLIP_PLANE_BOTTOM: planeDp[i] = vtxW[i] + vtxY[i]; break;
			case ClipPlanes::CLIP_PLANE_TOP:    planeDp[i] = vtxW[i] - vtxY[i]; break;
			case ClipPlanes::CLIP_PLANE_NEAR:   planeDp[i] = vtxW[i] - _mmw_set1_ps(mNearDist); break;
			}
		}

		// Look at FP sign and determine if tri is inside, outside or straddles the frustum plane
		__mw inside = _mmw_andnot_ps(planeDp[0], _mmw_andnot_ps(planeDp[1], ~planeDp[2]));
		__mw outside = planeDp[0] & planeDp[1] & planeDp[2];
		unsigned int inMask = (unsigned int)_mmw_movemask_ps(inside);
		unsigned int outMask = (unsigned int)_mmw_movemask_ps(outside);
		straddleMask = (~outMask) & (~inMask);
		triMask &= ~outMask;
	}

	FORCE_INLINE void ClipTriangleAndAddToBuffer(__mw *vtxX, __mw *vtxY, __mw *vtxW, __m128 *clippedTrisBuffer, int &clipWriteIdx, unsigned int &triMask, unsigned int triClipMask, ClipPlanes clipPlaneMask)
	{
		if (!triClipMask)
			return;

		// Inside test all 3 triangle vertices against all active frustum planes
		unsigned int straddleMask[5];
		TestClipPlane<ClipPlanes::CLIP_PLANE_NEAR>(vtxX, vtxY, vtxW, straddleMask[0], triMask, clipPlaneMask);
		TestClipPlane<ClipPlanes::CLIP_PLANE_LEFT>(vtxX, vtxY, vtxW, straddleMask[1], triMask, clipPlaneMask);
		TestClipPlane<ClipPlanes::CLIP_PLANE_RIGHT>(vtxX, vtxY, vtxW, straddleMask[2], triMask, clipPlaneMask);
		TestClipPlane<ClipPlanes::CLIP_PLANE_BOTTOM>(vtxX, vtxY, vtxW, straddleMask[3], triMask, clipPlaneMask);
		TestClipPlane<ClipPlanes::CLIP_PLANE_TOP>(vtxX, vtxY, vtxW, straddleMask[4], triMask, clipPlaneMask);

		// Clip triangle against straddling planes and add to the clipped triangle buffer
		__m128 vtxBuf[2][8];
		unsigned int clipMask = (straddleMask[0] | straddleMask[1] | straddleMask[2] | straddleMask[3] | straddleMask[4]) & (triClipMask & triMask);
		while (clipMask)
		{
			// Find and setup next triangle to clip
			unsigned int triIdx = find_clear_lsb(&clipMask);
			unsigned int triBit = (1U << triIdx);
			assert(triIdx < SIMD_LANES);

			int bufIdx = 0;
			int nClippedVerts = 3;
			for (int i = 0; i < 3; i++)
				vtxBuf[0][i] = _mm_setr_ps(vtxX[i].mw_f32[triIdx], vtxY[i].mw_f32[triIdx], vtxW[i].mw_f32[triIdx], 1.0f);

			// Clip triangle with straddling planes. 
			for (int i = 0; i < 5; ++i)
			{
				if ((straddleMask[i] & triBit) && (clipPlaneMask & (1 << i)))
				{
					nClippedVerts = ClipPolygon(vtxBuf[bufIdx ^ 1], vtxBuf[bufIdx], mCSFrustumPlanes[i], nClippedVerts);
					bufIdx ^= 1;
				}
			}

			if (nClippedVerts >= 3)
			{
				// Write the first triangle back into the list of currently processed triangles
				for (int i = 0; i < 3; i++)
				{
					vtxX[i].mw_f32[triIdx] = vtxBuf[bufIdx][i].m128_f32[0];
					vtxY[i].mw_f32[triIdx] = vtxBuf[bufIdx][i].m128_f32[1];
					vtxW[i].mw_f32[triIdx] = vtxBuf[bufIdx][i].m128_f32[2];
				}
				// Write the remaining triangles into the clip buffer and process them next loop iteration
				for (int i = 2; i < nClippedVerts - 1; i++)
				{
					clippedTrisBuffer[clipWriteIdx * 3 + 0] = vtxBuf[bufIdx][0];
					clippedTrisBuffer[clipWriteIdx * 3 + 1] = vtxBuf[bufIdx][i];
					clippedTrisBuffer[clipWriteIdx * 3 + 2] = vtxBuf[bufIdx][i + 1];
					clipWriteIdx = (clipWriteIdx + 1) & (MAX_CLIPPED - 1);
				}
			}
			else // Kill triangles that was removed by clipping
				triMask &= ~triBit;
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Common SSE/AVX input assembly functions, note that there are specialized gathers for the general case in the SSE/AVX specific files
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<int N> FORCE_INLINE void VtxFetch4(__mw *v, const unsigned int *inTrisPtr, int triVtx, const float *inVtx, int numLanes)
	{
		// Fetch 4 vectors (matching 1 sse part of the SIMD register), and continue to the next
		const int ssePart = (SIMD_LANES / 4) - N;
		for (int k = 0; k < 4; k++)
		{
			int lane = 4 * ssePart + k;
			if (numLanes > lane)
				v[k] = _mmw_insertf32x4_ps(v[k], _mm_loadu_ps(&inVtx[inTrisPtr[lane * 3 + triVtx] << 2]), ssePart);
		}
		VtxFetch4<N - 1>(v, inTrisPtr, triVtx, inVtx, numLanes);
	}
	template<> FORCE_INLINE void VtxFetch4<0>(__mw *v, const unsigned int *inTrisPtr, int triVtx, const float *inVtx, int numLanes) {}

	FORCE_INLINE void GatherVerticesFast(__mw *vtxX, __mw *vtxY, __mw *vtxW, const float *inVtx, const unsigned int *inTrisPtr, int numLanes)
	{
		// This function assumes that the vertex layout is four packed x, y, z, w-values.
		// Since the layout is known we can get some additional performance by using a 
		// more optimized gather strategy.
		assert(numLanes >= 1);

		// Gather vertices 
		__mw v[4], swz[4];
		for (int i = 0; i < 3; i++)
		{
			// Load 4 (x,y,z,w) vectors per SSE part of the SIMD register (so 4 vectors for SSE, 8 vectors for AVX)
			// this fetch uses templates to unroll the loop
			VtxFetch4<SIMD_LANES / 4>(v, inTrisPtr, i, inVtx, numLanes);

			// Transpose each individual SSE part of the SSE/AVX register (similar to _MM_TRANSPOSE4_PS)
			swz[0] = _mmw_shuffle_ps(v[0], v[1], 0x44);
			swz[2] = _mmw_shuffle_ps(v[0], v[1], 0xEE);
			swz[1] = _mmw_shuffle_ps(v[2], v[3], 0x44);
			swz[3] = _mmw_shuffle_ps(v[2], v[3], 0xEE);

			vtxX[i] = _mmw_shuffle_ps(swz[0], swz[1], 0x88);
			vtxY[i] = _mmw_shuffle_ps(swz[0], swz[1], 0xDD);
			vtxW[i] = _mmw_shuffle_ps(swz[2], swz[3], 0xDD);
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Rasterization functions
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	FORCE_INLINE void ComputeBoundingBox(__mwi &bbminX, __mwi &bbminY, __mwi &bbmaxX, __mwi &bbmaxY, const __mw *vX, const __mw *vY, const ScissorRect *scissor)
	{
		static const __mwi SIMD_PAD_W_MASK = _mmw_set1_epi32(~(TILE_WIDTH - 1));
		static const __mwi SIMD_PAD_H_MASK = _mmw_set1_epi32(~(TILE_HEIGHT - 1));

		// Find Min/Max vertices
		bbminX = _mmw_cvttps_epi32(_mmw_min_ps(vX[0], _mmw_min_ps(vX[1], vX[2])));
		bbminY = _mmw_cvttps_epi32(_mmw_min_ps(vY[0], _mmw_min_ps(vY[1], vY[2])));
		bbmaxX = _mmw_cvttps_epi32(_mmw_max_ps(vX[0], _mmw_max_ps(vX[1], vX[2])));
		bbmaxY = _mmw_cvttps_epi32(_mmw_max_ps(vY[0], _mmw_max_ps(vY[1], vY[2])));

		// Clamp to tile boundaries
		bbminX = _mmw_max_epi32(bbminX & SIMD_PAD_W_MASK, _mmw_set1_epi32(scissor->mMinX));
		bbmaxX = _mmw_min_epi32((bbmaxX + TILE_WIDTH) & SIMD_PAD_W_MASK, _mmw_set1_epi32(scissor->mMaxX));
		bbminY = _mmw_max_epi32(bbminY & SIMD_PAD_H_MASK, _mmw_set1_epi32(scissor->mMinY));
		bbmaxY = _mmw_min_epi32((bbmaxY + TILE_HEIGHT) & SIMD_PAD_H_MASK, _mmw_set1_epi32(scissor->mMaxY));
	}

	FORCE_INLINE void SortVertices(__mw *vX, __mw *vY)
	{
		// Rotate the triangle in the winding order until v0 is the vertex with lowest Y value
		for (int i = 0; i < 2; i++)
		{
			__mw ey1 = vY[1] - vY[0];
			__mw ey2 = vY[2] - vY[0];
			__mw swapMask = (ey1 | ey2 | simd_cast<__mw>(_mmw_cmpeq_epi32(simd_cast<__mwi>(ey2), SIMD_BITS_ZERO)));
			__mw sX, sY;
			sX = _mmw_blendv_ps(vX[2], vX[0], swapMask);
			vX[0] = _mmw_blendv_ps(vX[0], vX[1], swapMask);
			vX[1] = _mmw_blendv_ps(vX[1], vX[2], swapMask);
			vX[2] = sX;
			sY = _mmw_blendv_ps(vY[2], vY[0], swapMask);
			vY[0] = _mmw_blendv_ps(vY[0], vY[1], swapMask);
			vY[1] = _mmw_blendv_ps(vY[1], vY[2], swapMask);
			vY[2] = sY;
		}
	}

	FORCE_INLINE void ComputeDepthPlane(const __mw *pVtxX, const __mw *pVtxY, const __mw *pVtxZ, __mw &zPixelDx, __mw &zPixelDy) const
	{
		// Setup z(x,y) = z0 + dx*x + dy*y screen space depth plane equation
		__mw x2 = pVtxX[2] - pVtxX[0];
		__mw x1 = pVtxX[1] - pVtxX[0];
		__mw y1 = pVtxY[1] - pVtxY[0];
		__mw y2 = pVtxY[2] - pVtxY[0];
		__mw z1 = pVtxZ[1] - pVtxZ[0];
		__mw z2 = pVtxZ[2] - pVtxZ[0];
		__mw d = _mmw_set1_ps(1.0f) / _mmw_fmsub_ps(x1, y2, y1 * x2);
		zPixelDx = _mmw_fmsub_ps(z1, y2, y1 * z2) * d;
		zPixelDy = _mmw_fmsub_ps(x1, z2, z1 * x2) * d;
	}

	FORCE_INLINE void UpdateTileQuick(int tileIdx, const __mwi &coverage, const __mw &zTriv)
	{
		// Update heuristic used in the paper "Masked Software Occlusion Culling", 
		// good balance between performance and accuracy
		STATS_ADD(mStats.mOccluders.mNumTilesUpdated, 1);
		assert(tileIdx >= 0 && tileIdx < mTilesWidth*mTilesHeight);

		__mwi mask = mMaskedHiZBuffer[tileIdx].mMask;
		__mw *zMin = mMaskedHiZBuffer[tileIdx].mZMin;

		// Swizzle coverage mask to 8x4 subtiles and test if any subtiles are not covered at all
		__mwi rastMask = _mmw_transpose_epi8(coverage);
		__mwi deadLane = _mmw_cmpeq_epi32(rastMask, SIMD_BITS_ZERO);

		// Mask out all subtiles failing the depth test (don't update these subtiles)
		deadLane |= _mmw_srai_epi32(simd_cast<__mwi>(zTriv - zMin[0]), 31);
		rastMask = _mmw_andnot_epi32(deadLane, rastMask);

		// Use distance heuristic to discard layer 1 if incoming triangle is significantly nearer to observer
		// than the buffer contents. See Section 3.2 in "Masked Software Occlusion Culling"
		__mwi coveredLane = _mmw_cmpeq_epi32(rastMask, SIMD_BITS_ONE);
		__mw diff = _mmw_fmsub_ps(zMin[1], _mmw_set1_ps(2.0f), zTriv + zMin[0]);
		__mwi discardLayerMask = _mmw_andnot_epi32(deadLane, _mmw_srai_epi32(simd_cast<__mwi>(diff), 31) | coveredLane);

		// Update the mask with incoming triangle coverage
		mask = _mmw_andnot_epi32(discardLayerMask, mask) | rastMask;

		__mwi maskFull = _mmw_cmpeq_epi32(mask, SIMD_BITS_ONE);

		// Compute new value for zMin[1]. This has one of four outcomes: zMin[1] = min(zMin[1], zTriv),  zMin[1] = zTriv, 
		// zMin[1] = FLT_MAX or unchanged, depending on if the layer is updated, discarded, fully covered, or not updated
		__mw opA = _mmw_blendv_ps(zTriv, zMin[1], simd_cast<__mw>(deadLane));
		__mw opB = _mmw_blendv_ps(zMin[1], zTriv, simd_cast<__mw>(discardLayerMask));
		__mw z1min = _mmw_min_ps(opA, opB);
		zMin[1] = _mmw_blendv_ps(z1min, _mmw_set1_ps(FLT_MAX), simd_cast<__mw>(maskFull));

		// Propagate zMin[1] back to zMin[0] if tile was fully covered, and update the mask
		zMin[0] = _mmw_blendv_ps(zMin[0], z1min, simd_cast<__mw>(maskFull));
		mMaskedHiZBuffer[tileIdx].mMask = _mmw_andnot_epi32(maskFull, mask);
	}

	FORCE_INLINE void UpdateTileAccurate(int tileIdx, const __mwi &coverage, const __mw &zTriv)
	{
		assert(tileIdx >= 0 && tileIdx < mTilesWidth*mTilesHeight);

		__mw *zMin = mMaskedHiZBuffer[tileIdx].mZMin;
		__mwi &mask = mMaskedHiZBuffer[tileIdx].mMask;

		// Swizzle coverage mask to 8x4 subtiles
		__mwi rastMask = _mmw_transpose_epi8(coverage);

		// Perform individual depth tests with layer 0 & 1 and mask out all failing pixels 
		__mw sdist0 = zMin[0] - zTriv;
		__mw sdist1 = zMin[1] - zTriv;
		__mwi sign0 = _mmw_srai_epi32(simd_cast<__mwi>(sdist0), 31);
		__mwi sign1 = _mmw_srai_epi32(simd_cast<__mwi>(sdist1), 31);
		__mwi triMask = rastMask & (_mmw_andnot_epi32(mask, sign0) | (mask & sign1));

		// Early out if no pixels survived the depth test (this test is more accurate than
		// the early culling test in TraverseScanline())
		__mwi t0 = _mmw_cmpeq_epi32(triMask, SIMD_BITS_ZERO);
		__mwi t0inv = ~t0; // TODO: change to != instead of using not?
		if (_mmw_testz_epi32(t0inv, t0inv))
			return;

		STATS_ADD(mStats.mOccluders.mNumTilesUpdated, 1);

		__mw zTri = _mmw_blendv_ps(zTriv, zMin[0], simd_cast<__mw>(t0));

		// Test if incoming triangle completely overwrites layer 0 or 1
		__mwi layerMask0 = _mmw_andnot_epi32(triMask, ~mask);
		__mwi layerMask1 = _mmw_andnot_epi32(triMask, mask);
		__mwi lm0 = _mmw_cmpeq_epi32(layerMask0, SIMD_BITS_ZERO);
		__mwi lm1 = _mmw_cmpeq_epi32(layerMask1, SIMD_BITS_ZERO);
		__mw z0 = _mmw_blendv_ps(zMin[0], zTri, simd_cast<__mw>(lm0));
		__mw z1 = _mmw_blendv_ps(zMin[1], zTri, simd_cast<__mw>(lm1));

		// Compute distances used for merging heuristic
		__mw d0 = abs(sdist0);
		__mw d1 = abs(sdist1);
		__mw d2 = abs(z0 - z1);

		// Find minimum distance
		__mwi c01 = simd_cast<__mwi>(d0 - d1);
		__mwi c02 = simd_cast<__mwi>(d0 - d2);
		__mwi c12 = simd_cast<__mwi>(d1 - d2);
		// Two tests indicating which layer the incoming triangle will merge with or 
		// overwrite. d0min indicates that the triangle will overwrite layer 0, and 
		// d1min flags that the triangle will overwrite layer 1.
		__mwi d0min = (c01 & c02) | (lm0 | t0);
		__mwi d1min = _mmw_andnot_epi32(d0min, c12 | lm1);

		///////////////////////////////////////////////////////////////////////////////
		// Update depth buffer entry. NOTE: we always merge into layer 0, so if the 
		// triangle should be merged with layer 1, we first swap layer 0 & 1 and then
		// merge into layer 0.
		///////////////////////////////////////////////////////////////////////////////

		// Update mask based on which layer the triangle overwrites or was merged into
		__mw inner = _mmw_blendv_ps(simd_cast<__mw>(triMask), simd_cast<__mw>(layerMask1), simd_cast<__mw>(d0min));
		mask = simd_cast<__mwi>(_mmw_blendv_ps(inner, simd_cast<__mw>(layerMask0), simd_cast<__mw>(d1min)));

		// Update the zMin[0] value. There are four outcomes: overwrite with layer 1,
		// merge with layer 1, merge with zTri or overwrite with layer 1 and then merge
		// with zTri.
		__mw e0 = _mmw_blendv_ps(z0, z1, simd_cast<__mw>(d1min));
		__mw e1 = _mmw_blendv_ps(z1, zTri, simd_cast<__mw>(d1min | d0min));
		zMin[0] = _mmw_min_ps(e0, e1);

		// Update the zMin[1] value. There are three outcomes: keep current value,
		// overwrite with zTri, or overwrite with z1
		__mw z1t = _mmw_blendv_ps(zTri, z1, simd_cast<__mw>(d0min));
		zMin[1] = _mmw_blendv_ps(z1t, z0, simd_cast<__mw>(d1min));
	}

	template<bool TEST_Z, int MID_VTX_RIGHT, int N_EDGES>
	FORCE_INLINE int TraverseScanline(int tileIdx, int tileIdxEnd, const __mwi *events, const __mw &zTriMin, const __mw &zTriMax, const __mw &iz0, float zx)
	{
		// Floor edge events to integer pixel coordinates (shift out fixed point bits)
		__mwi r0, r1, r2;
		r0 = _mmw_max_epi32(_mmw_srai_epi32(events[0], SLOPE_FP_BITS), SIMD_BITS_ZERO);
		r1 = _mmw_max_epi32(_mmw_srai_epi32(events[1], SLOPE_FP_BITS), SIMD_BITS_ZERO);
		if (N_EDGES == 3)
			r2 = _mmw_max_epi32(_mmw_srai_epi32(events[2], SLOPE_FP_BITS), SIMD_BITS_ZERO);

		__mw z0 = iz0;
		for (;;)
		{
			if (TEST_Z)
				STATS_ADD(mStats.mOccludees.mNumTilesTraversed, 1);
			else
				STATS_ADD(mStats.mOccluders.mNumTilesTraversed, 1);

			// Perform a coarse test to quickly discard occluded tiles
#if QUICK_MASK != 0
			// Only use the reference layer (layer 0) to cull as it is always conservative
			__mw zMinBuf = mMaskedHiZBuffer[tileIdx].mZMin[0];
#else
			// Compute zMin for the overlapped layers 
			__mwi mask = mMaskedHiZBuffer[tileIdx].mMask;
			__mw zMin0 = _mmw_blendv_ps(mMaskedHiZBuffer[tileIdx].mZMin[0], mMaskedHiZBuffer[tileIdx].mZMin[1], simd_cast<__mw>(_mmw_cmpeq_epi32(mask, _mmw_set1_epi32(~0))));
			__mw zMin1 = _mmw_blendv_ps(mMaskedHiZBuffer[tileIdx].mZMin[1], mMaskedHiZBuffer[tileIdx].mZMin[0], simd_cast<__mw>(_mmw_cmpeq_epi32(mask, _mmw_setzero_epi32())));
			__mw zMinBuf = _mmw_min_ps(zMin0, zMin1);
#endif
			__mw dist0 = zTriMax - zMinBuf;
			if (_mmw_movemask_ps(dist0) != SIMD_ALL_LANES_MASK)
			{
				// Compute coverage mask for entire 32xN using shift operations
				__mwi accumulatedMask;
				__mwi m0 = _mmw_sllv_ones(r0);	// Trick, no need to guard against < 0 because of saturated sub
				__mwi m1 = _mmw_sllv_ones(r1);
				if (N_EDGES == 3)
				{
					__mwi m2 = _mmw_sllv_ones(r2);
					if (MID_VTX_RIGHT)
						accumulatedMask = _mmw_andnot_epi32(m0, _mmw_andnot_epi32(m1, m2));
					else
						accumulatedMask = _mmw_andnot_epi32(m0, m1 & m2);
				}
				else
					accumulatedMask = _mmw_andnot_epi32(m0, m1);

				if (TEST_Z)
				{
					// Perform a conservative visibility test (test zMax against buffer for each covered 8x4 subtile)
					__mw zSubTileMax = _mmw_min_ps(z0, zTriMax);
					__mwi zPass = simd_cast<__mwi>(_mmw_cmpge_ps(zSubTileMax, zMinBuf));

					__mwi rastMask = _mmw_transpose_epi8(accumulatedMask);
					__mwi deadLane = _mmw_cmpeq_epi32(rastMask, SIMD_BITS_ZERO);
					zPass = _mmw_andnot_epi32(deadLane, zPass);

					if (!_mmw_testz_epi32(zPass, zPass))
						return CullingResult::VISIBLE;
				}
				else
				{
					// Compute interpolated min for each 8x4 subtile and update the masked hierarchical z buffer entry
					__mw zSubTileMin = _mmw_max_ps(z0, zTriMin);
#if QUICK_MASK != 0
					UpdateTileQuick(tileIdx, accumulatedMask, zSubTileMin);
#else 
					UpdateTileAccurate(tileIdx, accumulatedMask, zSubTileMin);
#endif
				}
			}

			// Update buffer address, interpolate z and edge events
			tileIdx++;
			if (tileIdx >= tileIdxEnd)
				break;
			z0 += zx;
			r0 = _mmw_subs_epu16(r0, SIMD_TILE_WIDTH);	// Trick, use sub saturated to avoid checking against < 0 for shift (values should fit in 16 bits)
			r1 = _mmw_subs_epu16(r1, SIMD_TILE_WIDTH);
			if (N_EDGES == 3)
				r2 = _mmw_subs_epu16(r2, SIMD_TILE_WIDTH);
		}

		return TEST_Z ? CullingResult::OCCLUDED : CullingResult::VISIBLE;
	}

	template<bool TEST_Z, int MID_VTX_RIGHT, int TIGHT_TRAVERSAL>
	FORCE_INLINE int RasterizeTriangle(unsigned int triIdx, int bbWidth, int tileRowIdx, int tileMidRowIdx, int tileEndRowIdx, const __mwi *eventStart, const __mwi *slope, const __mwi *slopeTileDelta, const __mw &zTriMin, const __mw &zTriMax, __mw &z0, float zx, float zy)
	{
		if (TEST_Z)
			STATS_ADD(mStats.mOccludees.mNumRasterizedTriangles, 1);
		else
			STATS_ADD(mStats.mOccluders.mNumRasterizedTriangles, 1);

		int cullResult;

		// Get deltas used to increment edge events each time we traverse one scanline of tiles
		__mwi triSlopeTileDelta[3];
		triSlopeTileDelta[0] = _mmw_set1_epi32(slopeTileDelta[0].mw_i32[triIdx]);
		triSlopeTileDelta[1] = _mmw_set1_epi32(slopeTileDelta[1].mw_i32[triIdx]);
		triSlopeTileDelta[2] = _mmw_set1_epi32(slopeTileDelta[2].mw_i32[triIdx]);

		// Setup edge events for first batch of SIMD_LANES scanlines
		__mwi triEvent[3];
		triEvent[0] = _mmw_set1_epi32(eventStart[0].mw_i32[triIdx]) + _mmw_mullo_epi32(SIMD_LANE_IDX, _mmw_set1_epi32(slope[0].mw_i32[triIdx]));
		triEvent[1] = _mmw_set1_epi32(eventStart[1].mw_i32[triIdx]) + _mmw_mullo_epi32(SIMD_LANE_IDX, _mmw_set1_epi32(slope[1].mw_i32[triIdx]));
		triEvent[2] = _mmw_set1_epi32(eventStart[2].mw_i32[triIdx]) + _mmw_mullo_epi32(SIMD_LANE_IDX, _mmw_set1_epi32(slope[2].mw_i32[triIdx]));

		// For big triangles track start & end tile for each scanline and only traverse the valid region
		int startDelta, endDelta, topDelta, startEvent, endEvent, topEvent;
		if (TIGHT_TRAVERSAL)
		{
			startDelta = slopeTileDelta[2].mw_i32[triIdx];
			endDelta = slopeTileDelta[0].mw_i32[triIdx];
			topDelta = slopeTileDelta[1].mw_i32[triIdx];

			// Compute conservative bounds for the edge events over a 32xN tile
			startEvent = eventStart[2].mw_i32[triIdx] + min(0, startDelta);
			endEvent = eventStart[0].mw_i32[triIdx] + max(0, endDelta) + (TILE_WIDTH << SLOPE_FP_BITS);
			if (MID_VTX_RIGHT)
				topEvent = eventStart[1].mw_i32[triIdx] + max(0, topDelta) + (TILE_WIDTH << SLOPE_FP_BITS);
			else
				topEvent = eventStart[1].mw_i32[triIdx] + min(0, topDelta);
		}

		__mwi events[2];
		// Traverse the bottom half of the triangle
		while (tileRowIdx < tileMidRowIdx)
		{
			events[0] = triEvent[0];
			events[1] = triEvent[2];
			if (TIGHT_TRAVERSAL)
			{
				// Compute tighter start and endpoints to avoid traversing lots of empty space
				int start = max(0, min(bbWidth - 1, startEvent >> (TILE_WIDTH_SHIFT + SLOPE_FP_BITS)));
				int end = min(bbWidth, ((int)endEvent >> (TILE_WIDTH_SHIFT + SLOPE_FP_BITS)));
				events[0] -= start << (TILE_WIDTH_SHIFT + SLOPE_FP_BITS);
				events[1] -= start << (TILE_WIDTH_SHIFT + SLOPE_FP_BITS);
				__mw zOffset = z0 + (float)(start)*zx;

				// Traverse the scanline and update the masked hierarchical z buffer
				cullResult = TraverseScanline<TEST_Z, false, 2>(tileRowIdx + start, tileRowIdx + end, events, zTriMin, zTriMax, zOffset, zx);
				startEvent += startDelta;
				endEvent += endDelta;
			}
			else
				cullResult = TraverseScanline<TEST_Z, false, 2>(tileRowIdx, tileRowIdx + bbWidth, events, zTriMin, zTriMax, z0, zx);

			if (TEST_Z && cullResult == CullingResult::VISIBLE) // Early out if performing occlusion query
				return CullingResult::VISIBLE;

			// move to the next scanline of tiles, update edge events and interpolate z
			tileRowIdx += mTilesWidth;
			z0 += zy;
			triEvent[0] += triSlopeTileDelta[0];
			triEvent[2] += triSlopeTileDelta[2];
		}

		// Traverse the middle scanline of tiles. We must consider all three edges only in this region
		if (tileRowIdx < tileEndRowIdx)
		{
			if (TIGHT_TRAVERSAL)
			{
				// For large triangles, switch the traversal start / end to account for the upper side edge
				endEvent = MID_VTX_RIGHT ? topEvent : endEvent;
				endDelta = MID_VTX_RIGHT ? topDelta : endDelta;
				startEvent = MID_VTX_RIGHT ? startEvent : topEvent;
				startDelta = MID_VTX_RIGHT ? startDelta : topDelta;
				startEvent += startDelta;
				endEvent += endDelta;
			}

			// Traverse the scanline and update the masked hierarchical z buffer. TODO: could compute tighter bounds here as well
			if (MID_VTX_RIGHT)
				cullResult = TraverseScanline<TEST_Z, true, 3>(tileRowIdx, tileRowIdx + bbWidth, triEvent, zTriMin, zTriMax, z0, zx);
			else
				cullResult = TraverseScanline<TEST_Z, false, 3>(tileRowIdx, tileRowIdx + bbWidth, triEvent, zTriMin, zTriMax, z0, zx);

			if (TEST_Z && cullResult == CullingResult::VISIBLE) // Early out if performing occlusion query
				return CullingResult::VISIBLE;

			tileRowIdx += mTilesWidth;
		}

		// Traverse the top half of the triangle
		if (tileRowIdx < tileEndRowIdx)
		{
			// move to the next scanline of tiles, update edge events and interpolate z
			z0 += zy;
			triEvent[MID_VTX_RIGHT + 0] += triSlopeTileDelta[MID_VTX_RIGHT + 0];
			triEvent[MID_VTX_RIGHT + 1] += triSlopeTileDelta[MID_VTX_RIGHT + 1];

			for (;;)
			{
				events[0] = triEvent[MID_VTX_RIGHT + 0];
				events[1] = triEvent[MID_VTX_RIGHT + 1];
				if (TIGHT_TRAVERSAL)
				{
					// Compute tighter start and endpoints to avoid traversing lots of empty space
					int start = max(0, min(bbWidth - 1, startEvent >> (TILE_WIDTH_SHIFT + SLOPE_FP_BITS)));
					int end = min(bbWidth, ((int)endEvent >> (TILE_WIDTH_SHIFT + SLOPE_FP_BITS)));
					events[0] -= start << (TILE_WIDTH_SHIFT + SLOPE_FP_BITS);
					events[1] -= start << (TILE_WIDTH_SHIFT + SLOPE_FP_BITS);
					__mw zOffset = z0 + (float)(start)*zx;

					// Traverse the scanline and update the masked hierarchical z buffer
					cullResult = TraverseScanline<TEST_Z, false, 2>(tileRowIdx + start, tileRowIdx + end, events, zTriMin, zTriMax, zOffset, zx);
					startEvent += startDelta;
					endEvent += endDelta;
				}
				else
					cullResult = TraverseScanline<TEST_Z, false, 2>(tileRowIdx, tileRowIdx + bbWidth, events, zTriMin, zTriMax, z0, zx);

				if (TEST_Z && cullResult == CullingResult::VISIBLE) // Early out if performing occlusion query
					return CullingResult::VISIBLE;

				// move to the next scanline of tiles, update edge events and interpolate z
				tileRowIdx += mTilesWidth;
				if (tileRowIdx >= tileEndRowIdx)
					break;
				z0 += zy;
				triEvent[MID_VTX_RIGHT + 0] += triSlopeTileDelta[MID_VTX_RIGHT + 0];
				triEvent[MID_VTX_RIGHT + 1] += triSlopeTileDelta[MID_VTX_RIGHT + 1];
			}
		}

		return TEST_Z ? CullingResult::OCCLUDED : CullingResult::VISIBLE;
	}

	template<bool TEST_Z, bool FAST_GATHER>
	int RenderTriangles(const float *inVtx, const unsigned int *inTris, int nTris, ClipPlanes clipPlaneMask, const ScissorRect *scissor, const VertexLayout &vtxLayout)
	{
		assert(mMaskedHiZBuffer != nullptr);

		if (TEST_Z)
			STATS_ADD(mStats.mOccludees.mNumProcessedTriangles, 1);
		else
			STATS_ADD(mStats.mOccluders.mNumProcessedTriangles, 1);

		int clipHead = 0;
		int clipTail = 0;
		__m128 clipTriBuffer[MAX_CLIPPED * 3];
		int cullResult = CullingResult::VIEW_CULLED;

		// Setup fullscreen scissor rect as default
		scissor = scissor == nullptr ? &mFullscreenScissor : scissor;

		const unsigned int *inTrisPtr = inTris;
		int numLanes = SIMD_LANES;
		int triIndex = 0;
		while (triIndex < nTris || clipHead != clipTail)
		{
			//////////////////////////////////////////////////////////////////////////////
			// Assemble triangles from the index list and clip if necessary
			//////////////////////////////////////////////////////////////////////////////
			__mw vtxX[3], vtxY[3], vtxW[3];
			unsigned int triMask = SIMD_ALL_LANES_MASK, triClipMask = SIMD_ALL_LANES_MASK;

			if (clipHead != clipTail)
			{
				int clippedTris = clipHead > clipTail ? clipHead - clipTail : MAX_CLIPPED + clipHead - clipTail;
				clippedTris = min(clippedTris, SIMD_LANES);

				// Fill out SIMD registers by fetching more triangles. 
				numLanes = max(0, min(SIMD_LANES - clippedTris, nTris - triIndex));
				if (numLanes > 0) {
					if (FAST_GATHER)
						GatherVerticesFast(vtxX, vtxY, vtxW, inVtx, inTrisPtr, numLanes);
					else
						GatherVertices(vtxX, vtxY, vtxW, inVtx, inTrisPtr, numLanes, vtxLayout);
				}

				for (int clipTri = numLanes; clipTri < numLanes + clippedTris; clipTri++)
				{
					int triIdx = clipTail * 3;
					for (int i = 0; i < 3; i++)
					{
						vtxX[i].mw_f32[clipTri] = clipTriBuffer[triIdx + i].m128_f32[0];
						vtxY[i].mw_f32[clipTri] = clipTriBuffer[triIdx + i].m128_f32[1];
						vtxW[i].mw_f32[clipTri] = clipTriBuffer[triIdx + i].m128_f32[2];
					}
					clipTail = (clipTail + 1) & (MAX_CLIPPED-1);
				}

				triIndex += numLanes;
				inTrisPtr += numLanes * 3;

				triMask = (1U << (clippedTris + numLanes)) - 1;
				triClipMask = (1U << numLanes) - 1; // Don't re-clip already clipped triangles
			}
			else
			{
				numLanes = min(SIMD_LANES, nTris - triIndex);
				triMask = (1U << numLanes) - 1;
				triClipMask = triMask;

				if (FAST_GATHER)
					GatherVerticesFast(vtxX, vtxY, vtxW, inVtx, inTrisPtr, numLanes);
				else
					GatherVertices(vtxX, vtxY, vtxW, inVtx, inTrisPtr, numLanes, vtxLayout);

				triIndex += SIMD_LANES;
				inTrisPtr += SIMD_LANES*3;
			}

			if (clipPlaneMask != ClipPlanes::CLIP_PLANE_NONE)
				ClipTriangleAndAddToBuffer(vtxX, vtxY, vtxW, clipTriBuffer, clipHead, triMask, triClipMask, clipPlaneMask);

			if (triMask == 0x0)
				continue;

			//////////////////////////////////////////////////////////////////////////////
			// Project, transform to screen space and perform backface culling. Note 
			// that we use z = 1.0 / vtx.w for depth, which means that z = 0 is far and
			// z = 1 is near. We must also use a greater than depth test, and in effect
			// everything is reversed compared to regular z implementations.
			//////////////////////////////////////////////////////////////////////////////

			__mw pVtxX[3], pVtxY[3], pVtxZ[3];

			// Project vertices and transform to screen space. Round to nearest integer pixel coordinate
			for (int i = 0; i < 3; i++)
			{
				__mw rcpW = _mmw_set1_ps(1.0f) / vtxW[i];

				// The rounding modes are set to match HW rasterization with OpenGL. In practice our samples are placed
				// in the (1,0) corner of each pixel, while HW rasterizer uses (0.5, 0.5). We get (1,0) because of the 
				// floor used when interpolating along triangle edges. The rounding modes match an offset of (0.5, -0.5)
				pVtxX[i] = _mmw_ceil_ps(_mmw_fmadd_ps(vtxX[i] * mHalfWidth, rcpW, mCenterX));
				pVtxY[i] = _mmw_floor_ps(_mmw_fmadd_ps(vtxY[i] * mHalfHeight, rcpW, mCenterY));
				pVtxZ[i] = rcpW;
			}

			// Perform backface test. 
			__mw triArea1 = (pVtxX[1] - pVtxX[0]) * (pVtxY[2] - pVtxY[0]);
			__mw triArea2 = (pVtxX[0] - pVtxX[2]) * (pVtxY[0] - pVtxY[1]);
			__mw triArea = triArea1 - triArea2;
			triMask &= _mmw_movemask_ps(_mmw_cmpgt_ps(triArea, _mmw_setzero_ps()));

			if (triMask == 0x0)
				continue;

			//////////////////////////////////////////////////////////////////////////////
			// Compute bounding box and clamp to tile coordinates
			//////////////////////////////////////////////////////////////////////////////

			__mwi bbPixelMinX, bbPixelMinY, bbPixelMaxX, bbPixelMaxY;
			ComputeBoundingBox(bbPixelMinX, bbPixelMinY, bbPixelMaxX, bbPixelMaxY, pVtxX, pVtxY, scissor);

			// Clamp bounding box to tiles (it's already padded in computeBoundingBox)
			__mwi bbTileMinX = _mmw_srai_epi32(bbPixelMinX, TILE_WIDTH_SHIFT);
			__mwi bbTileMinY = _mmw_srai_epi32(bbPixelMinY, TILE_HEIGHT_SHIFT);
			__mwi bbTileMaxX = _mmw_srai_epi32(bbPixelMaxX, TILE_WIDTH_SHIFT);
			__mwi bbTileMaxY = _mmw_srai_epi32(bbPixelMaxY, TILE_HEIGHT_SHIFT);
			__mwi bbTileSizeX = bbTileMaxX - bbTileMinX;
			__mwi bbTileSizeY = bbTileMaxY - bbTileMinY;

			// Cull triangles with zero bounding box
			triMask &= ~_mmw_movemask_ps(simd_cast<__mw>((bbTileSizeX - 1) | (bbTileSizeY - 1))) & SIMD_ALL_LANES_MASK;
			if (triMask == 0x0)
				continue;

			if (!TEST_Z)
				cullResult = CullingResult::VISIBLE;

			//////////////////////////////////////////////////////////////////////////////
			// Set up screen space depth plane
			//////////////////////////////////////////////////////////////////////////////

			__mw zPixelDx, zPixelDy;
			ComputeDepthPlane(pVtxX, pVtxY, pVtxZ, zPixelDx, zPixelDy);

			// Compute z value at min corner of bounding box. Offset to make sure z is conservative for all 8x4 subtiles
			__mw bbMinXV0 = _mmw_cvtepi32_ps(bbPixelMinX) - pVtxX[0];
			__mw bbMinYV0 = _mmw_cvtepi32_ps(bbPixelMinY) - pVtxY[0];
			__mw zPlaneOffset = _mmw_fmadd_ps(zPixelDx, bbMinXV0, _mmw_fmadd_ps(zPixelDy, bbMinYV0, pVtxZ[0]));
			__mw zTileDx = zPixelDx * _mmw_set1_ps((float)TILE_WIDTH);
			__mw zTileDy = zPixelDy * _mmw_set1_ps((float)TILE_HEIGHT);
			if (TEST_Z)
				zPlaneOffset += _mmw_max_ps(_mmw_setzero_ps(), zPixelDx*(float)SUB_TILE_WIDTH) + _mmw_max_ps(_mmw_setzero_ps(), zPixelDy*(float)SUB_TILE_HEIGHT);
			else
				zPlaneOffset += _mmw_min_ps(_mmw_setzero_ps(), zPixelDx*(float)SUB_TILE_WIDTH) + _mmw_min_ps(_mmw_setzero_ps(), zPixelDy*(float)SUB_TILE_HEIGHT);

			// Compute Zmin and Zmax for the triangle (used to narrow the range for difficult tiles)
			__mw zMin = _mmw_min_ps(pVtxZ[0], _mmw_min_ps(pVtxZ[1], pVtxZ[2]));
			__mw zMax = _mmw_max_ps(pVtxZ[0], _mmw_max_ps(pVtxZ[1], pVtxZ[2]));

			//////////////////////////////////////////////////////////////////////////////
			// Sort vertices (v0 has lowest Y, and the rest is in winding order) and
			// compute edges. Also find the middle vertex and compute tile
			//////////////////////////////////////////////////////////////////////////////

			SortVertices(pVtxX, pVtxY);

			// Compute edges
			__mw edgeX[3] = { pVtxX[1] - pVtxX[0], pVtxX[2] - pVtxX[1], pVtxX[2] - pVtxX[0] };
			__mw edgeY[3] = { pVtxY[1] - pVtxY[0], pVtxY[2] - pVtxY[1], pVtxY[2] - pVtxY[0] };

			// Classify if the middle vertex is on the left or right and compute its position
			int midVtxRight = ~_mmw_movemask_ps(edgeY[1]);
			__mw midPixelX = _mmw_blendv_ps(pVtxX[1], pVtxX[2], edgeY[1]);
			__mw midPixelY = _mmw_blendv_ps(pVtxY[1], pVtxY[2], edgeY[1]);
			__mwi midTileY = _mmw_srai_epi32(_mmw_max_epi32(_mmw_cvttps_epi32(midPixelY), SIMD_BITS_ZERO), TILE_HEIGHT_SHIFT);
			midTileY = _mmw_max_epi32(bbTileMinY, _mmw_min_epi32(bbTileMaxY, midTileY));

			//////////////////////////////////////////////////////////////////////////////
			// Edge slope setup - Note we do not conform to DX/GL rasterization rules
			//////////////////////////////////////////////////////////////////////////////

			// Compute floating point slopes
			__mw slope[3];
			slope[0] = edgeX[0] / edgeY[0];
			slope[1] = edgeX[1] / edgeY[1];
			slope[2] = edgeX[2] / edgeY[2];

			// Modify slope of horizontal edges to make sure they mask out pixels above/below the edge. The slope is set to screen
			// width to mask out all pixels above or below the horizontal edge. We must also add a small bias to acount for that 
			// vertices may end up off screen due to clipping. We're assuming that the round off error is no bigger than 1.0
			__mw horizontalSlopeDelta = _mmw_set1_ps((float)mWidth + 2.0f*(GUARD_BAND_PIXEL_SIZE + 1.0f));
			slope[0] = _mmw_blendv_ps(slope[0], horizontalSlopeDelta, _mmw_cmpeq_ps(edgeY[0], _mmw_setzero_ps()));
			slope[1] = _mmw_blendv_ps(slope[1], -horizontalSlopeDelta, _mmw_cmpeq_ps(edgeY[1], _mmw_setzero_ps()));

			// Convert floaing point slopes to fixed point
			__mwi slopeFP[3];
			slopeFP[0] = _mmw_cvttps_epi32(slope[0] * SIMD_SLOPE_FP_SCALE);
			slopeFP[1] = _mmw_cvttps_epi32(slope[1] * SIMD_SLOPE_FP_SCALE);
			slopeFP[2] = _mmw_cvttps_epi32(slope[2] * SIMD_SLOPE_FP_SCALE);

			// Fan out edge slopes to avoid (rare) cracks at vertices. We increase right facing slopes 
			// by 1 LSB, which results in overshooting vertices slightly, increasing triangle coverage. 
			// e0 is always right facing, e1 depends on if the middle vertex is on the left or right
			slopeFP[0] = slopeFP[0] + 1;
			slopeFP[1] = slopeFP[1] + _mmw_srli_epi32(~simd_cast<__mwi>(edgeY[1]), 31);

			// Compute slope deltas for an SIMD_LANES scanline step (tile height)
			__mwi slopeTileDelta[3];
			slopeTileDelta[0] = _mmw_slli_epi32(slopeFP[0], TILE_HEIGHT_SHIFT);
			slopeTileDelta[1] = _mmw_slli_epi32(slopeFP[1], TILE_HEIGHT_SHIFT);
			slopeTileDelta[2] = _mmw_slli_epi32(slopeFP[2], TILE_HEIGHT_SHIFT);

			// Compute edge events for the bottom of the bounding box, or for the middle tile in case of 
			// the edge originating from the middle vertex.
			__mwi xDiffi[2], yDiffi[2];
			xDiffi[0] = _mmw_slli_epi32(_mmw_cvtps_epi32(pVtxX[0]) - bbPixelMinX, SLOPE_FP_BITS);
			xDiffi[1] = _mmw_slli_epi32(_mmw_cvtps_epi32(midPixelX) - bbPixelMinX, SLOPE_FP_BITS);
			yDiffi[0] = _mmw_cvtps_epi32(pVtxY[0]) - bbPixelMinY;
			yDiffi[1] = _mmw_cvtps_epi32(midPixelY) - _mmw_slli_epi32(midTileY, TILE_HEIGHT_SHIFT);

			__mwi eventStart[3];
			eventStart[0] = xDiffi[0] - _mmw_mullo_epi32(slopeFP[0], yDiffi[0]);
			eventStart[1] = xDiffi[1] - _mmw_mullo_epi32(slopeFP[1], yDiffi[1]);
			eventStart[2] = xDiffi[0] - _mmw_mullo_epi32(slopeFP[2], yDiffi[0]);

			//////////////////////////////////////////////////////////////////////////////
			// Split bounding box into bottom - middle - top region.
			//////////////////////////////////////////////////////////////////////////////

			__mwi bbBottomIdx = bbTileMinX + _mmw_mullo_epi32(bbTileMinY, _mmw_set1_epi32(mTilesWidth));
			__mwi bbTopIdx = bbTileMinX + _mmw_mullo_epi32(bbTileMinY + bbTileSizeY, _mmw_set1_epi32(mTilesWidth));
			__mwi bbMidIdx = bbTileMinX + _mmw_mullo_epi32(midTileY, _mmw_set1_epi32(mTilesWidth));

			//////////////////////////////////////////////////////////////////////////////
			// Loop over non-culled triangle and change SIMD axis to per-pixel
			//////////////////////////////////////////////////////////////////////////////
			while (triMask)
			{
				unsigned int triIdx = find_clear_lsb(&triMask);
				int triMidVtxRight = (midVtxRight >> triIdx) & 1;

				// Get Triangle Zmin zMax
				__mw zTriMax = _mmw_set1_ps(zMax.mw_f32[triIdx]);
				__mw zTriMin = _mmw_set1_ps(zMin.mw_f32[triIdx]);

				// Setup Zmin value for first set of 8x4 subtiles
				__mw z0 = _mmw_fmadd_ps(_mmw_set1_ps(zPixelDx.mw_f32[triIdx]), SIMD_SUB_TILE_COL_OFFSET_F,
					_mmw_fmadd_ps(_mmw_set1_ps(zPixelDy.mw_f32[triIdx]), SIMD_SUB_TILE_ROW_OFFSET_F, _mmw_set1_ps(zPlaneOffset.mw_f32[triIdx])));
				float zx = zTileDx.mw_f32[triIdx];
				float zy = zTileDy.mw_f32[triIdx];

				// Get dimension of bounding box bottom, mid & top segments
				int bbWidth = bbTileSizeX.mw_i32[triIdx];
				int bbHeight = bbTileSizeY.mw_i32[triIdx];
				int tileRowIdx = bbBottomIdx.mw_i32[triIdx];
				int tileMidRowIdx = bbMidIdx.mw_i32[triIdx];
				int tileEndRowIdx = bbTopIdx.mw_i32[triIdx];

				if (bbWidth > BIG_TRIANGLE && bbHeight > BIG_TRIANGLE) // For big triangles we use a more expensive but tighter traversal algorithm
				{
					if (triMidVtxRight)
						cullResult &= RasterizeTriangle<TEST_Z, 1, 1>(triIdx, bbWidth, tileRowIdx, tileMidRowIdx, tileEndRowIdx, eventStart, slopeFP, slopeTileDelta, zTriMin, zTriMax, z0, zx, zy);
					else
						cullResult &= RasterizeTriangle<TEST_Z, 0, 1>(triIdx, bbWidth, tileRowIdx, tileMidRowIdx, tileEndRowIdx, eventStart, slopeFP, slopeTileDelta, zTriMin, zTriMax, z0, zx, zy);
				}
				else
				{
					if (triMidVtxRight)
						cullResult &= RasterizeTriangle<TEST_Z, 1, 0>(triIdx, bbWidth, tileRowIdx, tileMidRowIdx, tileEndRowIdx, eventStart, slopeFP, slopeTileDelta, zTriMin, zTriMax, z0, zx, zy);
					else
						cullResult &= RasterizeTriangle<TEST_Z, 0, 0>(triIdx, bbWidth, tileRowIdx, tileMidRowIdx, tileEndRowIdx, eventStart, slopeFP, slopeTileDelta, zTriMin, zTriMax, z0, zx, zy);
				}

				if (TEST_Z && cullResult == CullingResult::VISIBLE)
					return CullingResult::VISIBLE;
			}
		}

		return cullResult;
	}

	CullingResult RenderTriangles(const float *inVtx, const unsigned int *inTris, int nTris, ClipPlanes clipPlaneMask, const ScissorRect *scissor, const VertexLayout &vtxLayout) override
	{
		if (vtxLayout.mStride == 16 && vtxLayout.mOffsetY == 4 && vtxLayout.mOffsetW == 12)
			return (CullingResult)RenderTriangles<false, true>(inVtx, inTris, nTris, clipPlaneMask, scissor, vtxLayout);

		return (CullingResult)RenderTriangles<false, false>(inVtx, inTris, nTris, clipPlaneMask, scissor, vtxLayout);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Occlusion query functions
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	CullingResult TestTriangles(const float *inVtx, const unsigned int *inTris, int nTris, ClipPlanes clipPlaneMask, const ScissorRect *scissor, const VertexLayout &vtxLayout)override
	{
		if (vtxLayout.mStride == 16 && vtxLayout.mOffsetY == 4 && vtxLayout.mOffsetW == 12)
			return (CullingResult)RenderTriangles<true, true>(inVtx, inTris, nTris, clipPlaneMask, scissor, vtxLayout);

		return (CullingResult)RenderTriangles<true, false>(inVtx, inTris, nTris, clipPlaneMask, scissor, vtxLayout);
	}

	CullingResult TestRect(float xmin, float ymin, float xmax, float ymax, float wmin) const override
	{
		STATS_ADD(mStats.mOccludees.mNumProcessedRectangles, 1);
		assert(mMaskedHiZBuffer != nullptr);

		static const __m128  SIMD_CEIL_OFFSET = _mm_setr_ps(0.0f, 1.0f, 0.0f, 1.0f);
		static const __m128i SIMD_TILE_PAD = _mm_setr_epi32(0, TILE_WIDTH, 0, TILE_HEIGHT);
		static const __m128i SIMD_TILE_PAD_MASK = _mm_setr_epi32(~(TILE_WIDTH - 1), ~(TILE_WIDTH - 1), ~(TILE_HEIGHT - 1), ~(TILE_HEIGHT - 1));
		static const __m128i SIMD_SUB_TILE_PAD = _mm_setr_epi32(0, SUB_TILE_WIDTH, 0, SUB_TILE_HEIGHT);
		static const __m128i SIMD_SUB_TILE_PAD_MASK = _mm_setr_epi32(~(SUB_TILE_WIDTH - 1), ~(SUB_TILE_WIDTH - 1), ~(SUB_TILE_HEIGHT - 1), ~(SUB_TILE_HEIGHT - 1));

		//////////////////////////////////////////////////////////////////////////////
		// Compute screen space bounding box and guard for out of bounds
		//////////////////////////////////////////////////////////////////////////////

		__m128  pixelBBox = _mm_setr_ps(xmin, xmax, ymin, ymax) * mIHalfSize + mICenter;
		__m128i pixelBBoxi = _mm_cvttps_epi32(pixelBBox);
		pixelBBoxi = _mmx_max_epi32(_mm_setzero_si128(), _mmx_min_epi32(mIScreenSize, pixelBBoxi));

		//////////////////////////////////////////////////////////////////////////////
		// Pad bounding box to (32xN) tiles. Tile BB is used for looping / traversal
		//////////////////////////////////////////////////////////////////////////////
		__m128i tileBBoxi = (pixelBBoxi + SIMD_TILE_PAD) & SIMD_TILE_PAD_MASK;
		int txMin = tileBBoxi.m128i_i32[0] >> TILE_WIDTH_SHIFT;
		int txMax = tileBBoxi.m128i_i32[1] >> TILE_WIDTH_SHIFT;
		int tileRowIdx = (tileBBoxi.m128i_i32[2] >> TILE_HEIGHT_SHIFT)*mTilesWidth;
		int tileRowIdxEnd = (tileBBoxi.m128i_i32[3] >> TILE_HEIGHT_SHIFT)*mTilesWidth;

		if (tileBBoxi.m128i_i32[0] == tileBBoxi.m128i_i32[1] || tileBBoxi.m128i_i32[2] == tileBBoxi.m128i_i32[3])
			return CullingResult::VIEW_CULLED;

		///////////////////////////////////////////////////////////////////////////////
		// Pad bounding box to (8x4) subtiles. Skip SIMD lanes outside the subtile BB
		///////////////////////////////////////////////////////////////////////////////
		__m128i subTileBBoxi = (pixelBBoxi + SIMD_SUB_TILE_PAD) & SIMD_SUB_TILE_PAD_MASK;
		__mwi stxmin = _mmw_set1_epi32(subTileBBoxi.m128i_i32[0] - 1); // - 1 to be able to use GT test
		__mwi stymin = _mmw_set1_epi32(subTileBBoxi.m128i_i32[2] - 1); // - 1 to be able to use GT test
		__mwi stxmax = _mmw_set1_epi32(subTileBBoxi.m128i_i32[1]);
		__mwi stymax = _mmw_set1_epi32(subTileBBoxi.m128i_i32[3]);

		// Setup pixel coordinates used to discard lanes outside subtile BB
		__mwi startPixelX = SIMD_SUB_TILE_COL_OFFSET + tileBBoxi.m128i_i32[0];
		__mwi pixelY = SIMD_SUB_TILE_ROW_OFFSET + tileBBoxi.m128i_i32[2];

		//////////////////////////////////////////////////////////////////////////////
		// Compute z from w. Note that z is reversed order, 0 = far, 1 = near, which
		// means we use a greater than test, so zMax is used to test for visibility.
		//////////////////////////////////////////////////////////////////////////////
		__mw zMax = _mmw_set1_ps(1.0f) / wmin;

		for (;;)
		{
			__mwi pixelX = startPixelX;
			for (int tx = txMin;;)
			{
				STATS_ADD(mStats.mOccludees.mNumTilesTraversed, 1);

				int tileIdx = tileRowIdx + tx;
				assert(tileIdx >= 0 && tileIdx < mTilesWidth*mTilesHeight);

				// Fetch zMin from masked hierarchical Z buffer
#if QUICK_MASK != 0
				__mw zBuf = mMaskedHiZBuffer[tileIdx].mZMin[0];
#else
				__mwi mask = mMaskedHiZBuffer[tileIdx].mMask;
				__mw zMin0 = _mmw_blendv_ps(mMaskedHiZBuffer[tileIdx].mZMin[0], mMaskedHiZBuffer[tileIdx].mZMin[1], simd_cast<__mw>(_mmw_cmpeq_epi32(mask, _mmw_set1_epi32(~0))));
				__mw zMin1 = _mmw_blendv_ps(mMaskedHiZBuffer[tileIdx].mZMin[1], mMaskedHiZBuffer[tileIdx].mZMin[0], simd_cast<__mw>(_mmw_cmpeq_epi32(mask, _mmw_setzero_epi32())));
				__mw zBuf = _mmw_min_ps(zMin0, zMin1);
#endif
				// Perform conservative greater than test against hierarchical Z buffer (zMax >= zBuf means the subtile is visible)
				__mwi zPass = simd_cast<__mwi>(_mmw_cmpge_ps(zMax, zBuf));	//zPass = zMax >= zBuf ? ~0 : 0

				// Mask out lanes corresponding to subtiles outside the bounding box
				__mwi bboxTestMin = _mmw_cmpgt_epi32(pixelX, stxmin) & _mmw_cmpgt_epi32(pixelY, stymin);
				__mwi bboxTestMax = _mmw_cmpgt_epi32(stxmax, pixelX) & _mmw_cmpgt_epi32(stymax, pixelY);
				__mwi boxMask = bboxTestMin & bboxTestMax;
				zPass = zPass & boxMask;

				// If not all tiles failed the conservative z test we can immediately terminate the test
				if (!_mmw_testz_epi32(zPass, zPass))
					return CullingResult::VISIBLE;

				if (++tx >= txMax)
					break;
				pixelX += TILE_WIDTH;
			}

			tileRowIdx += mTilesWidth;
			if (tileRowIdx >= tileRowIdxEnd)
				break;
			pixelY += TILE_HEIGHT;
		}

		return CullingResult::OCCLUDED;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Debugging and statistics
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	MaskedOcclusionCulling::Implementation GetImplementation() override
	{
		return gInstructionSet;
	}

	void ComputePixelDepthBuffer(float *depthData) override
	{
		assert(mMaskedHiZBuffer != nullptr);
		for (int y = 0; y < mHeight; y++)
		{
			for (int x = 0; x < mWidth; x++)
			{
				// Compute 32xN tile index (SIMD value offset)
				int tx = x / TILE_WIDTH;
				int ty = y / TILE_HEIGHT;
				int tileIdx = ty * mTilesWidth + tx;

				// Compute 8x4 subtile index (SIMD lane offset)
				int stx = (x % TILE_WIDTH) / SUB_TILE_WIDTH;
				int sty = (y % TILE_HEIGHT) / SUB_TILE_HEIGHT;
				int subTileIdx = sty * 4 + stx;

				// Compute pixel index in subtile (bit index in 32-bit word)
				int px = (x % SUB_TILE_WIDTH);
				int py = (y % SUB_TILE_HEIGHT);
				int bitIdx = py * 8 + px;

				int pixelLayer = (mMaskedHiZBuffer[tileIdx].mMask.mw_i32[subTileIdx] >> bitIdx) & 1;
				float pixelDepth = mMaskedHiZBuffer[tileIdx].mZMin[pixelLayer].mw_f32[subTileIdx];

				depthData[y * mWidth + x] = pixelDepth;
			}
		}
	}

	OcclusionCullingStatistics GetStatistics() override
	{
		return mStats;
	}

};
