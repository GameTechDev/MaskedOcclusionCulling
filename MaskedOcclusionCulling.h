////////////////////////////////////////////////////////////////////////////////
// Copyright 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.  You may obtain a copy
// of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
// License for the specific language governing permissions and limitations
// under the License.
////////////////////////////////////////////////////////////////////////////////
#pragma once

/*!
 *  \file MaskedOcclusionCulling.h
 *  \brief Masked Occlusion Culling
 * 
 *  General information
 *   - Input to all API functions are (x,y,w) clip-space coordinates (x positive left, y positive up, w positive away from camera).
 *     We entirely skip the z component and instead compute it as 1 / w, see next bullet. For TestRect the input is NDC (x/w, y/w).
 *   - We use a simple z = 1 / w transform, which is a bit faster than OGL/DX depth transforms. Thus, depth is REVERSED and z = 0 at
 *     the far plane and z = inf at w = 0. We also have to use a GREATER depth function, which explains why all the conservative
 *     tests will be reversed compared to what you might be used to (for example zMaxTri >= zMinBuffer is a visibility test)
 *   - We support different layouts for vertex data (basic AoS and SoA), but note that it's beneficial to store the position data
 *     as tightly in memory as possible to reduce cache misses. Big strides are bad, so it's beneficial to keep position as a separate
 *     stream (rather than bundled with attributes) or to keep a copy of the position data for the occlusion culling system.
 *   - The resolution width must be a multiple of 8 and height a multiple of 4.
 *   - The hierarchical Z buffer is stored OpenGL-style with the y axis pointing up. This includes the scissor box.
 *   - This code is only tested with Visual Studio 2015, but should hopefully be easy to port to other compilers.
 */


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Defines used to configure the implementation
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef QUICK_MASK
/*!
 * Configure the algorithm used for updating and merging hierarchical z buffer entries. If QUICK_MASK
 * is defined to 1, use the algorithm from the paper "Masked Software Occlusion Culling", which has good
 * balance between performance and low leakage. If QUICK_MASK is defined to 0, use the algorithm from
 * "Masked Depth Culling for Graphics Hardware" which has less leakage, but also lower performance.
 */
#define QUICK_MASK          1

#endif

#ifndef USE_D3D
/*!
 * Configures the library for use with Direct3D (default) or OpenGL rendering. This changes whether the 
 * screen space Y axis points downwards (D3D) or upwards (OGL), and is primarily important in combination 
 * with the PRECISE_COVERAGE define, where this is important to ensure correct rounding and tie-breaker
 * behaviour. It also affects the ScissorRect screen space coordinates and the memory layout of the buffer 
 * returned by ComputePixelDepthBuffer().
 */
#define USE_D3D             1

#endif

#ifndef PRECISE_COVERAGE
/*!
 * Define PRECISE_COVERAGE to 1 to more closely match GPU rasterization rules. The increased precision comes
 * at a cost of slightly lower performance.
 */
#define PRECISE_COVERAGE    0

#endif

#ifndef ENABLE_STATS
/*!
 * Define ENABLE_STATS to 1 to gather various statistics during occlusion culling. Can be used for profiling 
 * and debugging. Note that enabling this function will reduce performance significantly.
 */
#define ENABLE_STATS        0

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Masked occlusion culling class
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class MaskedOcclusionCulling 
{
public:

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Memory management callback functions
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	typedef void *(*pfnAlignedAlloc)(size_t alignment, size_t size);
	typedef void  (*pfnAlignedFree) (void *ptr);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Enums
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	enum Implementation 
	{
		SSE2 = 0,
		SSE41 = 1,
		AVX2 = 2,
		AVX512 = 3
	};

	enum CullingResult
	{
		VISIBLE = 0x0,
		OCCLUDED = 0x1,
		VIEW_CULLED = 0x3
	};

	enum ClipPlanes
	{
		CLIP_PLANE_NONE = 0x00,
		CLIP_PLANE_NEAR = 0x01,
		CLIP_PLANE_LEFT = 0x02,
		CLIP_PLANE_RIGHT = 0x04,
		CLIP_PLANE_BOTTOM = 0x08,
		CLIP_PLANE_TOP = 0x10,
		CLIP_PLANE_SIDES = (CLIP_PLANE_LEFT | CLIP_PLANE_RIGHT | CLIP_PLANE_BOTTOM | CLIP_PLANE_TOP),
		CLIP_PLANE_ALL = (CLIP_PLANE_LEFT | CLIP_PLANE_RIGHT | CLIP_PLANE_BOTTOM | CLIP_PLANE_TOP | CLIP_PLANE_NEAR)
	};

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Structs
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	/*!
	 * Used to specify custom vertex layout. Memory offsets to y and z coordinates are set through 
	 * mOffsetY and mOffsetW, and vertex stride is given by mStride. It's possible to configure both 
	 * AoS and SoA layouts. Note that large strides may cause more cache misses and decrease 
	 * performance. It is advicable to store position data as compactly in memory as possible.
	 */
	struct VertexLayout
	{
		VertexLayout() {}
		VertexLayout(int stride, int offsetY, int offsetZW) :
			mStride(stride), mOffsetY(offsetY), mOffsetW(offsetZW) {}

		int mStride;  //< byte stride between vertices
		int mOffsetY; //< byte offset from X to Y coordinate
		union {
			int mOffsetZ; //!< byte offset from X to Z coordinate
			int mOffsetW; //!< byte offset from X to W coordinate
		};
	};

	/*!
	 * Used to control scissoring during rasterization. Note that we only provide coarse scissor support. 
	 * The scissor box x coordinates must be a multiple of 32, and the y coordinates a multiple of 8. 
	 * Scissoring is mainly meant as a means of enabling binning (sort middle) rasterizers in case
	 * application developers want to use that approach for multithreading.
	 */
	struct ScissorRect
	{
		ScissorRect() {}
		ScissorRect(int minX, int minY, int maxX, int maxY) :
			mMinX(minX), mMinY(minY), mMaxX(maxX), mMaxY(maxY) {}

		int mMinX;	//!< Screen space X coordinate for left side of scissor rect, inclusive and must be a multiple of 32
		int mMinY;	//!< Screen space Y coordinate for bottom side of scissor rect, inclusive and must be a multiple of 8
		int mMaxX;	//!< Screen space X coordinate for right side of scissor rect, <B>non</B> inclusive and must be a multiple of 32
		int mMaxY;	//!< Screen space Y coordinate for top side of scissor rect, <B>non</B> inclusive and must be a multiple of 8
	};

	/*!
	 * Used to specify storage area for a binlist, containing triangles. This struct is used for binning 
	 * and multithreading. The host application is responsible for allocating memory for the binlists.
	 */
	struct TriList
	{
		unsigned int mNumTriangles; //!< Maximum number of triangles that may be stored in mPtr
		unsigned int mTriIdx;       //!< Index of next triangle to be written, clear before calling BinTriangles to start from the beginning of the list
		float		 *mPtr;         //!< Scratchpad buffer allocated by the host application
	};
	struct OcclusionCullingStatistics
	{
		struct
		{
			long long mNumProcessedTriangles;
			long long mNumRasterizedTriangles;
			long long mNumTilesTraversed;
			long long mNumTilesUpdated;
		} mOccluders;

		struct
		{
			long long mNumProcessedRectangles;
			long long mNumProcessedTriangles;
			long long mNumRasterizedTriangles;
			long long mNumTilesTraversed;
		} mOccludees;
	};

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Functions
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	/*!
	 * \brief Creates a new object with default state, no z buffer attached/allocated.
	 */
	static MaskedOcclusionCulling *Create();
	
	/*!
	 * \brief Creates a new object with default state, no z buffer attached/allocated.
	 * \param memAlloc Pointer to a callback function used when allocating memory 
	 * \param memFree Pointer to a callback function used when freeing memory 
	 */
	static MaskedOcclusionCulling *Create(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree);

	/*!
	 * \brief Destroys an object and frees the z buffer memory. Note that you cannot 
	 * use the delete operator, and should rather use this function to free up memory.
	 */
	static void Destroy(MaskedOcclusionCulling *moc);

	/*!
	 * \brief Sets the resolution of the hierarchical depth buffer. This function will
	 *        re-allocate the current depth buffer (if present). The contents of the
	 *        buffer is undefined until ClearBuffer() is called.
	 *
	 * \param witdh The width of the buffer in pixels, must be a multiple of 8
	 * \param height The height of the buffer in pixels, must be a multiple of 4
	 */
	virtual void SetResolution(unsigned int width, unsigned int height) = 0;

	/*!
	* \brief Gets the resolution of the hierarchical depth buffer. 
	*
	* \param witdh Output: The width of the buffer in pixels
	* \param height Output: The height of the buffer in pixels
	*/
	virtual void GetResolution(unsigned int &width, unsigned int &height) = 0;

	/*!
	 * \brief Returns the tile size for the current implementation.
	 *
	 * \param nBinsW Number of vertical bins, the screen is divided into nBinsW x nBinsH
	 *        rectangular bins.
	 * \param nBinsH Number of horizontal bins, the screen is divided into nBinsW x nBinsH
	 *        rectangular bins.
	 * \param outBinWidth Output: The width of the single bin in pixels (except for the 
	 *        rightmost bin width, which is extended to resolution width)
	 * \param outBinHeight Output: The height of the single bin in pixels (except for the 
	 *        bottommost bin height, which is extended to resolution height)
	 */
	virtual void ComputeBinWidthHeight( unsigned int nBinsW, unsigned int nBinsH, unsigned int & outBinWidth, unsigned int & outBinHeight ) = 0;

    /*!
	 * \brief Sets the distance for the near clipping plane. Default is nearDist = 0.
	 *
	 * \param nearDist The distance to the near clipping plane, given as clip space w
	 */
	virtual void SetNearClipPlane(float nearDist) = 0;

	/*!
	* \brief Gets the distance for the near clipping plane. 
	*/
	virtual float GetNearClipPlane() = 0;

	/*!
	 * \brief Clears the hierarchical depth buffer.
	 */
	virtual void ClearBuffer() = 0;

	/*! 
	 * \brief Renders a mesh of occluder triangles and updates the hierarchical z buffer
	 *        with conservative depth values.
	 *
	 * This function is optimized for vertex layouts with stride 16 and y and w
	 * offsets of 4 and 12 bytes, respectively.
	 *
	 * \param inVtx Pointer to an array of input vertices, should point to the x component
	 *        of the first vertex. The input vertices are given as (x,y,w) cooordinates
	 *        in clip space. The memory layout can be changed using vtxLayout.
	 * \param inTris Pointer to an arrray of vertex indices. Each triangle is created 
	 *        from three indices consecutively fetched from the array.
	 * \param nTris The number of triangles to render (inTris must contain atleast 3*nTris
	 *        entries)
	 * \param modelToClipMatrix all vertices will be transformed by this matrix before
	 *        performing projection. If nullptr is passed the transform step will be skipped
	 * \param clipPlaneMask A mask indicating which clip planes should be considered by the
	 *        triangle clipper. Can be used as an optimization if your application can 
	 *        determine (for example during culling) that a group of triangles does not 
	 *        intersect a certein frustum plane. However, setting an incorrect mask may 
	 *        cause out of bounds memory accesses.
	 * \param scissor A scissor rectangle used to limit the active screen area. Note that
	 *        scissoring is only meant as a means of threading an implementation. The 
	 *        scissor rectangle coordinates must be a multiple of the tile size (32x8).
	 *        This argument is optional. Setting scissor to nullptr disables scissoring.
	 * \param vtxLayout A struct specifying the vertex layout (see struct for detailed 
	 *        description). For best performance, it is advicable to store position data
	 *        as compactly in memory as possible.
	 * \return Will return VIEW_CULLED if all triangles are either outside the frustum or
	 *         backface culled, returns VISIBLE otherwise.
	 */
	virtual CullingResult RenderTriangles(const float *inVtx, const unsigned int *inTris, int nTris, const float *modelToClipMatrix = nullptr, ClipPlanes clipPlaneMask = CLIP_PLANE_ALL, const ScissorRect *scissor = nullptr, const VertexLayout &vtxLayout = VertexLayout(16, 4, 12)) = 0;

	/*!
	 * \brief Occlusion query for a rectangle with a given depth. The rectangle is given 
	 *        in normalized device coordinates where (x,y) coordinates between [-1,1] map 
	 *        to the visible screen area. The query uses a GREATER_EQUAL (reversed) depth 
	 *        test meaning that depth values equal to the contents of the depth buffer are
	 *        counted as visible.
	 *
	 * \param xmin NDC coordinate of the left side of the rectangle.
	 * \param ymin NDC coordinate of the bottom side of the rectangle.
	 * \param xmax NDC coordinate of the right side of the rectangle.
	 * \param ymax NDC coordinate of the top side of the rectangle.
	 * \param ymax NDC coordinate of the top side of the rectangle.
	 * \param wmin Clip space W coordinate for the rectangle.
	 * \return The query will return VISIBLE if the rectangle may be visible, OCCLUDED
	 *         if the rectangle is occluded by a previously rendered  object, or VIEW_CULLED
	 *         if the rectangle is outside the view frustum.
	 */
	virtual CullingResult TestRect(float xmin, float ymin, float xmax, float ymax, float wmin) const = 0;

	/*!
	 * \brief This function is similar to RenderTriangles(), but performs an occlusion
	 *        query instead and does not update the hierarchical z buffer. The query uses 
	 *        a GREATER_EQUAL (reversed) depth test meaning that depth values equal to the 
	 *        contents of the depth buffer are counted as visible.
	 *
	 * This function is optimized for vertex layouts with stride 16 and y and w
	 * offsets of 4 and 12 bytes, respectively.
	 *
	 * \param inVtx Pointer to an array of input vertices, should point to the x component
	 *        of the first vertex. The input vertices are given as (x,y,w) cooordinates
	 *        in clip space. The memory layout can be changed using vtxLayout.
	 * \param inTris Pointer to an arrray of triangle indices. Each triangle is created 
	 *        from three indices consecutively fetched from the array.
	 * \param nTris The number of triangles to render (inTris must contain atleast 3*nTris
	 *        entries)
	 * \param modelToClipMatrix all vertices will be transformed by this matrix before
	 *        performing projection. If nullptr is passed the transform step will be skipped
	 * \param clipPlaneMask A mask indicating which clip planes should be considered by the
	 *        triangle clipper. Can be used as an optimization if your application can
	 *        determine (for example during culling) that a group of triangles does not
	 *        intersect a certein frustum plane. However, setting an incorrect mask may
	 *        cause out of bounds memory accesses.
	 * \param scissor A scissor rectangle used to limit the active screen area. Note that
	 *        scissoring is only meant as a means of threading an implementation. The
	 *        scissor rectangle coordinates must be a multiple of the tile size (32x8).
	 *        This argument is optional. Setting scissor to nullptr disables scissoring.
	 * \param vtxLayout A struct specifying the vertex layout (see struct for detailed 
	 *        description). For best performance, it is advicable to store position data
	 *        as compactly in memory as possible.
	 * \return The query will return VISIBLE if the triangle mesh may be visible, OCCLUDED
	 *         if the mesh is occluded by a previously rendered object, or VIEW_CULLED if all
	 *         triangles are entirely outside the view frustum or backface culled.
	 */
	virtual CullingResult TestTriangles(const float *inVtx, const unsigned int *inTris, int nTris, const float *modelToClipMatrix = nullptr, ClipPlanes clipPlaneMask = CLIP_PLANE_ALL, const ScissorRect *scissor = nullptr, const VertexLayout &vtxLayout = VertexLayout(16, 4, 12)) = 0;

	/*!
	 * \brief Perform input assembly, clipping , projection, triangle setup, and write
	 *        triangles to the screen space bins they overlap. This function can be used to
	 *        distribute work for threading (See the CullingThreadpool class for an example)
	 *
	 * \param inVtx Pointer to an array of input vertices, should point to the x component
	 *        of the first vertex. The input vertices are given as (x,y,w) cooordinates
	 *        in clip space. The memory layout can be changed using vtxLayout.
	 * \param inTris Pointer to an arrray of vertex indices. Each triangle is created
	 *        from three indices consecutively fetched from the array.
	 * \param nTris The number of triangles to render (inTris must contain atleast 3*nTris
	 *        entries)
	 * \param triLists Pointer to an array of TriList objects with one TriList object per
	 *        bin. If a triangle overlaps a bin, it will be written to the corresponding
	 *        trilist. Note that this method appends the triangles to the current list, to
	 *        start writing from the beginning of the list, set triList.mTriIdx = 0
	 * \param nBinsW Number of vertical bins, the screen is divided into nBinsW x nBinsH
	 *        rectangular bins.
	 * \param nBinsH Number of horizontal bins, the screen is divided into nBinsW x nBinsH
	 *        rectangular bins.
	 * \param modelToClipMatrix all vertices will be transformed by this matrix before
	 *        performing projection. If nullptr is passed the transform step will be skipped
	 * \param clipPlaneMask A mask indicating which clip planes should be considered by the
	 *        triangle clipper. Can be used as an optimization if your application can
	 *        determine (for example during culling) that a group of triangles does not
	 *        intersect a certein frustum plane. However, setting an incorrect mask may
	 *        cause out of bounds memory accesses.
	 * \param vtxLayout A struct specifying the vertex layout (see struct for detailed
	 *        description). For best performance, it is advicable to store position data
	 *        as compactly in memory as possible.
	 */
	virtual void BinTriangles(const float *inVtx, const unsigned int *inTris, int nTris, TriList *triLists, unsigned int nBinsW, unsigned int nBinsH, const float *modelToClipMatrix = nullptr, ClipPlanes clipPlaneMask = CLIP_PLANE_ALL, const VertexLayout &vtxLayout = VertexLayout(16, 4, 12)) = 0;

	/*!
	 * \brief Renders all occluder triangles in a trilist. This function can be used in
	 *        combination with BinTriangles() to create a threded (binning) rasterizer. The
	 *        bins can be processed independently by different threads without risking writing
	 *        to overlapping memory regions.
	 *
	 * \param triLists A triangle list, filled using the BinTriangles() function that is to
	 *        be rendered.
	 * \param scissor A scissor box limiting the rendering region to the bin. The size of each
	 *        bin must be a multiple of 32x8 pixels due to implementation constrants. For a
	 *        render target with (width, height) resolution and (nBinsW, nBinsH) bins, the
	 *        size of a bin is:
	 *          binWidth = (width / nBinsW) - (width / nBinsW) % 32;
	 *          binHeight = (height / nBinsH) - (height / nBinsH) % 8;
	 *        The last row and column of tiles have a different size:
	 *          lastColBinWidth = width - (nBinsW-1)*binWidth;
	 *          lastRowBinHeight = height - (nBinsH-1)*binHeight;
	 */
	virtual void RenderTrilist(const TriList &triList, const ScissorRect *scissor) = 0;

	/*!
	 * \brief Creates a per-pixel depth buffer from the hierarchical z buffer representation.
	 *        Intended for visualizing the hierarchical depth buffer for debugging. The 
	 *        buffer is written in scanline order, from the top to bottom (D3D) or bottom to 
	 *        top (OGL) of the surface. See the USE_D3D define.
	 *
	 * \param depthData Pointer to memory where the per-pixel depth data is written. Must
	 *        hold storage for atleast width*height elements as set by setResolution.
	 */
	virtual void ComputePixelDepthBuffer(float *depthData) = 0;
	
	/*!
	 * \brief Fetch occlusion culling statistics, returns zeroes if ENABLE_STATS define is
	 *        not defined. The statistics can be used for profiling or debugging.
	 */
	virtual OcclusionCullingStatistics GetStatistics() = 0;

	/*!
	 * \brief Returns the implementation (CPU instruction set) version of this object.
	 */
	virtual Implementation GetImplementation() = 0;

	/*!
	 * \brief Utility function for transforming vertices and outputting them to an (x,y,z,w)
	 *        format suitable for the occluder rasterization and occludee testing functions.
	 *
	 * \param mtx Pointer to matrix data. The matrix should column major for post 
	 *        multiplication (OGL) and row major for pre-multiplication (DX). This is 
	 *        consistent with OpenGL / DirectX behavior.
	 * \param inVtx Pointer to an array of input vertices. The input vertices are given as
	 *        (x,y,z) cooordinates. The memory layout can be changed using vtxLayout.
	 * \param xfVtx Pointer to an array to store transformed vertices. The transformed
	 *        vertices are always stored as array of structs (AoS) (x,y,z,w) packed in memory.
	 * \param nVtx Number of vertices to transform.
	 * \param vtxLayout A struct specifying the vertex layout (see struct for detailed 
	 *        description). For best performance, it is advicable to store position data
	 *        as compactly in memory as possible. Note that for this function, the
	 *        w-component is assumed to be 1.0.
	 */
	static void TransformVertices(const float *mtx, const float *inVtx, float *xfVtx, unsigned int nVtx, const VertexLayout &vtxLayout = VertexLayout(12, 4, 8));

protected:
	pfnAlignedAlloc mAlignedAllocCallback;
	pfnAlignedFree  mAlignedFreeCallback;

	mutable OcclusionCullingStatistics mStats;

	virtual ~MaskedOcclusionCulling() {}
};
