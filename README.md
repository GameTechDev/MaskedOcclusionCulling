# MaskedOcclusionCulling

This code accompanies the research paper ["Masked Software Occlusion Culling"](http://fileadmin.cs.lth.se/graphics/research/papers/2016/culling/), 
and implements an efficient alternative to the hierarchical depth buffer algorithm. Our algorithm decouples depth values and coverage, and operates 
directly on the hierarchical depth buffer. It lets us efficiently parallelize both coverage computations and hierarchical depth buffer updates.

## Requirements

This code uses AVX2 specific instructions and will only run on AVX2 capable CPUs. Supporting machines with older instruction sets, such as SSE, 
is therefore *not* just a simple matter of changing the tile size. It may be possible to make an efficient SSE implementation, 
but we currently have no such plans.

## <a name="cs"></a>Notes on coordinate systems and winding

Most inputs are given as clip space (x,y,w) coordinates assuming the same right handed coordinate system as used by DirectX and OpenGL (x positive right, y 
positive up and w positive in the view direction). Note that we use the clip space w coordinate for depth and disregard the z coordinate. Internally our 
masked hierarchical depth buffer stores *depth = 1 / w*. Backface culling is **always enabled**, with counterclockwise winding considered front-facing. 

The `TestRect()` function is an exception and instead accepts normalized device coordinates (NDC), *(x' = x/w, y' = y/w)*, where the visible screen region
maps to the range [-1,1] for *x'* and *y'* (x positive right and y positive up). Again, this is consistent with both DirectX and OpenGL behavior.

Finally, the screen space coordinate system used internally to access our hierarchical depth buffer follows OpenGL convensions (y positive up), which is 
**not** consistent with DirectX (y positive down). This does not matter during typical use, but if you wish to query/visualize our internal depth representation 
using the `ComputePixelDepthBuffer()` function, you should be aware of this.

## API / Tutorial

We have made an effort to keep the API as simple and minimal as possible. The rendering functions are quite similar to submitting DirectX or OpenGL drawcalls 
and we hope they will feel natural to anyone with graphics programming experience. In the following we will use the example project as a tutorial to showcase 
the API. Please refer to the documentation in the header file for further details.

### State 

We begin by creating a new instance of the occlusion culling object. The default constructor initializes the object to a default state.

```C++
MaskedOcclusionCulling moc;
```

The created object is empty has no hierarchical depth buffer attached, so we must first allocate a buffer using the `SetResolution()` function. This function can 
also be used later to resize the hierarchical depth buffer, causing it to be re-allocated. Note that the resolution width must be a multiple of 8, and the height 
a multiple of 4. This is a limitation of the occlusion culling algorithm.

```C++
int width = 1920;
int height = 1080;
moc.SetResolution(width, height);   // Set full HD resolution
```
After setting the resolution we can start rendering occluders and performing occlusion queries. We must first clear the hierarchical depth buffer

```C++
// Clear hierarchical depth buffer to far depth
moc.ClearDepthBuffer();
```

**Optional** The `SetNearClipPlane()` function can be used to configure the distance to the near clipping plane to make the occlusion culling renderer match your DX/GL 
renderer. The default value for the near plane is 0 which should work as expected unless your application relies on having onscreen geometry clipped by
the near plane.

```C++
float nearClipDist = 1.0f;
moc.SetNearClipPlane(nearClipDist); // Set near clipping dist (optional)
```

### Occluder rendering

The `RenderTriangles()` function renders triangle meshes to the hierarchical depth buffer. Similar to DirectX/OpenGL, meshes are constructed from a vertex array 
and an triangle index array. By default, the vertices are given as *(x,y,z,w)* floating point clip space coordinates, but the *z*-coordinate is ignored and 
instead we use *depth = 1 / w*. We expose a `TransformVertices()` utility function to transform vertices from *(x,y,z,1)* model/world space to *(x,y,z,w)* clip 
space, but you can use your own transform code as well. For more information on the `TransformVertices()` function, please refer to the documentaiton in the 
header file.

The triangle index array is identical to a DirectX or OpenGL triangle list and connects vertices to form triangles. Every three indices in the array form a new 
triangle, so the size of the array must be a multiple of 3. Note that we only support triangle lists, and we currently have no plans on supporting other primitives
such as strips or fans.

```C++
struct ClipSpaceVertex {float x, y, w};

// Create an example triangle . Note that the z component of each vertex is unused
ClipspaceVertex triVerts[] = { { 5, 0, 0, 10 }, { 30, 0, 0, 20 }, { 10, 50, 0, 40 } };
unsigned int triIndices[] = { 0, 1, 2 };
unsigned int nTris = 1;

// Render an example triangle 
moc.RenderTriangles(triVerts, triIndices, nTris);
```

The `RenderTriangles()` accepts an additional parameter to optimize polygon clipping. The calling application may disable any clipping plane if it can 
guarantee that the mesh does not intersect said clipping plane. In the example below we have a quad which is entirely on screen, and we can disable 
all clipping planes. **Warning** it is unsafe to incorrectly disable clipping planes and this may cause the program to crash or perform out of bounds 
memory accesses. Consider this a power user feature (use `CLIP_PLANE_ALL` to clip against the full frustum when in doubt).

```C++
// Create a quad completely within the view frustum
ClipspaceVertex quadVerts[] 
	= { { -150, -150, 0, 200 },{ -10, -65, 0, 75 },{ 0, 0, 0, 20 },{ -40, 10, 0, 50 } };
unsigned int quadIndices[] = { 0, 1, 2, 0, 2, 3 };
unsigned int nTris = 2;

// Render the quad. As an optimization, indicate that clipping is not required 
moc.RenderTriangles((float*)quadVerts, quadIndices, nTris, CLIP_PLANE_NONE);
```

Finally, the `RenderTriangles()` supports configurable vertex layout. The code so far has used an array of structs (AoS) layout based on the `ClipSpaceVertex`, 
but you may use the `VertexLayout` struct to configure the memory layout of the vertex data. Note that the vertex pointer passed to the `RenderTriangles()` 
should point at the *x* coordinate of the first vertex, so there is no x coordinate offset specified in the struct.

```C++	
struct VertexLayout
{
	int mStride;  // Stride between vertices
	int mOffsetY; // Offset to vertex y coordinate
	int mOffsetW; // Offset to vertex w coordinate
};
```

For example, you can configure a struct of arrays (SoA) layout as follows

```C++
// A triangle specified on AoS array of structs (AoS) form
float AoSVerts[] = {
	 10, 10,   7, // x-coordinates
	-10, -7, -10, // y-coordinates
	 10, 10,  10  // w-coordinates
};

// Set vertex layout (stride, y offset, w offset)
VertexLayout AoSVertexLayout(sizeof(float), 3 * sizeof(float), 6 * sizeof(float));

// Render triangle with AoS layout
moc.RenderTriangles((float*)AoSVerts, triIndices, 1, CLIP_PLANE_ALL, nullptr, AoSVertexLayout);
```

Vertex layout may affect occlusion culling performance. We have seen no large performance impact when using either SoA or AoS layout, but generally speaking the 
vertex position data should be packed as compactly into memory as possible to minimize number of cache misses. It is, for example, not advicable to bundle vertex 
position data together with normals, texture coordinates, etc. and using a large stride. 

### Occlusion queries

After rendering a few occluder meshes you can begin to perform occlusion queries. There are two functions for occlusion queries, called `TestTriangles()` and
`TestRect()`. The `TestTriangles()` function is identical to `RenderTriangles()` with the exception being that it performs an occlusion query and does not 
update the hierarchical depth buffer. The result of the occlusion query is returned as an enum, which indicates if the triangles are `VISIBLE`, `OCCLUDED`, or were 
`VIEW_CULLED`. Here, `VIEW_CULLED` means that all triangles were either frustum or back face culling, so no occlusion culling test had to be performed.

```C++	
// A triangle that is partly, but not completely, overlapped by the quad rendered before
ClipspaceVertex oqTriVerts[] = { { 0, 50, 0, 200 },{ -60, -60, 0, 200 },{ 20, -40, 0, 200 } };
unsigned int oqTriIndices[] = { 0, 1, 2 };
unsigned int nTris = 1;

// Perform an occlusion query. The triangle is visible and the query should return VISIBLE
CullingResult result = moc.TestTriangles((float*)oqTriVerts, oqTriIndices, nTris);
```

The `TestRect()` function performs an occlusion query for a rectangular screen space region with a given depth. It can be used to, for example, quickly test
the projected bounding box of an object to determine if the entire object is visible or not. The function is considerably faster than `TestTriangles()` becuase 
it does not require input assembly, clipping, or triangle rasterization. The queries are typically less accurate as screen space bounding rectangles tend to 
grow quite large, but we've personally seen best overall performance using this type of culling.

```C++	
// Perform an occlusion query testing if a rectangle is visible. The rectangle is completely 
// behind the previously drawn quad, so the query should indicate that it's occluded
result = moc.TestRect(-0.6f, -0.6f, -0.4f, -0.4f, 100);
```

Unlike the other functions the input to `TestRect()` is normalized device coordinates (NDC). Normalized device coordinates are projected clip space coordinates 
*(x' = x/w, y' = y/w)* and the visible screen maps to the range [-1,1] for both the *x'* and *y'* coordinate. The w coordinate is still given in clip space, 
however. It is up to the application to compute projected bounding rectangles from the object's bounding shapes.

### Debugging and visualization

We expose a utility function, `ComputePixelDepthBuffer()` that can be used to visualize the hierarchical depth buffer used internally by the occlusion culling 
system. The function fills in a complete per-pixel depth buffer, but the internal representation is hierarchical with just two depth values and a mask stored per 
tile. It is not reasonable to expect the image to completely match the exact depth buffer, and you may notice some areas where backrgound objects leak through 
the foreground. Leakage is part of the algorithm (and one reason for the high performance), and we have 
not found it to be problematic. However, if you experience issues due to leakage you may want to disable the `QUICK_MASK` define, described in more detail in the 
section on [hierarchical depth buffer updates](#update).

```C++
// Compute a per pixel depth buffer from the hierarchical depth buffer, used for visualization.
float *perPixelZBuffer = new float[width * height];
moc.ComputePixelDepthBuffer(perPixelZBuffer);
```

We also support basic instrumentation to help with profiling and debugging. By defining `ENABLE_STATS` in the header file, the occlusion culling code will 
gather statistics about the number of occluders rendered and occlusion queries performed. For more details about the statistics, see the 
`OcclusionCullingStatistics` struct. The statistics can be queried using the `GetStatistics()` function, which will simply return a zeroed struct if `ENABLE_STATS` 
is not defined. Note that instrumentation reduce performance somewhat and should generally be disabled in release builds.

```C++
OcclusionCullingStatistics stats = moc.GetStatistics();
```
## <a name="update"></a>Hierarchical depth buffer update algorithm and render order

The implementation contains two update algorithms / heuristics for the hierarchical depth buffer, one focused on speed and one focused on accuracy. The 
active algorithm can be configured using the `QUICK_MASK` define. Setting the define (default) enables algorithm is described in the research paper 
["Masked Software Occlusion Culling"](http://fileadmin.cs.lth.se/graphics/research/papers/2016/culling/), which has a good balance between low leakage 
and good performance. Not defining `QUICK_MASK` enables the mergine heuristic used in the paper ["Masked depth culling for graphics hardware"](http://dl.acm.org/citation.cfm?id=2818138). 
It is more accurate, with less leakage, but also has lower performance. 

If you experience problems due to leakage you may want to use the more accurate update algorithm. However, rendering order can also affect the quality 
of the hierarchical depth buffer, with the best order being rendering objects front-to-back. We perform early depth culling tests during occluder 
rendering, so rendering in front-to-back order will not only improve quality, but also greatly improve performance of occluder rendering. If your scene 
is stored in a hierarchical data structure, it is often possible to modify the traversal algorithm to traverse nodes in approximate front-to-back order, 
see the research paper ["Masked Software Occlusion Culling"](http://fileadmin.cs.lth.se/graphics/research/papers/2016/culling/) for an example.

## Interleaving occluder rendering and occlusion queries

This implementation supports *light weight* switching between occluder rendering and occlusion queries. While it is still possible to do occlusion culling
as a standard two pass algorithm (first render all occluders, then perform all queries) it is typically beneficial to interleave occluder rendering with 
queries. 

This is especially powerful when rendering objects in front-to-back order. After drawing the first few occluder triangles, you can start performing 
occlusion queries, and if the occlusion query indicate that an object is occluded there is no need to draw the occluder mesh for that object. This 
can greatly improve the performance of the occlusion culling pass in itself. As described in further detail in the research paper 
["Masked Software Occlusion Culling"](http://fileadmin.cs.lth.se/graphics/research/papers/2016/culling/), this may be used to perform early exits in 
BVH traversal code.

## Multi-threading and scissoring

We have no plans on making multi-threading a part of the libraray. However, all occluder rendering and occlusion query functions are thread safe except 
for the hierarchical depth buffer updates. Using a binning (sort middle) rasterizer it one way to ensure that all threads works on a separate part of the 
hierarchical depth buffer, and can be use as a means of multi-threading. As shown in the example below, we expose basic support for binning through a coarse 
scissor rectangle in the `RenderTriangles()` function. Note that scissoring is meant as a means of multi-threading, and we therefore do not support fine 
grained scissor rectangles. The x coordinate must be a multiple of 32, and the y coordinate a multiple of 8. The scissor box is given in screen space 
coordinates, please see the section on [coordinate systems](#cs) for details.

```C++	
// Create scissor rectangles for the left and right half of the screen.
ScissorRect scissorLeft  = {0, 0, 960, 1080};
ScissorRect scissorRight = {960, 0, 1920, 1080};

moc.RenderTriangles(triVerts, triIndices, nTris, CLIP_PLANE_ALL, &scissorLeft);  // thread 0
moc.RenderTriangles(triVerts, triIndices, nTris, CLIP_PLANE_ALL, &scissorRight); // thread 1
```
## Memory management

As shown in the example below, you may optionally provide callback functions for allocating and freeing memory when constructing a 
`MaskedOcclusionCulling` object. The functions must support aligned allocations (at least to 32 byte boundaries). 

```C++
void *alignedAllocCallback(size_t alignment, size_t size)
{
	...
}

void alignedFreeCallback(void *ptr)
{
	...
}

MaskedOcclusionCulling moc(alignedAllocCallback, alignedFreeCallback);
```


## Differences to the research paper

This code does not exactly match implementation used in ["Masked Software Occlusion Culling"](http://fileadmin.cs.lth.se/graphics/research/papers/2016/culling/), 
and performance may vary slightly from what is presented in the research paper. We aimed for making the API as simple as possible and have removed many 
limitations, in particular requirements on input data being aligned to SIMD boundaries. This affects performance slightly in both directions. Unaligned 
loads and gathers are more costly, but unaligned data may be packed more efficiently in memory leading to fewer cache misses. 

## License agreement

Copyright (c) 2016, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
- Neither the name of Intel Corporation nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

