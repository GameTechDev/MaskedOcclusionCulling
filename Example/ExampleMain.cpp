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
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <intrin.h>
#include <algorithm>

#include "../MaskedOcclusionCulling.h"

////////////////////////////////////////////////////////////////////////////////////////
// Code to detect AVX2 support. This code is based on the intel's example:
// https://software.intel.com/en-us/articles/how-to-detect-new-instruction-support-in-the-4th-generation-intel-core-processor-family
////////////////////////////////////////////////////////////////////////////////////////
void run_cpuid(int eax, int ecx, int* abcd)
{
	__cpuidex(abcd, eax, ecx);
}
int check_xcr0_ymm()
{
	int xcr0;
	xcr0 = (int)_xgetbv(0); 
	return ((xcr0 & 6) == 6); 
}
int check_4th_gen_intel_core_features()
{
	int abcd[4];
	int fma_movbe_osxsave_mask = ((1 << 12) | (1 << 22) | (1 << 27));
	int avx2_bmi12_mask = (1 << 5) | (1 << 3) | (1 << 8);
	run_cpuid(1, 0, abcd);
	if ((abcd[2] & fma_movbe_osxsave_mask) != fma_movbe_osxsave_mask)
		return 0;
	if (!check_xcr0_ymm())
		return 0;
	run_cpuid(7, 0, abcd);
	if ((abcd[1] & avx2_bmi12_mask) != avx2_bmi12_mask)
		return 0;
	run_cpuid(0x80000001, 0, abcd);
	if ((abcd[2] & (1 << 5)) == 0)
		return 0;
	return 1;
}

////////////////////////////////////////////////////////////////////////////////////////
// Image utility functions, minimal BMP writer and depth buffer tone mapping
////////////////////////////////////////////////////////////////////////////////////////

static void WriteBMP(const char *filename, const unsigned char *data, int w, int h)
{
	short header[] = { 0x4D42, 0, 0, 0, 0, 26, 0, 12, 0, (short)w, (short)h, 1, 24 };
	FILE *f = fopen(filename, "wb");
	fwrite(header, 1, sizeof(header), f);
	fwrite(data, 1, w * h * 3, f);
	fclose(f);
}

static void TonemapDepth(float *depth, unsigned char *image, int w, int h)
{
	// Find min/max w coordinate (discard cleared pixels)
	float minW = FLT_MAX, maxW = 0.0f;
	for (int i = 0; i < w*h; ++i)
	{
		if (depth[i] > 0.0f)
		{
			minW = std::min(minW, depth[i]);
			maxW = std::max(maxW, depth[i]);
		}
	}

	// Tonemap depth values
	for (int i = 0; i < w*h; ++i)
	{
		int intensity = 0;
		if (depth[i] > 0)
			intensity = (unsigned char)(223.0*(depth[i] - minW) / (maxW - minW) + 32.0);
			
		image[i * 3 + 0] = intensity;
		image[i * 3 + 1] = intensity;
		image[i * 3 + 2] = intensity;
	}
}

////////////////////////////////////////////////////////////////////////////////////////
// Example code
////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	// Detect AVX2 support (required)
	if (!check_4th_gen_intel_core_features())
	{
		printf("ERROR: this code requires AVX2 support\n");
		exit(1);
	}

	// Flush denorms to zero to avoid performance issues with small values
	_mm_setcsr(_mm_getcsr() | 0x8040);

	MaskedOcclusionCulling moc;

	////////////////////////////////////////////////////////////////////////////////////////
	// Setup and state related code
	////////////////////////////////////////////////////////////////////////////////////////

	// Setup a 1920 x 1080 rendertarget with near clip plane at w = 1.0
	const int width = 1920, height = 1080;
	moc.SetResolution(width, height);
	moc.SetNearClipPlane(1.0f);

	// Clear the depth buffer
	moc.ClearBuffer();

	////////////////////////////////////////////////////////////////////////////////////////
	// Render some occluders
	////////////////////////////////////////////////////////////////////////////////////////
	struct ClipspaceVertex { float x, y, z, w; };

	// A triangle that intersects the view frustum. Note that the z component of each vertex is unused
	ClipspaceVertex triVerts[] = { { 5, 0, 0, 10 },{ 30, 0, 0, 20 },{ 10, 50, 0, 40 } };
	unsigned int triIndices[] = { 0, 1, 2 };

	// Render the triangle
	moc.RenderTriangles((float*)triVerts, triIndices, 1);

	// A quad completely within the view frustum
	ClipspaceVertex quadVerts[] = { { -150, -150, 0, 200 },{ -10, -65, 0, 75 },{ 0, 0, 0, 20 },{ -40, 10, 0, 50 } };
	unsigned int quadIndices[] = { 0, 1, 2, 0, 2, 3 };

	// Render the quad. As an optimization, indicate that clipping is not required as it is 
	// completely inside the view frustum
	moc.RenderTriangles((float*)quadVerts, quadIndices, 2, MaskedOcclusionCulling::CLIP_PLANE_NONE);

	// A triangle specified on AoS array of structs (AoS) form
	float AoSVerts[] = {
		 10, 10,   7, // x-coordinates
		-10, -7, -10, // y-coordinates
		 10, 10,  10  // w-coordinates
	};

	// Set vertex layout (stride, y offset, w offset)
	MaskedOcclusionCulling::VertexLayout AoSVertexLayout(sizeof(float), 3 * sizeof(float), 6 * sizeof(float));

	// Render triangle with AoS layout
	moc.RenderTriangles((float*)AoSVerts, triIndices, 1, MaskedOcclusionCulling::CLIP_PLANE_ALL, nullptr, AoSVertexLayout);


	////////////////////////////////////////////////////////////////////////////////////////
	// Perform some occlusion queries
	////////////////////////////////////////////////////////////////////////////////////////

	// A triangle, partly overlapped by the quad
	ClipspaceVertex oqTriVerts[] = { { 0, 50, 0, 200 },{ -60, -60, 0, 200 },{ 20, -40, 0, 200 } };
	unsigned int oqTriIndices[] = { 0, 1, 2 };

	// Perform an occlusion query. The triangle is visible and the query should return VISIBLE
	MaskedOcclusionCulling::CullingResult result;
	result = moc.TestTriangles((float*)oqTriVerts, oqTriIndices, 1);
	if (result == MaskedOcclusionCulling::VISIBLE)
		printf("Tested triangle is VISIBLE\n");
	else if (result == MaskedOcclusionCulling::OCCLUDED)
		printf("Tested triangle is OCCLUDED\n");
	else if (result == MaskedOcclusionCulling::VIEW_CULLED)
		printf("Tested triangle is outside view frustum\n");

	// Render the occlusion query triangle to show its position
	moc.RenderTriangles((float*)oqTriVerts, oqTriIndices, 1);


	// Perform an occlusion query testing if a rectangle is visible. The rectangle is completely 
	// behind the previously drawn quad, so the query should indicate that it's occluded
	result = moc.TestRect(-0.6f, -0.6f, -0.4f, -0.4f, 100);
	if (result == MaskedOcclusionCulling::VISIBLE)
		printf("Tested rect is VISIBLE\n");
	else if (result == MaskedOcclusionCulling::OCCLUDED)
		printf("Tested rect is OCCLUDED\n");
	else if (result == MaskedOcclusionCulling::VIEW_CULLED)
		printf("Tested rect is outside view frustum\n");

	// Compute a per pixel depth buffer from the hierarchical depth buffer, used for visualization.
	float *perPixelZBuffer = new float[width * height];
	moc.ComputePixelDepthBuffer(perPixelZBuffer);

	// Tonemap the image
	unsigned char *image = new unsigned char[width * height * 3];
	TonemapDepth(perPixelZBuffer, image, width, height);
	WriteBMP("image.bmp", image, width, height);
}
