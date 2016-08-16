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
#define _USE_MATH_DEFINES
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <intrin.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <chrono>

#include "../MaskedOcclusionCulling.h"

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
// Tutorial example code
////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	// Flush denorms to zero to avoid performance issues with small values
	_mm_setcsr(_mm_getcsr() | 0x8040);

	MaskedOcclusionCulling *moc = MaskedOcclusionCulling::Create();

	////////////////////////////////////////////////////////////////////////////////////////
	// Print which version (instruction set) is being used
	////////////////////////////////////////////////////////////////////////////////////////

	MaskedOcclusionCulling::Implementation implementation = moc->GetImplementation();
	switch (implementation) {
		case MaskedOcclusionCulling::SSE2: printf("Using SSE2 version\n"); break;
		case MaskedOcclusionCulling::SSE41: printf("Using SSE41 version\n"); break;
		case MaskedOcclusionCulling::AVX2: printf("Using AVX2 version\n"); break;
	}

	////////////////////////////////////////////////////////////////////////////////////////
	// Setup and state related code
	////////////////////////////////////////////////////////////////////////////////////////

	// Setup a 1920 x 1080 rendertarget with near clip plane at w = 1.0
	const int width = 1920, height = 1080;
	moc->SetResolution(width, height);
	moc->SetNearClipPlane(1.0f);

	// Clear the depth buffer
	moc->ClearBuffer();

	////////////////////////////////////////////////////////////////////////////////////////
	// Render some occluders
	////////////////////////////////////////////////////////////////////////////////////////
	struct ClipspaceVertex { float x, y, z, w; };

	// A triangle that intersects the view frustum
	ClipspaceVertex triVerts[] = { { 5, 0, 0, 10 }, { 30, 0, 0, 20 }, { 10, 50, 0, 40 } };
	unsigned int triIndices[] = { 0, 1, 2 };

	// Render the triangle
	moc->RenderTriangles((float*)triVerts, triIndices, 1);

	// A quad completely within the view frustum
	ClipspaceVertex quadVerts[] = { { -150, -150, 0, 200 }, { -10, -65, 0, 75 }, { 0, 0, 0, 20 }, { -40, 10, 0, 50 } };
	unsigned int quadIndices[] = { 0, 1, 2, 0, 2, 3 };

	// Render the quad. As an optimization, indicate that clipping is not required as it is 
	// completely inside the view frustum
	moc->RenderTriangles((float*)quadVerts, quadIndices, 2, MaskedOcclusionCulling::CLIP_PLANE_NONE);

	// A triangle specified on struct of arrays (SoA) form
	float SoAVerts[] = {
		 10, 10,   7, // x-coordinates
		-10, -7, -10, // y-coordinates
		 10, 10,  10  // w-coordinates
	};

	// Set vertex layout (stride, y offset, w offset)
	MaskedOcclusionCulling::VertexLayout SoAVertexLayout(sizeof(float), 3 * sizeof(float), 6 * sizeof(float));

	// Render triangle with SoA layout
	moc->RenderTriangles((float*)SoAVerts, triIndices, 1, MaskedOcclusionCulling::CLIP_PLANE_ALL, nullptr, SoAVertexLayout);


	////////////////////////////////////////////////////////////////////////////////////////
	// Perform some occlusion queries
	////////////////////////////////////////////////////////////////////////////////////////

	// A triangle, partly overlapped by the quad
	ClipspaceVertex oqTriVerts[] = { { 0, 50, 0, 200 }, { -60, -60, 0, 200 }, { 20, -40, 0, 200 } };
	unsigned int oqTriIndices[] = { 0, 1, 2 };

	// Perform an occlusion query. The triangle is visible and the query should return VISIBLE
	MaskedOcclusionCulling::CullingResult result;
	result = moc->TestTriangles((float*)oqTriVerts, oqTriIndices, 1);
	if (result == MaskedOcclusionCulling::VISIBLE)
		printf("Tested triangle is VISIBLE\n");
	else if (result == MaskedOcclusionCulling::OCCLUDED)
		printf("Tested triangle is OCCLUDED\n");
	else if (result == MaskedOcclusionCulling::VIEW_CULLED)
		printf("Tested triangle is outside view frustum\n");

	// Render the occlusion query triangle to show its position
	moc->RenderTriangles((float*)oqTriVerts, oqTriIndices, 1);


	// Perform an occlusion query testing if a rectangle is visible. The rectangle is completely 
	// behind the previously drawn quad, so the query should indicate that it's occluded
	result = moc->TestRect(-0.6f, -0.6f, -0.4f, -0.4f, 100);
	if (result == MaskedOcclusionCulling::VISIBLE)
		printf("Tested rect is VISIBLE\n");
	else if (result == MaskedOcclusionCulling::OCCLUDED)
		printf("Tested rect is OCCLUDED\n");
	else if (result == MaskedOcclusionCulling::VIEW_CULLED)
		printf("Tested rect is outside view frustum\n");

	// Compute a per pixel depth buffer from the hierarchical depth buffer, used for visualization.
	float *perPixelZBuffer = new float[width * height];
	moc->ComputePixelDepthBuffer(perPixelZBuffer);

	// Tonemap the image
	unsigned char *image = new unsigned char[width * height * 3];
	TonemapDepth(perPixelZBuffer, image, width, height);
	WriteBMP("image.bmp", image, width, height);
	delete [] image;

	// Destroy occlusion culling object and free hierarchical z-buffer
	MaskedOcclusionCulling::Destroy(moc);

#ifndef _DEBUG
	// In release builds, run a simple rasterizer/fillrate benchmark
	void Benchmark();
	Benchmark();
#endif
}

////////////////////////////////////////////////////////////////////////////////////////
// Simple random triangle rasterizer benchmark
////////////////////////////////////////////////////////////////////////////////////////

inline float frand() { return (float)rand() / (float)RAND_MAX; }

void GenerateRandomTriangles(float *verts, unsigned int *triIdx, int nTris, int size, float width, float height)
{
	for (int idx = 0; idx < nTris; ++idx)
	{
		triIdx[idx * 3 + 0] = idx * 3 + 0;
		triIdx[idx * 3 + 1] = idx * 3 + 1;
		triIdx[idx * 3 + 2] = idx * 3 + 2;

		while (true)
		{
			float vtx[3][3] = { { 0, 0, 1 },{ size * 2 / (float)width, 0, 1 },{ 0, size * 2 / (float)height, 1 } };
			float offset[3] = { frand()*2.0f - 1.0f, frand()*2.0f - 1.0f, 0 };
			float rotation = frand() * 2 * (float)M_PI;

			float myz = (float)(nTris - idx);
			bool triOk = true;
			float rvtx[3][3];
			for (int i = 0; i < 3; ++i)
			{
				rvtx[i][0] = cos(rotation)*vtx[i][0] - sin(rotation)*vtx[i][1] + offset[0];
				rvtx[i][1] = sin(rotation)*vtx[i][0] + cos(rotation)*vtx[i][1] + offset[1];
				rvtx[i][2] = 1;

				if (rvtx[i][0] < -1.0f || rvtx[i][0] > 1.0f || rvtx[i][1] < -1.0f || rvtx[i][1] > 1.0f)
					triOk = false;

				int vtxIdx = idx * 3 + i;
				verts[vtxIdx * 4 + 0] = rvtx[i][0] * myz;
				verts[vtxIdx * 4 + 1] = rvtx[i][1] * myz;
				verts[vtxIdx * 4 + 2] = 0.0f;
				verts[vtxIdx * 4 + 3] = rvtx[i][2] * myz;
			}
			if (triOk)
				break;
		}
	}
}

double BenchmarkTriangles(float *verts, unsigned int *tris, int numTriangles, MaskedOcclusionCulling *moc)
{
	moc->ClearBuffer();
	auto before = std::chrono::high_resolution_clock::now();
	moc->RenderTriangles(verts, tris, numTriangles, MaskedOcclusionCulling::CLIP_PLANE_NONE);
	auto after = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<double>(after - before).count();
}

void Benchmark()
{
	// Flush denorms to zero to avoid performance issues with small values
	_mm_setcsr(_mm_getcsr() | 0x8040);

	MaskedOcclusionCulling *moc = MaskedOcclusionCulling::Create();

	////////////////////////////////////////////////////////////////////////////////////////
	// Setup and state related code
	////////////////////////////////////////////////////////////////////////////////////////

	// Setup a 1024 x 1024 rendertarget with near clip plane at w = 1.0
	const int width = 1024, height = 1024;
	moc->SetResolution(width, height);
	moc->SetNearClipPlane(1.0f);

	////////////////////////////////////////////////////////////////////////////////////////
	// Create randomized triangles for back-to-front and front-to-back rendering
	////////////////////////////////////////////////////////////////////////////////////////

	const int numTriangles = 1000 * 256;
	const int sizes[] = { 10, 20, 30, 40, 50, 75, 100, 200, 300, 400, 500 };
	int numSizes = sizeof(sizes) / sizeof(int);

	printf("Generating randomized triangles");
	std::vector<unsigned int *> trisBtF, trisFtB;
	std::vector<float *>		verts;
	for (int i = 0; i < numSizes; ++i)
	{
		float *pVerts = new float[numTriangles * 4 * 3];
		unsigned int *pTris = new unsigned int[numTriangles * 3];
		GenerateRandomTriangles(pVerts, pTris, numTriangles, sizes[i], 1024.0f, 1024.0f);
		verts.push_back(pVerts);
		trisBtF.push_back(pTris);
		printf(".");
	}

	////////////////////////////////////////////////////////////////////////////////////////
	// Perform benchmarks
	////////////////////////////////////////////////////////////////////////////////////////

	printf("\n\nSingle threaded back-to-front rendering (%d kTris)\n", numTriangles / 1000);
	printf("----\n");
	for (int i = 0; i < numSizes; ++i)
	{
		int size = sizes[i];
		double t = BenchmarkTriangles(verts[i], trisBtF[i], numTriangles, moc);
		double GPixelsPerSecond = (double)numTriangles*size*size / (2.0 * 1e9 * t);
		double MTrisPerSecond = (double)numTriangles / (1e6 * t);
		printf("Tri: %3dx%3d - Time: %7.2f ms, MTris/s: %5.2f, GPixels/s: %5.2f \n", size, size, t * 1000.0f, MTrisPerSecond, GPixelsPerSecond);
	}

}
