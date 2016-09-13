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
#include <random>

#include "../MaskedOcclusionCulling.h"

std::mt19937 gRnd;
std::uniform_real_distribution<float> gRndUniform(0, 1);

////////////////////////////////////////////////////////////////////////////////////////
// DX 11 functions
////////////////////////////////////////////////////////////////////////////////////////

#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>

#pragma comment (lib, "d3d11.lib")
#pragma comment (lib, "d3dcompiler.lib")

ID3D11Texture2D				*textureZ;
ID3D11Texture2D				*textureCol, *staging;
ID3D11DepthStencilView		*textureDSV;
ID3D11RenderTargetView		*textureRTV;
ID3D11Device				*device;
ID3D11DeviceContext			*context;

ID3D11RasterizerState		*rastState;
ID3D11DepthStencilState		*DSState;
ID3D11InputLayout			*layout;
ID3D11VertexShader			*VS;
ID3D11PixelShader			*PS;
std::vector<ID3D11Buffer*>	vBuffers;
ID3D11Query					*endQuery;

#define D3DVERIFY(X) if (X != S_OK) exit(1);

void InitD3D(unsigned int width, unsigned int height)
{
	const char *shader =
		"float4 VShader(float4 position : POSITION) : SV_POSITION { return position; }"
		"float4 PShader(float4 position : SV_POSITION) : SV_TARGET { return 1.0f - position.z; }";

	D3D_FEATURE_LEVEL fLevel;
	D3DVERIFY(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device, &fLevel, &context));

	D3D11_TEXTURE2D_DESC tDesc;
	tDesc.Width = width;
	tDesc.Height = height;
	tDesc.MipLevels = tDesc.ArraySize = 1;
	tDesc.SampleDesc.Count = 1;
	tDesc.SampleDesc.Quality = 0;
	tDesc.CPUAccessFlags = 0;
	tDesc.MiscFlags = 0;
	tDesc.Usage = D3D11_USAGE_DEFAULT;

	tDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	tDesc.BindFlags = D3D11_BIND_RENDER_TARGET;
	D3DVERIFY(device->CreateTexture2D(&tDesc, nullptr, &textureCol));

	tDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	tDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	D3DVERIFY(device->CreateTexture2D(&tDesc, nullptr, &textureZ));

	tDesc.Usage = D3D11_USAGE_STAGING;
	tDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	tDesc.BindFlags = 0;
	tDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	D3DVERIFY(device->CreateTexture2D(&tDesc, nullptr, &staging));

	D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc;
	dsvDesc.Flags = 0;
	dsvDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	dsvDesc.Texture2D.MipSlice = 0;
	D3DVERIFY(device->CreateDepthStencilView(textureZ, &dsvDesc, &textureDSV));

	D3D11_RENDER_TARGET_VIEW_DESC rtvDesc;
	rtvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
	rtvDesc.Texture2D.MipSlice = 0;
	D3DVERIFY(device->CreateRenderTargetView(textureCol, &rtvDesc, &textureRTV));

	// Set the viewport
	D3D11_VIEWPORT viewport;
	viewport.TopLeftX = 0;
	viewport.TopLeftY = 0;
	viewport.Width = (float)width;
	viewport.Height = (float)height;
	viewport.MinDepth = 0;
	viewport.MaxDepth = 1;
	context->RSSetViewports(1, &viewport);

	// load and compile the two shaders
	ID3D10Blob *VSBlob, *PSBlob;
	D3DVERIFY(D3DCompile(shader, strlen(shader), "shader", nullptr, nullptr, "VShader", "vs_5_0", 0, 0, &VSBlob, nullptr));
	D3DVERIFY(D3DCompile(shader, strlen(shader), "shader", nullptr, nullptr, "PShader", "ps_5_0", 0, 0, &PSBlob, nullptr));

	// encapsulate both shaders into shader objects
	D3DVERIFY(device->CreateVertexShader(VSBlob->GetBufferPointer(), VSBlob->GetBufferSize(), NULL, &VS));
	D3DVERIFY(device->CreatePixelShader(PSBlob->GetBufferPointer(), PSBlob->GetBufferSize(), NULL, &PS));

	// set the shader objects
	context->VSSetShader(VS, 0, 0);
	context->PSSetShader(PS, 0, 0);

	// create the input layout object
	D3D11_INPUT_ELEMENT_DESC ied[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
	};

	D3DVERIFY(device->CreateInputLayout(ied, 1, VSBlob->GetBufferPointer(), VSBlob->GetBufferSize(), &layout));
	context->IASetInputLayout(layout);

	D3D11_RASTERIZER_DESC rDesc;
	rDesc.FillMode = D3D11_FILL_SOLID;
	rDesc.CullMode = D3D11_CULL_NONE;
	rDesc.FrontCounterClockwise = false;
	rDesc.DepthBias = 0;
	rDesc.DepthBiasClamp = 0;
	rDesc.SlopeScaledDepthBias = 0;
	rDesc.DepthClipEnable = false;
	rDesc.ScissorEnable = false;
	rDesc.MultisampleEnable = false;
	rDesc.AntialiasedLineEnable = false;
	device->CreateRasterizerState(&rDesc, &rastState);
	context->RSSetState(rastState);

	D3D11_DEPTH_STENCIL_DESC dsDesc;
	ZeroMemory(&dsDesc, sizeof(D3D11_DEPTH_STENCIL_DESC));
	dsDesc.DepthEnable = true;
	dsDesc.StencilEnable = false;
	dsDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
	dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	device->CreateDepthStencilState(&dsDesc, &DSState);
	context->OMSetDepthStencilState(DSState, 0);

	D3D11_QUERY_DESC qDesc;
	qDesc.Query = D3D11_QUERY_EVENT;
	qDesc.MiscFlags = 0;
	D3DVERIFY(device->CreateQuery(&qDesc, &endQuery));
}

bool D3DValidateTriangle(float *verts, MaskedOcclusionCulling *moc)
{
	unsigned int width, height;
	moc->GetResolution(width, height);

	////////////////////////////////////////////////////////////////////////////////////////
	// Draw triangle using our framework and read back depth buffer
	////////////////////////////////////////////////////////////////////////////////////////

	// Draw triangle using our framework
	unsigned int indices[3] = { 0,1,2 };
	moc->ClearBuffer();
	moc->RenderTriangles(verts, indices, 1);

	// Read back result
	float *depthBuffer = new float[width*height];
	moc->ComputePixelDepthBuffer(depthBuffer);

	////////////////////////////////////////////////////////////////////////////////////////
	// Draw triangle using DirectX 11 and read back color image
	////////////////////////////////////////////////////////////////////////////////////////

	// Create D3D buffer
	ID3D11Buffer *buf = nullptr;

	D3D11_SUBRESOURCE_DATA iData;
	iData.pSysMem = verts;
	iData.SysMemPitch = 0;
	iData.SysMemSlicePitch = 0;

	D3D11_BUFFER_DESC bDesc;
	bDesc.Usage = D3D11_USAGE_DEFAULT;
	bDesc.ByteWidth = 3 * 4 * sizeof(float);
	bDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bDesc.CPUAccessFlags = bDesc.MiscFlags = bDesc.StructureByteStride = 0;
	D3DVERIFY(device->CreateBuffer(&bDesc, &iData, &buf));

	// Clear renderbuffers
	float clearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
	context->OMSetRenderTargets(1, &textureRTV, textureDSV);
	context->ClearRenderTargetView(textureRTV, clearColor);
	context->ClearDepthStencilView(textureDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);

	// Draw triangle using D3D
	UINT stride = sizeof(float) * 4, offset = 0;
	context->IASetVertexBuffers(0, 1, &buf, &stride, &offset);
	context->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	context->Draw(3, 0);

	buf->Release();

	// Read back resulting D3D image 
	unsigned char *d3dimg = new unsigned char[width * height * 4];
	D3D11_MAPPED_SUBRESOURCE map;
	context->CopyResource(staging, textureCol);
	context->Map(staging, 0, D3D11_MAP_READ, 0, &map);
	memcpy(d3dimg, map.pData, width*height * 4);
	context->Unmap(staging, 0);

	////////////////////////////////////////////////////////////////////////////////////////
	// Compare rasterized coverage
	////////////////////////////////////////////////////////////////////////////////////////

	unsigned char *pD3D = (unsigned char*)map.pData;
	bool identical = true;
	for (unsigned int y = 0; y < height; ++y)
	{
		for (unsigned int x = 0; x < width; ++x)
		{
			bool d3dcov = (d3dimg[(x + y*width) * 4] + d3dimg[(x + y*width) * 4 + 1] + d3dimg[(x + y*width) * 4 + 2]) != 0;
			bool ourcov = depthBuffer[x + y*width] != -1.0f;

			if (d3dcov != ourcov)
				identical = false;
		}
	}

	delete depthBuffer;
	delete[] d3dimg;

	return identical;
}

////////////////////////////////////////////////////////////////////////////////////////
// Random triangle generator
////////////////////////////////////////////////////////////////////////////////////////

void RandomTriangle(float *verts)
{
	float xprod = -1;
	while (xprod < 0.0f)
	{
		for (unsigned int i = 0; i < 3; ++i)
		{
			verts[i * 4 + 0] = gRndUniform(gRnd) * 2.0f - 1.0f;
			verts[i * 4 + 1] = gRndUniform(gRnd) * 2.0f - 1.0f;
			verts[i * 4 + 2] = 0.0f;
			verts[i * 4 + 3] = 1.0f;
		}

		// Test if triangle is front facing (ccw winded)
		float v0x = verts[4] - verts[0];
		float v0y = verts[5] - verts[1];
		float v1x = verts[8] - verts[0];
		float v1y = verts[9] - verts[1];
		xprod = (v0x * v1y) - (v0y * v1x);
	}
}

////////////////////////////////////////////////////////////////////////////////////////
// Main: Validate a large number of randomized triangles
////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	const int width = 2048, height = 2048;

	// Flush denorms to zero to avoid performance issues with small values
	_mm_setcsr(_mm_getcsr() | 0x8040);

	// Initialize directx
	InitD3D(width, height);

	MaskedOcclusionCulling *moc = MaskedOcclusionCulling::Create();

	////////////////////////////////////////////////////////////////////////////////////////
	// Setup and state related code
	////////////////////////////////////////////////////////////////////////////////////////

	// Setup a rendertarget with near clip plane at w = 1.0
	moc->SetResolution(width, height);
	moc->SetNearClipPlane(1.0f);

	float verts[3*4];
	static const int nTriangles = 1000;
	int nPassed = 0;
	for (int i = 0; i < nTriangles; ++i)
	{
		RandomTriangle(verts);
		bool pass = D3DValidateTriangle(verts, moc);
		if (pass)
			nPassed++;
		else
			printf("Testing triangle %d... FALIED\n", i);

		if (i % 100 == 0)
			printf("Testing triangle %d\n", i);
	}
	printf("%d / %d triangles passed\n", nPassed, nTriangles);
}