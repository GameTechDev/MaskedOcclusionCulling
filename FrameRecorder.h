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
  * \file FrameRecorder.h
  * \brief Masked occlusion culling recorder class (set ENABLE_RECORDER to 1 to enable)
  *
  * Masked occlusion culling recorder class (To enable, set ENABLE_RECORDER to 1 in MaskedOcclusionCulling.h)
  *
  * Enables gathering and storing all triangle rendering and all testing calls and their results to a file, for
  * later playback and performance testing.
  * Usage info:
  *  - Calling MaskedOcclusionCulling::RecorderStart with a file name will open the file and start recording all subsequent
  *    triangle rendering and any testing calls including the test results, and MaskedOcclusionCulling::RecorderStop will 
  *    stop recording and close the file.
  *  - ClearBuffer-s are not recorded so if recording multiple frames, Start/RecorderStop is needed for each frame. For 
  *    correctness testing, the recording should be started around or after ClearBuffer, before any other "Render" calls.
  *  - BinTriangles and RenderTrilist calls are NOT recorded; If using a custom multithreaded rendering, one should 
  *    record input triangles by manually calling MaskedOcclusionCulling::RecordRenderTriangles - see 
  *    CullingThreadpool::RenderTriangles for an example. 
  *    This is done intentionally in order to get raw input triangles so we can performance test and optimize various
  *    thread pool approaches.
  */

#include "MaskedOcclusionCulling.h"

#if ENABLE_RECORDER

#include <fstream>
#include <mutex>
#include <vector>

class FrameRecorder
{
	std::ofstream       mOutStream;

protected:
	friend class MaskedOcclusionCulling;
	FrameRecorder( std::ofstream && outStream );

public:
	~FrameRecorder( );

public:
	void RecordRenderTriangles( MaskedOcclusionCulling::CullingResult cullingResult, const float *inVtx, const unsigned int *inTris, int nTris, const float *modelToClipMatrix, MaskedOcclusionCulling::ClipPlanes clipPlaneMask, MaskedOcclusionCulling::BackfaceWinding bfWinding, const MaskedOcclusionCulling::VertexLayout &vtxLayout );
	void RecordTestRect( MaskedOcclusionCulling::CullingResult cullingResult, float xmin, float ymin, float xmax, float ymax, float wmin );
	void RecordTestTriangles( MaskedOcclusionCulling::CullingResult cullingResult, const float *inVtx, const unsigned int *inTris, int nTris, const float *modelToClipMatrix, MaskedOcclusionCulling::ClipPlanes clipPlaneMask, MaskedOcclusionCulling::BackfaceWinding bfWinding, const MaskedOcclusionCulling::VertexLayout &vtxLayout );
};

struct FrameRecord
{
	struct TrianglesEntry
	{
		MaskedOcclusionCulling::CullingResult	mCullingResult;
		std::vector<float>						mVertices;
		std::vector<unsigned int>				mTriangles;
		float									mModelToClipMatrix[16];
		MaskedOcclusionCulling::BackfaceWinding	mbfWinding;
		MaskedOcclusionCulling::VertexLayout	mVertexLayout;
		MaskedOcclusionCulling::ClipPlanes		mClipPlaneMask;
		bool									mHasModelToClipMatrix;
		bool									mHasScissorRect;
	};
	struct RectEntry
	{
		MaskedOcclusionCulling::CullingResult mCullingResult;
		float mXMin;
		float mYMin;
		float mXMax;
		float mYMax;
		float mWMin;
    };

	// list of type&index pairs for playback ( type 0 is RenderTriangles, type 1 is TestRect, type 2 is TestTriangles )
	std::vector< std::pair< char, int > >	mPlaybackOrder;

	std::vector< TrianglesEntry >			mTriangleEntries;
	std::vector< RectEntry >				mRectEntries;

    FrameRecord( ) = default;
    FrameRecord( const FrameRecord & other ) = default;

    FrameRecord( FrameRecord && other )
    {
        mPlaybackOrder = std::move( other.mPlaybackOrder );
        mTriangleEntries = std::move( other.mTriangleEntries );
        mRectEntries = std::move( other.mRectEntries );
    }

	void Reset( )
	{
		mPlaybackOrder.clear();
		mTriangleEntries.clear();
		mRectEntries.clear();
	}

	static bool Load( const char * inputFilePath, FrameRecord & outRecording );
};

#endif // #if ENABLE_RECORDER