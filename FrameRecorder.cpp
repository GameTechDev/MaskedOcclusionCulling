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
#include "FrameRecorder.h"

#if ENABLE_RECORDER
#include <assert.h>
#include <algorithm>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Masked occlusion culling 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool MaskedOcclusionCulling::RecorderStart( const char * outputFilePath ) const
{
    std::lock_guard<std::mutex> lock( mRecorderMutex );

    assert( mRecorder == nullptr ); // forgot to call RecorderStop?
    if( mRecorder != nullptr )
        return false;

    std::ofstream outStream( outputFilePath, std::ios::out | std::ios::trunc | std::ios::binary );
    if( !outStream.is_open( ) )
        return false;
    mRecorder = new FrameRecorder( std::move( outStream ), *this );
    return true;
}

void MaskedOcclusionCulling::RecorderStop( ) const
{
    std::lock_guard<std::mutex> lock( mRecorderMutex );

    delete mRecorder;
    mRecorder = nullptr;
}

void MaskedOcclusionCulling::RecordRenderTriangles( const float *inVtx, const unsigned int *inTris, int nTris, const float *modelToClipMatrix, ClipPlanes clipPlaneMask, BackfaceWinding bfWinding, const VertexLayout &vtxLayout, CullingResult cullingResult )
{
    std::lock_guard<std::mutex> lock( mRecorderMutex );
    if( mRecorder != nullptr ) 
        mRecorder->RecordRenderTriangles( cullingResult, inVtx, inTris, nTris, modelToClipMatrix, clipPlaneMask, bfWinding, vtxLayout );
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Masked occlusion culling recorder
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrameRecorder::FrameRecorder( std::ofstream && outStream, const MaskedOcclusionCulling & moc ) : mOutStream( std::move( outStream ) )
{
    assert( mOutStream.is_open( ) );

    // for file verification purposes
    unsigned int fileHeader = 0xA701B600;
    mOutStream.write( (const char *)&fileHeader, sizeof( fileHeader ) );

    // save some of the MOC states (we can override them for the playback)
    float nearClipPlane = moc.GetNearClipPlane();
    unsigned int width;
    unsigned int height;
    moc.GetResolution( width, height );

    mOutStream.write( (const char *)&nearClipPlane, sizeof( nearClipPlane ) );
    mOutStream.write( (const char *)&width, sizeof( width ) );
    mOutStream.write( (const char *)&height, sizeof( height ) );
}

FrameRecorder::~FrameRecorder( )
{
    // end of file marker
    char footer = 0x7F;
    mOutStream.write( &footer, 1 );

    mOutStream.close( );
}

static void WriteTriangleRecording( std::ofstream & outStream, MaskedOcclusionCulling::CullingResult cullingResult, const float *inVtx, const unsigned int *inTris, int nTris, const float *modelToClipMatrix, MaskedOcclusionCulling::ClipPlanes clipPlaneMask, MaskedOcclusionCulling::BackfaceWinding bfWinding, const MaskedOcclusionCulling::VertexLayout & vtxLayout )
{
    // write culling result
    outStream.write( (const char*)&cullingResult, sizeof( cullingResult ) );

    unsigned int minVIndex = 0xffffffff;
    unsigned int maxVIndex = 0;
    for( int i = 0; i < nTris; i++ )
    {
        const unsigned int & a = inTris[i * 3 + 0];
        const unsigned int & b = inTris[i * 3 + 1];
        const unsigned int & c = inTris[i * 3 + 2];
        minVIndex = std::min( std::min( minVIndex, a ), std::min( b, c ) );
        maxVIndex = std::max( std::max( maxVIndex, a ), std::max( b, c ) );
    }

    // write actually used vertex count
    int vertexCount = ( maxVIndex < minVIndex ) ? ( 0 ) : ( maxVIndex - minVIndex + 1 );
    outStream.write( (const char*)&vertexCount, sizeof( vertexCount ) );

    // nothing more to write? early exit
    if( vertexCount == 0 )
        return;

    // write vertex size
    int vertexSize = vtxLayout.mStride;
    outStream.write( (const char*)&vertexSize, sizeof( vertexSize ) );

    // write vertices
    outStream.write( ( (const char*)inVtx ) + minVIndex*vertexSize, vertexSize * ( vertexCount ) );

    // write triangle count
    outStream.write( (const char *)&nTris, sizeof( nTris ) );

    // write indices with adjusted offset
    for( int i = 0; i < nTris; i++ )
    {
        unsigned int triangleIndices[3];
        triangleIndices[0] = inTris[i * 3 + 0] - minVIndex;
        triangleIndices[1] = inTris[i * 3 + 1] - minVIndex;
        triangleIndices[2] = inTris[i * 3 + 2] - minVIndex;
        outStream.write( (const char *)triangleIndices, sizeof( triangleIndices ) );
    }

    // write model to clip matrix (if any)
    char hasMatrix = ( modelToClipMatrix != nullptr ) ? ( 1 ) : ( 0 );
    outStream.write( (const char *)&hasMatrix, sizeof( hasMatrix ) );
    if( hasMatrix )
        outStream.write( (const char *)modelToClipMatrix, 16 * sizeof( float ) );

    outStream.write( (const char *)&clipPlaneMask, sizeof( clipPlaneMask ) );

    outStream.write( (const char*)&bfWinding, sizeof( bfWinding ) );

    // write vertex layout
    outStream.write( (const char *)&vtxLayout, sizeof( vtxLayout ) );
}

static bool ReadTriangleRecording( FrameRecording::TrianglesEntry & outEntry, std::ifstream & inStream )
{
    // read culling result
    inStream.read( (char*)&outEntry.mCullingResult, sizeof( outEntry.mCullingResult ) );
    if( inStream.gcount( ) != sizeof( outEntry.mCullingResult ) )
    {
        assert( false );
        return false;
    }

    // read used vertex count
    int vertexCount = 0;
    inStream.read( (char*)&vertexCount, sizeof( vertexCount ) );
    if( inStream.gcount( ) != sizeof( vertexCount ) )
    {
        assert( false );
        return false;
    }

    // nothing in the recording? that's ok, just exit
    if( vertexCount == 0 )
    {
        outEntry.mVertices.clear( );
        outEntry.mTriangles.clear( );
        return true;
    }


    // read vertex size
    int vertexSize = 0;
    inStream.read( (char*)&vertexSize, sizeof( vertexSize ) );
    if( inStream.gcount( ) != sizeof( vertexSize ) )
    {
        assert( false );
        return false;
    }

    // read vertices
    outEntry.mVertices.resize( vertexSize / 4 * vertexCount );  // pre-allocate data
    inStream.read( (char*) outEntry.mVertices.data(), vertexSize * vertexCount );
    if( inStream.gcount( ) != (vertexSize * vertexCount) )
    {
        assert( false );
        return false;
    }

    // read triangle count
    int triangleCount = 0;
    inStream.read( (char*)&triangleCount, sizeof( triangleCount ) );
    if( inStream.gcount( ) != sizeof( triangleCount ) )
    {
        assert( false );
        return false;
    }

    outEntry.mTriangles.resize( triangleCount * 3 );
    inStream.read( (char*)outEntry.mTriangles.data( ), triangleCount * 3 * 4 );
    if( inStream.gcount( ) != ( triangleCount * 3 * 4 ) )
    {
        assert( false );
        return false;
    }

    // read matrix (if any)
    char hasMatrix = 0;
    inStream.read( (char*)&hasMatrix, sizeof( hasMatrix ) );
    if( inStream.gcount( ) != sizeof( hasMatrix ) )
    {
        assert( false );
        return false;
    }

    if( (outEntry.mHasModelToClipMatrix = (hasMatrix != 0)) )
    {
        inStream.read( (char*)outEntry.mModelToClipMatrix, 16 * sizeof( float ) );
        if( inStream.gcount( ) != 16 * sizeof( float ) )
        {
            assert( false );
            return false;
        }
    }
    else
    {
        memset( outEntry.mModelToClipMatrix, 0, 16 * sizeof( float ) );
    }

    inStream.read( (char*)&outEntry.mClipPlaneMask, sizeof( outEntry.mClipPlaneMask ) );
    if( inStream.gcount( ) != sizeof( outEntry.mClipPlaneMask ) )
    {
        assert( false );
        return false;
    }

    // read triangle cull winding
    inStream.read( (char*)&outEntry.mbfWinding, sizeof( outEntry.mbfWinding ) );

    // read vertex layout
    inStream.read( (char*)&outEntry.mVertexLayout, sizeof( outEntry.mVertexLayout ) );
    if( inStream.gcount( ) != sizeof( outEntry.mVertexLayout ) )
    {
        assert( false );
        return false;
    }
    if( outEntry.mVertexLayout.mStride != vertexSize )
    {
        assert( false );
        return false;
    }
    return true;
}

void FrameRecorder::RecordClearBuffer( )
{
    assert( mOutStream.is_open( ) );

    char header = 3;
    mOutStream.write( &header, 1 );
}

void FrameRecorder::RecordRenderTriangles( MaskedOcclusionCulling::CullingResult cullingResult, const float *inVtx, const unsigned int *inTris, int nTris, const float *modelToClipMatrix, MaskedOcclusionCulling::ClipPlanes clipPlaneMask, MaskedOcclusionCulling::BackfaceWinding bfWinding, const MaskedOcclusionCulling::VertexLayout & vtxLayout )
{
    assert( mOutStream.is_open( ) );

    char header = 0;
    mOutStream.write( &header, 1 );
    WriteTriangleRecording( mOutStream, cullingResult, inVtx, inTris, nTris, modelToClipMatrix, clipPlaneMask, bfWinding, vtxLayout );
}

void FrameRecorder::RecordTestRect( MaskedOcclusionCulling::CullingResult cullingResult, float xmin, float ymin, float xmax, float ymax, float wmin )
{
    assert( mOutStream.is_open( ) );

    char header = 1;
    mOutStream.write( &header, 1 );

    mOutStream.write( (const char*)&cullingResult, sizeof( cullingResult ) );
    mOutStream.write( (const char*)&xmin, sizeof( xmin ) );
    mOutStream.write( (const char*)&ymin, sizeof( ymin ) );
    mOutStream.write( (const char*)&xmax, sizeof( xmax ) );
    mOutStream.write( (const char*)&ymax, sizeof( ymax ) );
    mOutStream.write( (const char*)&wmin, sizeof( wmin ) );
}

void FrameRecorder::RecordTestTriangles( MaskedOcclusionCulling::CullingResult cullingResult, const float *inVtx, const unsigned int *inTris, int nTris, const float *modelToClipMatrix, MaskedOcclusionCulling::ClipPlanes clipPlaneMask, MaskedOcclusionCulling::BackfaceWinding bfWinding, const MaskedOcclusionCulling::VertexLayout & vtxLayout )
{
    assert( mOutStream.is_open( ) );

    char header = 2;
    mOutStream.write( &header, 1 );
    WriteTriangleRecording( mOutStream, cullingResult, inVtx, inTris, nTris, modelToClipMatrix, clipPlaneMask, bfWinding, vtxLayout );
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Masked occlusion culling recording
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool FrameRecording::Load( const char * inputFilePath, FrameRecording & outRecording )
{
    outRecording.Reset();

    std::ifstream inStream( inputFilePath, std::ios::binary );
    if( !inStream.is_open( ) )
    {
        return false;
    }
    
    // for file verification purposes
    unsigned int fileHeader = 0;
    inStream.read( (char *)&fileHeader, sizeof( fileHeader ) );
    if( (inStream.gcount() != 4) || (fileHeader != 0xA701B600) )
    {
        return false;
    }

    inStream.read( (char *)&outRecording.mNearClipPlane, sizeof( outRecording.mNearClipPlane ) );
    inStream.read( (char *)&outRecording.mResolutionWidth, sizeof( outRecording.mResolutionWidth ) );
    inStream.read( (char *)&outRecording.mResolutionHeight, sizeof( outRecording.mResolutionHeight ) );

    bool continueLoading = true;
    while( continueLoading )
    {
        char chunkHeader = 0;
        inStream.read( (char *)&chunkHeader, sizeof( chunkHeader ) );
        if( (inStream.gcount() != 1) )
        {
            assert( false );
            outRecording.Reset();
            return false;
        }
        switch( chunkHeader )
        {
            case( 0 ):      // RenderTriangles
            case( 2 ):      // TestTriangles
            {
                outRecording.mTriangleEntries.push_back( TrianglesEntry() );
                int triangleEntryIndex = (int)outRecording.mTriangleEntries.size( )-1;
                if( !ReadTriangleRecording( outRecording.mTriangleEntries[triangleEntryIndex], inStream ) )
                { 
                    assert( false );
                    outRecording.Reset( );
                    return false;
                }
                outRecording.mPlaybackOrder.push_back( std::make_pair( chunkHeader, triangleEntryIndex )  );
            } break;
            case( 1 ):      // TestRect
            {
                outRecording.mRectEntries.push_back( RectEntry( ) );
                int rectEntryIndex = (int)outRecording.mRectEntries.size( )-1;

                // read rectangle in one go
                inStream.read( (char*)&outRecording.mRectEntries[rectEntryIndex], sizeof( outRecording.mRectEntries[rectEntryIndex] ) );
                if( inStream.gcount( ) != sizeof( outRecording.mRectEntries[rectEntryIndex] ) )
                {
                    assert( false );
                    outRecording.Reset( );
                    return false;
                }
                outRecording.mPlaybackOrder.push_back( std::make_pair( chunkHeader, rectEntryIndex ) );
            } break;
            case( 3 ):      // ClearBuffer
            {
                outRecording.mPlaybackOrder.push_back( std::make_pair( 3, -1 ) );
            } break;
            case( 0x7F ):   // eOF
            {
                continueLoading = false;
                return true;
            } break;
            default:
            {
                assert( false );
                outRecording.Reset( );
                return false;
            }
        }
    }

    assert( false );    // we should never get here
    return true;
}

#endif // #if ENABLE_RECORDER