############################# Project  setup ###############################
cmake_minimum_required( VERSION 2.6 )
project( Example )
set(CMAKE_SUPPRESS_REGENERATION true)

set( MOC_AVX512_FILES ../MaskedOcclusionCullingAVX512.cpp )
set( MOC_AVX2_FILES ../MaskedOcclusionCullingAVX2.cpp )
set( MOC_SSE_FILES ../MaskedOcclusionCulling.cpp ../CullingThreadpool.cpp )
set( MOC_INCLUDE_FILES ../MaskedOcclusionCulling.h ../CullingThreadpool.h ../CompilerSpecific.inl ../MaskedOcclusionCullingCommon.inl )
set( MOC_FILES ${MOC_AVX512_FILES} ${MOC_AVX2_FILES} ${MOC_SSE_FILES} ${MOC_INCLUDE_FILES} )

########################## Common compiler flags ###########################
set(COMMON_CXX_FLAGS "-Wall -std=c++0x -std=c++11 -O3 -m64" )
set(CMAKE_CXX_FLAGS "${COMMON_CXX_FLAGS}")

################### Setup AVX512 version ####################
SET_SOURCE_FILES_PROPERTIES( ${MOC_AVX512_FILES} PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512dq -mavx2 -mfma -msse4.1" )

################### Setup AVX2 version ####################
SET_SOURCE_FILES_PROPERTIES( ${MOC_AVX2_FILES} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -msse4.1" )

########################## Setup SSE4/2 versions ###########################
SET_SOURCE_FILES_PROPERTIES( ${MOC_SSE_FILES} PROPERTIES COMPILE_FLAGS "-msse4.1" )

########################## Build executable ###########################
add_executable( Example ${MOC_FILES} ExampleMain.cpp )