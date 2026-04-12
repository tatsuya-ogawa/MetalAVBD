//
//  ShaderTypes.h
//  MetalAVBD
//
//  Created by Tatsuya Ogawa on 2026/04/07.
//

//
//  Header containing types and enum constants shared between Metal shaders and Swift/ObjC source
//
#ifndef ShaderTypes_h
#define ShaderTypes_h

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define NS_CLOSED_ENUM(_type, _name) NS_ENUM(_type, _name)
typedef metal::int32_t EnumBackingType;
#else
#import <Foundation/Foundation.h>
typedef NSInteger EnumBackingType;
#endif

#include <simd/simd.h>
#include "AVBDComputeTypes.h"

typedef NS_CLOSED_ENUM(EnumBackingType, AVBDRenderShape)
{
    AVBDRenderShapeBox    = 0,
    AVBDRenderShapeSphere = 1,
    AVBDRenderShapeTorus  = 2
};

typedef NS_CLOSED_ENUM(EnumBackingType, AVBDAxisType)
{
    AVBDAxisTypeFaceA = 0,
    AVBDAxisTypeFaceB = 1,
    AVBDAxisTypeEdge  = 2
};

typedef NS_CLOSED_ENUM(EnumBackingType, AVBDCollisionFeaturePrefix)
{
    AVBDCollisionFeaturePrefixSphereSphere = 3 << 24,
    AVBDCollisionFeaturePrefixSphereBox    = 4 << 24,
    AVBDCollisionFeaturePrefixTorusSphere  = 5 << 24,
    AVBDCollisionFeaturePrefixTorusBox     = 6 << 24,
    AVBDCollisionFeaturePrefixTorusTorus   = 7 << 24
};

#define AVBD_TORUS_APPROX_SPHERE_COUNT_MIN 4
#define AVBD_TORUS_APPROX_SPHERE_COUNT_DEFAULT 16
#define AVBD_TORUS_APPROX_SPHERE_COUNT_MAX 32
#define AVBD_TORUS_APPROX_SPHERE_RADIUS_SCALE_DEFAULT 2.0f
#define AVBD_COLLISION_MAX_POLY_VERTS 16
#define AVBD_COLLISION_SAT_AXIS_EPSILON 1.0e-6f
#define AVBD_COLLISION_PLANE_EPSILON 1.0e-5f
#define AVBD_COLLISION_CONTACT_MERGE_DIST_SQ 1.0e-6f
#define AVBD_COLLISION_CONTACT_PENALTY_START 100.0f

typedef NS_ENUM(EnumBackingType, BufferIndex)
{
    BufferIndexMeshPositions = 0,
    BufferIndexMeshNormals   = 1,
    BufferIndexUniforms      = 2,
    BufferIndexInstances     = 3
};

typedef NS_ENUM(EnumBackingType, VertexAttribute)
{
    VertexAttributePosition      = 0,
    VertexAttributeNormal        = 1,
    VertexAttributeModelColumn0  = 2,
    VertexAttributeModelColumn1  = 3,
    VertexAttributeModelColumn2  = 4,
    VertexAttributeModelColumn3  = 5,
    VertexAttributeColor         = 6,
    VertexAttributeShapeParams   = 7
};

typedef struct
{
    matrix_float4x4 viewProjectionMatrix;
} Uniforms;

typedef struct
{
    matrix_float4x4 modelMatrix;
    vector_float4 renderColor;
    vector_float4 shapeParams;
} InstanceUniforms;

#endif /* ShaderTypes_h */
