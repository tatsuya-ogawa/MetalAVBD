//
//  Shaders.metal
//  MetalAVBD
//
//  Created by Tatsuya Ogawa on 2026/04/07.
//

// File for Metal kernel and shader functions

#include <metal_stdlib>
#include <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;

typedef struct
{
    float3 position [[attribute(VertexAttributePosition)]];
    float3 normal [[attribute(VertexAttributeNormal)]];
    float4 modelColumn0 [[attribute(VertexAttributeModelColumn0)]];
    float4 modelColumn1 [[attribute(VertexAttributeModelColumn1)]];
    float4 modelColumn2 [[attribute(VertexAttributeModelColumn2)]];
    float4 modelColumn3 [[attribute(VertexAttributeModelColumn3)]];
    float4 color [[attribute(VertexAttributeColor)]];
    float4 shapeParams [[attribute(VertexAttributeShapeParams)]];
} Vertex;

typedef struct
{
    float4 position [[position]];
    float3 normal;
    float4 color;
} ColorInOut;

typedef struct
{
    float4 position [[position]];
    float3 worldPosition;
    float4 color;
    uint meshIndex;
} SDFDebugInOut;

struct AVBDSDFDebugTextureSet {
    array<texture3d<float, access::sample>, AVBD_MAX_COLLISION_MESH_SDFS> sdf [[id(0)]];
};

constant sampler avbdDebugSDFSampler(coord::normalized,
                                     address::clamp_to_edge,
                                     filter::linear);

static float3 sdf_debug_texcoord(float3 pointLocal,
                                 const device AVBDGPUCollisionMeshInfo &meshInfo)
{
    float3 extent = max(meshInfo.sdfLocalMaxBounds.xyz - meshInfo.sdfLocalMinBounds.xyz, float3(1.0e-5f));
    return (pointLocal - meshInfo.sdfLocalMinBounds.xyz) / extent;
}

static bool sdf_debug_point_in_bounds(float3 pointLocal,
                                      const device AVBDGPUCollisionMeshInfo &meshInfo)
{
    return all(pointLocal >= meshInfo.sdfLocalMinBounds.xyz) && all(pointLocal <= meshInfo.sdfLocalMaxBounds.xyz);
}

static float sdf_debug_sample_value(texture3d<float, access::sample> sdfTexture,
                                    float3 pointLocal,
                                    const device AVBDGPUCollisionMeshInfo &meshInfo)
{
    float3 texCoord = sdf_debug_texcoord(pointLocal, meshInfo);
    if (any(texCoord < 0.0f) || any(texCoord > 1.0f)) {
        return FLT_MAX;
    }
    return sdfTexture.sample(avbdDebugSDFSampler, texCoord).r;
}

static float3 sdf_debug_local_normal(texture3d<float, access::sample> sdfTexture,
                                     float3 pointLocal,
                                     const device AVBDGPUCollisionMeshInfo &meshInfo)
{
    float3 eps = max(meshInfo.sdfVoxelSize.xyz, float3(1.0e-4f));
    float sdfXPos = sdf_debug_sample_value(sdfTexture, pointLocal + float3(eps.x, 0.0f, 0.0f), meshInfo);
    float sdfXNeg = sdf_debug_sample_value(sdfTexture, pointLocal - float3(eps.x, 0.0f, 0.0f), meshInfo);
    float sdfYPos = sdf_debug_sample_value(sdfTexture, pointLocal + float3(0.0f, eps.y, 0.0f), meshInfo);
    float sdfYNeg = sdf_debug_sample_value(sdfTexture, pointLocal - float3(0.0f, eps.y, 0.0f), meshInfo);
    float sdfZPos = sdf_debug_sample_value(sdfTexture, pointLocal + float3(0.0f, 0.0f, eps.z), meshInfo);
    float sdfZNeg = sdf_debug_sample_value(sdfTexture, pointLocal - float3(0.0f, 0.0f, eps.z), meshInfo);

    if (!isfinite(sdfXPos)) sdfXPos = 1.0f;
    if (!isfinite(sdfXNeg)) sdfXNeg = 1.0f;
    if (!isfinite(sdfYPos)) sdfYPos = 1.0f;
    if (!isfinite(sdfYNeg)) sdfYNeg = 1.0f;
    if (!isfinite(sdfZPos)) sdfZPos = 1.0f;
    if (!isfinite(sdfZNeg)) sdfZNeg = 1.0f;

    return normalize(float3(
        (sdfXPos - sdfXNeg) / max(2.0f * eps.x, 1.0e-5f),
        (sdfYPos - sdfYNeg) / max(2.0f * eps.y, 1.0e-5f),
        (sdfZPos - sdfZNeg) / max(2.0f * eps.z, 1.0e-5f)
    ));
}

static float3 sdf_debug_world_normal(float3 localNormal,
                                     const device AVBDGPUCollisionMeshInfo &meshInfo)
{
    float3x3 invLinear = float3x3(
        meshInfo.sdfInvTransform.columns[0].xyz,
        meshInfo.sdfInvTransform.columns[1].xyz,
        meshInfo.sdfInvTransform.columns[2].xyz
    );
    return normalize(transpose(invLinear) * localNormal);
}

vertex ColorInOut vertexShader(Vertex in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]])
{
    ColorInOut out;
    float4x4 modelMatrix = float4x4(in.modelColumn0, in.modelColumn1, in.modelColumn2, in.modelColumn3);
    float3 localPosition = in.position;
    float3 localNormal = in.normal;
    int renderShape = int(rint(in.shapeParams.x));

    if (renderShape == AVBDRenderShapeTorus) {
        float u = in.position.x;
        float v = in.position.y;
        float majorRadius = in.shapeParams.y;
        float minorRadius = in.shapeParams.z;
        float cu = cos(u);
        float su = sin(u);
        float cv = cos(v);
        float sv = sin(v);

        localPosition = float3((majorRadius + minorRadius * cv) * cu,
                               (majorRadius + minorRadius * cv) * su,
                               minorRadius * sv);
        localNormal = float3(cu * cv, su * cv, sv);
    }

    float4 worldPosition = modelMatrix * float4(localPosition, 1.0);

    out.position = uniforms.viewProjectionMatrix * worldPosition;
    out.normal = normalize((modelMatrix * float4(localNormal, 0.0)).xyz);
    out.color = in.color;

    return out;
}

fragment float4 fragmentShader(ColorInOut in [[stage_in]])
{
    float3 normal = normalize(in.normal);
    float3 lightDirection = normalize(float3(0.35, 0.55, 0.76));
    float lighting = 0.30 + 0.70 * saturate(dot(normal, lightDirection));
    return float4(in.color.rgb * lighting, in.color.a);
}

vertex SDFDebugInOut sdfDebugVertexShader(Vertex in [[stage_in]],
                                          constant Uniforms & uniforms [[buffer(BufferIndexUniforms)]])
{
    SDFDebugInOut out;
    float4x4 modelMatrix = float4x4(in.modelColumn0, in.modelColumn1, in.modelColumn2, in.modelColumn3);
    float4 worldPosition = modelMatrix * float4(in.position, 1.0);
    out.position = uniforms.viewProjectionMatrix * worldPosition;
    out.worldPosition = worldPosition.xyz;
    out.color = in.color;
    out.meshIndex = uint(max(in.shapeParams.y, 0.0));
    return out;
}

fragment float4 sdfDebugFragmentShader(
    SDFDebugInOut in [[stage_in]],
    constant Uniforms & uniforms [[buffer(0)]],
    device const AVBDGPUCollisionMeshInfo *meshInfos [[buffer(1)]],
    constant AVBDSDFDebugTextureSet &meshSDFSet [[buffer(2)]]
)
{
    const device AVBDGPUCollisionMeshInfo &meshInfo = meshInfos[in.meshIndex];
    uint sdfResourceIndex = uint(max(meshInfo.sdfResourceIndex, 0));
    texture3d<float, access::sample> sdfTexture = meshSDFSet.sdf[sdfResourceIndex];

    float3 worldRayDirection = normalize(in.worldPosition - uniforms.cameraWorldPosition);
    float3 localRayOrigin = (meshInfo.sdfInvTransform * float4(in.worldPosition, 1.0f)).xyz;
    float3 localRayDirection = normalize((meshInfo.sdfInvTransform * float4(worldRayDirection, 0.0f)).xyz);
    float minVoxel = max(min(meshInfo.sdfVoxelSize.x, min(meshInfo.sdfVoxelSize.y, meshInfo.sdfVoxelSize.z)), 1.0e-4f);
    float hitThreshold = minVoxel * 0.75f;
    float travel = hitThreshold;

    for (uint stepIndex = 0; stepIndex < 96; ++stepIndex) {
        float3 localPoint = localRayOrigin + localRayDirection * travel;
        if (!sdf_debug_point_in_bounds(localPoint, meshInfo)) {
            break;
        }

        float signedDistance = sdf_debug_sample_value(sdfTexture, localPoint, meshInfo);
        if (!isfinite(signedDistance)) {
            break;
        }

        if (abs(signedDistance) <= hitThreshold) {
            float3 localNormal = sdf_debug_local_normal(sdfTexture, localPoint, meshInfo);
            float3 worldNormal = sdf_debug_world_normal(localNormal, meshInfo);
            float3 lightDirection = normalize(float3(0.35, 0.55, 0.76));
            float lighting = 0.25 + 0.75 * saturate(dot(worldNormal, lightDirection));
            float contour = 0.55 + 0.45 * abs(sin(travel * 2.5));
            float alpha = clamp(in.color.a, 0.0f, 1.0f);
            return float4(in.color.rgb * lighting * contour, alpha);
        }

        travel += max(abs(signedDistance), minVoxel * 0.5f);
    }

    discard_fragment();
    return float4(0.0f);
}
