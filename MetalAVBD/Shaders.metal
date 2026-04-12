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
