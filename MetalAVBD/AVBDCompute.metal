//
//  AVBDCompute.metal
//  MetalAVBD
//
//  Compute shader kernels for AVBD rigid body solver.
//  Ported from the CPU AVBDSolver (Gauss-Seidel → Jacobi-parallel).
//

#include <metal_stdlib>
#include <simd/simd.h>
#import "AVBDComputeTypes.h"
#import "ShaderTypes.h"

using namespace metal;

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

// Quaternion multiply: q * r
static float4 quat_mul(float4 q, float4 r) {
    return float4(
        q.w * r.x + q.x * r.w + q.y * r.z - q.z * r.y,
        q.w * r.y - q.x * r.z + q.y * r.w + q.z * r.x,
        q.w * r.z + q.x * r.y - q.y * r.x + q.z * r.w,
        q.w * r.w - q.x * r.x - q.y * r.y - q.z * r.z
    );
}

// Quaternion conjugate
static float4 quat_conj(float4 q) {
    return float4(-q.x, -q.y, -q.z, q.w);
}

// Rotate vector by quaternion
static float3 quat_act(float4 q, float3 v) {
    float3 t = 2.0f * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Rotation difference as scaled axis: 2 * imag(a * inv(b))
static float3 quat_delta(float4 a, float4 b) {
    float4 d = quat_mul(a, quat_conj(b));
    return d.xyz * 2.0f;
}

// q + angular velocity * dt
static float4 quat_add_angular(float4 q, float3 v) {
    float4 dq = float4(v.x, v.y, v.z, 0.0f);
    float4 result = q + quat_mul(dq, q) * 0.5f;
    return normalize(result);
}

// Transform local point to world
static float3 xform(float3 pos, float4 quat, float3 local) {
    return quat_act(quat, local) + pos;
}

// 3x3 matrix operations (stored as 3 row float3s)
struct Mat3 {
    float3 r0, r1, r2;

    Mat3() : r0(0), r1(0), r2(0) {}
    Mat3(float3 r0, float3 r1, float3 r2) : r0(r0), r1(r1), r2(r2) {}

    static Mat3 identity() {
        return Mat3(float3(1,0,0), float3(0,1,0), float3(0,0,1));
    }

    static Mat3 diag(float a, float b, float c) {
        return Mat3(float3(a,0,0), float3(0,b,0), float3(0,0,c));
    }

    static Mat3 diag3(float3 v) {
        return diag(v.x, v.y, v.z);
    }

    float3 col(int c) const {
        return float3(r0[c], r1[c], r2[c]);
    }

    float3 mul(float3 v) const {
        return float3(dot(r0, v), dot(r1, v), dot(r2, v));
    }

    Mat3 operator+(Mat3 b) const { return Mat3(r0 + b.r0, r1 + b.r1, r2 + b.r2); }
    Mat3 operator-(Mat3 b) const { return Mat3(r0 - b.r0, r1 - b.r1, r2 - b.r2); }
    Mat3 operator*(float s) const { return Mat3(r0 * s, r1 * s, r2 * s); }
    Mat3 operator/(float s) const { return Mat3(r0 / s, r1 / s, r2 / s); }
};

static Mat3 mat3_mul(Mat3 a, Mat3 b) {
    return Mat3(
        float3(dot(a.r0, b.col(0)), dot(a.r0, b.col(1)), dot(a.r0, b.col(2))),
        float3(dot(a.r1, b.col(0)), dot(a.r1, b.col(1)), dot(a.r1, b.col(2))),
        float3(dot(a.r2, b.col(0)), dot(a.r2, b.col(1)), dot(a.r2, b.col(2)))
    );
}

static Mat3 mat3_transpose(Mat3 m) {
    return Mat3(m.col(0), m.col(1), m.col(2));
}

static Mat3 skew(float3 r) {
    return Mat3(
        float3(0, -r.z, r.y),
        float3(r.z, 0, -r.x),
        float3(-r.y, r.x, 0)
    );
}

static Mat3 outer(float3 a, float3 b) {
    return Mat3(b * a.x, b * a.y, b * a.z);
}

static Mat3 diagonalize_mat(Mat3 m) {
    return Mat3::diag(length(m.col(0)), length(m.col(1)), length(m.col(2)));
}

static float sign_f(float x) {
    return x < 0 ? -1.0f : (x > 0 ? 1.0f : 0.0f);
}

static float damping_factor(float damping, float dt) {
    return damping <= 0.0f ? 1.0f : exp(-damping * dt);
}

static float3 clamp_vec(float3 v, float lo, float hi) {
    return float3(clamp(v.x, lo, hi), clamp(v.y, lo, hi), clamp(v.z, lo, hi));
}

static float3 min_vec(float3 v, float s) {
    return float3(min(v.x, s), min(v.y, s), min(v.z, s));
}

static float3 abs_vec(float3 v) {
    return float3(abs(v.x), abs(v.y), abs(v.z));
}

constant int AVBD_GPU_AXIS_FACE_A = AVBDAxisTypeFaceA;
constant int AVBD_GPU_AXIS_FACE_B = AVBDAxisTypeFaceB;
constant int AVBD_GPU_AXIS_EDGE = AVBDAxisTypeEdge;
constant int AVBD_GPU_RENDER_SHAPE_BOX = AVBDRenderShapeBox;
constant int AVBD_GPU_RENDER_SHAPE_SPHERE = AVBDRenderShapeSphere;
constant int AVBD_GPU_RENDER_SHAPE_TORUS = AVBDRenderShapeTorus;
constant int AVBD_GPU_FEATURE_SPHERE_SPHERE = AVBDCollisionFeaturePrefixSphereSphere;
constant int AVBD_GPU_FEATURE_SPHERE_BOX = AVBDCollisionFeaturePrefixSphereBox;
constant int AVBD_GPU_FEATURE_TORUS_SPHERE = AVBDCollisionFeaturePrefixTorusSphere;
constant int AVBD_GPU_FEATURE_TORUS_BOX = AVBDCollisionFeaturePrefixTorusBox;
constant int AVBD_GPU_FEATURE_TORUS_TORUS = AVBDCollisionFeaturePrefixTorusTorus;
constant int AVBD_GPU_FEATURE_PRIMITIVE_MESH = AVBDCollisionFeaturePrefixPrimitiveMesh;
constant int AVBD_GPU_MAX_POLY_VERTS = AVBD_COLLISION_MAX_POLY_VERTS;
constant int AVBD_GPU_TORUS_APPROX_SPHERE_COUNT_MAX = AVBD_TORUS_APPROX_SPHERE_COUNT_MAX;
constant int AVBD_GPU_TORUS_APPROX_SPHERE_COUNT_DEFAULT = AVBD_TORUS_APPROX_SPHERE_COUNT_DEFAULT;
constant float AVBD_GPU_TORUS_APPROX_SPHERE_RADIUS_SCALE_DEFAULT = AVBD_TORUS_APPROX_SPHERE_RADIUS_SCALE_DEFAULT;
constant float AVBD_GPU_SAT_AXIS_EPSILON = AVBD_COLLISION_SAT_AXIS_EPSILON;
constant float AVBD_GPU_PLANE_EPSILON = AVBD_COLLISION_PLANE_EPSILON;
constant float AVBD_GPU_CONTACT_MERGE_DIST_SQ = AVBD_COLLISION_CONTACT_MERGE_DIST_SQ;
constant float AVBD_GPU_CONTACT_PENALTY_START = AVBD_COLLISION_CONTACT_PENALTY_START;

struct AVBDGPUOBBMetal {
    float3 center;
    float4 rotation;
    float3 halfSize;
    float3 axis[3];
};

struct AVBDGPUSatAxisMetal {
    int type;
    int indexA;
    int indexB;
    float separation;
    float3 normalAB;
    bool valid;
};

struct AVBDGPUFaceFrameMetal {
    float3 normal;
    float3 center;
    float3 u;
    float3 v;
    float extentU;
    float extentV;
};

struct AVBDGPUContactBuilderMetal {
    AVBDGPUContact contacts[AVBD_MAX_CONTACTS_PER_PAIR_BURST];
    float3 midpoints[AVBD_MAX_CONTACTS_PER_PAIR_BURST];
    int count;
};

static int torus_approx_sphere_count(constant AVBDGPUSolverParams &params);
static float torus_outer_radius(device const AVBDGPUBody &body);
static float torus_minor_radius(device const AVBDGPUBody &body);
static float torus_approx_sphere_radius(device const AVBDGPUBody &body, constant AVBDGPUSolverParams &params);
static float torus_major_radius(device const AVBDGPUBody &body);

// Derive (bodyA, bodyB) from an upper-triangle pair index.
// For n bodies, pairs (i,j) with i<j are enumerated row-major:
//   tid 0 → (0,1), 1 → (0,2), ..., n-2 → (0,n-1), n-1 → (1,2), ...
static void upper_triangle_pair(uint tid, int n, thread int &outA, thread int &outB) {
    if (n < 2) {
        outA = -1;
        outB = -1;
        return;
    }

    float fn = float(n);
    float fk = float(tid);
    int i = int(floor((2.0f * fn - 1.0f - sqrt((2.0f * fn - 1.0f) * (2.0f * fn - 1.0f) - 8.0f * fk)) * 0.5f));
    // Clamp to valid range (floating-point rounding protection)
    i = clamp(i, 0, n - 2);
    int rowStart = i * (2 * n - i - 1) / 2;
    // If we overshot due to rounding, step back until the row covers tid.
    while (i > 0 && rowStart > int(tid)) {
        i--;
        rowStart = i * (2 * n - i - 1) / 2;
    }
    // If sqrt rounding kept us on the previous row at an exact row boundary, step forward.
    while (i < n - 2) {
        int nextRow = i + 1;
        int nextRowStart = nextRow * (2 * n - nextRow - 1) / 2;
        if (nextRowStart > int(tid)) {
            break;
        }
        i = nextRow;
        rowStart = nextRowStart;
    }
    int j = int(tid) - rowStart + i + 1;
    outA = i;
    outB = j;
}

// Check if bodyB is in bodyA's collision exclusion list.
static bool is_excluded(device AVBDGPUCollisionExclusion *exclusions, int bodyA, int bodyB) {
    int countA = exclusions[bodyA].excludeCount;
    for (int i = 0; i < countA; i++) {
        if (exclusions[bodyA].excludeIndices[i] == bodyB) return true;
    }
    return false;
}

static int reserve_contact_range(device AVBDGPUContactAllocator &allocator, int requestedCount, thread int &baseIndex) {
    if (requestedCount <= 0) return 0;
    int index = atomic_fetch_add_explicit(&allocator.nextContactIndex, requestedCount, memory_order_relaxed);
    if (index >= allocator.contactCapacity) return 0;
    int grantedCount = min(requestedCount, allocator.contactCapacity - index);
    baseIndex = index;
    return grantedCount;
}

static int reserve_recent_pair_range(device AVBDGPURecentPairCacheState &state, int requestedCount, thread int &baseIndex) {
    if (requestedCount <= 0) return 0;
    int index = atomic_fetch_add_explicit(&state.count, requestedCount, memory_order_relaxed);
    if (index >= state.capacity) return 0;
    int grantedCount = min(requestedCount, state.capacity - index);
    baseIndex = index;
    return grantedCount;
}

static int reserve_derived_pair_range(device AVBDGPUDerivedPairCandidateState &state, int requestedCount, thread int &baseIndex) {
    if (requestedCount <= 0) return 0;
    int index = atomic_fetch_add_explicit(&state.count, requestedCount, memory_order_relaxed);
    if (index >= state.capacity) return 0;
    int grantedCount = min(requestedCount, state.capacity - index);
    baseIndex = index;
    return grantedCount;
}

static int reserve_active_manifold_range(device AVBDGPUActiveManifoldListState &state, int requestedCount, thread int &baseIndex) {
    if (requestedCount <= 0) return 0;
    int index = atomic_fetch_add_explicit(&state.count, requestedCount, memory_order_relaxed);
    if (index >= state.capacity) return 0;
    int grantedCount = min(requestedCount, state.capacity - index);
    baseIndex = index;
    return grantedCount;
}

static int reserve_primitive_mesh_pair_range(device AVBDGPUPrimitiveMeshPairListState &state,
                                             int requestedCount,
                                             thread int &baseIndex) {
    if (requestedCount <= 0) return 0;
    int index = atomic_fetch_add_explicit(&state.count, requestedCount, memory_order_relaxed);
    if (index >= state.capacity) return 0;
    int grantedCount = min(requestedCount, state.capacity - index);
    baseIndex = index;
    return grantedCount;
}

static int reserve_mesh_mesh_pair_range(device AVBDGPUMeshMeshPairListState &state,
                                        int requestedCount,
                                        thread int &baseIndex) {
    if (requestedCount <= 0) return 0;
    int index = atomic_fetch_add_explicit(&state.count, requestedCount, memory_order_relaxed);
    if (index >= state.capacity) return 0;
    int grantedCount = min(requestedCount, state.capacity - index);
    baseIndex = index;
    return grantedCount;
}

static float abs_dot(float3 a, float3 b) {
    return abs(dot(a, b));
}

static Mat3 orthonormal_basis(float3 normal) {
    float3 t1 = abs(normal.x) > abs(normal.z)
        ? float3(-normal.y, normal.x, 0.0f)
        : float3(0.0f, -normal.z, normal.y);
    float len = length(t1);
    t1 = len > 1.0e-8f ? t1 / len : float3(1, 0, 0);
    float3 t2 = cross(normal, t1);
    return Mat3(normal, t1, t2);
}

static void make_obb(device const AVBDGPUBody &body, thread AVBDGPUOBBMetal &box) {
    box.center = body.positionLin;
    box.rotation = normalize(body.positionAng);
    box.halfSize = body.size * 0.5f;
    box.axis[0] = quat_act(box.rotation, float3(1, 0, 0));
    box.axis[1] = quat_act(box.rotation, float3(0, 1, 0));
    box.axis[2] = quat_act(box.rotation, float3(0, 0, 1));
}

static int torus_approx_sphere_count(constant AVBDGPUSolverParams &params) {
    int requestedCount = params.torusApproxSphereCount > 0 ? params.torusApproxSphereCount : AVBD_GPU_TORUS_APPROX_SPHERE_COUNT_DEFAULT;
    return clamp(requestedCount, 4, AVBD_GPU_TORUS_APPROX_SPHERE_COUNT_MAX);
}

static float body_radius(device const AVBDGPUBody &body, constant AVBDGPUSolverParams &params) {
    if (body.renderShape == AVBD_GPU_RENDER_SHAPE_SPHERE) {
        return max(body.size.x, max(body.size.y, body.size.z)) * 0.5f;
    }
    if (body.renderShape == AVBD_GPU_RENDER_SHAPE_TORUS) {
        return torus_major_radius(body) + torus_approx_sphere_radius(body, params);
    }
    return length(body.size * 0.5f);
}

static float torus_outer_radius(device const AVBDGPUBody &body) {
    float outerDiameter = max(min(body.size.x, body.size.y), 0.0f);
    return outerDiameter * 0.5f;
}

static float torus_minor_radius(device const AVBDGPUBody &body) {
    float outerDiameter = torus_outer_radius(body) * 2.0f;
    float tubeDiameter = max(min(body.size.z, outerDiameter), 0.0f);
    return tubeDiameter * 0.5f;
}

static float torus_approx_sphere_radius(device const AVBDGPUBody &body, constant AVBDGPUSolverParams &params) {
    float scale = params.torusApproxSphereRadiusScale > 0.0f ? params.torusApproxSphereRadiusScale : AVBD_GPU_TORUS_APPROX_SPHERE_RADIUS_SCALE_DEFAULT;
    return torus_minor_radius(body) * scale;
}

static float torus_major_radius(device const AVBDGPUBody &body) {
    return max(torus_outer_radius(body) - torus_minor_radius(body), 0.0f);
}

static float3 torus_sphere_local_center(device const AVBDGPUBody &body, int index, constant AVBDGPUSolverParams &params) {
    float angle = (2.0f * M_PI_F * float(index)) / float(torus_approx_sphere_count(params));
    float majorRadius = torus_major_radius(body);
    return float3(cos(angle) * majorRadius, sin(angle) * majorRadius, 0.0f);
}

static float3 torus_sphere_world_center(device const AVBDGPUBody &body, int index, constant AVBDGPUSolverParams &params) {
    return quat_act(body.positionAng, torus_sphere_local_center(body, index, params)) + body.positionLin;
}

static void make_body_aabb(device const AVBDGPUBody &body,
                           constant AVBDGPUSolverParams &params,
                           thread float3 &minBounds,
                           thread float3 &maxBounds) {
    if (body.renderShape == AVBD_GPU_RENDER_SHAPE_BOX) {
        AVBDGPUOBBMetal box;
        make_obb(body, box);
        float3 extent =
            abs(box.axis[0]) * box.halfSize.x +
            abs(box.axis[1]) * box.halfSize.y +
            abs(box.axis[2]) * box.halfSize.z;
        minBounds = box.center - extent;
        maxBounds = box.center + extent;
        return;
    }

    float radius = body_radius(body, params);
    float3 extent = float3(radius);
    minBounds = body.positionLin - extent;
    maxBounds = body.positionLin + extent;
}

static bool aabb_overlaps(float3 minA,
                          float3 maxA,
                          float3 minB,
                          float3 maxB) {
    return minA.x <= maxB.x && maxA.x >= minB.x &&
           minA.y <= maxB.y && maxA.y >= minB.y &&
           minA.z <= maxB.z && maxA.z >= minB.z;
}

static bool test_axis(thread const AVBDGPUOBBMetal &boxA,
                      thread const AVBDGPUOBBMetal &boxB,
                      float3 delta,
                      float3 axis,
                      int type,
                      int indexA,
                      int indexB,
                      thread AVBDGPUSatAxisMetal &best)
{
    float lenSq = length_squared(axis);
    if (lenSq < AVBD_GPU_SAT_AXIS_EPSILON) {
        return true;
    }

    float3 n = axis / sqrt(lenSq);
    if (dot(n, delta) < 0.0f) {
        n = -n;
    }

    float distance = abs(dot(delta, n));
    float rA = boxA.halfSize.x * abs_dot(n, boxA.axis[0])
        + boxA.halfSize.y * abs_dot(n, boxA.axis[1])
        + boxA.halfSize.z * abs_dot(n, boxA.axis[2]);
    float rB = boxB.halfSize.x * abs_dot(n, boxB.axis[0])
        + boxB.halfSize.y * abs_dot(n, boxB.axis[1])
        + boxB.halfSize.z * abs_dot(n, boxB.axis[2]);

    float separation = distance - (rA + rB);
    if (separation > 0.0f) {
        return false;
    }

    if (!best.valid || separation > best.separation) {
        best.valid = true;
        best.type = type;
        best.indexA = indexA;
        best.indexB = indexB;
        best.separation = separation;
        best.normalAB = n;
    }

    return true;
}

static float3 support_point(thread const AVBDGPUOBBMetal &box, float3 dir) {
    float sx = dot(dir, box.axis[0]) >= 0.0f ? 1.0f : -1.0f;
    float sy = dot(dir, box.axis[1]) >= 0.0f ? 1.0f : -1.0f;
    float sz = dot(dir, box.axis[2]) >= 0.0f ? 1.0f : -1.0f;

    return box.center
        + box.axis[0] * (box.halfSize.x * sx)
        + box.axis[1] * (box.halfSize.y * sy)
        + box.axis[2] * (box.halfSize.z * sz);
}

static float3 safe_normalize(float3 value, float3 fallback) {
    float lenSq = length_squared(value);
    if (lenSq <= 1.0e-12f) {
        return fallback;
    }
    return value * rsqrt(lenSq);
}

static bool ray_triangle_intersection(float3 origin,
                                      float3 direction,
                                      float maxDistance,
                                      float3 v0,
                                      float3 v1,
                                      float3 v2,
                                      thread float &outT,
                                      thread float3 &outNormal) {
    float3 edge01 = v1 - v0;
    float3 edge02 = v2 - v0;
    float3 faceNormal = cross(edge01, edge02);
    float normalLengthSq = length_squared(faceNormal);
    if (normalLengthSq <= 1.0e-12f) {
        return false;
    }

    float3 p = cross(direction, edge02);
    float det = dot(edge01, p);
    if (abs(det) <= 1.0e-8f) {
        return false;
    }

    float invDet = 1.0f / det;
    float3 t = origin - v0;
    float u = dot(t, p) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    float3 q = cross(t, edge01);
    float v = dot(direction, q) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) {
        return false;
    }

    float hitT = dot(edge02, q) * invDet;
    if (hitT < 0.0f || hitT > maxDistance) {
        return false;
    }

    outT = hitT;
    outNormal = faceNormal * rsqrt(normalLengthSq);
    return true;
}

struct AVBDCollisionMeshSDFSet {
    array<texture3d<float, access::sample>, AVBD_MAX_COLLISION_MESH_SDFS> coarse [[id(0)]];
    array<texture3d<float, access::sample>, AVBD_MAX_COLLISION_MESH_SDFS> atlas [[id(64)]];
    array<texture3d<uint, access::read>, AVBD_MAX_COLLISION_MESH_SDFS> indirection [[id(128)]];
};

constant float3 AVBD_GPU_SDF_RAY_DIRECTION = float3(0.728492f, 0.514883f, 0.452173f);
constant int AVBD_COLLISION_MESH_SDF_BRICK_DIM = 8;
constant int AVBD_COLLISION_MESH_SDF_GUARD_VOXELS = 1;
constant int AVBD_COLLISION_MESH_SDF_STORED_BRICK_DIM = 10;
constexpr sampler avbd_sdf_sampler(coord::normalized, filter::linear, address::clamp_to_edge);

static float point_segment_distance(float3 point, float3 a, float3 b) {
    float3 ab = b - a;
    float denom = max(dot(ab, ab), 1.0e-12f);
    float t = clamp(dot(point - a, ab) / denom, 0.0f, 1.0f);
    return length(point - (a + ab * t));
}

static float point_triangle_unsigned_distance(float3 point, float3 v0, float3 v1, float3 v2) {
    float3 edge0 = v1 - v0;
    float3 edge1 = v2 - v0;
    float3 v0ToPoint = point - v0;

    float a = dot(edge0, edge0);
    float b = dot(edge0, edge1);
    float c = dot(edge1, edge1);
    float d = dot(edge0, v0ToPoint);
    float e = dot(edge1, v0ToPoint);
    float det = a * c - b * b;
    float s = b * e - c * d;
    float t = b * d - a * e;

    if (s + t <= det) {
        if (s < 0.0f) {
            if (t < 0.0f) {
                return min(point_segment_distance(point, v0, v1), point_segment_distance(point, v0, v2));
            }
            return point_segment_distance(point, v0, v2);
        }
        if (t < 0.0f) {
            return point_segment_distance(point, v0, v1);
        }
    } else {
        if (s < 0.0f) {
            return point_segment_distance(point, v1, v2);
        }
        if (t < 0.0f) {
            return point_segment_distance(point, v1, v2);
        }
    }

    float invDet = 1.0f / max(det, 1.0e-12f);
    s *= invDet;
    t *= invDet;
    float3 closestPoint = v0 + edge0 * s + edge1 * t;
    return length(point - closestPoint);
}

static bool point_in_collision_mesh_sdf_bounds(float3 pointLocal, device const AVBDGPUCollisionMeshInfo &meshInfo) {
    return all(pointLocal >= meshInfo.sdfLocalMinBounds.xyz) && all(pointLocal <= meshInfo.sdfLocalMaxBounds.xyz);
}

static float3 collision_mesh_sdf_texcoord(float3 pointLocal, device const AVBDGPUCollisionMeshInfo &meshInfo) {
    float3 extent = max(meshInfo.sdfLocalMaxBounds.xyz - meshInfo.sdfLocalMinBounds.xyz, float3(1.0e-5f));
    return (pointLocal - meshInfo.sdfLocalMinBounds.xyz) / extent;
}

static float sample_collision_mesh_sdf_value(constant AVBDCollisionMeshSDFSet &meshSDFSet,
                                             int sdfResourceIndex,
                                             float3 pointLocal,
                                             device const AVBDGPUCollisionMeshInfo &meshInfo) {
    float3 texCoord = collision_mesh_sdf_texcoord(pointLocal, meshInfo);
    if (any(texCoord < 0.0f) || any(texCoord > 1.0f)) {
        return FLT_MAX;
    }

    texture3d<float, access::sample> coarseTexture = meshSDFSet.coarse[sdfResourceIndex];
    texture3d<float, access::sample> atlasTexture = meshSDFSet.atlas[sdfResourceIndex];
    texture3d<uint, access::read> indirectionTexture = meshSDFSet.indirection[sdfResourceIndex];

    float3 resolution = max(float3(meshInfo.sdfResolution.xyz), float3(1.0f));
    float3 sampleCoord = clamp(texCoord * resolution - 0.5f, float3(0.0f), resolution - float3(1.0f));
    int3 brickIdx = int3(floor(sampleCoord / float(AVBD_COLLISION_MESH_SDF_BRICK_DIM)));
    int3 brickGrid = int3(indirectionTexture.get_width(), indirectionTexture.get_height(), indirectionTexture.get_depth());
    if (any(brickIdx < 0) || any(brickIdx >= brickGrid)) {
        return coarseTexture.sample(avbd_sdf_sampler, texCoord).r;
    }

    uint brickAtlasIdx = indirectionTexture.read(uint3(brickIdx)).r;
    if (brickAtlasIdx == 0xFFFFFFFFu) {
        return coarseTexture.sample(avbd_sdf_sampler, texCoord).r;
    }

    float3 localPos = sampleCoord - float3(brickIdx * AVBD_COLLISION_MESH_SDF_BRICK_DIM);
    int atlasBricksAcross = max(int(atlasTexture.get_width()) / AVBD_COLLISION_MESH_SDF_STORED_BRICK_DIM, 1);
    int atlasBricksDown = max(int(atlasTexture.get_height()) / AVBD_COLLISION_MESH_SDF_STORED_BRICK_DIM, 1);
    int atlasLayerStride = atlasBricksAcross * atlasBricksDown;
    int atlasIndex = int(brickAtlasIdx);
    int atlasBrickZ = atlasIndex / atlasLayerStride;
    int atlasBrickY = (atlasIndex / atlasBricksAcross) % atlasBricksDown;
    int atlasBrickX = atlasIndex % atlasBricksAcross;
    float3 atlasCoord = float3(
        atlasBrickX * AVBD_COLLISION_MESH_SDF_STORED_BRICK_DIM,
        atlasBrickY * AVBD_COLLISION_MESH_SDF_STORED_BRICK_DIM,
        atlasBrickZ * AVBD_COLLISION_MESH_SDF_STORED_BRICK_DIM
    ) + float(AVBD_COLLISION_MESH_SDF_GUARD_VOXELS) + localPos;
    float3 atlasSize = float3(atlasTexture.get_width(), atlasTexture.get_height(), atlasTexture.get_depth());
    return atlasTexture.sample(avbd_sdf_sampler, (atlasCoord + 0.5f) / atlasSize).r;
}

static float3 collision_mesh_sdf_local_normal(constant AVBDCollisionMeshSDFSet &meshSDFSet,
                                              int sdfResourceIndex,
                                              float3 pointLocal,
                                              device const AVBDGPUCollisionMeshInfo &meshInfo) {
    float3 eps = max(meshInfo.sdfVoxelSize.xyz, float3(1.0e-4f));
    float sdfXPos = sample_collision_mesh_sdf_value(meshSDFSet, sdfResourceIndex, pointLocal + float3(eps.x, 0.0f, 0.0f), meshInfo);
    float sdfXNeg = sample_collision_mesh_sdf_value(meshSDFSet, sdfResourceIndex, pointLocal - float3(eps.x, 0.0f, 0.0f), meshInfo);
    float sdfYPos = sample_collision_mesh_sdf_value(meshSDFSet, sdfResourceIndex, pointLocal + float3(0.0f, eps.y, 0.0f), meshInfo);
    float sdfYNeg = sample_collision_mesh_sdf_value(meshSDFSet, sdfResourceIndex, pointLocal - float3(0.0f, eps.y, 0.0f), meshInfo);
    float sdfZPos = sample_collision_mesh_sdf_value(meshSDFSet, sdfResourceIndex, pointLocal + float3(0.0f, 0.0f, eps.z), meshInfo);
    float sdfZNeg = sample_collision_mesh_sdf_value(meshSDFSet, sdfResourceIndex, pointLocal - float3(0.0f, 0.0f, eps.z), meshInfo);

    if (!isfinite(sdfXPos)) sdfXPos = 1.0f;
    if (!isfinite(sdfXNeg)) sdfXNeg = 1.0f;
    if (!isfinite(sdfYPos)) sdfYPos = 1.0f;
    if (!isfinite(sdfYNeg)) sdfYNeg = 1.0f;
    if (!isfinite(sdfZPos)) sdfZPos = 1.0f;
    if (!isfinite(sdfZNeg)) sdfZNeg = 1.0f;

    float3 gradient = float3(
        (sdfXPos - sdfXNeg) / max(2.0f * eps.x, 1.0e-5f),
        (sdfYPos - sdfYNeg) / max(2.0f * eps.y, 1.0e-5f),
        (sdfZPos - sdfZNeg) / max(2.0f * eps.z, 1.0e-5f)
    );
    return safe_normalize(gradient, float3(0.0f, 1.0f, 0.0f));
}

static float3 collision_mesh_sdf_world_normal(float3 localNormal,
                                              device const AVBDGPUCollisionMeshInfo &meshInfo) {
    float3x3 invLinear = float3x3(
        meshInfo.sdfInvTransform.columns[0].xyz,
        meshInfo.sdfInvTransform.columns[1].xyz,
        meshInfo.sdfInvTransform.columns[2].xyz
    );
    float3 worldNormal = transpose(invLinear) * localNormal;
    return safe_normalize(worldNormal, float3(0.0f, 1.0f, 0.0f));
}

static float max_component(float3 value) {
    return max(value.x, max(value.y, value.z));
}

struct AVBDGPUCollisionMeshVoxelRangeMetal {
    int3 minCoord;
    int3 maxCoord;
    int3 resolution;
    float3 originLocal;
    float3 voxelSize;
    int valid;
};

static int3 collision_mesh_voxel_resolution(device const AVBDGPUCollisionMeshInfo &meshInfo) {
    return max(meshInfo.sdfResolution.xyz, int3(1));
}

static int3 collision_mesh_voxel_range_counts(thread const AVBDGPUCollisionMeshVoxelRangeMetal &range) {
    if (!range.valid) {
        return int3(0);
    }
    return max(range.maxCoord - range.minCoord + 1, int3(0));
}

static int collision_mesh_voxel_range_sample_count(thread const AVBDGPUCollisionMeshVoxelRangeMetal &range) {
    int3 counts = collision_mesh_voxel_range_counts(range);
    return counts.x * counts.y * counts.z;
}

static int hydroelastic_voxel_iteration_step(thread const AVBDGPUCollisionMeshVoxelRangeMetal &range,
                                             int maxSamples) {
    int3 counts = collision_mesh_voxel_range_counts(range);
    int safeMaxSamples = max(maxSamples, 1);
    int step = 1;
    while ((((counts.x + step - 1) / step) *
            ((counts.y + step - 1) / step) *
            ((counts.z + step - 1) / step)) > safeMaxSamples) {
        step += 1;
    }
    return step;
}

static float3 world_aabb_corner(float3 worldMin, float3 worldMax, int cornerIndex) {
    return float3(
        (cornerIndex & 1) != 0 ? worldMax.x : worldMin.x,
        (cornerIndex & 2) != 0 ? worldMax.y : worldMin.y,
        (cornerIndex & 4) != 0 ? worldMax.z : worldMin.z
    );
}

static float hydroelastic_effective_surface_diff(float sdfA,
                                                 float sdfB,
                                                 constant AVBDGPUSolverParams &params) {
    if (sdfA < 0.0f && sdfB < 0.0f) {
        return params.hydroelasticInteriorWeight * (sdfA - sdfB);
    }
    return sdfA - sdfB;
}

static bool world_aabb_to_collision_mesh_voxel_range(
    device const AVBDGPUCollisionMeshInfo &meshInfo,
    float3 worldMin,
    float3 worldMax,
    int padVoxels,
    thread AVBDGPUCollisionMeshVoxelRangeMetal &outRange)
{
    outRange.valid = 0;
    outRange.resolution = collision_mesh_voxel_resolution(meshInfo);
    outRange.originLocal = meshInfo.sdfLocalMinBounds.xyz;
    outRange.voxelSize = max(meshInfo.sdfVoxelSize.xyz, float3(1.0e-5f));
    outRange.minCoord = int3(0);
    outRange.maxCoord = int3(-1);

    if (any(worldMin > worldMax)) {
        return false;
    }

    float3 boundsMin = float3(FLT_MAX);
    float3 boundsMax = float3(-FLT_MAX);
    for (int cornerIndex = 0; cornerIndex < 8; ++cornerIndex) {
        float3 localCorner = (meshInfo.sdfInvTransform * float4(world_aabb_corner(worldMin, worldMax, cornerIndex), 1.0f)).xyz;
        boundsMin = min(boundsMin, localCorner);
        boundsMax = max(boundsMax, localCorner);
    }

    float3 voxelMinF = floor((boundsMin - outRange.originLocal) / outRange.voxelSize) - float3(float(padVoxels));
    float3 voxelMaxF = ceil((boundsMax - outRange.originLocal) / outRange.voxelSize) + float3(float(padVoxels));

    int3 voxelMin = clamp(int3(voxelMinF), int3(0), outRange.resolution - 1);
    int3 voxelMax = clamp(int3(voxelMaxF), int3(0), outRange.resolution - 1);
    if (any(voxelMin > voxelMax)) {
        return false;
    }

    outRange.minCoord = voxelMin;
    outRange.maxCoord = voxelMax;
    outRange.valid = 1;
    return true;
}

static bool world_aabb_to_collision_mesh_voxel_cell_range(
    device const AVBDGPUCollisionMeshInfo &meshInfo,
    float3 worldMin,
    float3 worldMax,
    int padVoxels,
    thread AVBDGPUCollisionMeshVoxelRangeMetal &outRange)
{
    if (!world_aabb_to_collision_mesh_voxel_range(meshInfo, worldMin, worldMax, padVoxels, outRange)) {
        return false;
    }
    if (any(outRange.resolution < int3(2))) {
        outRange.valid = 0;
        return false;
    }

    int3 maxCellCoord = outRange.resolution - 2;
    outRange.minCoord = clamp(outRange.minCoord, int3(0), maxCellCoord);
    outRange.maxCoord = clamp(outRange.maxCoord - 1, int3(0), maxCellCoord);
    if (any(outRange.minCoord > outRange.maxCoord)) {
        outRange.valid = 0;
        return false;
    }
    outRange.valid = 1;
    return true;
}

static float3 collision_mesh_voxel_center_local(
    thread const AVBDGPUCollisionMeshVoxelRangeMetal &range,
    int3 voxelCoord)
{
    return range.originLocal + (float3(voxelCoord) + 0.5f) * range.voxelSize;
}

[[maybe_unused]] static float3 collision_mesh_voxel_center_world(
    device const AVBDGPUCollisionMeshInfo &meshInfo,
    thread const AVBDGPUCollisionMeshVoxelRangeMetal &range,
    int3 voxelCoord)
{
    return (meshInfo.sdfTransform * float4(collision_mesh_voxel_center_local(range, voxelCoord), 1.0f)).xyz;
}

static float3 collision_mesh_voxel_corner_local(
    thread const AVBDGPUCollisionMeshVoxelRangeMetal &range,
    int3 voxelCoord,
    int cornerIndex)
{
    int3 cornerOffset = int3(cornerIndex & 1, (cornerIndex >> 1) & 1, (cornerIndex >> 2) & 1);
    return range.originLocal + float3(voxelCoord + cornerOffset) * range.voxelSize;
}

// ──────────────────────────────────────────────────────────────
// Iso voxel detection and MC centroid extraction
// ──────────────────────────────────────────────────────────────

// Edge connectivity: 12 edges of a cube, each connecting two corner vertices.
// Corner ordering: 0=(0,0,0) 1=(1,0,0) 2=(1,1,0) 3=(0,1,0)
//                  4=(0,0,1) 5=(1,0,1) 6=(1,1,1) 7=(0,1,1)
constant int mc_edge_corners[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},   // bottom edges
    {4, 5}, {5, 6}, {6, 7}, {7, 4},   // top edges
    {0, 4}, {1, 5}, {2, 6}, {3, 7}    // vertical edges
};

// Check if a voxel cell contains a zero crossing of the effective surface
// and if so, return the MC polygon centroid as a world-space sample point.
static bool evaluate_iso_voxel_and_centroid(
    constant AVBDCollisionMeshSDFSet &meshSDFSet,
    device const AVBDGPUCollisionMeshInfo &meshInfoDriver,
    device const AVBDGPUCollisionMeshInfo &meshInfoOther,
    thread const AVBDGPUCollisionMeshVoxelRangeMetal &driverRange,
    int3 voxelCoord,
    bool driverIsA,
    float bandWidth,
    constant AVBDGPUSolverParams &params,
    thread float3 &outWorldPoint)
{
    int sdfResDriver = meshInfoDriver.sdfResourceIndex;
    int sdfResOther = meshInfoOther.sdfResourceIndex;
    if (sdfResDriver < 0 || sdfResDriver >= AVBD_MAX_COLLISION_MESH_SDFS ||
        sdfResOther < 0 || sdfResOther >= AVBD_MAX_COLLISION_MESH_SDFS) {
        return false;
    }

    float cornerDiff[8];
    float3 cornerWorld[8];
    float minDiff = FLT_MAX, maxDiff = -FLT_MAX;
    bool nearBand = false;

    for (int c = 0; c < 8; ++c) {
        float3 localDriver = collision_mesh_voxel_corner_local(driverRange, voxelCoord, c);
        float3 world = (meshInfoDriver.sdfTransform * float4(localDriver, 1.0f)).xyz;
        float3 localOther = (meshInfoOther.sdfInvTransform * float4(world, 1.0f)).xyz;
        if (!point_in_collision_mesh_sdf_bounds(localOther, meshInfoOther)) {
            return false;
        }

        float sdfD = sample_collision_mesh_sdf_value(meshSDFSet, sdfResDriver, localDriver, meshInfoDriver);
        float sdfO = sample_collision_mesh_sdf_value(meshSDFSet, sdfResOther, localOther, meshInfoOther);
        if (!isfinite(sdfD) || !isfinite(sdfO)) {
            return false;
        }

        cornerWorld[c] = world;
        float effectiveDiff = driverIsA
            ? hydroelastic_effective_surface_diff(sdfD, sdfO, params)
            : hydroelastic_effective_surface_diff(sdfO, sdfD, params);
        cornerDiff[c] = effectiveDiff;
        minDiff = min(minDiff, effectiveDiff);
        maxDiff = max(maxDiff, effectiveDiff);
        nearBand = nearBand || (sdfD <= bandWidth && sdfO <= bandWidth);
    }

    if (!nearBand || minDiff > 0.0f || maxDiff < 0.0f) {
        return false;
    }

    // Find zero-crossing points on edges and compute centroid
    float3 centroid = float3(0.0f);
    int crossCount = 0;

    for (int e = 0; e < 12; ++e) {
        int c0 = mc_edge_corners[e][0];
        int c1 = mc_edge_corners[e][1];
        float v0 = cornerDiff[c0];
        float v1 = cornerDiff[c1];
        if ((v0 < 0.0f) == (v1 < 0.0f)) {
            continue;
        }
        float t = clamp(v0 / (v0 - v1), 0.0f, 1.0f);
        centroid += mix(cornerWorld[c0], cornerWorld[c1], t);
        crossCount++;
    }

    if (crossCount < 3) {
        return false;
    }

    outWorldPoint = centroid / float(crossCount);
    return true;
}

static int mesh_mesh_sample_axis_count(float extent, float bandWidth) {
    return extent > bandWidth * 1.5f ? 3 : 1;
}

static bool evaluate_mesh_mesh_hydroelastic_sample(
    constant AVBDCollisionMeshSDFSet &meshSDFSet,
    device const AVBDGPUCollisionMeshInfo &meshInfoA,
    device const AVBDGPUCollisionMeshInfo &meshInfoB,
    float3 worldPoint,
    bool swapBodyOrder,
    float bandWidth,
    constant AVBDGPUSolverParams &params,
    thread float3 &outXA,
    thread float3 &outXB,
    thread float3 &outNormal,
    thread float3 &outMidpoint,
    thread float &outSeparation,
    thread float &outScore)
{
    int sdfResourceIndexA = meshInfoA.sdfResourceIndex;
    int sdfResourceIndexB = meshInfoB.sdfResourceIndex;
    if (sdfResourceIndexA < 0 || sdfResourceIndexA >= AVBD_MAX_COLLISION_MESH_SDFS ||
        sdfResourceIndexB < 0 || sdfResourceIndexB >= AVBD_MAX_COLLISION_MESH_SDFS) {
        return false;
    }

    float3 pointLocalA = (meshInfoA.sdfInvTransform * float4(worldPoint, 1.0f)).xyz;
    float3 pointLocalB = (meshInfoB.sdfInvTransform * float4(worldPoint, 1.0f)).xyz;
    if (!point_in_collision_mesh_sdf_bounds(pointLocalA, meshInfoA) ||
        !point_in_collision_mesh_sdf_bounds(pointLocalB, meshInfoB)) {
        return false;
    }

    float sdfA = sample_collision_mesh_sdf_value(meshSDFSet, sdfResourceIndexA, pointLocalA, meshInfoA);
    float sdfB = sample_collision_mesh_sdf_value(meshSDFSet, sdfResourceIndexB, pointLocalB, meshInfoB);
    if (!isfinite(sdfA) || !isfinite(sdfB) || sdfA > bandWidth || sdfB > bandWidth) {
        return false;
    }

    float effectiveDiff = hydroelastic_effective_surface_diff(sdfA, sdfB, params);
    if (abs(effectiveDiff) > bandWidth) {
        return false;
    }

    float3 localNormalA = collision_mesh_sdf_local_normal(meshSDFSet, sdfResourceIndexA, pointLocalA, meshInfoA);
    float3 localNormalB = collision_mesh_sdf_local_normal(meshSDFSet, sdfResourceIndexB, pointLocalB, meshInfoB);
    float3 worldNormalA = collision_mesh_sdf_world_normal(localNormalA, meshInfoA);
    float3 worldNormalB = collision_mesh_sdf_world_normal(localNormalB, meshInfoB);

    float3 surfacePointA = (meshInfoA.sdfTransform * float4(pointLocalA - localNormalA * sdfA, 1.0f)).xyz;
    float3 surfacePointB = (meshInfoB.sdfTransform * float4(pointLocalB - localNormalB * sdfB, 1.0f)).xyz;
    float3 normalAB = safe_normalize(worldNormalB - worldNormalA, worldNormalB);

    float3 xA = surfacePointA;
    float3 xB = surfacePointB;
    float3 normal = normalAB;
    if (swapBodyOrder) {
        xA = surfacePointB;
        xB = surfacePointA;
        normal = -normalAB;
    }

    float separation = dot(normal, xA - xB);
    if (separation > bandWidth) {
        return false;
    }

    float penetration = max(-separation, 0.0f);
    float depthBias = max(-min(sdfA, sdfB), 0.0f);
    float closeness = max(bandWidth - abs(effectiveDiff), 0.0f);

    outXA = xA;
    outXB = xB;
    outNormal = normal;
    outMidpoint = 0.5f * (xA + xB);
    outSeparation = separation;
    outScore = penetration + depthBias * 0.25f + closeness * 0.05f;
    return outScore > 0.0f || separation <= bandWidth * 0.25f;
}

static void get_face_axes(thread const AVBDGPUOBBMetal &box,
                          int axisIndex,
                          thread float3 &u,
                          thread float3 &v,
                          thread float &extentU,
                          thread float &extentV)
{
    if (axisIndex == 0) {
        u = box.axis[1];
        v = box.axis[2];
        extentU = box.halfSize.y;
        extentV = box.halfSize.z;
    } else if (axisIndex == 1) {
        u = box.axis[0];
        v = box.axis[2];
        extentU = box.halfSize.x;
        extentV = box.halfSize.z;
    } else {
        u = box.axis[0];
        v = box.axis[1];
        extentU = box.halfSize.x;
        extentV = box.halfSize.y;
    }
}

static void build_box_face_sample_points(device const AVBDGPUBody &body,
                                         float3 directionToMesh,
                                         thread float3 samplePoints[AVBD_MAX_CONTACTS_PER_PAIR],
                                         thread int sampleSeeds[AVBD_MAX_CONTACTS_PER_PAIR],
                                         thread int &sampleCount) {
    AVBDGPUOBBMetal box;
    make_obb(body, box);

    float3 dir = safe_normalize(directionToMesh, box.axis[1]);
    float projection0 = abs(dot(dir, box.axis[0]));
    float projection1 = abs(dot(dir, box.axis[1]));
    float projection2 = abs(dot(dir, box.axis[2]));

    int axisIndex = 0;
    float bestProjection = projection0;
    if (projection1 > bestProjection) {
        axisIndex = 1;
        bestProjection = projection1;
    }
    if (projection2 > bestProjection) {
        axisIndex = 2;
    }

    float faceSign = dot(dir, box.axis[axisIndex]) >= 0.0f ? 1.0f : -1.0f;
    float3 faceCenter = box.center + box.axis[axisIndex] * (box.halfSize[axisIndex] * faceSign);
    float3 u, v;
    float extentU, extentV;
    get_face_axes(box, axisIndex, u, v, extentU, extentV);

    samplePoints[0] = faceCenter + u * extentU + v * extentV;
    samplePoints[1] = faceCenter - u * extentU + v * extentV;
    samplePoints[2] = faceCenter - u * extentU - v * extentV;
    samplePoints[3] = faceCenter + u * extentU - v * extentV;
    sampleSeeds[0] = (axisIndex << 4) | 0;
    sampleSeeds[1] = (axisIndex << 4) | 1;
    sampleSeeds[2] = (axisIndex << 4) | 2;
    sampleSeeds[3] = (axisIndex << 4) | 3;
    sampleCount = 4;
}

static void build_primitive_mesh_sample_points(device const AVBDGPUBody &body,
                                               float3 meshCenter,
                                               constant AVBDGPUSolverParams &params,
                                               thread float3 samplePoints[AVBD_MAX_CONTACTS_PER_PAIR],
                                               thread int sampleSeeds[AVBD_MAX_CONTACTS_PER_PAIR],
                                               thread int &sampleCount) {
    float3 bodyToMesh = meshCenter - body.positionLin;
    float3 bodyDir = safe_normalize(bodyToMesh, float3(0.0f, 1.0f, 0.0f));

    if (body.renderShape == AVBD_GPU_RENDER_SHAPE_BOX) {
        build_box_face_sample_points(body, bodyToMesh, samplePoints, sampleSeeds, sampleCount);
        return;
    }

    if (body.renderShape == AVBD_GPU_RENDER_SHAPE_TORUS) {
        int torusSphereCount = torus_approx_sphere_count(params);
        int desiredCount = min(AVBD_MAX_CONTACTS_PER_PAIR, torusSphereCount);
        float angleStep = (2.0f * M_PI_F) / float(torusSphereCount);
        float3 localDir = quat_act(quat_conj(body.positionAng), bodyDir);
        float theta = atan2(localDir.y, localDir.x);
        if (theta < 0.0f) theta += 2.0f * M_PI_F;
        int centerIndex = int(round(theta / angleStep)) % torusSphereCount;
        int offsets[AVBD_MAX_CONTACTS_PER_PAIR] = { 0, 1, -1, 2 };
        float torusSphereRadius = torus_approx_sphere_radius(body, params);

        for (int sampleIndex = 0; sampleIndex < desiredCount; ++sampleIndex) {
            int torusSphereIndex = centerIndex + offsets[sampleIndex];
            while (torusSphereIndex < 0) {
                torusSphereIndex += torusSphereCount;
            }
            while (torusSphereIndex >= torusSphereCount) {
                torusSphereIndex -= torusSphereCount;
            }

            float3 torusSphereCenter = torus_sphere_world_center(body, torusSphereIndex, params);
            float3 sampleDir = safe_normalize(meshCenter - torusSphereCenter, bodyDir);
            samplePoints[sampleIndex] = torusSphereCenter + sampleDir * torusSphereRadius;
            sampleSeeds[sampleIndex] = torusSphereIndex & 0xFF;
        }

        sampleCount = desiredCount;
        return;
    }

    samplePoints[0] = body.positionLin + bodyDir * body_radius(body, params);
    sampleSeeds[0] = 0;
    sampleCount = 1;
}

static AVBDGPUFaceFrameMetal build_face_frame(thread const AVBDGPUOBBMetal &box, int axisIndex, float3 outwardNormal) {
    float faceSign = dot(outwardNormal, box.axis[axisIndex]) >= 0.0f ? 1.0f : -1.0f;
    AVBDGPUFaceFrameMetal frame;
    frame.normal = box.axis[axisIndex] * faceSign;
    get_face_axes(box, axisIndex, frame.u, frame.v, frame.extentU, frame.extentV);
    frame.center = box.center + frame.normal * box.halfSize[axisIndex];
    return frame;
}

static int choose_incident_face_axis(thread const AVBDGPUOBBMetal &box, float3 referenceNormal) {
    int axis = 0;
    float best = -FLT_MAX;
    for (int i = 0; i < 3; i++) {
        float d = abs_dot(box.axis[i], referenceNormal);
        if (d > best) {
            best = d;
            axis = i;
        }
    }
    return axis;
}

static void build_incident_face(thread const AVBDGPUOBBMetal &box,
                                int axisIndex,
                                float3 referenceNormal,
                                thread float3 verts[AVBD_GPU_MAX_POLY_VERTS],
                                thread int &count)
{
    float faceSign = dot(box.axis[axisIndex], referenceNormal) > 0.0f ? -1.0f : 1.0f;
    float3 faceNormal = box.axis[axisIndex] * faceSign;
    float3 faceCenter = box.center + faceNormal * box.halfSize[axisIndex];
    float3 u, v;
    float extentU, extentV;
    get_face_axes(box, axisIndex, u, v, extentU, extentV);

    verts[0] = faceCenter + u * extentU + v * extentV;
    verts[1] = faceCenter - u * extentU + v * extentV;
    verts[2] = faceCenter - u * extentU - v * extentV;
    verts[3] = faceCenter + u * extentU - v * extentV;
    count = 4;
}

static int clip_polygon_against_plane(thread const float3 inVerts[AVBD_GPU_MAX_POLY_VERTS],
                                      int inCount,
                                      float3 planeNormal,
                                      float planeOffset,
                                      thread float3 outVerts[AVBD_GPU_MAX_POLY_VERTS])
{
    if (inCount <= 0) {
        return 0;
    }

    int outCount = 0;
    float3 a = inVerts[inCount - 1];
    float da = dot(planeNormal, a) - planeOffset;

    for (int i = 0; i < inCount; i++) {
        float3 b = inVerts[i];
        float db = dot(planeNormal, b) - planeOffset;
        bool aInside = da <= AVBD_GPU_PLANE_EPSILON;
        bool bInside = db <= AVBD_GPU_PLANE_EPSILON;

        if (aInside != bInside) {
            float t = 0.0f;
            float denom = da - db;
            if (abs(denom) > AVBD_GPU_SAT_AXIS_EPSILON) {
                t = clamp(da / denom, 0.0f, 1.0f);
            }
            if (outCount < AVBD_GPU_MAX_POLY_VERTS) {
                outVerts[outCount++] = a + (b - a) * t;
            }
        }

        if (bInside && outCount < AVBD_GPU_MAX_POLY_VERTS) {
            outVerts[outCount++] = b;
        }

        a = b;
        da = db;
    }

    return outCount;
}

static AVBDGPUContact make_contact(device const AVBDGPUBody &bodyA,
                                   device const AVBDGPUBody &bodyB,
                                   float3 xA,
                                   float3 xB,
                                   int featureKey)
{
    AVBDGPUContact contact;
    contact.rA = quat_act(quat_conj(bodyA.positionAng), xA - bodyA.positionLin);
    contact.rB = quat_act(quat_conj(bodyB.positionAng), xB - bodyB.positionLin);
    contact.C0 = float3(0);
    contact.penalty = float3(0);
    contact.lambda = float3(0);
    contact.featureKey = featureKey;
    contact.active = 0;
    contact.stick = 0;
    return contact;
}

static void append_contact(thread AVBDGPUContactBuilderMetal &builder,
                           device const AVBDGPUBody &bodyA,
                           device const AVBDGPUBody &bodyB,
                           float3 xA,
                           float3 xB,
                           int featureKey)
{
    float3 midpoint = (xA + xB) * 0.5f;
    for (int i = 0; i < builder.count; i++) {
        if (length_squared(midpoint - builder.midpoints[i]) < AVBD_GPU_CONTACT_MERGE_DIST_SQ) {
            return;
        }
    }
    if (builder.count >= AVBD_MAX_CONTACTS_PER_PAIR_BURST) {
        return;
    }
    builder.contacts[builder.count] = make_contact(bodyA, bodyB, xA, xB, featureKey);
    builder.midpoints[builder.count] = midpoint;
    builder.count++;
}

static AVBDGPUContact make_mesh_contact(device const AVBDGPUBody &bodyA,
                                        float3 xA,
                                        float3 xMeshWorld,
                                        int featureKey)
{
    AVBDGPUContact contact;
    contact.rA = quat_act(quat_conj(bodyA.positionAng), xA - bodyA.positionLin);
    contact.rB = xMeshWorld;
    contact.C0 = float3(0);
    contact.penalty = float3(0);
    contact.lambda = float3(0);
    contact.featureKey = featureKey;
    contact.active = 0;
    contact.stick = 0;
    return contact;
}

static void append_mesh_contact(thread AVBDGPUContactBuilderMetal &builder,
                                device const AVBDGPUBody &bodyA,
                                float3 xA,
                                float3 xMeshWorld,
                                int featureKey)
{
    float3 midpoint = (xA + xMeshWorld) * 0.5f;
    for (int i = 0; i < builder.count; i++) {
        if (length_squared(midpoint - builder.midpoints[i]) < AVBD_GPU_CONTACT_MERGE_DIST_SQ) {
            return;
        }
    }
    if (builder.count >= AVBD_MAX_CONTACTS_PER_PAIR) {
        return;
    }
    builder.contacts[builder.count] = make_mesh_contact(bodyA, xA, xMeshWorld, featureKey);
    builder.midpoints[builder.count] = midpoint;
    builder.count++;
}

static void append_primitive_mesh_contact(thread AVBDGPUContactBuilderMetal &builder,
                                          device const AVBDGPUBody &bodyA,
                                          device const AVBDGPUBody &bodyB,
                                          float3 xA,
                                          float3 xMeshWorld,
                                          int featureKey)
{
    float3 midpoint = (xA + xMeshWorld) * 0.5f;
    for (int i = 0; i < builder.count; i++) {
        if (length_squared(midpoint - builder.midpoints[i]) < AVBD_GPU_CONTACT_MERGE_DIST_SQ) {
            return;
        }
    }
    if (builder.count >= AVBD_MAX_CONTACTS_PER_PAIR) {
        return;
    }
    builder.contacts[builder.count] = make_contact(bodyA, bodyB, xA, xMeshWorld, featureKey);
    builder.midpoints[builder.count] = midpoint;
    builder.count++;
}

static void finalize_builder_contacts(device const AVBDGPUBody &bodyA,
                                      device const AVBDGPUBody &bodyB,
                                      thread AVBDGPUContactBuilderMetal &builder,
                                      thread Mat3 &basis,
                                      float collisionMargin)
{
    for (int i = 0; i < builder.count; i++) {
        float3 xA = quat_act(bodyA.positionAng, builder.contacts[i].rA) + bodyA.positionLin;
        float3 xB = quat_act(bodyB.positionAng, builder.contacts[i].rB) + bodyB.positionLin;
        float3 diff = xA - xB;
        builder.contacts[i].C0 = float3(dot(basis.r0, diff), dot(basis.r1, diff), dot(basis.r2, diff)) + float3(collisionMargin, 0, 0);
        builder.contacts[i].penalty = float3(AVBD_GPU_CONTACT_PENALTY_START);
        builder.contacts[i].active = 1;
    }
}

static void finalize_builder_mesh_contacts(device const AVBDGPUBody &bodyA,
                                           thread AVBDGPUContactBuilderMetal &builder,
                                           thread Mat3 &basis,
                                           float collisionMargin)
{
    for (int i = 0; i < builder.count; i++) {
        float3 xA = quat_act(bodyA.positionAng, builder.contacts[i].rA) + bodyA.positionLin;
        float3 diff = xA - builder.contacts[i].rB;
        builder.contacts[i].C0 = float3(dot(basis.r0, diff), dot(basis.r1, diff), dot(basis.r2, diff)) + float3(collisionMargin, 0, 0);
        builder.contacts[i].penalty = float3(AVBD_GPU_CONTACT_PENALTY_START);
        builder.contacts[i].active = 1;
    }
}

static int primitive_mesh_feature_key(int meshIndex, int triangleIndex, int sampleSeed) {
    int encoded = ((meshIndex * 73856093) ^ (triangleIndex * 19349663) ^ (sampleSeed * 83492791)) & 0x00FFFFFF;
    return AVBD_GPU_FEATURE_PRIMITIVE_MESH | encoded;
}

static void support_edge(thread const AVBDGPUOBBMetal &box,
                         int axisIndex,
                         float3 dir,
                         thread float3 &edgeA,
                         thread float3 &edgeB)
{
    int axis1 = (axisIndex + 1) % 3;
    int axis2 = (axisIndex + 2) % 3;
    float sign1 = dot(dir, box.axis[axis1]) >= 0.0f ? 1.0f : -1.0f;
    float sign2 = dot(dir, box.axis[axis2]) >= 0.0f ? 1.0f : -1.0f;
    float3 edgeCenter = box.center
        + box.axis[axis1] * (box.halfSize[axis1] * sign1)
        + box.axis[axis2] * (box.halfSize[axis2] * sign2);
    edgeA = edgeCenter - box.axis[axisIndex] * box.halfSize[axisIndex];
    edgeB = edgeCenter + box.axis[axisIndex] * box.halfSize[axisIndex];
}

static void closest_points_on_segments(float3 p0,
                                       float3 p1,
                                       float3 q0,
                                       float3 q1,
                                       thread float3 &outA,
                                       thread float3 &outB)
{
    float3 d1 = p1 - p0;
    float3 d2 = q1 - q0;
    float3 r = p0 - q0;
    float a = dot(d1, d1);
    float e = dot(d2, d2);
    float f = dot(d2, r);
    float s = 0.0f;
    float t = 0.0f;

    if (a <= AVBD_GPU_SAT_AXIS_EPSILON && e <= AVBD_GPU_SAT_AXIS_EPSILON) {
        outA = p0;
        outB = q0;
        return;
    }

    if (a <= AVBD_GPU_SAT_AXIS_EPSILON) {
        t = clamp(f / e, 0.0f, 1.0f);
    } else {
        float c = dot(d1, r);
        if (e <= AVBD_GPU_SAT_AXIS_EPSILON) {
            s = clamp(-c / a, 0.0f, 1.0f);
        } else {
            float b = dot(d1, d2);
            float denom = a * e - b * b;
            if (abs(denom) > AVBD_GPU_SAT_AXIS_EPSILON) {
                s = clamp((b * f - c * e) / denom, 0.0f, 1.0f);
            }
            t = (b * s + f) / e;
            if (t < 0.0f) {
                t = 0.0f;
                s = clamp(-c / a, 0.0f, 1.0f);
            } else if (t > 1.0f) {
                t = 1.0f;
                s = clamp((b - c) / a, 0.0f, 1.0f);
            }
        }
    }

    outA = p0 + d1 * s;
    outB = q0 + d2 * t;
}

static void build_face_manifold(device const AVBDGPUBody &bodyA,
                                device const AVBDGPUBody &bodyB,
                                thread const AVBDGPUOBBMetal &boxA,
                                thread const AVBDGPUOBBMetal &boxB,
                                bool referenceIsA,
                                int referenceAxis,
                                float3 normalAB,
                                thread AVBDGPUContactBuilderMetal &builder)
{
    thread const AVBDGPUOBBMetal &referenceBox = referenceIsA ? boxA : boxB;
    thread const AVBDGPUOBBMetal &incidentBox = referenceIsA ? boxB : boxA;
    float3 referenceOutward = referenceIsA ? normalAB : -normalAB;
    AVBDGPUFaceFrameMetal referenceFace = build_face_frame(referenceBox, referenceAxis, referenceOutward);
    int incidentAxis = choose_incident_face_axis(incidentBox, referenceFace.normal);

    float3 polyA[AVBD_GPU_MAX_POLY_VERTS];
    float3 polyB[AVBD_GPU_MAX_POLY_VERTS];
    int count = 0;
    build_incident_face(incidentBox, incidentAxis, referenceFace.normal, polyA, count);
    count = clip_polygon_against_plane(polyA, count, referenceFace.u, dot(referenceFace.u, referenceFace.center) + referenceFace.extentU, polyB);
    if (count <= 0) return;
    count = clip_polygon_against_plane(polyB, count, -referenceFace.u, dot(-referenceFace.u, referenceFace.center) + referenceFace.extentU, polyA);
    if (count <= 0) return;
    count = clip_polygon_against_plane(polyA, count, referenceFace.v, dot(referenceFace.v, referenceFace.center) + referenceFace.extentV, polyB);
    if (count <= 0) return;
    count = clip_polygon_against_plane(polyB, count, -referenceFace.v, dot(-referenceFace.v, referenceFace.center) + referenceFace.extentV, polyA);
    if (count <= 0) return;

    int featurePrefix = (referenceIsA ? AVBD_GPU_AXIS_FACE_A : AVBD_GPU_AXIS_FACE_B) << 24;
    featurePrefix |= (referenceAxis & 0xFF) << 16;
    featurePrefix |= (incidentAxis & 0xFF) << 8;

    for (int i = 0; i < count && builder.count < AVBD_MAX_CONTACTS_PER_PAIR_BURST; i++) {
        float3 pIncident = polyA[i];
        float distance = dot(pIncident - referenceFace.center, referenceFace.normal);
        if (distance > AVBD_GPU_PLANE_EPSILON) {
            continue;
        }

        float3 pReference = pIncident - referenceFace.normal * distance;
        float3 xA = referenceIsA ? pReference : pIncident;
        float3 xB = referenceIsA ? pIncident : pReference;
        append_contact(builder, bodyA, bodyB, xA, xB, featurePrefix | (i & 0xFF));
    }

    if (builder.count == 0) {
        append_contact(builder, bodyA, bodyB, support_point(boxA, normalAB), support_point(boxB, -normalAB), featurePrefix);
    }
}

static void build_edge_contact(device const AVBDGPUBody &bodyA,
                               device const AVBDGPUBody &bodyB,
                               thread const AVBDGPUOBBMetal &boxA,
                               thread const AVBDGPUOBBMetal &boxB,
                               int axisA,
                               int axisB,
                               float3 normalAB,
                               thread AVBDGPUContactBuilderMetal &builder)
{
    float3 a0, a1, b0, b1;
    support_edge(boxA, axisA, normalAB, a0, a1);
    support_edge(boxB, axisB, -normalAB, b0, b1);
    float3 xA, xB;
    closest_points_on_segments(a0, a1, b0, b1, xA, xB);

    int featureKey = (AVBD_GPU_AXIS_EDGE << 24) | ((axisA & 0xFF) << 8) | (axisB & 0xFF);
    append_contact(builder, bodyA, bodyB, xA, xB, featureKey);

    if (builder.count == 0) {
        append_contact(builder, bodyA, bodyB, support_point(boxA, normalAB), support_point(boxB, -normalAB), featureKey);
    }
}

static void closest_point_on_box(thread const AVBDGPUOBBMetal &box,
                                 float3 point,
                                 thread float3 &closestPoint,
                                 thread float3 &normalToBox)
{
    float3 localPoint = quat_act(quat_conj(box.rotation), point - box.center);
    float3 clamped = clamp(localPoint, -box.halfSize, box.halfSize);
    float3 deltaLocal = clamped - localPoint;
    float deltaLenSq = length_squared(deltaLocal);
    if (deltaLenSq > AVBD_GPU_SAT_AXIS_EPSILON) {
        closestPoint = quat_act(box.rotation, clamped) + box.center;
        normalToBox = quat_act(box.rotation, deltaLocal / sqrt(deltaLenSq));
        return;
    }

    float dx = box.halfSize.x - abs(localPoint.x);
    float dy = box.halfSize.y - abs(localPoint.y);
    float dz = box.halfSize.z - abs(localPoint.z);
    int axis = 0;
    if (dy < dx && dy <= dz) {
        axis = 1;
    } else if (dz < dx && dz < dy) {
        axis = 2;
    }

    float3 normalLocal = float3(0.0f);
    float signToFace = 1.0f;
    if (axis == 0) {
        signToFace = localPoint.x >= 0.0f ? 1.0f : -1.0f;
        clamped.x = box.halfSize.x * signToFace;
        normalLocal.x = signToFace;
    } else if (axis == 1) {
        signToFace = localPoint.y >= 0.0f ? 1.0f : -1.0f;
        clamped.y = box.halfSize.y * signToFace;
        normalLocal.y = signToFace;
    } else {
        signToFace = localPoint.z >= 0.0f ? 1.0f : -1.0f;
        clamped.z = box.halfSize.z * signToFace;
        normalLocal.z = signToFace;
    }

    closestPoint = quat_act(box.rotation, clamped) + box.center;
    normalToBox = quat_act(box.rotation, normalLocal);
}

static bool collide_sphere_sphere_gpu(device const AVBDGPUBody &bodyA,
                                      device const AVBDGPUBody &bodyB,
                                      thread AVBDGPUContactBuilderMetal &builder,
                                      thread Mat3 &basis,
                                      constant AVBDGPUSolverParams &params,
                                      float collisionMargin)
{
    float radiusA = body_radius(bodyA, params);
    float radiusB = body_radius(bodyB, params);
    float3 delta = bodyB.positionLin - bodyA.positionLin;
    float distSq = length_squared(delta);
    float radius = radiusA + radiusB;
    if (distSq > radius * radius) {
        return false;
    }

    float3 normalAB = distSq > AVBD_GPU_SAT_AXIS_EPSILON ? delta / sqrt(distSq) : float3(1, 0, 0);
    basis = orthonormal_basis(-normalAB);

    float3 xA = bodyA.positionLin + normalAB * radiusA;
    float3 xB = bodyB.positionLin - normalAB * radiusB;
    append_contact(builder, bodyA, bodyB, xA, xB, AVBD_GPU_FEATURE_SPHERE_SPHERE);
    finalize_builder_contacts(bodyA, bodyB, builder, basis, collisionMargin);

    return builder.count > 0;
}

static bool collide_sphere_box_gpu(device const AVBDGPUBody &sphereBody,
                                   device const AVBDGPUBody &boxBody,
                                   bool sphereIsBodyA,
                                   thread AVBDGPUContactBuilderMetal &builder,
                                   thread Mat3 &basis,
                                   constant AVBDGPUSolverParams &params,
                                   float collisionMargin)
{
    AVBDGPUOBBMetal box;
    make_obb(boxBody, box);

    float3 closestPoint;
    float3 sphereToBoxNormal;
    closest_point_on_box(box, sphereBody.positionLin, closestPoint, sphereToBoxNormal);

    float sphereRadius = body_radius(sphereBody, params);
    float3 offset = closestPoint - sphereBody.positionLin;
    if (length_squared(offset) > sphereRadius * sphereRadius) {
        return false;
    }

    float3 normalAB = sphereIsBodyA ? sphereToBoxNormal : -sphereToBoxNormal;
    basis = orthonormal_basis(-normalAB);

    float3 spherePoint = sphereBody.positionLin + sphereToBoxNormal * sphereRadius;
    float3 xA = sphereIsBodyA ? spherePoint : closestPoint;
    float3 xB = sphereIsBodyA ? closestPoint : spherePoint;
    append_contact(builder,
                   sphereIsBodyA ? sphereBody : boxBody,
                   sphereIsBodyA ? boxBody : sphereBody,
                   xA,
                   xB,
                   AVBD_GPU_FEATURE_SPHERE_BOX);
    finalize_builder_contacts(sphereIsBodyA ? sphereBody : boxBody,
                              sphereIsBodyA ? boxBody : sphereBody,
                              builder,
                              basis,
                              collisionMargin);

    return builder.count > 0;
}

static bool sphere_sphere_contact(float3 centerA,
                                  float radiusA,
                                  float3 centerB,
                                  float radiusB,
                                  thread float3 &normalAB,
                                  thread float3 &xA,
                                  thread float3 &xB,
                                  thread float &penetration)
{
    float3 delta = centerB - centerA;
    float distSq = length_squared(delta);
    float radius = radiusA + radiusB;
    if (distSq > radius * radius) {
        return false;
    }

    float dist = sqrt(max(distSq, 0.0f));
    normalAB = dist > AVBD_GPU_SAT_AXIS_EPSILON ? delta / dist : float3(1, 0, 0);
    xA = centerA + normalAB * radiusA;
    xB = centerB - normalAB * radiusB;
    penetration = radius - dist;
    return true;
}

static bool sphere_box_contact(float3 sphereCenter,
                               float sphereRadius,
                               thread const AVBDGPUOBBMetal &box,
                               thread float3 &normalFromSphereToBox,
                               thread float3 &spherePoint,
                               thread float3 &boxPoint,
                               thread float &penetration)
{
    closest_point_on_box(box, sphereCenter, boxPoint, normalFromSphereToBox);
    float3 offset = boxPoint - sphereCenter;
    float distSq = length_squared(offset);
    if (distSq > sphereRadius * sphereRadius) {
        return false;
    }

    spherePoint = sphereCenter + normalFromSphereToBox * sphereRadius;
    penetration = sphereRadius - sqrt(max(distSq, 0.0f));
    return true;
}

static bool collide_torus_body_gpu(device const AVBDGPUBody &torusBody,
                                   device const AVBDGPUBody &otherBody,
                                   bool torusIsBodyA,
                                   thread AVBDGPUContactBuilderMetal &builder,
                                   thread Mat3 &basis,
                                   constant AVBDGPUSolverParams &params,
                                   float collisionMargin)
{
    float torusSphereRadius = torus_approx_sphere_radius(torusBody, params);
    float3 bestNormalAB = float3(1, 0, 0);
    float bestPenetration = -FLT_MAX;
    int torusSphereCount = torus_approx_sphere_count(params);

    if (otherBody.renderShape == AVBD_GPU_RENDER_SHAPE_SPHERE) {
        float otherRadius = body_radius(otherBody, params);
        for (int torusSphereIndex = 0; torusSphereIndex < torusSphereCount; torusSphereIndex++) {
            float3 torusSphereCenter = torus_sphere_world_center(torusBody, torusSphereIndex, params);
            float3 normalFromTorusToOther;
            float3 xTorus, xOther;
            float penetration;
            if (!sphere_sphere_contact(torusSphereCenter, torusSphereRadius, otherBody.positionLin, otherRadius, normalFromTorusToOther, xTorus, xOther, penetration)) {
                continue;
            }

            float3 normalAB = torusIsBodyA ? normalFromTorusToOther : -normalFromTorusToOther;
            if (torusIsBodyA) {
                append_contact(builder, torusBody, otherBody, xTorus, xOther, AVBD_GPU_FEATURE_TORUS_SPHERE | (torusSphereIndex & 0xFF));
            } else {
                append_contact(builder, otherBody, torusBody, xOther, xTorus, AVBD_GPU_FEATURE_TORUS_SPHERE | (torusSphereIndex & 0xFF));
            }

            if (penetration > bestPenetration) {
                bestPenetration = penetration;
                bestNormalAB = normalAB;
            }
        }
    } else if (otherBody.renderShape == AVBD_GPU_RENDER_SHAPE_BOX) {
        AVBDGPUOBBMetal box;
        make_obb(otherBody, box);

        for (int torusSphereIndex = 0; torusSphereIndex < torusSphereCount; torusSphereIndex++) {
            float3 torusSphereCenter = torus_sphere_world_center(torusBody, torusSphereIndex, params);
            float3 normalFromTorusToOther;
            float3 xTorus, xOther;
            float penetration;
            if (!sphere_box_contact(torusSphereCenter, torusSphereRadius, box, normalFromTorusToOther, xTorus, xOther, penetration)) {
                continue;
            }

            float3 normalAB = torusIsBodyA ? normalFromTorusToOther : -normalFromTorusToOther;
            if (torusIsBodyA) {
                append_contact(builder, torusBody, otherBody, xTorus, xOther, AVBD_GPU_FEATURE_TORUS_BOX | (torusSphereIndex & 0xFF));
            } else {
                append_contact(builder, otherBody, torusBody, xOther, xTorus, AVBD_GPU_FEATURE_TORUS_BOX | (torusSphereIndex & 0xFF));
            }

            if (penetration > bestPenetration) {
                bestPenetration = penetration;
                bestNormalAB = normalAB;
            }
        }
    } else if (otherBody.renderShape == AVBD_GPU_RENDER_SHAPE_TORUS) {
        float torusMajorRadius = torus_major_radius(torusBody);
        float otherSphereRadius = torus_approx_sphere_radius(otherBody, params);
        float otherMajorRadius = torus_major_radius(otherBody);
        float4 otherQuatInv = quat_conj(otherBody.positionAng);
        float reach = torusSphereRadius + otherSphereRadius + params.collisionMargin;
        float reachSq = reach * reach;
        float angleStep = 2.0f * M_PI_F / float(torusSphereCount);
        float invAngleStep = 1.0f / angleStep;
        float cosStep = cos(angleStep);
        float sinStep = sin(angleStep);
        float3 torusAxisX = quat_act(torusBody.positionAng, float3(1, 0, 0));
        float3 torusAxisY = quat_act(torusBody.positionAng, float3(0, 1, 0));
        float3 otherAxisX = quat_act(otherBody.positionAng, float3(1, 0, 0));
        float3 otherAxisY = quat_act(otherBody.positionAng, float3(0, 1, 0));
        float torusCos = 1.0f;
        float torusSin = 0.0f;

        for (int torusSphereIndex = 0; torusSphereIndex < torusSphereCount; torusSphereIndex++) {
            float3 torusSphereCenter = torusBody.positionLin
                + (torusAxisX * torusCos + torusAxisY * torusSin) * torusMajorRadius;

            // --- 1st stage: Distance to major circle pruning ---
            float3 localPos = quat_act(otherQuatInv, torusSphereCenter - otherBody.positionLin);
            float rhoSq = dot(localPos.xy, localPos.xy);
            float rho = sqrt(max(rhoSq, 0.0f));
            float radialDelta = rho - otherMajorRadius;
            float distToCircleSq = radialDelta * radialDelta + localPos.z * localPos.z;
            if (distToCircleSq <= reachSq) {
                // --- 2nd stage: Range limiting using angle ---
                int halfWindow = torusSphereCount / 2;
                int centerIdx = 0;
                int candidateCount = torusSphereCount;
                int startIdx = 0;
                if (otherMajorRadius > AVBD_GPU_SAT_AXIS_EPSILON && rho > AVBD_GPU_SAT_AXIS_EPSILON) {
                    float denom = 2.0f * rho * otherMajorRadius;
                    float cosLimit = (rhoSq + otherMajorRadius * otherMajorRadius + localPos.z * localPos.z - reachSq) / denom;
                    if (cosLimit > -1.0f) {
                        float angleRadius = acos(clamp(cosLimit, -1.0f, 1.0f));
                        halfWindow = min(torusSphereCount / 2, int(ceil(angleRadius * invAngleStep)) + 1);
                    }

                    float theta = atan2(localPos.y, localPos.x);
                    if (theta < 0.0f) theta += 2.0f * M_PI_F;
                    centerIdx = int(round(theta * invAngleStep)) % torusSphereCount;
                    candidateCount = min(torusSphereCount, halfWindow * 2 + 1);
                    startIdx = centerIdx - halfWindow;
                }

                float startAngle = angleStep * float(startIdx);
                float otherCos = cos(startAngle);
                float otherSin = sin(startAngle);
                int rawOtherSphereIndex = startIdx;
                for (int candidateIndex = 0; candidateIndex < candidateCount; candidateIndex++) {
                    int otherSphereIndex = rawOtherSphereIndex;
                    if (otherSphereIndex < 0) {
                        otherSphereIndex += torusSphereCount;
                    } else if (otherSphereIndex >= torusSphereCount) {
                        otherSphereIndex -= torusSphereCount;
                    }

                    float3 otherSphereCenter = otherBody.positionLin
                        + (otherAxisX * otherCos + otherAxisY * otherSin) * otherMajorRadius;
                    float3 normalFromTorusToOther;
                    float3 xTorus, xOther;
                    float penetration;
                    if (sphere_sphere_contact(torusSphereCenter, torusSphereRadius, otherSphereCenter, otherSphereRadius, normalFromTorusToOther, xTorus, xOther, penetration)) {
                        int featureKey = AVBD_GPU_FEATURE_TORUS_TORUS | ((torusSphereIndex & 0xFF) << 8) | (otherSphereIndex & 0xFF);
                        float3 normalAB = torusIsBodyA ? normalFromTorusToOther : -normalFromTorusToOther;
                        if (torusIsBodyA) {
                            append_contact(builder, torusBody, otherBody, xTorus, xOther, featureKey);
                        } else {
                            append_contact(builder, otherBody, torusBody, xOther, xTorus, featureKey);
                        }

                        if (penetration > bestPenetration) {
                            bestPenetration = penetration;
                            bestNormalAB = normalAB;
                        }
                    }

                    rawOtherSphereIndex++;
                    float nextOtherCos = otherCos * cosStep - otherSin * sinStep;
                    float nextOtherSin = otherSin * cosStep + otherCos * sinStep;
                    otherCos = nextOtherCos;
                    otherSin = nextOtherSin;
                }
            }

            float nextTorusCos = torusCos * cosStep - torusSin * sinStep;
            float nextTorusSin = torusSin * cosStep + torusCos * sinStep;
            torusCos = nextTorusCos;
            torusSin = nextTorusSin;
        }
    }

    if (builder.count <= 0) {
        return false;
    }

    basis = orthonormal_basis(-bestNormalAB);
    if (torusIsBodyA) {
        finalize_builder_contacts(torusBody, otherBody, builder, basis, collisionMargin);
    } else {
        finalize_builder_contacts(otherBody, torusBody, builder, basis, collisionMargin);
    }
    return true;
}

static bool collide_bodies_gpu(device const AVBDGPUBody &bodyA,
                               device const AVBDGPUBody &bodyB,
                               thread AVBDGPUContactBuilderMetal &builder,
                               thread Mat3 &basis,
                               constant AVBDGPUSolverParams &params,
                               float collisionMargin)
{
    if (bodyA.renderShape == AVBD_GPU_RENDER_SHAPE_SPHERE && bodyB.renderShape == AVBD_GPU_RENDER_SHAPE_SPHERE) {
        return collide_sphere_sphere_gpu(bodyA, bodyB, builder, basis, params, collisionMargin);
    }
    if (bodyA.renderShape == AVBD_GPU_RENDER_SHAPE_SPHERE && bodyB.renderShape == AVBD_GPU_RENDER_SHAPE_BOX) {
        return collide_sphere_box_gpu(bodyA, bodyB, true, builder, basis, params, collisionMargin);
    }
    if (bodyA.renderShape == AVBD_GPU_RENDER_SHAPE_BOX && bodyB.renderShape == AVBD_GPU_RENDER_SHAPE_SPHERE) {
        return collide_sphere_box_gpu(bodyB, bodyA, false, builder, basis, params, collisionMargin);
    }
    if (bodyA.renderShape == AVBD_GPU_RENDER_SHAPE_TORUS) {
        return collide_torus_body_gpu(bodyA, bodyB, true, builder, basis, params, collisionMargin);
    }
    if (bodyB.renderShape == AVBD_GPU_RENDER_SHAPE_TORUS) {
        return collide_torus_body_gpu(bodyB, bodyA, false, builder, basis, params, collisionMargin);
    }

    AVBDGPUOBBMetal boxA, boxB;
    make_obb(bodyA, boxA);
    make_obb(bodyB, boxB);
    float3 delta = boxB.center - boxA.center;

    AVBDGPUSatAxisMetal bestFace;
    bestFace.type = AVBD_GPU_AXIS_FACE_A;
    bestFace.indexA = -1;
    bestFace.indexB = -1;
    bestFace.separation = -FLT_MAX;
    bestFace.normalAB = float3(0);
    bestFace.valid = false;

    AVBDGPUSatAxisMetal bestEdge = bestFace;
    bestEdge.type = AVBD_GPU_AXIS_EDGE;

    for (int i = 0; i < 3; i++) {
        if (!test_axis(boxA, boxB, delta, boxA.axis[i], AVBD_GPU_AXIS_FACE_A, i, -1, bestFace)) {
            return false;
        }
    }
    for (int i = 0; i < 3; i++) {
        if (!test_axis(boxA, boxB, delta, boxB.axis[i], AVBD_GPU_AXIS_FACE_B, -1, i, bestFace)) {
            return false;
        }
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (!test_axis(boxA, boxB, delta, cross(boxA.axis[i], boxB.axis[j]), AVBD_GPU_AXIS_EDGE, i, j, bestEdge)) {
                return false;
            }
        }
    }

    if (!bestFace.valid) {
        return false;
    }

    AVBDGPUSatAxisMetal best = bestFace;
    if (bestEdge.valid && 0.95f * bestEdge.separation > bestFace.separation + 0.01f) {
        best = bestEdge;
    }

    basis = orthonormal_basis(-best.normalAB);

    if (best.type == AVBD_GPU_AXIS_EDGE) {
        build_edge_contact(bodyA, bodyB, boxA, boxB, best.indexA, best.indexB, best.normalAB, builder);
    } else if (best.type == AVBD_GPU_AXIS_FACE_A) {
        build_face_manifold(bodyA, bodyB, boxA, boxB, true, best.indexA, best.normalAB, builder);
    } else {
        build_face_manifold(bodyA, bodyB, boxA, boxB, false, best.indexB, best.normalAB, builder);
    }

    for (int i = 0; i < builder.count; i++) {
        float3 xA = quat_act(boxA.rotation, builder.contacts[i].rA) + bodyA.positionLin;
        float3 xB = quat_act(boxB.rotation, builder.contacts[i].rB) + bodyB.positionLin;
        float3 diff = xA - xB;
        builder.contacts[i].C0 = float3(dot(basis.r0, diff), dot(basis.r1, diff), dot(basis.r2, diff)) + float3(collisionMargin, 0, 0);
        builder.contacts[i].penalty = float3(AVBD_GPU_CONTACT_PENALTY_START);
        builder.contacts[i].active = 1;
    }

    return builder.count > 0;
}

static bool reserve_adjacency_slot(device atomic_int *count, thread int &slot) {
    int expected = atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
    if (expected < AVBD_MAX_CONSTRAINTS_PER_BODY) {
        slot = expected;
        return true;
    }
    return false;
}

static void add_joint_adjacency(device AVBDGPUAdjacency *adjacency, int bodyIdx, int jointIdx) {
    if (bodyIdx < 0) return;
    int slot = 0;
    if (reserve_adjacency_slot(&adjacency[bodyIdx].jointCount, slot)) {
        adjacency[bodyIdx].jointIndices[slot] = jointIdx;
    }
}

static void add_spring_adjacency(device AVBDGPUAdjacency *adjacency, int bodyIdx, int springIdx) {
    if (bodyIdx < 0) return;
    int slot = 0;
    if (reserve_adjacency_slot(&adjacency[bodyIdx].springCount, slot)) {
        adjacency[bodyIdx].springIndices[slot] = springIdx;
    }
}

static void add_manifold_adjacency(device AVBDGPUAdjacency *adjacency, int bodyIdx, int manifoldIdx) {
    if (bodyIdx < 0) return;
    int slot = 0;
    if (reserve_adjacency_slot(&adjacency[bodyIdx].manifoldCount, slot)) {
        adjacency[bodyIdx].manifoldIndices[slot] = manifoldIdx;
    }
}

// Geometric stiffness for ball-socket
static Mat3 geom_stiffness_bs(int k, float3 v) {
    Mat3 m = Mat3::diag(-v[k], -v[k], -v[k]);
    // Add v outer e_k
    m.r0[k] += v[0];
    m.r1[k] += v[1];
    m.r2[k] += v[2];
    return m;
}

// ─────────────────────────────────────────────────────────────
// 6×6 LDLT solve (inline, identical to CPU version)
// ─────────────────────────────────────────────────────────────
static void solve_6x6(Mat3 aLin, Mat3 aAng, Mat3 aCross,
                       float3 bLin, float3 bAng,
                       thread float3 &xLin, thread float3 &xAng)
{
    float A11 = aLin.r0[0];
    float A21 = aLin.r1[0], A22 = aLin.r1[1];
    float A31 = aLin.r2[0], A32 = aLin.r2[1], A33 = aLin.r2[2];
    float A41 = aCross.r0[0], A42 = aCross.r0[1], A43 = aCross.r0[2], A44 = aAng.r0[0];
    float A51 = aCross.r1[0], A52 = aCross.r1[1], A53 = aCross.r1[2], A54 = aAng.r1[0], A55 = aAng.r1[1];
    float A61 = aCross.r2[0], A62 = aCross.r2[1], A63 = aCross.r2[2], A64 = aAng.r2[0], A65 = aAng.r2[1], A66 = aAng.r2[2];

    if (abs(A11) < 1e-12f) A11 = 1e-12f;

    float L21 = A21 / A11;
    float L31 = A31 / A11;
    float L41 = A41 / A11;
    float L51 = A51 / A11;
    float L61 = A61 / A11;

    float D1 = A11;
    float D2 = A22 - L21 * L21 * D1;
    if (abs(D2) < 1e-12f) D2 = 1e-12f;

    float L32 = (A32 - L21 * L31 * D1) / D2;
    float L42 = (A42 - L21 * L41 * D1) / D2;
    float L52 = (A52 - L21 * L51 * D1) / D2;
    float L62 = (A62 - L21 * L61 * D1) / D2;

    float D3 = A33 - (L31 * L31 * D1 + L32 * L32 * D2);
    if (abs(D3) < 1e-12f) D3 = 1e-12f;

    float L43 = (A43 - L31 * L41 * D1 - L32 * L42 * D2) / D3;
    float L53 = (A53 - L31 * L51 * D1 - L32 * L52 * D2) / D3;
    float L63 = (A63 - L31 * L61 * D1 - L32 * L62 * D2) / D3;

    float D4 = A44 - (L41 * L41 * D1 + L42 * L42 * D2 + L43 * L43 * D3);
    if (abs(D4) < 1e-12f) D4 = 1e-12f;

    float L54 = (A54 - L41 * L51 * D1 - L42 * L52 * D2 - L43 * L53 * D3) / D4;
    float L64 = (A64 - L41 * L61 * D1 - L42 * L62 * D2 - L43 * L63 * D3) / D4;

    float D5 = A55 - (L51 * L51 * D1 + L52 * L52 * D2 + L53 * L53 * D3 + L54 * L54 * D4);
    if (abs(D5) < 1e-12f) D5 = 1e-12f;

    float L65 = (A65 - L51 * L61 * D1 - L52 * L62 * D2 - L53 * L63 * D3 - L54 * L64 * D4) / D5;

    float D6 = A66 - (L61 * L61 * D1 + L62 * L62 * D2 + L63 * L63 * D3 + L64 * L64 * D4 + L65 * L65 * D5);
    if (abs(D6) < 1e-12f) D6 = 1e-12f;

    float y1 = bLin[0];
    float y2 = bLin[1] - L21 * y1;
    float y3 = bLin[2] - L31 * y1 - L32 * y2;
    float y4 = bAng[0] - L41 * y1 - L42 * y2 - L43 * y3;
    float y5 = bAng[1] - L51 * y1 - L52 * y2 - L53 * y3 - L54 * y4;
    float y6 = bAng[2] - L61 * y1 - L62 * y2 - L63 * y3 - L64 * y4 - L65 * y5;

    float z1 = y1 / D1, z2 = y2 / D2, z3 = y3 / D3;
    float z4 = y4 / D4, z5 = y5 / D5, z6 = y6 / D6;

    xAng[2] = z6;
    xAng[1] = z5 - L65 * xAng[2];
    xAng[0] = z4 - L54 * xAng[1] - L64 * xAng[2];
    xLin[2] = z3 - L43 * xAng[0] - L53 * xAng[1] - L63 * xAng[2];
    xLin[1] = z2 - L32 * xLin[2] - L42 * xAng[0] - L52 * xAng[1] - L62 * xAng[2];
    xLin[0] = z1 - L21 * xLin[1] - L31 * xLin[2] - L41 * xAng[0] - L51 * xAng[1] - L61 * xAng[2];
}

// ─────────────────────────────────────────────────────────────
// GPU broadphase and adjacency rebuild
// ─────────────────────────────────────────────────────────────
kernel void avbd_reset_adjacency(
    device AVBDGPUAdjacency *adjacency [[buffer(0)]],
    device AVBDGPUContactAllocator *contactAllocator [[buffer(1)]],
    device AVBDGPURecentPairCacheState *nextRecentPairState [[buffer(2)]],
    device AVBDGPUActiveManifoldListState *activeManifoldState [[buffer(3)]],
    device AVBDGPUDerivedPairCandidateState *derivedPairState [[buffer(4)]],
    constant AVBDGPUSolverParams &params [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid == 0) {
        atomic_store_explicit(&contactAllocator[0].nextContactIndex, 0, memory_order_relaxed);
        atomic_store_explicit(&nextRecentPairState[0].count, 0, memory_order_relaxed);
        atomic_store_explicit(&activeManifoldState[0].count, 0, memory_order_relaxed);
        atomic_store_explicit(&derivedPairState[0].count, 0, memory_order_relaxed);
    }

    if (tid >= uint(params.bodyCount)) return;

    device AVBDGPUAdjacency &adj = adjacency[tid];
    atomic_store_explicit(&adj.jointCount, 0, memory_order_relaxed);
    atomic_store_explicit(&adj.springCount, 0, memory_order_relaxed);
    atomic_store_explicit(&adj.manifoldCount, 0, memory_order_relaxed);
}


static inline void compact_broadphase_batch(
    bool shouldCachePair,
    int encodedPair,
    bool shouldProcessDerivedPair,
    device int *nextRecentPairIndices,
    device AVBDGPURecentPairCacheState *nextRecentPairState,
    device int *derivedPairIndices,
    device AVBDGPUDerivedPairCandidateState *derivedPairState,
    uint lid,
    uint simd_size);

kernel void avbd_broadphase_full(
    device AVBDGPUBody *bodies [[buffer(0)]],
    device AVBDGPUCollisionExclusion *exclusions [[buffer(1)]],
    device int *nextRecentPairIndices [[buffer(2)]],
    device AVBDGPURecentPairCacheState *nextRecentPairState [[buffer(3)]],
    device int *derivedPairIndices [[buffer(4)]],
    device AVBDGPUDerivedPairCandidateState *derivedPairState [[buffer(5)]],
    constant AVBDGPUSolverParams &params [[buffer(6)]],
    uint lid [[thread_index_in_threadgroup]],
    uint tid [[thread_position_in_grid]],
    uint simd_size [[threads_per_simdgroup]])
{

    int n = params.bodyCount;
    int totalPairs = n * (n - 1) / 2;
    bool validThread = int(tid) < totalPairs;
    bool shouldCachePair = false;
    bool shouldProcessDerivedPair = false;
    int encodedPair = -1;
    int bodyA = -1, bodyB = -1;

    if (validThread) {
        upper_triangle_pair(tid, n, bodyA, bodyB);

        if (bodyA >= 0 && bodyB >= 0 && bodyA < n && bodyB < n
            && !(bodies[bodyA].mass <= 0.0f && bodies[bodyB].mass <= 0.0f)
            && !is_excluded(exclusions, bodyA, bodyB))
        {
            float radiusA = body_radius(bodies[bodyA], params);
            float radiusB = body_radius(bodies[bodyB], params);
            float3 dp = bodies[bodyA].positionLin - bodies[bodyB].positionLin;
            float radius = radiusA + radiusB;
            float relativeSpeed = length(bodies[bodyA].velocityLin - bodies[bodyB].velocityLin);
            float gravityExpansion = 0.5f * abs(params.gravity) * params.cacheTimeHorizon * params.cacheTimeHorizon;
            float predictiveMargin = params.cacheMargin + relativeSpeed * params.cacheTimeHorizon + gravityExpansion;
            float cacheRadius = radius + predictiveMargin;
            float distSq = length_squared(dp);

            if (distSq <= cacheRadius * cacheRadius) {
                shouldCachePair = true;
                encodedPair = (bodyA << 16) | bodyB;

                if (distSq <= radius * radius) {
                    shouldProcessDerivedPair = true;
                }
            }
        }
    }

    compact_broadphase_batch(
        shouldCachePair, encodedPair, shouldProcessDerivedPair,
        nextRecentPairIndices, nextRecentPairState,
        derivedPairIndices, derivedPairState,
        lid, simd_size);
}

kernel void avbd_broadphase_partial(
    device AVBDGPUBody *bodies [[buffer(0)]],
    device int *currentRecentPairIndices [[buffer(1)]],
    device AVBDGPURecentPairCacheState *currentRecentPairState [[buffer(2)]],
    device int *nextRecentPairIndices [[buffer(3)]],
    device AVBDGPURecentPairCacheState *nextRecentPairState [[buffer(4)]],
    device int *derivedPairIndices [[buffer(5)]],
    device AVBDGPUDerivedPairCandidateState *derivedPairState [[buffer(6)]],
    constant AVBDGPUSolverParams &params [[buffer(7)]],
    uint lid [[thread_index_in_threadgroup]],
    uint tid [[thread_position_in_grid]],
    uint simd_size [[threads_per_simdgroup]])
{

    int currentRecentPairCount = min(atomic_load_explicit(&currentRecentPairState[0].count, memory_order_relaxed), currentRecentPairState[0].capacity);
    bool validThread = tid < uint(currentRecentPairCount);
    bool shouldCachePair = false;
    bool shouldProcessDerivedPair = false;
    int encodedPair = -1;
    int bodyA = -1, bodyB = -1;

    if (validThread) {
        encodedPair = currentRecentPairIndices[tid];
        bodyA = encodedPair >> 16;
        bodyB = encodedPair & 0xFFFF;

        if (bodyA >= 0 && bodyB >= 0 && bodyA < params.bodyCount && bodyB < params.bodyCount) {
            float radiusA = body_radius(bodies[bodyA], params);
            float radiusB = body_radius(bodies[bodyB], params);
            float3 dp = bodies[bodyA].positionLin - bodies[bodyB].positionLin;
            float radius = radiusA + radiusB;
            float relativeSpeed = length(bodies[bodyA].velocityLin - bodies[bodyB].velocityLin);
            float gravityExpansion = 0.5f * abs(params.gravity) * params.cacheTimeHorizon * params.cacheTimeHorizon;
            float predictiveMargin = params.cacheMargin + relativeSpeed * params.cacheTimeHorizon + gravityExpansion;
            float cacheRadius = radius + predictiveMargin;
            float distSq = length_squared(dp);

            if (distSq <= cacheRadius * cacheRadius) {
                shouldCachePair = true;

                if (distSq <= radius * radius) {
                    shouldProcessDerivedPair = true;
                }
            }
        }
    }

    compact_broadphase_batch(
        shouldCachePair, encodedPair, shouldProcessDerivedPair,
        nextRecentPairIndices, nextRecentPairState,
        derivedPairIndices, derivedPairState,
        lid, simd_size);
}

static inline void compact_broadphase_batch(
    bool shouldCachePair,
    int encodedPair,
    bool shouldProcessDerivedPair,
    device int *nextRecentPairIndices,
    device AVBDGPURecentPairCacheState *nextRecentPairState,
    device int *derivedPairIndices,
    device AVBDGPUDerivedPairCandidateState *derivedPairState,
    uint lid,
    uint simd_size)
{
    // Cached pair prefix sum
    int cachedPairLocalRank = simd_prefix_exclusive_sum(shouldCachePair ? 1 : 0);
    int cachedPairTotalCount = simd_broadcast(cachedPairLocalRank + (shouldCachePair ? 1 : 0), simd_size - 1);
    
    int cachedPairBaseIndex = 0;
    int cachedPairGrantedCount = 0;
    if (lid == 0 && cachedPairTotalCount > 0) {
        cachedPairGrantedCount = reserve_recent_pair_range(nextRecentPairState[0], cachedPairTotalCount, cachedPairBaseIndex);
    }
    cachedPairBaseIndex = simd_broadcast(cachedPairBaseIndex, 0);
    cachedPairGrantedCount = simd_broadcast(cachedPairGrantedCount, 0);

    // Derived pair prefix sum
    int derivedPairLocalRank = simd_prefix_exclusive_sum(shouldProcessDerivedPair ? 1 : 0);
    int derivedPairTotalCount = simd_broadcast(derivedPairLocalRank + (shouldProcessDerivedPair ? 1 : 0), simd_size - 1);

    int derivedPairBaseIndex = 0;
    int derivedPairGrantedCount = 0;
    if (lid == 0 && derivedPairTotalCount > 0) {
        derivedPairGrantedCount = reserve_derived_pair_range(derivedPairState[0], derivedPairTotalCount, derivedPairBaseIndex);
    }
    derivedPairBaseIndex = simd_broadcast(derivedPairBaseIndex, 0);
    derivedPairGrantedCount = simd_broadcast(derivedPairGrantedCount, 0);

    // Write-out
    if (shouldCachePair && cachedPairLocalRank < cachedPairGrantedCount) {
        nextRecentPairIndices[cachedPairBaseIndex + cachedPairLocalRank] = encodedPair;
    }
    if (shouldProcessDerivedPair && derivedPairLocalRank < derivedPairGrantedCount) {
        derivedPairIndices[derivedPairBaseIndex + derivedPairLocalRank] = encodedPair;
    }
}

kernel void avbd_build_primitive_manifolds(
    device AVBDGPUBody *bodies [[buffer(0)]],
    device int *derivedPairIndices [[buffer(1)]],
    device AVBDGPUDerivedPairCandidateState *derivedPairState [[buffer(2)]],
    device AVBDGPUManifold *manifolds [[buffer(3)]],
    device AVBDGPUContact *allContacts [[buffer(4)]],
    device AVBDGPUContactAllocator *contactAllocator [[buffer(5)]],
    device int *activeManifoldIndices [[buffer(6)]],
    device AVBDGPUActiveManifoldListState *activeManifoldState [[buffer(7)]],
    constant AVBDGPUSolverParams &params [[buffer(8)]],
    uint lid [[thread_index_in_threadgroup]],
    uint tid [[thread_position_in_grid]],
    uint simd_size [[threads_per_simdgroup]])
{
    int derivedPairCount = min(atomic_load_explicit(&derivedPairState[0].count, memory_order_relaxed), derivedPairState[0].capacity);
    bool validThread = tid < uint(derivedPairCount);

    // --- Collision detection (no atomics) ---
    int manifoldSlot = -1;
    AVBDGPUContactBuilderMetal builder;
    builder.count = 0;
    Mat3 basis = Mat3::identity();
    int bodyIdxA = -1, bodyIdxB = -1;

    if (validThread) {
        manifoldSlot = int(tid);
        int encodedPair = derivedPairIndices[tid];
        bodyIdxA = encodedPair >> 16;
        bodyIdxB = encodedPair & 0xFFFF;

        device AVBDGPUBody &bodyA = bodies[bodyIdxA];
        device AVBDGPUBody &bodyB = bodies[bodyIdxB];
        device AVBDGPUManifold &manifold = manifolds[manifoldSlot];

        manifold.bodyA = bodyIdxA;
        manifold.bodyB = bodyIdxB;
        manifold.contactCount = 0;
        manifold.contactBaseIndex = -1;
        manifold.active = 0;
        manifold.friction = sqrt(bodyA.friction * bodyB.friction);
        manifold.basisR0 = float3(1, 0, 0);
        manifold.basisR1 = float3(0, 1, 0);
        manifold.basisR2 = float3(0, 0, 1);

        float3 dp = bodyA.positionLin - bodyB.positionLin;
        float radiusA = body_radius(bodyA, params);
        float radiusB = body_radius(bodyB, params);
        float radius = radiusA + radiusB;
        float distSq = length_squared(dp);

        if (distSq <= radius * radius) {
            collide_bodies_gpu(bodyA, bodyB, builder, basis, params, params.collisionMargin);
        }
    }

    // --- Batch 2: Reserve contacts (SIMD optimized) ---
    int myContactOffset = simd_prefix_exclusive_sum(builder.count);
    int totalContacts = simd_broadcast(myContactOffset + builder.count, simd_size - 1);

    int contactBatchBaseIndex = 0;
    int contactBatchGrantedCount = 0;
    if (lid == 0 && totalContacts > 0) {
        contactBatchGrantedCount = reserve_contact_range(contactAllocator[0], totalContacts, contactBatchBaseIndex);
    }
    contactBatchBaseIndex = simd_broadcast(contactBatchBaseIndex, 0);
    contactBatchGrantedCount = simd_broadcast(contactBatchGrantedCount, 0);

    // --- Write manifold + contacts ---
    bool shouldEmitActiveManifold = false;
    if (manifoldSlot >= 0 && builder.count > 0) {
        int myGrantedContacts = min(builder.count, max(contactBatchGrantedCount - myContactOffset, 0));

        if (myGrantedContacts > 0) {
            device AVBDGPUManifold &manifold = manifolds[manifoldSlot];
            manifold.basisR0 = basis.r0;
            manifold.basisR1 = basis.r1;
            manifold.basisR2 = basis.r2;
            manifold.contactBaseIndex = contactBatchBaseIndex + myContactOffset;
            manifold.contactCount = myGrantedContacts;
            manifold.active = 1;
            shouldEmitActiveManifold = true;
            for (int i = 0; i < myGrantedContacts; i++) {
                allContacts[manifold.contactBaseIndex + i] = builder.contacts[i];
            }
        }
    }

    // --- Batch 3: Reserve active manifold slots (SIMD optimized) ---
    int myActiveRank = simd_prefix_exclusive_sum(shouldEmitActiveManifold ? 1 : 0);
    int totalActive = simd_broadcast(myActiveRank + (shouldEmitActiveManifold ? 1 : 0), simd_size - 1);

    int activeManifoldBaseIndex = 0;
    int activeManifoldGrantedCount = 0;
    if (lid == 0 && totalActive > 0) {
        activeManifoldGrantedCount = reserve_active_manifold_range(activeManifoldState[0], totalActive, activeManifoldBaseIndex);
    }
    activeManifoldBaseIndex = simd_broadcast(activeManifoldBaseIndex, 0);
    activeManifoldGrantedCount = simd_broadcast(activeManifoldGrantedCount, 0);

    if (shouldEmitActiveManifold && myActiveRank < activeManifoldGrantedCount) {
        activeManifoldIndices[activeManifoldBaseIndex + myActiveRank] = manifoldSlot;
    }
}

kernel void avbd_prepare_broadphase_indirect(
    device AVBDGPURecentPairCacheState *nextRecentPairState [[buffer(0)]],
    device AVBDGPUIndirectDispatchArgs *broadphaseIndirectArgs [[buffer(1)]],
    device AVBDGPUDerivedPairCandidateState *derivedPairState [[buffer(2)]],
    device AVBDGPUIndirectDispatchArgs *derivedPairIndirectArgs [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint simd_size [[threads_per_simdgroup]])
{
    if (tid != 0) return;

    int pairCount = min(atomic_load_explicit(&nextRecentPairState[0].count, memory_order_relaxed), nextRecentPairState[0].capacity);
    uint threadgroupsX = uint(max(pairCount, 0) + simd_size - 1) / simd_size;
    broadphaseIndirectArgs[0].threadgroupsPerGrid[0] = threadgroupsX;
    broadphaseIndirectArgs[0].threadgroupsPerGrid[1] = 1;
    broadphaseIndirectArgs[0].threadgroupsPerGrid[2] = 1;

    int derivedPairCount = min(atomic_load_explicit(&derivedPairState[0].count, memory_order_relaxed), derivedPairState[0].capacity);
    uint derivedTgX = uint(max(derivedPairCount, 0) + simd_size - 1) / simd_size;
    derivedPairIndirectArgs[0].threadgroupsPerGrid[0] = derivedTgX;
    derivedPairIndirectArgs[0].threadgroupsPerGrid[1] = 1;
    derivedPairIndirectArgs[0].threadgroupsPerGrid[2] = 1;
}

kernel void avbd_prepare_active_manifolds_indirect(
    device AVBDGPUActiveManifoldListState *activeManifoldState [[buffer(0)]],
    device AVBDGPUIndirectDispatchArgs *indirectArgs [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint simd_size [[threads_per_simdgroup]])
{
    if (tid != 0) return;

    int manifoldCount = min(atomic_load_explicit(&activeManifoldState[0].count, memory_order_relaxed), activeManifoldState[0].capacity);
    uint threadgroupsX = uint(max(manifoldCount, 0) + simd_size - 1) / simd_size;
    indirectArgs[0].threadgroupsPerGrid[0] = threadgroupsX;
    indirectArgs[0].threadgroupsPerGrid[1] = 1;
    indirectArgs[0].threadgroupsPerGrid[2] = 1;
}

kernel void avbd_initialize_mesh_sdf(
    texture3d<float, access::write> sdfTexture [[texture(0)]],
    uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= sdfTexture.get_width() || gid.y >= sdfTexture.get_height() || gid.z >= sdfTexture.get_depth()) {
        return;
    }
    sdfTexture.write(FLT_MAX, gid);
}

kernel void avbd_accumulate_mesh_sdf(
    device const float3 *vertices [[buffer(0)]],
    device const uint *indices [[buffer(1)]],
    texture3d<float, access::read_write> sdfTexture [[texture(0)]],
    device uint *insideCounts [[buffer(2)]],
    constant uint &triangleOffset [[buffer(3)]],
    constant uint &triangleChunkCount [[buffer(4)]],
    constant float3 &sdfOrigin [[buffer(5)]],
    constant float3 &voxelSize [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= sdfTexture.get_width() || gid.y >= sdfTexture.get_height() || gid.z >= sdfTexture.get_depth()) {
        return;
    }

    uint width = sdfTexture.get_width();
    uint height = sdfTexture.get_height();
    uint linearIndex = gid.x + gid.y * width + gid.z * width * height;
    float3 localPoint = sdfOrigin + (float3(gid) + 0.5f) * voxelSize;

    float minDistance = sdfTexture.read(gid).r;
    uint insideCount = insideCounts[linearIndex];
    for (uint triangleIndex = triangleOffset; triangleIndex < triangleOffset + triangleChunkCount; ++triangleIndex) {
        uint indexBase = triangleIndex * 3u;
        float3 v0 = vertices[indices[indexBase + 0u]];
        float3 v1 = vertices[indices[indexBase + 1u]];
        float3 v2 = vertices[indices[indexBase + 2u]];

        minDistance = min(minDistance, point_triangle_unsigned_distance(localPoint, v0, v1, v2));

        float hitT = 0.0f;
        float3 hitNormal = float3(0.0f);
        if (ray_triangle_intersection(localPoint, AVBD_GPU_SDF_RAY_DIRECTION, FLT_MAX, v0, v1, v2, hitT, hitNormal)) {
            insideCount += 1u;
        }
    }

    sdfTexture.write(minDistance, gid);
    insideCounts[linearIndex] = insideCount;
}

kernel void avbd_finalize_mesh_sdf(
    texture3d<float, access::read_write> sdfTexture [[texture(0)]],
    device const uint *insideCounts [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= sdfTexture.get_width() || gid.y >= sdfTexture.get_height() || gid.z >= sdfTexture.get_depth()) {
        return;
    }

    uint width = sdfTexture.get_width();
    uint height = sdfTexture.get_height();
    uint linearIndex = gid.x + gid.y * width + gid.z * width * height;
    float sdfValue = sdfTexture.read(gid).r;
    if (!isfinite(sdfValue)) {
        sdfValue = FLT_MAX;
    }
    if ((insideCounts[linearIndex] & 1u) != 0u) {
        sdfValue = -sdfValue;
    }
    sdfTexture.write(sdfValue, gid);
}

kernel void avbd_broadphase_primitive_mesh(
    device const AVBDGPUBody *bodies [[buffer(0)]],
    device const AVBDGPUCollisionMeshInfo *meshInfos [[buffer(1)]],
    device AVBDGPUPrimitiveMeshPair *candidatePairs [[buffer(2)]],
    device AVBDGPUPrimitiveMeshPairListState *candidateState [[buffer(3)]],
    device const int *meshOwnerBodyMask [[buffer(4)]],
    constant AVBDGPUSolverParams &params [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    int bodyCount = params.bodyCount;
    int meshCount = params.meshCount;
    int pairCount = bodyCount * meshCount;
    if (tid >= uint(pairCount) || bodyCount <= 0 || meshCount <= 0) return;

    int bodyIndex = int(tid) / meshCount;
    int meshIndex = int(tid) - bodyIndex * meshCount;

    if (meshOwnerBodyMask[bodyIndex] != 0) {
        return;
    }

    if (meshInfos[meshIndex].ownerBodyIndex == bodyIndex) {
        return;
    }

    float3 primitiveMin;
    float3 primitiveMax;
    make_body_aabb(bodies[bodyIndex], params, primitiveMin, primitiveMax);

    float3 meshMin = meshInfos[meshIndex].minBounds.xyz;
    float3 meshMax = meshInfos[meshIndex].maxBounds.xyz;
    if (!aabb_overlaps(primitiveMin, primitiveMax, meshMin, meshMax)) {
        return;
    }

    int baseIndex = 0;
    if (reserve_primitive_mesh_pair_range(candidateState[0], 1, baseIndex) <= 0) {
        return;
    }

    candidatePairs[baseIndex].bodyIndex = bodyIndex;
    candidatePairs[baseIndex].meshIndex = meshIndex;
}

kernel void avbd_prepare_primitive_mesh_collision_indirect(
    device AVBDGPUPrimitiveMeshPairListState *candidateState [[buffer(0)]],
    device AVBDGPUIndirectDispatchArgs *indirectArgs [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint simd_size [[threads_per_simdgroup]])
{
    if (tid != 0) return;

    int pairCount = min(atomic_load_explicit(&candidateState[0].count, memory_order_relaxed), candidateState[0].capacity);
    uint threadgroupsX = uint(max(pairCount, 0) + simd_size - 1) / simd_size;
    indirectArgs[0].threadgroupsPerGrid[0] = threadgroupsX;
    indirectArgs[0].threadgroupsPerGrid[1] = 1;
    indirectArgs[0].threadgroupsPerGrid[2] = 1;
}

kernel void avbd_broadphase_mesh_mesh(
    device const AVBDGPUCollisionMeshInfo *meshInfos [[buffer(0)]],
    device AVBDGPUMeshMeshPair *candidatePairs [[buffer(1)]],
    device AVBDGPUMeshMeshPairListState *candidateState [[buffer(2)]],
    constant AVBDGPUSolverParams &params [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    int meshCount = params.meshCount;
    int pairCount = meshCount * (meshCount - 1) / 2;
    if (tid >= uint(pairCount) || meshCount <= 1) return;

    int meshIndexA = -1;
    int meshIndexB = -1;
    upper_triangle_pair(tid, meshCount, meshIndexA, meshIndexB);
    if (meshIndexA < 0 || meshIndexB < 0) return;

    int ownerBodyIndexA = meshInfos[meshIndexA].ownerBodyIndex;
    int ownerBodyIndexB = meshInfos[meshIndexB].ownerBodyIndex;
    if ((ownerBodyIndexA >= 0 && ownerBodyIndexA == ownerBodyIndexB) ||
        (ownerBodyIndexA < 0 && ownerBodyIndexB < 0)) {
        return;
    }

    float expandA = params.cacheMargin + params.collisionMargin + max_component(meshInfos[meshIndexA].sdfVoxelSize.xyz);
    float expandB = params.cacheMargin + params.collisionMargin + max_component(meshInfos[meshIndexB].sdfVoxelSize.xyz);
    float3 minA = meshInfos[meshIndexA].minBounds.xyz - float3(expandA);
    float3 maxA = meshInfos[meshIndexA].maxBounds.xyz + float3(expandA);
    float3 minB = meshInfos[meshIndexB].minBounds.xyz - float3(expandB);
    float3 maxB = meshInfos[meshIndexB].maxBounds.xyz + float3(expandB);
    if (!aabb_overlaps(minA, maxA, minB, maxB)) {
        return;
    }

    int baseIndex = 0;
    if (reserve_mesh_mesh_pair_range(candidateState[0], 1, baseIndex) <= 0) {
        return;
    }

    candidatePairs[baseIndex].meshIndexA = meshIndexA;
    candidatePairs[baseIndex].meshIndexB = meshIndexB;
}

kernel void avbd_prepare_mesh_mesh_collision_indirect(
    device AVBDGPUMeshMeshPairListState *candidateState [[buffer(0)]],
    device AVBDGPUIndirectDispatchArgs *indirectArgs [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint simd_size [[threads_per_simdgroup]])
{
    if (tid != 0) return;

    int pairCount = min(atomic_load_explicit(&candidateState[0].count, memory_order_relaxed), candidateState[0].capacity);
    uint threadgroupsX = uint(max(pairCount, 0) + simd_size - 1) / simd_size;
    indirectArgs[0].threadgroupsPerGrid[0] = threadgroupsX;
    indirectArgs[0].threadgroupsPerGrid[1] = 1;
    indirectArgs[0].threadgroupsPerGrid[2] = 1;
}

kernel void avbd_collide_mesh_mesh(
    device const AVBDGPUBody *bodies [[buffer(0)]],
    device const AVBDGPUCollisionMeshInfo *meshInfos [[buffer(1)]],
    device const AVBDGPUMeshMeshPair *candidatePairs [[buffer(2)]],
    device AVBDGPUMeshMeshPairListState *candidateState [[buffer(3)]],
    constant AVBDGPUSolverParams &params [[buffer(4)]],
    device AVBDGPUManifold *manifolds [[buffer(5)]],
    device AVBDGPUContact *allContacts [[buffer(6)]],
    device AVBDGPUContactAllocator *contactAllocator [[buffer(7)]],
    device int *activeManifoldIndices [[buffer(8)]],
    device AVBDGPUActiveManifoldListState *activeManifoldState [[buffer(9)]],
    constant AVBDCollisionMeshSDFSet &meshSDFSet [[buffer(10)]],
    device AVBDGPUMeshMeshIsoVoxelDebug *isoVoxelDebugEntries [[buffer(11)]],
    device AVBDGPUMeshMeshIsoVoxelCoord *isoVoxelCoords [[buffer(12)]],
    uint lid [[thread_index_in_threadgroup]],
    uint tid [[thread_position_in_grid]],
    uint simd_size [[threads_per_simdgroup]])
{
    int candidateCount = min(atomic_load_explicit(&candidateState[0].count, memory_order_relaxed), candidateState[0].capacity);
    bool trackIsoVoxelDebug = tid < uint(max(params.meshMeshIsoVoxelTrackedPairCapacity, 0));
    if (trackIsoVoxelDebug) {
        isoVoxelDebugEntries[tid].meshIndexA = -1;
        isoVoxelDebugEntries[tid].meshIndexB = -1;
        isoVoxelDebugEntries[tid].driverMeshIndex = -1;
        isoVoxelDebugEntries[tid].sampledVoxelCount = 0;
        isoVoxelDebugEntries[tid].candidateVoxelCount = 0;
        isoVoxelDebugEntries[tid].compactedVoxelCount = 0;
        isoVoxelDebugEntries[tid].sampleStride = 0;
        isoVoxelDebugEntries[tid].overflowed = 0;
        isoVoxelDebugEntries[tid].valid = 0;
    }
    bool validThread = tid < uint(candidateCount);

    int manifoldSlot = -1;
    AVBDGPUContactBuilderMetal builder;
    builder.count = 0;
    Mat3 basis = Mat3::identity();

    if (validThread) {
        const device AVBDGPUMeshMeshPair &pair = candidatePairs[tid];
        if (trackIsoVoxelDebug) {
            isoVoxelDebugEntries[tid].meshIndexA = pair.meshIndexA;
            isoVoxelDebugEntries[tid].meshIndexB = pair.meshIndexB;
        }
        if (pair.meshIndexA < 0 || pair.meshIndexA >= params.meshCount ||
            pair.meshIndexB < 0 || pair.meshIndexB >= params.meshCount) {
            validThread = false;
        }
    }

    if (validThread) {
        const device AVBDGPUMeshMeshPair &pair = candidatePairs[tid];
        const device AVBDGPUCollisionMeshInfo &meshInfoA = meshInfos[pair.meshIndexA];
        const device AVBDGPUCollisionMeshInfo &meshInfoB = meshInfos[pair.meshIndexB];

        float3 overlapMin = max(meshInfoA.minBounds.xyz, meshInfoB.minBounds.xyz);
        float3 overlapMax = min(meshInfoA.maxBounds.xyz, meshInfoB.maxBounds.xyz);
        if (any(overlapMin >= overlapMax)) {
            validThread = false;
        }

        int ownerBodyIndexA = meshInfoA.ownerBodyIndex;
        int ownerBodyIndexB = meshInfoB.ownerBodyIndex;
        if ((ownerBodyIndexA >= 0 && ownerBodyIndexA == ownerBodyIndexB) ||
            (ownerBodyIndexA < 0 && ownerBodyIndexB < 0)) {
            validThread = false;
        }
        if ((ownerBodyIndexA >= params.bodyCount) || (ownerBodyIndexB >= params.bodyCount)) {
            validThread = false;
        }

        bool swapBodyOrder = ownerBodyIndexA < 0 && ownerBodyIndexB >= 0;
        bool bodyAIsDynamicMesh = !swapBodyOrder;
        int dynamicBodyIndex = bodyAIsDynamicMesh ? ownerBodyIndexA : ownerBodyIndexB;
        int otherBodyIndex = bodyAIsDynamicMesh ? ownerBodyIndexB : ownerBodyIndexA;
        int staticMeshIndex = bodyAIsDynamicMesh ? pair.meshIndexB : pair.meshIndexA;

        manifoldSlot = params.meshMeshManifoldOffset + int(tid);
        device AVBDGPUManifold &manifold = manifolds[manifoldSlot];
        manifold.bodyA = dynamicBodyIndex;
        manifold.bodyB = otherBodyIndex >= 0 ? otherBodyIndex : -(staticMeshIndex + 1);
        manifold.contactCount = 0;
        manifold.contactBaseIndex = -1;
        manifold.active = 0;
        manifold.friction = 0.5f;
        manifold.basisR0 = float3(1, 0, 0);
        manifold.basisR1 = float3(0, 1, 0);
        manifold.basisR2 = float3(0, 0, 1);

        if (validThread && dynamicBodyIndex >= 0) {
            if (otherBodyIndex >= 0) {
                manifold.friction = 0.5f * (bodies[dynamicBodyIndex].friction + bodies[otherBodyIndex].friction);
            } else {
                manifold.friction = bodies[dynamicBodyIndex].friction;
            }
        } else {
            validThread = false;
        }

        if (validThread) {
            // ── Iso voxel-driven single-pass contact generation ──
            int maxIsoSamples = max(8, min(params.meshMeshMaxIsoVoxelSamples, 4096));
            bool reduceContacts = params.meshMeshReduceContacts != 0;
            float voxelBand = max(max_component(meshInfoA.sdfVoxelSize.xyz), max_component(meshInfoB.sdfVoxelSize.xyz));
            float bandWidth = max(params.collisionMargin * 2.0f + voxelBand, voxelBand * 2.5f);

            AVBDGPUCollisionMeshVoxelRangeMetal cellRangeA;
            AVBDGPUCollisionMeshVoxelRangeMetal cellRangeB;
            bool hasCellRangeA = world_aabb_to_collision_mesh_voxel_cell_range(meshInfoA, overlapMin, overlapMax, 1, cellRangeA);
            bool hasCellRangeB = world_aabb_to_collision_mesh_voxel_cell_range(meshInfoB, overlapMin, overlapMax, 1, cellRangeB);

            bool usedIsoVoxelPath = false;

            if (hasCellRangeA && hasCellRangeB) {
                int cellCountA = collision_mesh_voxel_range_sample_count(cellRangeA);
                int cellCountB = collision_mesh_voxel_range_sample_count(cellRangeB);
                bool driveFromA = cellCountA <= cellCountB;
                AVBDGPUCollisionMeshVoxelRangeMetal driverRange = driveFromA ? cellRangeA : cellRangeB;
                int3 driverCounts = collision_mesh_voxel_range_counts(driverRange);
                int sampleStride = hydroelastic_voxel_iteration_step(driverRange, maxIsoSamples);
                int sampleCountVX = max((driverCounts.x + sampleStride - 1) / sampleStride, 1);
                int sampleCountVY = max((driverCounts.y + sampleStride - 1) / sampleStride, 1);
                int sampleCountVZ = max((driverCounts.z + sampleStride - 1) / sampleStride, 1);

                // Debug tracking
                int sampledVoxelCount = 0;
                int candidateVoxelCount = 0;
                int compactedVoxelCount = 0;
                int debugOverflowed = 0;
                int coordBaseOffset = trackIsoVoxelDebug ? int(tid) * params.meshMeshIsoVoxelCoordsPerPair : 0;

                // Single-pass: use overlap geometry for stable basis, bin into quadrants
                float bestOverallScore = -FLT_MAX;
                float3 bestOverallNormal = float3(0.0f, 1.0f, 0.0f);
                float3 basisCenter = 0.5f * (overlapMin + overlapMax);

                // Build stable basis from overlap AABB: shortest axis ~ contact normal
                {
                    float3 overlapExtent = overlapMax - overlapMin;
                    int shortAxis = 0;
                    if (overlapExtent.y < overlapExtent.x && overlapExtent.y < overlapExtent.z) shortAxis = 1;
                    else if (overlapExtent.z < overlapExtent.x) shortAxis = 2;
                    float3 axisDir = float3(shortAxis == 0 ? 1.0f : 0.0f,
                                            shortAxis == 1 ? 1.0f : 0.0f,
                                            shortAxis == 2 ? 1.0f : 0.0f);
                    basis = orthonormal_basis(axisDir);
                }

                float quadScore[AVBD_MAX_CONTACTS_PER_PAIR];
                float3 quadXA[AVBD_MAX_CONTACTS_PER_PAIR];
                float3 quadXB[AVBD_MAX_CONTACTS_PER_PAIR];
                for (int qi = 0; qi < AVBD_MAX_CONTACTS_PER_PAIR; ++qi) {
                    quadScore[qi] = -FLT_MAX;
                }

                for (int vz = 0; vz < sampleCountVZ; ++vz) {
                    for (int vy = 0; vy < sampleCountVY; ++vy) {
                        for (int vx = 0; vx < sampleCountVX; ++vx) {
                            int3 voxelOffset = int3(
                                min(vx * sampleStride, driverCounts.x - 1),
                                min(vy * sampleStride, driverCounts.y - 1),
                                min(vz * sampleStride, driverCounts.z - 1)
                            );
                            int3 voxelCoord = driverRange.minCoord + voxelOffset;
                            sampledVoxelCount += 1;

                            float3 worldPoint;
                            if (!evaluate_iso_voxel_and_centroid(
                                    meshSDFSet,
                                    driveFromA ? meshInfoA : meshInfoB,
                                    driveFromA ? meshInfoB : meshInfoA,
                                    driverRange,
                                    voxelCoord,
                                    driveFromA,
                                    bandWidth,
                                    params,
                                    worldPoint)) {
                                continue;
                            }

                            candidateVoxelCount += 1;

                            if (trackIsoVoxelDebug) {
                                if (compactedVoxelCount < params.meshMeshIsoVoxelCoordsPerPair) {
                                    isoVoxelCoords[coordBaseOffset + compactedVoxelCount].voxelCoord = int4(voxelCoord, 0);
                                    compactedVoxelCount += 1;
                                } else {
                                    debugOverflowed = 1;
                                }
                            }

                            float3 xA, xB, normal, midpoint;
                            float separation, score;
                            if (!evaluate_mesh_mesh_hydroelastic_sample(
                                    meshSDFSet, meshInfoA, meshInfoB, worldPoint,
                                    swapBodyOrder, bandWidth, params,
                                    xA, xB, normal, midpoint, separation, score)) {
                                continue;
                            }

                            if (score > bestOverallScore) {
                                bestOverallScore = score;
                                bestOverallNormal = normal;
                            }

                            if (reduceContacts) {
                                // Quadrant binning
                                float2 tangentOffset = float2(
                                    dot(basis.r1, midpoint - basisCenter),
                                    dot(basis.r2, midpoint - basisCenter)
                                );
                                int quadrant = (tangentOffset.x >= 0.0f ? 1 : 0) | (tangentOffset.y >= 0.0f ? 2 : 0);
                                float quadrantScore = score + max(-separation, 0.0f) * 0.5f;
                                if (quadrantScore > quadScore[quadrant]) {
                                    quadScore[quadrant] = quadrantScore;
                                    quadXA[quadrant] = xA;
                                    quadXB[quadrant] = xB;
                                }
                            } else {
                                // No reduction: emit directly (up to contact limit)
                                if (builder.count < AVBD_MAX_CONTACTS_PER_PAIR) {
                                    if (otherBodyIndex >= 0) {
                                        append_primitive_mesh_contact(
                                            builder, bodies[dynamicBodyIndex], bodies[otherBodyIndex],
                                            xA, xB,
                                            primitive_mesh_feature_key(pair.meshIndexA ^ pair.meshIndexB, builder.count, int(tid))
                                        );
                                    } else {
                                        append_mesh_contact(
                                            builder, bodies[dynamicBodyIndex], xA, xB,
                                            primitive_mesh_feature_key(staticMeshIndex, builder.count, int(tid))
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                // Write debug info
                if (trackIsoVoxelDebug) {
                    isoVoxelDebugEntries[tid].driverMeshIndex = driveFromA ? pair.meshIndexA : pair.meshIndexB;
                    isoVoxelDebugEntries[tid].sampledVoxelCount = sampledVoxelCount;
                    isoVoxelDebugEntries[tid].candidateVoxelCount = candidateVoxelCount;
                    isoVoxelDebugEntries[tid].compactedVoxelCount = compactedVoxelCount;
                    isoVoxelDebugEntries[tid].sampleStride = sampleStride;
                    isoVoxelDebugEntries[tid].overflowed = debugOverflowed;
                    isoVoxelDebugEntries[tid].valid = 1;
                }

                // Emit reduced contacts
                if (reduceContacts && bestOverallScore > -FLT_MAX) {
                    usedIsoVoxelPath = true;
                    // Use best contact normal for the solver basis
                    basis = orthonormal_basis(bestOverallNormal);
                    if (otherBodyIndex >= 0) {
                        for (int quadrant = 0; quadrant < AVBD_MAX_CONTACTS_PER_PAIR; ++quadrant) {
                            if (quadScore[quadrant] <= -FLT_MAX) continue;
                            append_primitive_mesh_contact(
                                builder, bodies[dynamicBodyIndex], bodies[otherBodyIndex],
                                quadXA[quadrant], quadXB[quadrant],
                                primitive_mesh_feature_key(pair.meshIndexA ^ pair.meshIndexB, quadrant, int(tid))
                            );
                        }
                    } else {
                        for (int quadrant = 0; quadrant < AVBD_MAX_CONTACTS_PER_PAIR; ++quadrant) {
                            if (quadScore[quadrant] <= -FLT_MAX) continue;
                            append_mesh_contact(
                                builder, bodies[dynamicBodyIndex],
                                quadXA[quadrant], quadXB[quadrant],
                                primitive_mesh_feature_key(staticMeshIndex, quadrant, int(tid))
                            );
                        }
                    }
                } else if (!reduceContacts && builder.count > 0) {
                    usedIsoVoxelPath = true;
                }
            }

            // Fallback: grid sampling when iso voxel path produces nothing
            if (!usedIsoVoxelPath) {
                float3 overlapExtent = overlapMax - overlapMin;
                int sampleCountX = mesh_mesh_sample_axis_count(overlapExtent.x, bandWidth);
                int sampleCountY = mesh_mesh_sample_axis_count(overlapExtent.y, bandWidth);
                int sampleCountZ = mesh_mesh_sample_axis_count(overlapExtent.z, bandWidth);
                float3 sampleCountF = float3(float(sampleCountX), float(sampleCountY), float(sampleCountZ));

                float bestScore = -FLT_MAX;
                float3 bestNormal = float3(0.0f, 1.0f, 0.0f);
                float3 bestMidpoint = 0.5f * (overlapMin + overlapMax);
                float bestSeparation = FLT_MAX;

                for (int z = 0; z < sampleCountZ; ++z) {
                    for (int y = 0; y < sampleCountY; ++y) {
                        for (int x = 0; x < sampleCountX; ++x) {
                            float3 uv = (float3(float(x), float(y), float(z)) + 0.5f) / sampleCountF;
                            float3 worldPoint = mix(overlapMin, overlapMax, uv);
                            float3 xA, xB, normal, midpoint;
                            float separation, score;
                            if (!evaluate_mesh_mesh_hydroelastic_sample(
                                    meshSDFSet, meshInfoA, meshInfoB, worldPoint,
                                    swapBodyOrder, bandWidth, params,
                                    xA, xB, normal, midpoint, separation, score)) {
                                continue;
                            }
                            if (score > bestScore || (score == bestScore && separation < bestSeparation)) {
                                bestScore = score;
                                bestNormal = normal;
                                bestMidpoint = midpoint;
                                bestSeparation = separation;
                            }
                        }
                    }
                }

                if (bestScore > -FLT_MAX) {
                    basis = orthonormal_basis(bestNormal);
                    float fbQuadrantScore[AVBD_MAX_CONTACTS_PER_PAIR];
                    float3 fbQuadrantXA[AVBD_MAX_CONTACTS_PER_PAIR];
                    float3 fbQuadrantXB[AVBD_MAX_CONTACTS_PER_PAIR];
                    for (int qi = 0; qi < AVBD_MAX_CONTACTS_PER_PAIR; ++qi) {
                        fbQuadrantScore[qi] = -FLT_MAX;
                        fbQuadrantXA[qi] = float3(0.0f);
                        fbQuadrantXB[qi] = float3(0.0f);
                    }
                    for (int z = 0; z < sampleCountZ; ++z) {
                        for (int y = 0; y < sampleCountY; ++y) {
                            for (int x = 0; x < sampleCountX; ++x) {
                                float3 uv = (float3(float(x), float(y), float(z)) + 0.5f) / sampleCountF;
                                float3 worldPoint = mix(overlapMin, overlapMax, uv);
                                float3 xA, xB, normal, midpoint;
                                float separation, score;
                                if (!evaluate_mesh_mesh_hydroelastic_sample(
                                        meshSDFSet, meshInfoA, meshInfoB, worldPoint,
                                        swapBodyOrder, bandWidth, params,
                                        xA, xB, normal, midpoint, separation, score)) {
                                    continue;
                                }
                                float2 tangentOffset = float2(
                                    dot(basis.r1, midpoint - bestMidpoint),
                                    dot(basis.r2, midpoint - bestMidpoint)
                                );
                                int quadrant = (tangentOffset.x >= 0.0f ? 1 : 0) | (tangentOffset.y >= 0.0f ? 2 : 0);
                                float quadrantScore = score + max(-separation, 0.0f) * 0.5f;
                                if (quadrantScore > fbQuadrantScore[quadrant]) {
                                    fbQuadrantScore[quadrant] = quadrantScore;
                                    fbQuadrantXA[quadrant] = xA;
                                    fbQuadrantXB[quadrant] = xB;
                                }
                            }
                        }
                    }
                    if (otherBodyIndex >= 0) {
                        for (int quadrant = 0; quadrant < AVBD_MAX_CONTACTS_PER_PAIR; ++quadrant) {
                            if (fbQuadrantScore[quadrant] <= -FLT_MAX) continue;
                            append_primitive_mesh_contact(
                                builder, bodies[dynamicBodyIndex], bodies[otherBodyIndex],
                                fbQuadrantXA[quadrant], fbQuadrantXB[quadrant],
                                primitive_mesh_feature_key(pair.meshIndexA ^ pair.meshIndexB, quadrant, int(tid))
                            );
                        }
                    } else {
                        for (int quadrant = 0; quadrant < AVBD_MAX_CONTACTS_PER_PAIR; ++quadrant) {
                            if (fbQuadrantScore[quadrant] <= -FLT_MAX) continue;
                            append_mesh_contact(
                                builder, bodies[dynamicBodyIndex],
                                fbQuadrantXA[quadrant], fbQuadrantXB[quadrant],
                                primitive_mesh_feature_key(staticMeshIndex, quadrant, int(tid))
                            );
                        }
                    }
                }
            }

            // Write final debug counters
            if (trackIsoVoxelDebug) {
                isoVoxelDebugEntries[tid].emittedContactCount = builder.count;
                isoVoxelDebugEntries[tid].usedIsoVoxelPath = usedIsoVoxelPath ? 1 : 0;
            }

            if (builder.count > 0) {
                if (otherBodyIndex >= 0) {
                    finalize_builder_contacts(
                        bodies[dynamicBodyIndex],
                        bodies[otherBodyIndex],
                        builder,
                        basis,
                        params.collisionMargin
                    );
                    manifold.bodyB = otherBodyIndex;
                    manifold.friction = 0.5f * (bodies[dynamicBodyIndex].friction + bodies[otherBodyIndex].friction);
                } else {
                    finalize_builder_mesh_contacts(
                        bodies[dynamicBodyIndex],
                        builder,
                        basis,
                        params.collisionMargin
                    );
                    manifold.bodyB = -(staticMeshIndex + 1);
                    manifold.friction = bodies[dynamicBodyIndex].friction;
                }
                manifold.basisR0 = basis.r0;
                manifold.basisR1 = basis.r1;
                manifold.basisR2 = basis.r2;
            }
        }
    }

    int myContactOffset = simd_prefix_exclusive_sum(builder.count);
    int totalContacts = simd_broadcast(myContactOffset + builder.count, simd_size - 1);

    int contactBatchBaseIndex = 0;
    int contactBatchGrantedCount = 0;
    if (lid == 0 && totalContacts > 0) {
        contactBatchGrantedCount = reserve_contact_range(contactAllocator[0], totalContacts, contactBatchBaseIndex);
    }
    contactBatchBaseIndex = simd_broadcast(contactBatchBaseIndex, 0);
    contactBatchGrantedCount = simd_broadcast(contactBatchGrantedCount, 0);

    bool shouldEmitActiveManifold = false;
    if (validThread && manifoldSlot >= 0 && builder.count > 0) {
        int myGrantedContacts = min(builder.count, max(contactBatchGrantedCount - myContactOffset, 0));
        if (myGrantedContacts > 0) {
            device AVBDGPUManifold &manifold = manifolds[manifoldSlot];
            manifold.contactBaseIndex = contactBatchBaseIndex + myContactOffset;
            manifold.contactCount = myGrantedContacts;
            manifold.active = 1;
            shouldEmitActiveManifold = true;
            for (int i = 0; i < myGrantedContacts; ++i) {
                allContacts[manifold.contactBaseIndex + i] = builder.contacts[i];
            }
        }
    }

    int myActiveRank = simd_prefix_exclusive_sum(shouldEmitActiveManifold ? 1 : 0);
    int totalActive = simd_broadcast(myActiveRank + (shouldEmitActiveManifold ? 1 : 0), simd_size - 1);

    int activeManifoldBaseIndex = 0;
    int activeManifoldGrantedCount = 0;
    if (lid == 0 && totalActive > 0) {
        activeManifoldGrantedCount = reserve_active_manifold_range(activeManifoldState[0], totalActive, activeManifoldBaseIndex);
    }
    activeManifoldBaseIndex = simd_broadcast(activeManifoldBaseIndex, 0);
    activeManifoldGrantedCount = simd_broadcast(activeManifoldGrantedCount, 0);

    if (shouldEmitActiveManifold && myActiveRank < activeManifoldGrantedCount) {
        activeManifoldIndices[activeManifoldBaseIndex + myActiveRank] = manifoldSlot;
    }
}

kernel void avbd_collide_primitive_mesh(
    device AVBDGPUBody *bodies [[buffer(0)]],
    device const AVBDGPUCollisionMeshInfo *meshInfos [[buffer(1)]],
    device const AVBDGPUPrimitiveMeshPair *candidatePairs [[buffer(2)]],
    device AVBDGPUPrimitiveMeshPairListState *candidateState [[buffer(3)]],
    constant AVBDGPUSolverParams &params [[buffer(4)]],
    device const float3 *meshVertices [[buffer(5)]],
    device const uint *meshIndices [[buffer(6)]],
    device AVBDGPUManifold *manifolds [[buffer(7)]],
    device AVBDGPUContact *allContacts [[buffer(8)]],
    device AVBDGPUContactAllocator *contactAllocator [[buffer(9)]],
    device int *activeManifoldIndices [[buffer(10)]],
    device AVBDGPUActiveManifoldListState *activeManifoldState [[buffer(11)]],
    uint lid [[thread_index_in_threadgroup]],
    uint tid [[thread_position_in_grid]],
    uint simd_size [[threads_per_simdgroup]])
{
    int candidateCount = min(atomic_load_explicit(&candidateState[0].count, memory_order_relaxed), candidateState[0].capacity);
    bool validThread = tid < uint(candidateCount);

    int manifoldSlot = -1;
    int bodyIndex = -1;
    int bestMeshIndex = -1;
    float bestSignedDistance = FLT_MAX;
    AVBDGPUContactBuilderMetal builder;
    builder.count = 0;
    Mat3 basis = Mat3::identity();

    if (validThread) {
        const device AVBDGPUPrimitiveMeshPair &candidate = candidatePairs[tid];
        bodyIndex = candidate.bodyIndex;
        if (bodyIndex < 0 || bodyIndex >= params.bodyCount) {
            validThread = false;
        }
    }

    if (validThread) {
        const device AVBDGPUPrimitiveMeshPair &candidate = candidatePairs[tid];
        manifoldSlot = params.primitiveMeshManifoldOffset + int(tid);
        device AVBDGPUBody &body = bodies[bodyIndex];
        int meshOwnerBodyIndex = meshInfos[candidate.meshIndex].ownerBodyIndex;
        device AVBDGPUManifold &manifold = manifolds[manifoldSlot];
        manifold.bodyA = bodyIndex;
        manifold.bodyB = meshOwnerBodyIndex >= 0 ? meshOwnerBodyIndex : -(candidate.meshIndex + 1);
        manifold.contactCount = 0;
        manifold.contactBaseIndex = -1;
        manifold.active = 0;
        manifold.friction = meshOwnerBodyIndex >= 0 ? 0.5f * (body.friction + bodies[meshOwnerBodyIndex].friction) : body.friction;
        manifold.basisR0 = float3(1, 0, 0);
        manifold.basisR1 = float3(0, 1, 0);
        manifold.basisR2 = float3(0, 0, 1);

        if (body.mass > 0.0f) {
            float3 center = body.positionLin;

            const device AVBDGPUCollisionMeshInfo &meshInfo = meshInfos[candidate.meshIndex];
            if (candidate.meshIndex >= 0 && candidate.meshIndex < params.meshCount) {
                if (meshInfo.indexCount < 3 || meshInfo.vertexCount < 3) {
                    validThread = false;
                }
            } else {
                validThread = false;
            }

            if (validThread) {
                float3 meshCenter = 0.5f * (meshInfo.minBounds.xyz + meshInfo.maxBounds.xyz);
                float3 samplePoints[AVBD_MAX_CONTACTS_PER_PAIR];
                int sampleSeeds[AVBD_MAX_CONTACTS_PER_PAIR];
                int sampleCount = 0;
                build_primitive_mesh_sample_points(body, meshCenter, params, samplePoints, sampleSeeds, sampleCount);
                float bodyExtent = body_radius(body, params);
                float maxDistance = length(meshInfo.maxBounds.xyz - meshInfo.minBounds.xyz)
                    + bodyExtent * 2.0f
                    + params.collisionMargin
                    + 1.0e-2f;

                for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
                    float3 primitivePoint = samplePoints[sampleIndex];
                    float3 castDirection = safe_normalize(meshCenter - primitivePoint, safe_normalize(meshCenter - center, float3(0.0f, 1.0f, 0.0f)));
                    float3 rayOrigin = primitivePoint - castDirection * max(1.0e-3f, params.collisionMargin + 1.0e-3f);

                    bool foundHit = false;
                    float bestTriangleT = FLT_MAX;
                    float3 bestTriangleHit = float3(0.0f);
                    float3 bestTriangleNormal = float3(0.0f, 1.0f, 0.0f);
                    int bestTriangleIndex = -1;

                    int triangleCount = meshInfo.indexCount / 3;
                    for (int triangleIndex = 0; triangleIndex < triangleCount; ++triangleIndex) {
                        int indexBase = meshInfo.indexOffset + triangleIndex * 3;
                        uint i0 = meshIndices[indexBase + 0];
                        uint i1 = meshIndices[indexBase + 1];
                        uint i2 = meshIndices[indexBase + 2];
                        float3 v0 = (meshInfo.sdfTransform * float4(meshVertices[i0], 1.0f)).xyz;
                        float3 v1 = (meshInfo.sdfTransform * float4(meshVertices[i1], 1.0f)).xyz;
                        float3 v2 = (meshInfo.sdfTransform * float4(meshVertices[i2], 1.0f)).xyz;
                        float hitT;
                        float3 hitNormal;
                        if (!ray_triangle_intersection(rayOrigin,
                                                       castDirection,
                                                       maxDistance,
                                                       v0,
                                                       v1,
                                                       v2,
                                                       hitT,
                                                       hitNormal)) {
                            continue;
                        }

                        if (!foundHit || hitT < bestTriangleT) {
                            foundHit = true;
                            bestTriangleT = hitT;
                            bestTriangleHit = rayOrigin + castDirection * hitT;
                            bestTriangleNormal = hitNormal;
                            bestTriangleIndex = triangleIndex;
                        }
                    }

                    if (!foundHit) {
                        continue;
                    }

                    float3 outwardNormal = bestTriangleNormal;
                    if (dot(outwardNormal, primitivePoint - bestTriangleHit) < 0.0f) {
                        outwardNormal = -outwardNormal;
                    }

                    float signedDistance = dot(primitivePoint - bestTriangleHit, outwardNormal);
                    if (signedDistance >= params.collisionMargin) {
                        continue;
                    }

                    append_mesh_contact(
                        builder,
                        body,
                        primitivePoint,
                        bestTriangleHit,
                        primitive_mesh_feature_key(candidate.meshIndex, bestTriangleIndex, sampleSeeds[sampleIndex])
                    );

                    if (signedDistance < bestSignedDistance) {
                        bestSignedDistance = signedDistance;
                        bestMeshIndex = candidate.meshIndex;
                        basis = orthonormal_basis(outwardNormal);
                    }
                }
            }
        }

        if (bestMeshIndex >= 0 && builder.count > 0) {
            finalize_builder_mesh_contacts(body, builder, basis, params.collisionMargin);
            manifold.bodyB = -(bestMeshIndex + 1);
            manifold.friction = body.friction;
            manifold.basisR0 = basis.r0;
            manifold.basisR1 = basis.r1;
            manifold.basisR2 = basis.r2;
        }
    }

    int myContactOffset = simd_prefix_exclusive_sum(builder.count);
    int totalContacts = simd_broadcast(myContactOffset + builder.count, simd_size - 1);

    int contactBatchBaseIndex = 0;
    int contactBatchGrantedCount = 0;
    if (lid == 0 && totalContacts > 0) {
        contactBatchGrantedCount = reserve_contact_range(contactAllocator[0], totalContacts, contactBatchBaseIndex);
    }
    contactBatchBaseIndex = simd_broadcast(contactBatchBaseIndex, 0);
    contactBatchGrantedCount = simd_broadcast(contactBatchGrantedCount, 0);

    bool shouldEmitActiveManifold = false;
    if (validThread && manifoldSlot >= 0 && builder.count > 0) {
        int myGrantedContacts = min(builder.count, max(contactBatchGrantedCount - myContactOffset, 0));

        if (myGrantedContacts > 0) {
            device AVBDGPUManifold &manifold = manifolds[manifoldSlot];
            manifold.contactBaseIndex = contactBatchBaseIndex + myContactOffset;
            manifold.contactCount = myGrantedContacts;
            manifold.active = 1;
            shouldEmitActiveManifold = true;
            for (int i = 0; i < myGrantedContacts; i++) {
                allContacts[manifold.contactBaseIndex + i] = builder.contacts[i];
            }
        }
    }

    int myActiveRank = simd_prefix_exclusive_sum(shouldEmitActiveManifold ? 1 : 0);
    int totalActive = simd_broadcast(myActiveRank + (shouldEmitActiveManifold ? 1 : 0), simd_size - 1);

    int activeManifoldBaseIndex = 0;
    int activeManifoldGrantedCount = 0;
    if (lid == 0 && totalActive > 0) {
        activeManifoldGrantedCount = reserve_active_manifold_range(activeManifoldState[0], totalActive, activeManifoldBaseIndex);
    }
    activeManifoldBaseIndex = simd_broadcast(activeManifoldBaseIndex, 0);
    activeManifoldGrantedCount = simd_broadcast(activeManifoldGrantedCount, 0);

    if (shouldEmitActiveManifold && myActiveRank < activeManifoldGrantedCount) {
        activeManifoldIndices[activeManifoldBaseIndex + myActiveRank] = manifoldSlot;
    }
}

kernel void avbd_collide_primitive_mesh_sdf(
    device AVBDGPUBody *bodies [[buffer(0)]],
    device const AVBDGPUCollisionMeshInfo *meshInfos [[buffer(1)]],
    device const AVBDGPUPrimitiveMeshPair *candidatePairs [[buffer(2)]],
    device AVBDGPUPrimitiveMeshPairListState *candidateState [[buffer(3)]],
    constant AVBDGPUSolverParams &params [[buffer(4)]],
    device const float3 *meshVertices [[buffer(5)]],
    device const uint *meshIndices [[buffer(6)]],
    device AVBDGPUManifold *manifolds [[buffer(7)]],
    device AVBDGPUContact *allContacts [[buffer(8)]],
    device AVBDGPUContactAllocator *contactAllocator [[buffer(9)]],
    device int *activeManifoldIndices [[buffer(10)]],
    device AVBDGPUActiveManifoldListState *activeManifoldState [[buffer(11)]],
    constant AVBDCollisionMeshSDFSet &meshSDFSet [[buffer(12)]],
    uint lid [[thread_index_in_threadgroup]],
    uint tid [[thread_position_in_grid]],
    uint simd_size [[threads_per_simdgroup]])
{
    int candidateCount = min(atomic_load_explicit(&candidateState[0].count, memory_order_relaxed), candidateState[0].capacity);
    bool validThread = tid < uint(candidateCount);

    int manifoldSlot = -1;
    int bodyIndex = -1;
    int bestMeshIndex = -1;
    float bestSignedDistance = FLT_MAX;
    AVBDGPUContactBuilderMetal builder;
    builder.count = 0;
    Mat3 basis = Mat3::identity();

    if (validThread) {
        const device AVBDGPUPrimitiveMeshPair &candidate = candidatePairs[tid];
        bodyIndex = candidate.bodyIndex;
        if (bodyIndex < 0 || bodyIndex >= params.bodyCount) {
            validThread = false;
        }
    }

    if (validThread) {
        const device AVBDGPUPrimitiveMeshPair &candidate = candidatePairs[tid];
        manifoldSlot = params.primitiveMeshManifoldOffset + int(tid);
        device AVBDGPUBody &body = bodies[bodyIndex];
        int meshOwnerBodyIndex = meshInfos[candidate.meshIndex].ownerBodyIndex;
        device AVBDGPUManifold &manifold = manifolds[manifoldSlot];
        manifold.bodyA = bodyIndex;
        manifold.bodyB = meshOwnerBodyIndex >= 0 ? meshOwnerBodyIndex : -(candidate.meshIndex + 1);
        manifold.contactCount = 0;
        manifold.contactBaseIndex = -1;
        manifold.active = 0;
        manifold.friction = meshOwnerBodyIndex >= 0 ? 0.5f * (body.friction + bodies[meshOwnerBodyIndex].friction) : body.friction;
        manifold.basisR0 = float3(1, 0, 0);
        manifold.basisR1 = float3(0, 1, 0);
        manifold.basisR2 = float3(0, 0, 1);

        if (body.mass > 0.0f) {
            const device AVBDGPUCollisionMeshInfo &candidateMeshInfo = meshInfos[candidate.meshIndex];
            if (candidate.meshIndex >= 0 && candidate.meshIndex < params.meshCount) {
                if (candidateMeshInfo.indexCount < 3 || candidateMeshInfo.vertexCount < 3) {
                    validThread = false;
                }
            } else {
                validThread = false;
            }

            if (validThread) {
                float3 meshCenter = 0.5f * (candidateMeshInfo.minBounds.xyz + candidateMeshInfo.maxBounds.xyz);
                float3 samplePoints[AVBD_MAX_CONTACTS_PER_PAIR];
                int sampleSeeds[AVBD_MAX_CONTACTS_PER_PAIR];
                int sampleCount = 0;
                build_primitive_mesh_sample_points(body, meshCenter, params, samplePoints, sampleSeeds, sampleCount);
                int sdfResourceIndex = candidateMeshInfo.sdfResourceIndex;
                if (sdfResourceIndex < 0 || sdfResourceIndex >= AVBD_MAX_COLLISION_MESH_SDFS) {
                    validThread = false;
                } else {
                    for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
                        float3 primitivePoint = samplePoints[sampleIndex];
                        float4 worldPoint4 = candidateMeshInfo.sdfInvTransform * float4(primitivePoint, 1.0f);
                        float3 pointLocal = worldPoint4.xyz;
                        if (!point_in_collision_mesh_sdf_bounds(pointLocal, candidateMeshInfo)) {
                            continue;
                        }

                        float signedDistance = sample_collision_mesh_sdf_value(meshSDFSet, sdfResourceIndex, pointLocal, candidateMeshInfo);
                        if (!isfinite(signedDistance) || signedDistance >= params.collisionMargin) {
                            continue;
                        }

                        float3 localNormal = collision_mesh_sdf_local_normal(meshSDFSet, sdfResourceIndex, pointLocal, candidateMeshInfo);
                        float3 outwardNormal = collision_mesh_sdf_world_normal(localNormal, candidateMeshInfo);
                        float3 localSurfacePoint = pointLocal - localNormal * signedDistance;
                        float4 worldSurfacePoint4 = candidateMeshInfo.sdfTransform * float4(localSurfacePoint, 1.0f);
                        float3 bestSurfacePoint = worldSurfacePoint4.xyz;

                        if (meshOwnerBodyIndex >= 0) {
                            append_primitive_mesh_contact(
                                builder,
                                body,
                                bodies[meshOwnerBodyIndex],
                                primitivePoint,
                                bestSurfacePoint,
                                primitive_mesh_feature_key(candidate.meshIndex, sampleSeeds[sampleIndex], 0)
                            );
                        } else {
                            append_mesh_contact(
                                builder,
                                body,
                                primitivePoint,
                                bestSurfacePoint,
                                primitive_mesh_feature_key(candidate.meshIndex, sampleSeeds[sampleIndex], 0)
                            );
                        }

                        if (signedDistance < bestSignedDistance) {
                            bestSignedDistance = signedDistance;
                            bestMeshIndex = candidate.meshIndex;
                            basis = orthonormal_basis(outwardNormal);
                        }
                    }
                }
            }
        }

        if (bestMeshIndex >= 0 && builder.count > 0) {
            if (meshOwnerBodyIndex >= 0) {
                finalize_builder_contacts(body, bodies[meshOwnerBodyIndex], builder, basis, params.collisionMargin);
                manifold.bodyB = meshOwnerBodyIndex;
                manifold.friction = 0.5f * (body.friction + bodies[meshOwnerBodyIndex].friction);
            } else {
                finalize_builder_mesh_contacts(body, builder, basis, params.collisionMargin);
                manifold.bodyB = -(bestMeshIndex + 1);
                manifold.friction = body.friction;
            }
            manifold.basisR0 = basis.r0;
            manifold.basisR1 = basis.r1;
            manifold.basisR2 = basis.r2;
        }
    }

    int myContactOffset = simd_prefix_exclusive_sum(builder.count);
    int totalContacts = simd_broadcast(myContactOffset + builder.count, simd_size - 1);

    int contactBatchBaseIndex = 0;
    int contactBatchGrantedCount = 0;
    if (lid == 0 && totalContacts > 0) {
        contactBatchGrantedCount = reserve_contact_range(contactAllocator[0], totalContacts, contactBatchBaseIndex);
    }
    contactBatchBaseIndex = simd_broadcast(contactBatchBaseIndex, 0);
    contactBatchGrantedCount = simd_broadcast(contactBatchGrantedCount, 0);

    bool shouldEmitActiveManifold = false;
    if (validThread && manifoldSlot >= 0 && builder.count > 0) {
        int myGrantedContacts = min(builder.count, max(contactBatchGrantedCount - myContactOffset, 0));

        if (myGrantedContacts > 0) {
            device AVBDGPUManifold &manifold = manifolds[manifoldSlot];
            manifold.contactBaseIndex = contactBatchBaseIndex + myContactOffset;
            manifold.contactCount = myGrantedContacts;
            manifold.active = 1;
            shouldEmitActiveManifold = true;
            for (int i = 0; i < myGrantedContacts; i++) {
                allContacts[manifold.contactBaseIndex + i] = builder.contacts[i];
            }
        }
    }

    int myActiveRank = simd_prefix_exclusive_sum(shouldEmitActiveManifold ? 1 : 0);
    int totalActive = simd_broadcast(myActiveRank + (shouldEmitActiveManifold ? 1 : 0), simd_size - 1);

    int activeManifoldBaseIndex = 0;
    int activeManifoldGrantedCount = 0;
    if (lid == 0 && totalActive > 0) {
        activeManifoldGrantedCount = reserve_active_manifold_range(activeManifoldState[0], totalActive, activeManifoldBaseIndex);
    }
    activeManifoldBaseIndex = simd_broadcast(activeManifoldBaseIndex, 0);
    activeManifoldGrantedCount = simd_broadcast(activeManifoldGrantedCount, 0);

    if (shouldEmitActiveManifold && myActiveRank < activeManifoldGrantedCount) {
        activeManifoldIndices[activeManifoldBaseIndex + myActiveRank] = manifoldSlot;
    }
}

kernel void avbd_build_adjacency_constraints(
    device AVBDGPUJoint *joints [[buffer(0)]],
    device AVBDGPUSpring *springs [[buffer(1)]],
    device AVBDGPUAdjacency *adjacency [[buffer(2)]],
    constant AVBDGPUSolverParams &params [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < uint(params.jointCount)) {
        device AVBDGPUJoint &j = joints[tid];
        if (!j.broken) {
            add_joint_adjacency(adjacency, j.bodyA, int(tid));
            add_joint_adjacency(adjacency, j.bodyB, int(tid));
        }
    }

    if (tid < uint(params.springCount)) {
        device AVBDGPUSpring &s = springs[tid];
        add_spring_adjacency(adjacency, s.bodyA, int(tid));
        add_spring_adjacency(adjacency, s.bodyB, int(tid));
    }
}

kernel void avbd_build_adjacency_manifolds(
    device AVBDGPUManifold *manifolds [[buffer(0)]],
    device int *activeManifoldIndices [[buffer(1)]],
    device AVBDGPUActiveManifoldListState *activeManifoldState [[buffer(2)]],
    device AVBDGPUAdjacency *adjacency [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    int activeManifoldCount = min(atomic_load_explicit(&activeManifoldState[0].count, memory_order_relaxed), activeManifoldState[0].capacity);
    if (tid >= uint(activeManifoldCount)) return;

    int manifoldIndex = activeManifoldIndices[tid];
    device AVBDGPUManifold &m = manifolds[manifoldIndex];
    add_manifold_adjacency(adjacency, m.bodyA, manifoldIndex);
    if (m.bodyB >= 0) {
        add_manifold_adjacency(adjacency, m.bodyB, manifoldIndex);
    }
}

// ─────────────────────────────────────────────────────────────
// KERNEL 1: Forward Integration
//   Sets inertial targets and initial positions, then
//   forward-integrates dynamic bodies.
// ─────────────────────────────────────────────────────────────
kernel void avbd_forward_integrate(
    device AVBDGPUBody *bodies [[buffer(0)]],
    constant AVBDGPUSolverParams &params [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= uint(params.bodyCount)) return;

    device AVBDGPUBody &b = bodies[tid];
    float dt = params.dt;
    float gravity = params.gravity;

    // Compute inertial target
    b.inertialLin = b.positionLin + b.velocityLin * dt;
    if (b.mass > 0.0f) {
        b.inertialLin += float3(0, 0, gravity) * (dt * dt);
    }
    b.inertialAng = quat_add_angular(b.positionAng, b.velocityAng * dt);

    // Acceleration-based weight for position prediction
    float3 accel = (b.velocityLin - b.prevVelocityLin) / dt;
    float accelExt = accel.z * sign_f(gravity);
    float accelWeight = 0.0f;
    if (abs(gravity) > 1e-6f) {
        accelWeight = clamp(accelExt / abs(gravity), 0.0f, 1.0f);
    }
    // ensure accelWeight is not NaN
    if (!(accelWeight >= 0.0f)) accelWeight = 0.0f;

    // Snapshot initial state
    b.initialLin = b.positionLin;
    b.initialAng = b.positionAng;

    // Forward integrate dynamic bodies
    if (b.mass > 0.0f) {
        b.positionLin = b.positionLin + b.velocityLin * dt + float3(0, 0, gravity) * (accelWeight * dt * dt);
        b.positionAng = quat_add_angular(b.positionAng, b.velocityAng * dt);
    }
}

// ─────────────────────────────────────────────────────────────
// KERNEL 2: Initialize Joints
//   Computes C0, decays penalty & lambda for warm-start.
// ─────────────────────────────────────────────────────────────
kernel void avbd_init_joints(
    device AVBDGPUBody *bodies [[buffer(0)]],
    device AVBDGPUJoint *joints [[buffer(1)]],
    constant AVBDGPUSolverParams &params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= uint(params.jointCount)) return;

    device AVBDGPUJoint &j = joints[tid];
    if (j.broken) return;

    float3 posA, rA_local = j.rA;
    float4 quatA;
    if (j.bodyA >= 0) {
        posA = bodies[j.bodyA].initialLin;
        quatA = bodies[j.bodyA].initialAng;
    } else {
        posA = float3(0);
        quatA = float4(0, 0, 0, 1);
    }

    float3 posB = bodies[j.bodyB].initialLin;
    float4 quatB = bodies[j.bodyB].initialAng;

    float3 pA = xform(posA, quatA, rA_local);
    float3 pB = xform(posB, quatB, j.rB);
    j.C0Lin = pA - pB;

    float4 quatAForAng = (j.bodyA >= 0) ? quatA : float4(0, 0, 0, 1);
    j.C0Ang = quat_delta(quatAForAng, quatB) * j.torqueArm;

    j.lambdaLin *= params.alpha * params.gamma;
    j.lambdaAng *= params.alpha * params.gamma;
    j.penaltyLin = clamp_vec(j.penaltyLin * params.gamma, params.penaltyMin, params.penaltyMax);
    j.penaltyAng = clamp_vec(j.penaltyAng * params.gamma, params.penaltyMin, params.penaltyMax);
    j.penaltyLin = min_vec(j.penaltyLin, j.stiffnessLin);
    j.penaltyAng = min_vec(j.penaltyAng, j.stiffnessAng);
}

// ─────────────────────────────────────────────────────────────
// KERNEL 3: Body Solve (Jacobi iteration)
//   For each dynamic body: accumulate forces from joints,
//   springs, and manifolds, then solve 6×6 and update pose.
// ─────────────────────────────────────────────────────────────
kernel void avbd_body_solve(
    device AVBDGPUBody *bodies [[buffer(0)]],
    device AVBDGPUJoint *joints [[buffer(1)]],
    device AVBDGPUSpring *springs [[buffer(2)]],
    device AVBDGPUManifold *manifolds [[buffer(3)]],
    device AVBDGPUAdjacency *adjacency [[buffer(4)]],
    device AVBDGPUContact *allContacts [[buffer(5)]],
    constant AVBDGPUSolverParams &params [[buffer(6)]],
    device const int *bodyColors [[buffer(7)]],
    constant int &currentColor [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= uint(params.bodyCount)) return;
    if (bodyColors[tid] != currentColor) return;

    device AVBDGPUBody &body = bodies[tid];
    if (body.mass <= 0.0f) return;

    float dt = params.dt;
    float alpha = params.alpha;
    float dt2 = dt * dt;

    Mat3 mLin = Mat3::diag(body.mass, body.mass, body.mass);
    Mat3 mAng = Mat3::diag3(body.moment);

    Mat3 lhsLin = mLin / dt2;
    Mat3 lhsAng = mAng / dt2;
    Mat3 lhsCross;
    float3 rhsLin = (mLin / dt2).mul(body.positionLin - body.inertialLin);
    float3 rhsAng = (mAng / dt2).mul(quat_delta(body.positionAng, body.inertialAng));

    device AVBDGPUAdjacency &adj = adjacency[tid];

    // ── Accumulate joint forces ──
    int jointAdjCount = min(atomic_load_explicit(&adj.jointCount, memory_order_relaxed), AVBD_MAX_CONSTRAINTS_PER_BODY);
    for (int ji = 0; ji < jointAdjCount; ji++) {
        device AVBDGPUJoint &j = joints[adj.jointIndices[ji]];
        if (j.broken) continue;

        bool isA = (j.bodyA == int(tid));

        // Linear penalty
        if (length_squared(j.penaltyLin) > 0.0f) {
            Mat3 K = Mat3::diag3(j.penaltyLin);

            float3 posAFull, rA_local = j.rA;
            float4 quatAFull;
            if (j.bodyA >= 0) {
                posAFull = bodies[j.bodyA].positionLin;
                quatAFull = bodies[j.bodyA].positionAng;
            } else {
                posAFull = float3(0);
                quatAFull = float4(0, 0, 0, 1);
            }

            float3 pA = xform(posAFull, quatAFull, rA_local);
            float3 pB = xform(bodies[j.bodyB].positionLin, bodies[j.bodyB].positionAng, j.rB);
            float3 C = pA - pB;

            if (j.stiffnessLin >= 1e30f) {
                C -= j.C0Lin * alpha;
            }

            float3 F = K.mul(C) + j.lambdaLin;

            Mat3 jLin = isA ? Mat3::identity() : Mat3::identity() * (-1.0f);
            float3 rWorld;
            Mat3 jAng;
            if (isA) {
                rWorld = quat_act(quatAFull, rA_local);
                jAng = skew(-rWorld);
            } else {
                rWorld = quat_act(bodies[j.bodyB].positionAng, j.rB);
                jAng = skew(rWorld);
            }

            Mat3 jLinT = mat3_transpose(jLin);
            Mat3 jAngT = mat3_transpose(jAng);
            Mat3 jAngTk = mat3_mul(jAngT, K);

            lhsLin = lhsLin + mat3_mul(mat3_mul(jLinT, K), jLin);
            lhsAng = lhsAng + mat3_mul(jAngTk, jAng);
            lhsCross = lhsCross + mat3_mul(jAngTk, jLin);

            float3 r = isA ? quat_act(quatAFull, rA_local) : -quat_act(bodies[j.bodyB].positionAng, j.rB);
            Mat3 H = geom_stiffness_bs(0, r) * F[0]
                   + geom_stiffness_bs(1, r) * F[1]
                   + geom_stiffness_bs(2, r) * F[2];
            lhsAng = lhsAng + diagonalize_mat(H);

            rhsLin = rhsLin + jLinT.mul(F);
            rhsAng = rhsAng + jAngT.mul(F);
        }

        // Angular penalty
        if (length_squared(j.penaltyAng) > 0.0f) {
            Mat3 K = Mat3::diag3(j.penaltyAng);
            float4 quatAForAng2 = (j.bodyA >= 0) ? bodies[j.bodyA].positionAng : float4(0, 0, 0, 1);
            float3 C = quat_delta(quatAForAng2, bodies[j.bodyB].positionAng) * j.torqueArm;

            if (j.stiffnessAng >= 1e30f) {
                C -= j.C0Ang * alpha;
            }

            float3 F = K.mul(C) + j.lambdaAng;
            Mat3 jAng2 = isA ? Mat3::identity() * j.torqueArm : Mat3::identity() * (-j.torqueArm);

            lhsAng = lhsAng + mat3_mul(mat3_mul(mat3_transpose(jAng2), K), jAng2);
            rhsAng = rhsAng + mat3_transpose(jAng2).mul(F);
        }
    }

    // ── Accumulate spring forces ──
    int springAdjCount = min(atomic_load_explicit(&adj.springCount, memory_order_relaxed), AVBD_MAX_CONSTRAINTS_PER_BODY);
    for (int si = 0; si < springAdjCount; si++) {
        device AVBDGPUSpring &s = springs[adj.springIndices[si]];
        float3 pA = xform(bodies[s.bodyA].positionLin, bodies[s.bodyA].positionAng, s.rA);
        float3 pB = xform(bodies[s.bodyB].positionLin, bodies[s.bodyB].positionAng, s.rB);
        float3 d = pA - pB;
        float dLen = length(d);
        if (dLen <= 1.0e-6f) continue;

        float3 n = d / dLen;
        float C_spring = dLen - s.rest;
        float f = s.stiffness * C_spring;

        bool isBodyA = (s.bodyA == int(tid));
        float3 rWorld;
        float3 jLin, jAngV;
        if (isBodyA) {
            rWorld = quat_act(bodies[s.bodyA].positionAng, s.rA);
            jLin = n;
            jAngV = cross(rWorld, n);
        } else {
            rWorld = quat_act(bodies[s.bodyB].positionAng, s.rB);
            jLin = -n;
            jAngV = -cross(rWorld, n);
        }

        lhsLin = lhsLin + outer(jLin, jLin) * s.stiffness;
        lhsAng = lhsAng + outer(jAngV, jAngV) * s.stiffness;
        lhsCross = lhsCross + outer(jAngV, jLin) * s.stiffness;
        rhsLin = rhsLin + jLin * f;
        rhsAng = rhsAng + jAngV * f;
    }

    // ── Accumulate manifold (contact) forces ──
    int manifoldAdjCount = min(atomic_load_explicit(&adj.manifoldCount, memory_order_relaxed), AVBD_MAX_CONSTRAINTS_PER_BODY);
    for (int mi = 0; mi < manifoldAdjCount; mi++) {
        device AVBDGPUManifold &m = manifolds[adj.manifoldIndices[mi]];
        if (!m.active || m.contactCount == 0) continue;

        Mat3 basis(m.basisR0, m.basisR1, m.basisR2);
        bool isBodyA = (m.bodyA == int(tid));
        bool bodyBIsMesh = m.bodyB < 0;

        device AVBDGPUBody &bA = bodies[m.bodyA];
        float3 dqALin = bA.positionLin - bA.initialLin;
        float3 dqAAng = quat_delta(bA.positionAng, bA.initialAng);
        float3 dqBLin = float3(0.0f);
        float3 dqBAng = float3(0.0f);
        float4 bBQuat = float4(0.0f, 0.0f, 0.0f, 1.0f);
        if (!bodyBIsMesh) {
            device AVBDGPUBody &bB = bodies[m.bodyB];
            dqBLin = bB.positionLin - bB.initialLin;
            dqBAng = quat_delta(bB.positionAng, bB.initialAng);
            bBQuat = bB.positionAng;
        }

        for (int ci = 0; ci < m.contactCount; ci++) {
            device AVBDGPUContact &ct = allContacts[m.contactBaseIndex + ci];
            if (!ct.active) continue;

            float3 rAWorld = quat_act(bA.positionAng, ct.rA);
            float3 rBWorld = bodyBIsMesh ? ct.rB : quat_act(bBQuat, ct.rB);

            Mat3 jALin = basis;
            Mat3 jBLin = bodyBIsMesh ? Mat3(float3(0.0f), float3(0.0f), float3(0.0f)) : basis * (-1.0f);
            Mat3 jAAng = Mat3(cross(rAWorld, jALin.r0), cross(rAWorld, jALin.r1), cross(rAWorld, jALin.r2));
            Mat3 jBAng = bodyBIsMesh ? Mat3(float3(0.0f), float3(0.0f), float3(0.0f)) : Mat3(cross(rBWorld, jBLin.r0), cross(rBWorld, jBLin.r1), cross(rBWorld, jBLin.r2));

            Mat3 K = Mat3::diag3(ct.penalty);
            float3 C_ct = ct.C0 * (1.0f - alpha)
                + jALin.mul(dqALin) + jBLin.mul(dqBLin)
                + jAAng.mul(dqAAng) + jBAng.mul(dqBAng);
            float3 F_ct = K.mul(C_ct) + ct.lambda;

            // Clamp normal force (compression only)
            F_ct[0] = min(F_ct[0], 0.0f);
            // Friction cone
            float bounds = abs(F_ct[0]) * m.friction;
            float frictionScale = length(float2(F_ct[1], F_ct[2]));
            if (frictionScale > bounds && frictionScale > 0.0f) {
                F_ct[1] *= bounds / frictionScale;
                F_ct[2] *= bounds / frictionScale;
            }

            Mat3 jLin2 = isBodyA ? jALin : jBLin;
            Mat3 jAng2 = isBodyA ? jAAng : jBAng;
            Mat3 jLinT2 = mat3_transpose(jLin2);
            Mat3 jAngT2 = mat3_transpose(jAng2);
            Mat3 jAngTk2 = mat3_mul(jAngT2, K);

            lhsLin = lhsLin + mat3_mul(mat3_mul(jLinT2, K), jLin2);
            lhsAng = lhsAng + mat3_mul(jAngTk2, jAng2);
            lhsCross = lhsCross + mat3_mul(jAngTk2, jLin2);
            rhsLin = rhsLin + jLinT2.mul(F_ct);
            rhsAng = rhsAng + jAngT2.mul(F_ct);
        }
    }

    // ── Solve 6×6 system ──
    float3 dxLin, dxAng;
    solve_6x6(lhsLin, lhsAng, lhsCross, -rhsLin, -rhsAng, dxLin, dxAng);

    bool dxValid = (abs(dxLin.x) < 1e10f && abs(dxLin.y) < 1e10f && abs(dxLin.z) < 1e10f &&
                    abs(dxAng.x) < 1e10f && abs(dxAng.y) < 1e10f && abs(dxAng.z) < 1e10f);
    if (dxValid) {
        body.positionLin += dxLin;
        body.positionAng = quat_add_angular(body.positionAng, dxAng);
    }
}

// ─────────────────────────────────────────────────────────────
// KERNEL 4: Dual Update (penalty adaptation)
// ─────────────────────────────────────────────────────────────
kernel void avbd_dual_update_joints(
    device AVBDGPUBody *bodies [[buffer(0)]],
    device AVBDGPUJoint *joints [[buffer(1)]],
    constant AVBDGPUSolverParams &params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= uint(params.jointCount)) return;

    device AVBDGPUJoint &j = joints[tid];
    if (j.broken) return;

    float alpha = params.alpha;

    // Linear
    if (length_squared(j.penaltyLin) > 0.0f) {
        float3 posA, rA_local = j.rA;
        float4 quatA;
        if (j.bodyA >= 0) {
            posA = bodies[j.bodyA].positionLin;
            quatA = bodies[j.bodyA].positionAng;
        } else {
            posA = float3(0);
            quatA = float4(0, 0, 0, 1);
        }

        float3 pA = xform(posA, quatA, rA_local);
        float3 pB = xform(bodies[j.bodyB].positionLin, bodies[j.bodyB].positionAng, j.rB);
        float3 C = pA - pB;

        if (j.stiffnessLin >= 1e30f) {
            C -= j.C0Lin * alpha;
            Mat3 K = Mat3::diag3(j.penaltyLin);
            j.lambdaLin = K.mul(C) + j.lambdaLin;
        }

        j.penaltyLin = min_vec(j.penaltyLin + abs_vec(C) * params.betaLin,
                               min(j.stiffnessLin, params.penaltyMax));
    }

    // Angular
    if (length_squared(j.penaltyAng) > 0.0f) {
        float4 quatA2 = (j.bodyA >= 0) ? bodies[j.bodyA].positionAng : float4(0, 0, 0, 1);
        float3 C = quat_delta(quatA2, bodies[j.bodyB].positionAng) * j.torqueArm;

        if (j.stiffnessAng >= 1e30f) {
            C -= j.C0Ang * alpha;
            Mat3 K = Mat3::diag3(j.penaltyAng);
            j.lambdaAng = K.mul(C) + j.lambdaAng;
        }

        j.penaltyAng = min_vec(j.penaltyAng + abs_vec(C) * params.betaAng,
                               min(j.stiffnessAng, params.penaltyMax));
    }

    // Fracture check
    if (j.fracture < 1e30f && length_squared(j.lambdaAng) > j.fracture * j.fracture) {
        j.penaltyLin = float3(0);
        j.penaltyAng = float3(0);
        j.lambdaLin = float3(0);
        j.lambdaAng = float3(0);
        j.broken = 1;
    }
}

kernel void avbd_dual_update_manifolds(
    device AVBDGPUBody *bodies [[buffer(0)]],
    device AVBDGPUManifold *manifolds [[buffer(1)]],
    device int *activeManifoldIndices [[buffer(2)]],
    device AVBDGPUActiveManifoldListState *activeManifoldState [[buffer(3)]],
    device AVBDGPUContact *allContacts [[buffer(4)]],
    constant AVBDGPUSolverParams &params [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    int activeManifoldCount = min(atomic_load_explicit(&activeManifoldState[0].count, memory_order_relaxed), activeManifoldState[0].capacity);
    if (tid >= uint(activeManifoldCount)) return;

    device AVBDGPUManifold &m = manifolds[activeManifoldIndices[tid]];

    device AVBDGPUBody &bA = bodies[m.bodyA];
    bool bodyBIsMesh = m.bodyB < 0;

    Mat3 basis(m.basisR0, m.basisR1, m.basisR2);

    float3 dqALin = bA.positionLin - bA.initialLin;
    float3 dqAAng = quat_delta(bA.positionAng, bA.initialAng);
    float3 dqBLin = float3(0.0f);
    float3 dqBAng = float3(0.0f);
    float4 bBQuat = float4(0.0f, 0.0f, 0.0f, 1.0f);
    if (!bodyBIsMesh) {
        device AVBDGPUBody &bB = bodies[m.bodyB];
        dqBLin = bB.positionLin - bB.initialLin;
        dqBAng = quat_delta(bB.positionAng, bB.initialAng);
        bBQuat = bB.positionAng;
    }

    float alpha = params.alpha;

    for (int ci = 0; ci < m.contactCount; ci++) {
        device AVBDGPUContact &ct = allContacts[m.contactBaseIndex + ci];
        if (!ct.active) continue;

        float3 rAWorld = quat_act(bA.positionAng, ct.rA);
        float3 rBWorld = bodyBIsMesh ? ct.rB : quat_act(bBQuat, ct.rB);

        Mat3 jALin = basis;
        Mat3 jBLin = bodyBIsMesh ? Mat3(float3(0.0f), float3(0.0f), float3(0.0f)) : basis * (-1.0f);
        Mat3 jAAng = Mat3(cross(rAWorld, jALin.r0), cross(rAWorld, jALin.r1), cross(rAWorld, jALin.r2));
        Mat3 jBAng = bodyBIsMesh ? Mat3(float3(0.0f), float3(0.0f), float3(0.0f)) : Mat3(cross(rBWorld, jBLin.r0), cross(rBWorld, jBLin.r1), cross(rBWorld, jBLin.r2));

        Mat3 K = Mat3::diag3(ct.penalty);
        float3 C = ct.C0 * (1.0f - alpha)
            + jALin.mul(dqALin) + jBLin.mul(dqBLin)
            + jAAng.mul(dqAAng) + jBAng.mul(dqBAng);
        float3 F = K.mul(C) + ct.lambda;

        F[0] = min(F[0], 0.0f);
        float bounds = abs(F[0]) * m.friction;
        float frictionScale = length(float2(F[1], F[2]));
        if (frictionScale > bounds && frictionScale > 0.0f) {
            F[1] *= bounds / frictionScale;
            F[2] *= bounds / frictionScale;
        }

        ct.lambda = F;

        if (F[0] < 0.0f) {
            ct.penalty[0] = min(ct.penalty[0] + params.betaLin * abs(C[0]), params.penaltyMax);
        }
        if (frictionScale <= bounds) {
            ct.penalty[1] = min(ct.penalty[1] + params.betaLin * abs(C[1]), params.penaltyMax);
            ct.penalty[2] = min(ct.penalty[2] + params.betaLin * abs(C[2]), params.penaltyMax);
            ct.stick = (length(float2(C[1], C[2])) < params.stickThreshold) ? 1 : 0;
        }
    }
}

// ─────────────────────────────────────────────────────────────
// KERNEL 5: Finalize (velocity update)
// ─────────────────────────────────────────────────────────────
kernel void avbd_finalize(
    device AVBDGPUBody *bodies [[buffer(0)]],
    constant AVBDGPUSolverParams &params [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= uint(params.bodyCount)) return;

    device AVBDGPUBody &b = bodies[tid];
    float dt = params.dt;
    float linearDampingFactor = damping_factor(params.linearDamping, dt);
    float angularDampingFactor = damping_factor(params.angularDamping, dt);

    b.prevVelocityLin = b.velocityLin;
    if (b.mass > 0.0f) {
        b.velocityLin = (b.positionLin - b.initialLin) / dt;
        b.velocityAng = quat_delta(b.positionAng, b.initialAng) / dt;
        b.velocityLin *= linearDampingFactor;
        b.velocityAng *= angularDampingFactor;
    }
}

// ─────────────────────────────────────────────────────────────
// KERNEL 5b: Contact Warmstart
//   Matches newly detected contacts with previous-frame contacts
//   by (bodyA, bodyB, featureKey) and copies penalty/lambda/stick.
// ─────────────────────────────────────────────────────────────
kernel void avbd_warmstart_contacts(
    device AVBDGPUBody *bodies [[buffer(0)]],
    device AVBDGPUManifold *manifolds [[buffer(1)]],
    device AVBDGPUContact *contacts [[buffer(2)]],
    device int *activeManifoldIndices [[buffer(3)]],
    device AVBDGPUActiveManifoldListState *activeManifoldState [[buffer(4)]],
    device AVBDGPUManifold *prevManifolds [[buffer(5)]],
    device AVBDGPUContact *prevContacts [[buffer(6)]],
    device int *prevActiveManifoldIndices [[buffer(7)]],
    device AVBDGPUActiveManifoldListState *prevActiveManifoldState [[buffer(8)]],
    constant AVBDGPUSolverParams &params [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    int activeCount = min(atomic_load_explicit(&activeManifoldState[0].count, memory_order_relaxed), activeManifoldState[0].capacity);
    if (tid >= uint(activeCount)) return;

    int manifoldIdx = activeManifoldIndices[tid];
    device AVBDGPUManifold &m = manifolds[manifoldIdx];
    if (!m.active || m.contactCount == 0) return;

    // Search previous frame for matching manifold (same body pair)
    int prevActiveCount = min(atomic_load_explicit(&prevActiveManifoldState[0].count, memory_order_relaxed), prevActiveManifoldState[0].capacity);
    int prevMatchIdx = -1;
    for (int pi = 0; pi < prevActiveCount; pi++) {
        int prevMIdx = prevActiveManifoldIndices[pi];
        device AVBDGPUManifold &pm = prevManifolds[prevMIdx];
        if (pm.bodyA == m.bodyA && pm.bodyB == m.bodyB) {
            prevMatchIdx = prevMIdx;
            break;
        }
    }
    if (prevMatchIdx < 0) return;

    device AVBDGPUManifold &pm = prevManifolds[prevMatchIdx];
    Mat3 basis(m.basisR0, m.basisR1, m.basisR2);
    bool bodyBIsMesh = m.bodyB < 0;

    for (int ci = 0; ci < m.contactCount; ci++) {
        device AVBDGPUContact &ct = contacts[m.contactBaseIndex + ci];
        if (!ct.active) continue;

        for (int pci = 0; pci < pm.contactCount; pci++) {
            device AVBDGPUContact &pct = prevContacts[pm.contactBaseIndex + pci];
            if (pct.featureKey != ct.featureKey || !pct.active) continue;

            // Match found — copy warmstart data with decay
            ct.penalty = clamp(pct.penalty * params.gamma,
                               float3(params.penaltyMin),
                               float3(params.penaltyMax));
            ct.lambda = pct.lambda * params.alpha * params.gamma;

            if (pct.stick) {
                ct.rA = pct.rA;
                if (!bodyBIsMesh) {
                    ct.rB = pct.rB;
                }
                ct.stick = pct.stick;
                // Recompute C0 with old rA/rB and current body positions
                device AVBDGPUBody &bA = bodies[m.bodyA];
                float3 xA = quat_act(bA.positionAng, ct.rA) + bA.positionLin;
                float3 xB;
                if (bodyBIsMesh) {
                    xB = ct.rB;
                } else {
                    device AVBDGPUBody &bB = bodies[m.bodyB];
                    xB = quat_act(bB.positionAng, ct.rB) + bB.positionLin;
                }
                float3 diff = xA - xB;
                ct.C0 = float3(dot(basis.r0, diff), dot(basis.r1, diff), dot(basis.r2, diff))
                    + float3(params.collisionMargin, 0, 0);
            }
            break;
        }
    }
}

// ─────────────────────────────────────────────────────────────
// KERNEL 6: Write instance transforms for rendering
// ─────────────────────────────────────────────────────────────
constant int AVBD_NIL_COLOR_GROUP = (-2147483647 - 1);

inline float4 avbd_default_render_color_for_mass(float mass)
{
    return mass <= 0.0f ? float4(0.47f, 0.50f, 0.47f, 1.0f)
                        : float4(0.72f, 0.80f, 0.92f, 1.0f);
}

inline float4 avbd_render_color_for_group(int colorGroup)
{
    int paletteIndex = colorGroup % 5;
    if (paletteIndex < 0) {
        paletteIndex += 5;
    }

    switch (paletteIndex) {
        case 0: return float4(0.93f, 0.42f, 0.39f, 1.0f);
        case 1: return float4(0.97f, 0.71f, 0.30f, 1.0f);
        case 2: return float4(0.48f, 0.79f, 0.51f, 1.0f);
        case 3: return float4(0.35f, 0.73f, 0.92f, 1.0f);
        default: return float4(0.63f, 0.52f, 0.90f, 1.0f);
    }
}

inline float4 avbd_resolve_render_color(
    float4 explicitRenderColor,
    int explicitColorGroup,
    int computedColorGroup,
    float mass)
{
    if (explicitRenderColor.w >= 0.0f) {
        return explicitRenderColor;
    }
    if (explicitColorGroup != AVBD_NIL_COLOR_GROUP) {
        return avbd_render_color_for_group(explicitColorGroup);
    }
    if (mass <= 0.0f) {
        return avbd_default_render_color_for_mass(mass);
    }
    return avbd_render_color_for_group(computedColorGroup);
}

kernel void avbd_write_instances(
    device AVBDGPUBody *bodies [[buffer(0)]],
    device InstanceUniforms *instances [[buffer(1)]],
    device const float4 *renderColors [[buffer(2)]],
    device const int *renderColorGroups [[buffer(3)]],
    device const int *bodyColors [[buffer(4)]],
    constant AVBDGPUSolverParams &params [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= uint(params.bodyCount)) return;

    device AVBDGPUBody &b = bodies[tid];

    // Build model matrix = Translation * Rotation * Scale
    float4 q = normalize(b.positionAng);
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float xx = x*x, yy = y*y, zz = z*z;
    float xy = x*y, xz = x*z, yz = y*z;
    float wx = w*x, wy = w*y, wz = w*z;

    float3 sz = b.size;
    float4 c0 = float4((1 - 2*(yy+zz)) * sz.x, (2*(xy+wz)) * sz.x, (2*(xz-wy)) * sz.x, 0);
    float4 c1 = float4((2*(xy-wz)) * sz.y, (1 - 2*(xx+zz)) * sz.y, (2*(yz+wx)) * sz.y, 0);
    float4 c2 = float4((2*(xz+wy)) * sz.z, (2*(yz-wx)) * sz.z, (1 - 2*(xx+yy)) * sz.z, 0);
    float4 c3 = float4(b.positionLin, 1.0f);

    instances[tid].modelMatrix = float4x4(c0, c1, c2, c3);
    instances[tid].renderColor = avbd_resolve_render_color(
        renderColors[tid],
        renderColorGroups[tid],
        bodyColors[tid],
        b.mass
    );
    instances[tid].shapeParams = float4(float(b.renderShape),
                                        b.renderShape == AVBD_GPU_RENDER_SHAPE_TORUS ? torus_major_radius(b) : 0.0f,
                                        b.renderShape == AVBD_GPU_RENDER_SHAPE_TORUS ? torus_minor_radius(b) : 0.0f,
                                        0.0f);
}
