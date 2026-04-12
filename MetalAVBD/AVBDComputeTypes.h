//
//  AVBDComputeTypes.h
//  MetalAVBD
//
//  Shared types between Metal compute shaders and Swift host code.
//

#ifndef AVBDComputeTypes_h
#define AVBDComputeTypes_h

#include <simd/simd.h>

#ifdef __METAL_VERSION__
#define AVBD_CONSTANT constant
#else
#define AVBD_CONSTANT
#endif

// Shared compile-time limits used by both Metal shaders and Swift host code.
// Keep these as the single source of truth for fixed-size GPU layouts.
#define AVBD_MAX_CONSTRAINTS_PER_BODY 32
#define AVBD_MAX_CONTACTS_PER_PAIR 4
#define AVBD_MAX_CONTACTS_PER_PAIR_BURST (AVBD_MAX_CONTACTS_PER_PAIR * 2)
#define AVBD_BROADPHASE_THREADGROUP_SIZE 64
#define AVBD_DERIVED_THREADGROUP_SIZE 32
#define AVBD_MAX_COLLISIONS_PER_BODY 16

#ifdef __METAL_VERSION__
#define AVBD_ATOMIC_INT metal::atomic_int
#else
#define AVBD_ATOMIC_INT int
#endif

// Body state - SoA layout for GPU
typedef struct {
    vector_float3 positionLin;
    vector_float4 positionAng; // quaternion (x,y,z,w)
    vector_float3 initialLin;
    vector_float4 initialAng;
    vector_float3 inertialLin;
    vector_float4 inertialAng;
    vector_float3 velocityLin;
    vector_float3 velocityAng;
    vector_float3 prevVelocityLin;
    vector_float3 size;
    float mass;
    vector_float3 moment;
    float friction;
    int renderShape;
} AVBDGPUBody;

// Joint constraint data
typedef struct {
    int bodyA;          // -1 for world anchor
    int bodyB;
    vector_float3 rA;   // local anchor on A
    float stiffnessLin;
    vector_float3 rB;   // local anchor on B
    float stiffnessAng;
    vector_float3 C0Lin;
    float fracture;
    vector_float3 C0Ang;
    float torqueArm;
    vector_float3 penaltyLin;
    vector_float3 penaltyAng;
    vector_float3 lambdaLin;
    vector_float3 lambdaAng;
    int broken;
} AVBDGPUJoint;

// Spring constraint data
typedef struct {
    int bodyA;
    int bodyB;
    float stiffness;
    float rest;
    vector_float3 rA;
    vector_float3 rB;
} AVBDGPUSpring;

// Broadphase candidate pair
typedef struct {
    int bodyA;
    int bodyB;
} AVBDGPUCollisionPair;

// Contact point
typedef struct {
    vector_float3 rA;
    vector_float3 rB;
    vector_float3 C0;
    vector_float3 penalty;
    vector_float3 lambda;
    int featureKey;
    int active;
    int stick;
} AVBDGPUContact;

// Contact manifold between two bodies
typedef struct {
    int bodyA;
    int bodyB;
    int contactCount;
    int active;
    float friction;
    // 3x3 basis stored as 3 rows
    vector_float3 basisR0;
    vector_float3 basisR1;
    vector_float3 basisR2;
    int contactBaseIndex; // Index into the global contacts buffer
} AVBDGPUManifold;

// Per-body collision exclusion list (joint-connected and ignored pairs).
typedef struct {
    int excludeIndices[AVBD_MAX_CONSTRAINTS_PER_BODY];
    AVBD_ATOMIC_INT excludeCount;
} AVBDGPUCollisionExclusion;

// Global contact allocator state for broadphase compaction.
typedef struct {
    AVBD_ATOMIC_INT nextContactIndex;
    int contactCapacity;
} AVBDGPUContactAllocator;

// Global manifold allocator state for dynamic manifold slot allocation.
typedef struct {
    AVBD_ATOMIC_INT nextManifoldIndex;
    int manifoldCapacity;
} AVBDGPUManifoldAllocator;

// Cached broadphase pair list state for indirect dispatch.
typedef struct {
    AVBD_ATOMIC_INT count;
    int capacity;
} AVBDGPURecentPairCacheState;

// Derived narrowphase candidate list state for indirect dispatch.
typedef struct {
    AVBD_ATOMIC_INT count;
    int capacity;
} AVBDGPUDerivedPairCandidateState;

// Active manifold list state for manifold-only downstream passes.
typedef struct {
    AVBD_ATOMIC_INT count;
    int capacity;
} AVBDGPUActiveManifoldListState;

// Matches MTLDispatchThreadgroupsIndirectArguments layout.
typedef struct {
    unsigned int threadgroupsPerGrid[3];
} AVBDGPUIndirectDispatchArgs;

// Per-body adjacency: which joints/springs/manifolds act on this body
typedef struct {
    int jointIndices[AVBD_MAX_CONSTRAINTS_PER_BODY];
    AVBD_ATOMIC_INT jointCount;
    int springIndices[AVBD_MAX_CONSTRAINTS_PER_BODY];
    AVBD_ATOMIC_INT springCount;
    int manifoldIndices[AVBD_MAX_CONSTRAINTS_PER_BODY];
    AVBD_ATOMIC_INT manifoldCount;
} AVBDGPUAdjacency;

// Solver parameters
typedef struct {
    float dt;
    float gravity;
    float alpha;
    float betaLin;
    float betaAng;
    float gamma;
    int iterations;
    int bodyCount;
    int jointCount;
    int springCount;
    int manifoldCapacity;
    float collisionMargin;
    float cacheMargin;
    float cacheTimeHorizon;
    float penaltyMin;
    float penaltyMax;
    float stickThreshold;
    int torusApproxSphereCount;
    float torusApproxSphereRadiusScale;
} AVBDGPUSolverParams;

#endif /* AVBDComputeTypes_h */
