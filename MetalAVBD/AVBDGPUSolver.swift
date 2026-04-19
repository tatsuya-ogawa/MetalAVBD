//
//  AVBDGPUSolver.swift
//  MetalAVBD
//
//  GPU-accelerated AVBD solver using Metal compute shaders.
//  Mirrors the CPU AVBDSolver but runs the iteration loop on GPU.
//  Broadphase collision detection and contact generation also run on GPU.
//

import Foundation
import Metal
import QuartzCore
import simd

private let SWIFT_AVBD_MAX_CONTACTS_PER_PAIR = Int(AVBD_MAX_CONTACTS_PER_PAIR)
private let SWIFT_AVBD_MAX_CONTACTS_PER_PAIR_BURST = SWIFT_AVBD_MAX_CONTACTS_PER_PAIR * 2
private let SWIFT_AVBD_MAX_CONSTRAINTS_PER_BODY = Int(AVBD_MAX_CONSTRAINTS_PER_BODY)
private let gpuCollisionMargin: Float = 0.01
private let gpuPenaltyMin: Float = 1.0
private let gpuPenaltyMax: Float = 10_000_000_000.0
private let gpuStickThreshold: Float = 0.00001
private let gpuContactPenaltyStart: Float = Float(AVBD_COLLISION_CONTACT_PENALTY_START)
private let gpuMaxPolyVerts = Int(AVBD_COLLISION_MAX_POLY_VERTS)
private let gpuSatAxisEpsilon: Float = Float(AVBD_COLLISION_SAT_AXIS_EPSILON)
private let gpuPlaneEpsilon: Float = Float(AVBD_COLLISION_PLANE_EPSILON)
private let gpuContactMergeDistSq: Float = Float(AVBD_COLLISION_CONTACT_MERGE_DIST_SQ)
private let gpuNilRenderColor = SIMD4<Float>(0, 0, 0, -1)
private let gpuNilColorGroup: Int32 = Int32.min
private let gpuCollisionMeshSDFLongestAxisResolution: Int32 = 64
private let gpuCollisionMeshSDFTriangleChunkSize = 1024
private let gpuCollisionMeshSDFBrickDim = 8
private let gpuCollisionMeshSDFGuardVoxelCount = 1
private let gpuCollisionMeshSDFStoredBrickDim = gpuCollisionMeshSDFBrickDim + gpuCollisionMeshSDFGuardVoxelCount * 2
private let gpuCollisionMeshSDFAtlasBricksAcross = 64
private let gpuCollisionMeshSDFCoarseDownsampleFactor = 2
private let gpuCollisionMeshSDFMappedBrickDilation = 1
private let gpuMaxCollisionMeshSDFs = Int(AVBD_MAX_COLLISION_MESH_SDFS)
private let gpuCollisionMeshSDFAtlasTextureBaseIndex = gpuMaxCollisionMeshSDFs
private let gpuCollisionMeshSDFIndirectionTextureBaseIndex = gpuMaxCollisionMeshSDFs * 2
private let gpuMeshMeshIsoVoxelTrackedPairLimit = 1024
private let gpuMeshMeshIsoVoxelCoordsPerPair = 32

private struct AVBDGPUOBB {
    var center: SIMD3<Float>
    var rotation: simd_quatf
    var half: SIMD3<Float>
    var axis: [SIMD3<Float>]
}

private struct AVBDGPUSatAxis {
    var type: AVBDAxisType = .faceA
    var indexA = -1
    var indexB = -1
    var separation = -Float.greatestFiniteMagnitude
    var normalAB = SIMD3<Float>.zero
    var valid = false
}

private struct AVBDGPUFaceFrame {
    var normal = SIMD3<Float>.zero
    var center = SIMD3<Float>.zero
    var u = SIMD3<Float>.zero
    var v = SIMD3<Float>.zero
    var extentU: Float = 0
    var extentV: Float = 0
}

struct AVBDCollisionMeshBroadphaseMesh {
    var sdfResourceID: String = ""
    var ownerBodyIndex: Int = -1
    var localBoundsMin: SIMD3<Float>
    var localBoundsMax: SIMD3<Float>
    var transform: simd_float4x4 = matrix_identity_float4x4
    var positions: [SIMD3<Float>] = []
    var indices: [UInt32] = []
}

final class AVBDGPUSolver {
    struct MeshMeshIsoVoxelDebugEntry {
        var meshIndexA: Int
        var meshIndexB: Int
        var driverMeshIndex: Int
        var sampledVoxelCount: Int
        var candidateVoxelCount: Int
        var compactedVoxelCount: Int
        var sampleStride: Int
        var overflowed: Bool
        var emittedContactCount: Int
        var usedIsoVoxelPath: Bool
        var voxelCoords: [SIMD3<Int32>]
    }

    private struct CollisionMeshSDFResource {
        var coarseTexture: MTLTexture
        var atlasTexture: MTLTexture
        var indirectionTexture: MTLTexture
        var mappedBrickCount: Int
        var denseByteCount: Int
        var compactedByteCount: Int
    }

    struct CollisionMeshSDFCompactionData {
        var coarseResolution: SIMD3<Int>
        var atlasResolution: SIMD3<Int>
        var brickGrid: SIMD3<Int>
        var coarseData: [Float]
        var atlasData: [Float]
        var indirectionData: [UInt32]
        var mappedBrickCount: Int
        var denseByteCount: Int
        var compactedByteCount: Int
    }

    private struct CachedCollisionMeshSDF {
        var resource: CollisionMeshSDFResource
    }

    private struct CollisionMeshGeometryResource {
        var vertexOffset: Int
        var vertexCount: Int
        var indexOffset: Int
        var indexCount: Int
    }

    private static var collisionMeshSDFCache: [String: CachedCollisionMeshSDF] = [:]
    private static let collisionMeshSDFCacheLock = NSLock()

    let device: MTLDevice
    private let collisionMeshSDFCommandQueue: MTLCommandQueue?
    private let collisionMeshSDFArgumentEncoder: MTLArgumentEncoder?
    private var collisionMeshSDFArgumentBuffer: MTLBuffer?
    private var collisionMeshSDFResources: [CollisionMeshSDFResource] = []
    private var collisionMeshGeometryResources: [String: CollisionMeshGeometryResource] = [:]
    private var collisionMeshSDFResourceIndicesByKey: [String: Int] = [:]
    private var collisionMeshVertexCountUsed = 0
    private var collisionMeshIndexCountUsed = 0
    private(set) var collisionMeshSDFStatusText: String = "Idle"

    // Compute pipelines
    private let forwardIntegratePSO: MTLComputePipelineState
    private let resetAdjacencyPSO: MTLComputePipelineState
    private let broadphaseFullPSO: MTLComputePipelineState
    private let broadphasePartialPSO: MTLComputePipelineState
    private let prepareBroadphaseIndirectPSO: MTLComputePipelineState
    private let buildPrimitiveManifoldsPSO: MTLComputePipelineState
    private let prepareActiveManifoldsIndirectPSO: MTLComputePipelineState
    private let primitiveMeshBroadphasePSO: MTLComputePipelineState
    private let preparePrimitiveMeshCollisionIndirectPSO: MTLComputePipelineState
    private let collidePrimitiveMeshPSO: MTLComputePipelineState
    private let collidePrimitiveMeshSDFPSO: MTLComputePipelineState
    private let initializeCollisionMeshSDFPSO: MTLComputePipelineState
    private let accumulateCollisionMeshSDFPSO: MTLComputePipelineState
    private let finalizeCollisionMeshSDFPSO: MTLComputePipelineState
    private let meshMeshBroadphasePSO: MTLComputePipelineState
    private let prepareMeshMeshCollisionIndirectPSO: MTLComputePipelineState
    private let collideMeshMeshPSO: MTLComputePipelineState
    private let buildAdjacencyConstraintsPSO: MTLComputePipelineState
    private let buildAdjacencyManifoldsPSO: MTLComputePipelineState
    private let initJointsPSO: MTLComputePipelineState
    private let bodySolvePSO: MTLComputePipelineState
    private let dualUpdateJointsPSO: MTLComputePipelineState
    private let dualUpdateManifoldsPSO: MTLComputePipelineState
    private let warmstartContactsPSO: MTLComputePipelineState
    private let finalizePSO: MTLComputePipelineState
    private let writeInstancesPSO: MTLComputePipelineState

    // Solver parameters
    var dt: Float = 1.0 / 60.0
    var gravity: Float = -10.0
    var iterations: Int = 20
    var alpha: Float = 0.9
    var betaLin: Float = 100_000.0
    var betaAng: Float = 100.0
    var gamma: Float = 0.999
    var linearDamping: Float = 0.0
    var angularDamping: Float = 0.0
    var hydroelasticInteriorWeight: Float = 0.5
    var meshMeshMaxIsoVoxelSamples: Int = 512
    var meshMeshReduceContacts: Bool = true
    private var _broadphaseCacheMargin: Float = 0.1
    private var _broadphaseFullRefreshStepCount: Int = 60

    var broadphaseCacheMargin: Float {
        get { _broadphaseCacheMargin }
        set {
            _broadphaseCacheMargin = max(0.0, newValue)
            forceFullBroadphase = true
        }
    }

    var broadphaseFullRefreshStepCount: Int {
        get { _broadphaseFullRefreshStepCount }
        set {
            _broadphaseFullRefreshStepCount = max(0, newValue)
            forceFullBroadphase = true
        }
    }

    var enableContactWarmstart: Bool = true
    var enablePrimitiveMeshCollisions: Bool = true
    var enableMeshMeshCollisions: Bool = true

    // Extra capacity for dynamically thrown bodies
    private static let extraBodyCapacity = 50

    // Body data (authoritative on GPU after upload)
    private(set) var bodyCount: Int
    private var bodyCapacity: Int
    private var bodyBuffer: MTLBuffer
    private var bodyColorBuffer: MTLBuffer
    private var renderColorBuffer: MTLBuffer
    private var renderColorGroupBuffer: MTLBuffer
    private var numColors: Int
    private var jointBuffer: MTLBuffer
    private var springBuffer: MTLBuffer
    private var exclusionBuffer: MTLBuffer
    private var manifoldBuffer: MTLBuffer
    private var adjacencyBuffer: MTLBuffer
    private var contactBuffer: MTLBuffer
    private var contactAllocatorBuffer: MTLBuffer
    private var recentPairIndexBuffers: [MTLBuffer]
    private var recentPairStateBuffers: [MTLBuffer]
    private var recentPairIndirectBuffers: [MTLBuffer]
    private var derivedPairCandidateBuffer: MTLBuffer
    private var derivedPairStateBuffer: MTLBuffer
    private var derivedPairIndirectBuffer: MTLBuffer
    private var activeManifoldIndexBuffer: MTLBuffer
    private var activeManifoldStateBuffer: MTLBuffer
    private var activeManifoldIndirectBuffer: MTLBuffer
    private var prevManifoldBuffer: MTLBuffer
    private var prevContactBuffer: MTLBuffer
    private var prevActiveManifoldIndexBuffer: MTLBuffer
    private var prevActiveManifoldStateBuffer: MTLBuffer
    private var collisionMeshInfoBuffer: MTLBuffer
    private var collisionMeshOwnerBodyMaskBuffer: MTLBuffer
    private var collisionMeshVertexBuffer: MTLBuffer
    private var collisionMeshIndexBuffer: MTLBuffer
    private var primitiveMeshPairBuffer: MTLBuffer
    private var primitiveMeshPairStateBuffer: MTLBuffer
    private var primitiveMeshPairIndirectBuffer: MTLBuffer
    private var meshMeshPairBuffer: MTLBuffer
    private var meshMeshPairStateBuffer: MTLBuffer
    private var meshMeshPairIndirectBuffer: MTLBuffer
    private var meshMeshIsoVoxelDebugBuffer: MTLBuffer
    private var meshMeshIsoVoxelCoordBuffer: MTLBuffer
    private var paramsBuffer: MTLBuffer
    private var maxCollisions: Int
    private var bodyBodyManifoldCapacity: Int
    private var primitiveMeshManifoldCapacity: Int
    private var meshMeshManifoldCapacity: Int
    private var recentPairCapacity: Int
    private var contactCapacity: Int
    private var collisionMeshCapacity: Int
    private var collisionMeshCount: Int
    private var primitiveMeshPairCapacity: Int
    private var meshMeshPairCapacity: Int
    private var meshMeshIsoVoxelTrackedPairCapacity: Int
    private var recentPairCacheSlot: Int

    // CPU-side setup/reference copies for static constraints and initial upload
    private var cpuBodies: [AVBDGPUBody]
    private var cpuJoints: [AVBDGPUJoint]
    private var cpuSprings: [AVBDGPUSpring]
    private var cpuManifolds: [AVBDGPUManifold]
    private var cpuAdjacency: [AVBDGPUAdjacency]

    // Ignore-collision pairs
    private var ignorePairs: Set<UInt64> = []

    // Scene info for color reference
    private var bodyShapes: [AVBDRenderShape]
    private var computedBodyColorGroups: [Int32]
    private var stepsSinceFullBroadphase: Int
    private var forceFullBroadphase: Bool
    private(set) var lastBroadphaseUsedCache: Bool = false

    private var currentRecentPairIndexBuffer: MTLBuffer { recentPairIndexBuffers[recentPairCacheSlot] }
    private var nextRecentPairIndexBuffer: MTLBuffer { recentPairIndexBuffers[1 - recentPairCacheSlot] }
    private var currentRecentPairStateBuffer: MTLBuffer { recentPairStateBuffers[recentPairCacheSlot] }
    private var nextRecentPairStateBuffer: MTLBuffer { recentPairStateBuffers[1 - recentPairCacheSlot] }
    private var currentRecentPairIndirectBuffer: MTLBuffer { recentPairIndirectBuffers[recentPairCacheSlot] }
    private var nextRecentPairIndirectBuffer: MTLBuffer { recentPairIndirectBuffers[1 - recentPairCacheSlot] }

    private func encodeRenderColor(_ renderColor: SIMD4<Float>?) -> SIMD4<Float> {
        renderColor ?? gpuNilRenderColor
    }

    private func decodeRenderColor(_ renderColor: SIMD4<Float>) -> SIMD4<Float>? {
        renderColor.w < 0 ? nil : renderColor
    }

    private func encodeColorGroup(_ colorGroup: Int?) -> Int32 {
        guard let colorGroup else { return gpuNilColorGroup }
        return Int32(clamping: colorGroup)
    }

    private func decodeColorGroup(_ colorGroup: Int32) -> Int? {
        colorGroup == gpuNilColorGroup ? nil : Int(colorGroup)
    }

    func resolvedRenderColor(bodyIndex: Int) -> SIMD4<Float> {
        let renderColorPtr = renderColorBuffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: bodyCapacity)
        let renderColorGroupPtr = renderColorGroupBuffer.contents().bindMemory(to: Int32.self, capacity: bodyCapacity)
        let body = cpuBodies[bodyIndex]
        let explicitRenderColor = decodeRenderColor(renderColorPtr[bodyIndex])
        let explicitColorGroup = decodeColorGroup(renderColorGroupPtr[bodyIndex])
        let computedColorGroup = body.mass > 0 && explicitColorGroup == nil && bodyIndex < computedBodyColorGroups.count
            ? Int(computedBodyColorGroups[bodyIndex])
            : nil
        return avbdResolveRenderColor(
            renderColor: explicitRenderColor,
            colorGroup: explicitColorGroup ?? computedColorGroup,
            isStatic: body.mass <= 0
        )
    }

    var collisionMeshDebugInfoBuffer: MTLBuffer? {
        guard collisionMeshCount > 0 else { return nil }
        return collisionMeshInfoBuffer
    }

    var collisionMeshDebugSDFArgumentBuffer: MTLBuffer? {
        guard collisionMeshCount > 0,
              !collisionMeshSDFResources.isEmpty else {
            return nil
        }
        return collisionMeshSDFArgumentBuffer
    }

    var collisionMeshDebugTextures: [MTLTexture] {
        collisionMeshSDFResources
            .prefix(collisionMeshCount)
            .flatMap { [$0.coarseTexture, $0.atlasTexture, $0.indirectionTexture] }
    }

    func collisionMeshInfoSnapshot() -> [AVBDGPUCollisionMeshInfo] {
        guard collisionMeshCount > 0 else { return [] }
        let infoPtr = collisionMeshInfoBuffer.contents().bindMemory(
            to: AVBDGPUCollisionMeshInfo.self,
            capacity: collisionMeshCount
        )
        return Array(UnsafeBufferPointer(start: infoPtr, count: collisionMeshCount))
    }

    private func ensureCollisionMeshLocalBufferCapacity(requiredVertexCount: Int, requiredIndexCount: Int) {
        if collisionMeshVertexBuffer.length < MemoryLayout<SIMD3<Float>>.stride * max(requiredVertexCount, 1) {
            let oldBuffer = collisionMeshVertexBuffer
            collisionMeshVertexBuffer = device.makeBuffer(
                length: MemoryLayout<SIMD3<Float>>.stride * max(requiredVertexCount, 1),
                options: .storageModeShared
            )!
            if collisionMeshVertexCountUsed > 0 {
                memcpy(collisionMeshVertexBuffer.contents(), oldBuffer.contents(), MemoryLayout<SIMD3<Float>>.stride * collisionMeshVertexCountUsed)
            }
        }

        if collisionMeshIndexBuffer.length < MemoryLayout<UInt32>.stride * max(requiredIndexCount, 1) {
            let oldBuffer = collisionMeshIndexBuffer
            collisionMeshIndexBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * max(requiredIndexCount, 1),
                options: .storageModeShared
            )!
            if collisionMeshIndexCountUsed > 0 {
                memcpy(collisionMeshIndexBuffer.contents(), oldBuffer.contents(), MemoryLayout<UInt32>.stride * collisionMeshIndexCountUsed)
            }
        }
    }

    private func updateCollisionMeshSDFArgumentBufferBindings() {
        guard let collisionMeshSDFArgumentEncoder, !collisionMeshSDFResources.isEmpty else {
            return
        }
        if collisionMeshSDFArgumentBuffer == nil
            || collisionMeshSDFArgumentBuffer!.length < collisionMeshSDFArgumentEncoder.encodedLength {
            collisionMeshSDFArgumentBuffer = device.makeBuffer(
                length: collisionMeshSDFArgumentEncoder.encodedLength,
                options: .storageModeShared
            )
        }
        if let collisionMeshSDFArgumentBuffer {
            collisionMeshSDFArgumentEncoder.setArgumentBuffer(collisionMeshSDFArgumentBuffer, offset: 0)
            for textureIndex in 0..<gpuMaxCollisionMeshSDFs {
                let resource = textureIndex < collisionMeshSDFResources.count ? collisionMeshSDFResources[textureIndex] : nil
                collisionMeshSDFArgumentEncoder.setTexture(resource?.coarseTexture, index: textureIndex)
                collisionMeshSDFArgumentEncoder.setTexture(
                    resource?.atlasTexture,
                    index: gpuCollisionMeshSDFAtlasTextureBaseIndex + textureIndex
                )
                collisionMeshSDFArgumentEncoder.setTexture(
                    resource?.indirectionTexture,
                    index: gpuCollisionMeshSDFIndirectionTextureBaseIndex + textureIndex
                )
            }
        }
    }

    private static func cachedCollisionMeshSDF(for key: String) -> CachedCollisionMeshSDF? {
        collisionMeshSDFCacheLock.lock()
        defer { collisionMeshSDFCacheLock.unlock() }
        return collisionMeshSDFCache[key]
    }

    private static func storeCollisionMeshSDF(_ resource: CollisionMeshSDFResource, for key: String) {
        collisionMeshSDFCacheLock.lock()
        collisionMeshSDFCache[key] = CachedCollisionMeshSDF(resource: resource)
        collisionMeshSDFCacheLock.unlock()
    }

    @discardableResult
    func prepareCollisionMeshSDFCache(_ mesh: AVBDCollisionMeshBroadphaseMesh) -> Bool {
        let sdfPadding = Self.collisionMeshSDFPadding(for: mesh)
        let sdfLocalMinBounds = mesh.localBoundsMin - sdfPadding
        let sdfLocalMaxBounds = mesh.localBoundsMax + sdfPadding
        let sdfResolution = Self.collisionMeshSDFResolution(
            localMinBounds: sdfLocalMinBounds,
            localMaxBounds: sdfLocalMaxBounds
        )
        let resourceKey = Self.collisionMeshSDFCacheKey(
            mesh: mesh,
            localBoundsMin: sdfLocalMinBounds,
            localBoundsMax: sdfLocalMaxBounds,
            resolution: sdfResolution
        )

        if Self.cachedCollisionMeshSDF(for: resourceKey) != nil {
            return true
        }

        guard let sdfResource = buildCollisionMeshSDFTexture(
            localVertices: mesh.positions,
            indices: mesh.indices,
            localBoundsMin: sdfLocalMinBounds,
            localBoundsMax: sdfLocalMaxBounds,
            resolution: sdfResolution
        ) else {
            return false
        }

        Self.storeCollisionMeshSDF(sdfResource, for: resourceKey)
        return true
    }

    @discardableResult
    func prewarmCollisionMeshResource(_ mesh: AVBDCollisionMeshBroadphaseMesh) -> Bool {
        let sdfPadding = Self.collisionMeshSDFPadding(for: mesh)
        let sdfLocalMinBounds = mesh.localBoundsMin - sdfPadding
        let sdfLocalMaxBounds = mesh.localBoundsMax + sdfPadding
        let sdfResolution = Self.collisionMeshSDFResolution(
            localMinBounds: sdfLocalMinBounds,
            localMaxBounds: sdfLocalMaxBounds
        )
        let resourceKey = Self.collisionMeshSDFCacheKey(
            mesh: mesh,
            localBoundsMin: sdfLocalMinBounds,
            localBoundsMax: sdfLocalMaxBounds,
            resolution: sdfResolution
        )

        if collisionMeshGeometryResources[resourceKey] != nil,
           collisionMeshSDFResourceIndicesByKey[resourceKey] != nil {
            collisionMeshSDFStatusText = "Cached"
            return true
        }

        if collisionMeshGeometryResources[resourceKey] == nil {
            let vertexOffset = collisionMeshVertexCountUsed
            let indexOffset = collisionMeshIndexCountUsed
            ensureCollisionMeshLocalBufferCapacity(
                requiredVertexCount: vertexOffset + mesh.positions.count,
                requiredIndexCount: indexOffset + mesh.indices.count
            )
            let vertexPtr = collisionMeshVertexBuffer.contents().bindMemory(
                to: SIMD3<Float>.self,
                capacity: max(vertexOffset + mesh.positions.count, 1)
            )
            for vertexIndex in mesh.positions.indices {
                vertexPtr[vertexOffset + vertexIndex] = mesh.positions[vertexIndex]
            }
            let indexPtr = collisionMeshIndexBuffer.contents().bindMemory(
                to: UInt32.self,
                capacity: max(indexOffset + mesh.indices.count, 1)
            )
            for index in mesh.indices.indices {
                indexPtr[indexOffset + index] = UInt32(vertexOffset) + mesh.indices[index]
            }
            collisionMeshGeometryResources[resourceKey] = CollisionMeshGeometryResource(
                vertexOffset: vertexOffset,
                vertexCount: mesh.positions.count,
                indexOffset: indexOffset,
                indexCount: mesh.indices.count
            )
            collisionMeshVertexCountUsed += mesh.positions.count
            collisionMeshIndexCountUsed += mesh.indices.count
        }

        if collisionMeshSDFResourceIndicesByKey[resourceKey] == nil {
            let sdfResource: CollisionMeshSDFResource?
            if let cached = Self.cachedCollisionMeshSDF(for: resourceKey) {
                sdfResource = cached.resource
            } else {
                sdfResource = buildCollisionMeshSDFTexture(
                    localVertices: mesh.positions,
                    indices: mesh.indices,
                    localBoundsMin: sdfLocalMinBounds,
                    localBoundsMax: sdfLocalMaxBounds,
                    resolution: sdfResolution
                )
                if let sdfResource {
                    Self.storeCollisionMeshSDF(sdfResource, for: resourceKey)
                }
            }

            guard let sdfResource else {
                return false
            }

            collisionMeshSDFResourceIndicesByKey[resourceKey] = collisionMeshSDFResources.count
            collisionMeshSDFResources.append(sdfResource)
            updateCollisionMeshSDFArgumentBufferBindings()
        }

        collisionMeshSDFStatusText = "Cached"
        return true
    }

    @discardableResult
    func updateCollisionMeshInstances(_ meshes: [AVBDCollisionMeshBroadphaseMesh]) -> Bool {
        guard meshes.count == collisionMeshCount, collisionMeshCount > 0 else {
            return false
        }

        let infoPtr = collisionMeshInfoBuffer.contents().bindMemory(
            to: AVBDGPUCollisionMeshInfo.self,
            capacity: max(collisionMeshCapacity, 1)
        )

        for meshIndex in 0..<collisionMeshCount {
            let mesh = meshes[meshIndex]
            var meshInfo = infoPtr[meshIndex]
            if Int(meshInfo.vertexCount) != mesh.positions.count || Int(meshInfo.indexCount) != mesh.indices.count {
                return false
            }

            let localMinBounds = SIMD3<Float>(
                meshInfo.sdfLocalMinBounds.x,
                meshInfo.sdfLocalMinBounds.y,
                meshInfo.sdfLocalMinBounds.z
            )
            let localMaxBounds = SIMD3<Float>(
                meshInfo.sdfLocalMaxBounds.x,
                meshInfo.sdfLocalMaxBounds.y,
                meshInfo.sdfLocalMaxBounds.z
            )
            let worldBounds = Self.transformedBounds(
                localMinBounds: localMinBounds,
                localMaxBounds: localMaxBounds,
                transform: mesh.transform
            )

            meshInfo.ownerBodyIndex = Int32(mesh.ownerBodyIndex)
            meshInfo.minBounds = SIMD4<Float>(worldBounds.min, 0)
            meshInfo.maxBounds = SIMD4<Float>(worldBounds.max, 0)
            meshInfo.sdfTransform = mesh.transform
            meshInfo.sdfInvTransform = simd_inverse(mesh.transform)
            infoPtr[meshIndex] = meshInfo
        }

        forceFullBroadphase = true
        collisionMeshSDFStatusText = !collisionMeshSDFResources.isEmpty ? "Cached" : "Idle"
        refreshCollisionMeshOwnerBodyMask()
        return true
    }

    @discardableResult
    func appendCollisionMeshInstance(_ mesh: AVBDCollisionMeshBroadphaseMesh) -> Bool {
        guard collisionMeshCount < min(collisionMeshCapacity, gpuMaxCollisionMeshSDFs) else {
            return false
        }
        ensureCollisionCapacity(forMeshCount: collisionMeshCount + 1)

        let sdfPadding = Self.collisionMeshSDFPadding(for: mesh)
        let sdfLocalMinBounds = mesh.localBoundsMin - sdfPadding
        let sdfLocalMaxBounds = mesh.localBoundsMax + sdfPadding
        let sdfResolution = Self.collisionMeshSDFResolution(
            localMinBounds: sdfLocalMinBounds,
            localMaxBounds: sdfLocalMaxBounds
        )
        let resourceKey = Self.collisionMeshSDFCacheKey(
            mesh: mesh,
            localBoundsMin: sdfLocalMinBounds,
            localBoundsMax: sdfLocalMaxBounds,
            resolution: sdfResolution
        )

        guard let geometryResource = collisionMeshGeometryResources[resourceKey],
              let sdfResourceIndex = collisionMeshSDFResourceIndicesByKey[resourceKey] else {
            return false
        }

        let worldBounds = Self.transformedBounds(
            localMinBounds: sdfLocalMinBounds,
            localMaxBounds: sdfLocalMaxBounds,
            transform: mesh.transform
        )
        let sdfSize = max(sdfLocalMaxBounds - sdfLocalMinBounds, SIMD3<Float>(repeating: 1.0e-4))
        let sdfVoxelSize = SIMD3<Float>(
            sdfSize.x / Float(max(sdfResolution.x, 1)),
            sdfSize.y / Float(max(sdfResolution.y, 1)),
            sdfSize.z / Float(max(sdfResolution.z, 1))
        )

        let infoPtr = collisionMeshInfoBuffer.contents().bindMemory(
            to: AVBDGPUCollisionMeshInfo.self,
            capacity: max(collisionMeshCapacity, 1)
        )
        infoPtr[collisionMeshCount] = AVBDGPUCollisionMeshInfo(
            vertexOffset: Int32(geometryResource.vertexOffset),
            vertexCount: Int32(geometryResource.vertexCount),
            indexOffset: Int32(geometryResource.indexOffset),
            indexCount: Int32(geometryResource.indexCount),
            ownerBodyIndex: Int32(mesh.ownerBodyIndex),
            sdfResourceIndex: Int32(sdfResourceIndex),
            _reserved0: SIMD2<Int32>(repeating: 0),
            minBounds: SIMD4<Float>(worldBounds.min, 0),
            maxBounds: SIMD4<Float>(worldBounds.max, 0),
            sdfLocalMinBounds: SIMD4<Float>(sdfLocalMinBounds, 0),
            sdfLocalMaxBounds: SIMD4<Float>(sdfLocalMaxBounds, 0),
            sdfVoxelSize: SIMD4<Float>(sdfVoxelSize, 0),
            sdfResolution: SIMD4<Int32>(sdfResolution.x, sdfResolution.y, sdfResolution.z, 0),
            sdfTransform: mesh.transform,
            sdfInvTransform: simd_inverse(mesh.transform)
        )

        collisionMeshCount += 1
        forceFullBroadphase = true
        collisionMeshSDFStatusText = "Cached"
        refreshCollisionMeshOwnerBodyMask()
        return true
    }

    init?(device: MTLDevice, scene: AVBDScene) {
        self.device = device
        self.collisionMeshSDFCommandQueue = device.makeCommandQueue()

        guard let library = device.makeDefaultLibrary() else { return nil }

        func makeFunction(_ name: String) -> MTLFunction? {
            guard let function = library.makeFunction(name: name) else {
                print("Missing compute function: \(name)")
                return nil
            }
            return function
        }

        func makePSO(_ name: String) -> MTLComputePipelineState? {
            guard let function = makeFunction(name) else {
                return nil
            }
            return try? device.makeComputePipelineState(function: function)
        }

        guard let primitiveMeshSDFFunction = makeFunction("avbd_collide_primitive_mesh_sdf") else { return nil }
        self.collisionMeshSDFArgumentEncoder = primitiveMeshSDFFunction.makeArgumentEncoder(bufferIndex: 12)
        self.collisionMeshSDFArgumentBuffer = collisionMeshSDFArgumentEncoder.flatMap {
            device.makeBuffer(length: $0.encodedLength, options: .storageModeShared)
        }

        guard let fwd = makePSO("avbd_forward_integrate"),
              let ra  = makePSO("avbd_reset_adjacency"),
              let bpf = makePSO("avbd_broadphase_full"),
              let bpp = makePSO("avbd_broadphase_partial"),
              let pbi = makePSO("avbd_prepare_broadphase_indirect"),
              let bpm = makePSO("avbd_build_primitive_manifolds"),
              let pai = makePSO("avbd_prepare_active_manifolds_indirect"),
              let pmb = makePSO("avbd_broadphase_primitive_mesh"),
              let pmci = makePSO("avbd_prepare_primitive_mesh_collision_indirect"),
              let pcm = makePSO("avbd_collide_primitive_mesh"),
              let pmsdf = try? device.makeComputePipelineState(function: primitiveMeshSDFFunction),
              let smi = makePSO("avbd_initialize_mesh_sdf"),
              let sma = makePSO("avbd_accumulate_mesh_sdf"),
              let smf = makePSO("avbd_finalize_mesh_sdf"),
              let mmb = makePSO("avbd_broadphase_mesh_mesh"),
              let mmci = makePSO("avbd_prepare_mesh_mesh_collision_indirect"),
              let mcm = makePSO("avbd_collide_mesh_mesh"),
              let bac = makePSO("avbd_build_adjacency_constraints"),
              let bam = makePSO("avbd_build_adjacency_manifolds"),
              let ij  = makePSO("avbd_init_joints"),
              let bs  = makePSO("avbd_body_solve"),
              let duj = makePSO("avbd_dual_update_joints"),
              let dum = makePSO("avbd_dual_update_manifolds"),
              let wsc = makePSO("avbd_warmstart_contacts"),
              let fin = makePSO("avbd_finalize"),
              let wi  = makePSO("avbd_write_instances")
        else { return nil }

        forwardIntegratePSO = fwd
        resetAdjacencyPSO = ra
        broadphaseFullPSO = bpf
        broadphasePartialPSO = bpp
        prepareBroadphaseIndirectPSO = pbi
        buildPrimitiveManifoldsPSO = bpm
        prepareActiveManifoldsIndirectPSO = pai
        primitiveMeshBroadphasePSO = pmb
        preparePrimitiveMeshCollisionIndirectPSO = pmci
        collidePrimitiveMeshPSO = pcm
        collidePrimitiveMeshSDFPSO = pmsdf
        initializeCollisionMeshSDFPSO = smi
        accumulateCollisionMeshSDFPSO = sma
        finalizeCollisionMeshSDFPSO = smf
        meshMeshBroadphasePSO = mmb
        prepareMeshMeshCollisionIndirectPSO = mmci
        collideMeshMeshPSO = mcm
        buildAdjacencyConstraintsPSO = bac
        buildAdjacencyManifoldsPSO = bam
        initJointsPSO = ij
        bodySolvePSO = bs
        dualUpdateJointsPSO = duj
        dualUpdateManifoldsPSO = dum
        warmstartContactsPSO = wsc
        finalizePSO = fin
        writeInstancesPSO = wi

        // Build body data
        bodyCount = scene.bodies.count
        bodyShapes = scene.bodies.map { $0.renderShape }
        computedBodyColorGroups = Array(repeating: 0, count: bodyCount)

        cpuBodies = scene.bodies.map { body in
            let mass = avbdShapeMass(size: body.size, density: body.density, shape: body.renderShape)
            let moment = avbdShapeMoment(size: body.size, mass: mass, shape: body.renderShape)
            return AVBDGPUBody(
                positionLin: body.position,
                positionAng: body.orientation.vector,
                initialLin: body.position,
                initialAng: body.orientation.vector,
                inertialLin: body.position,
                inertialAng: body.orientation.vector,
                velocityLin: body.velocity,
                velocityAng: .zero,
                prevVelocityLin: body.velocity,
                size: body.size,
                mass: mass,
                moment: moment,
                friction: body.friction,
                renderShape: Int32(body.renderShape.rawValue)
            )
        }

        cpuJoints = []
        cpuSprings = []
        cpuManifolds = []
        cpuAdjacency = Array(repeating: AVBDGPUAdjacency(), count: max(bodyCount, 1))

        // Parse constraints
        for constraint in scene.constraints {
            switch constraint {
            case let .joint(bodyA, bodyB, anchorA, anchorB, stiffnessLin, stiffnessAng, fracture):
                let sizeA = scene.bodies[bodyA].size
                let sizeB = scene.bodies[bodyB].size
                let torqueArm = simd_length_squared(sizeA + sizeB)
                var joint = AVBDGPUJoint()
                joint.bodyA = Int32(bodyA)
                joint.bodyB = Int32(bodyB)
                joint.rA = anchorA
                joint.stiffnessLin = stiffnessLin.isInfinite ? 1e30 : stiffnessLin
                joint.rB = anchorB
                joint.stiffnessAng = stiffnessAng.isInfinite ? 1e30 : stiffnessAng
                joint.fracture = fracture == nil || fracture!.isInfinite ? 1e30 : fracture!
                joint.torqueArm = torqueArm
                cpuJoints.append(joint)

            case let .spring(bodyA, bodyB, anchorA, anchorB, stiffness, rest):
                var sp = AVBDGPUSpring()
                sp.bodyA = Int32(bodyA)
                sp.bodyB = Int32(bodyB)
                sp.rA = anchorA
                sp.rB = anchorB
                sp.stiffness = stiffness
                var computedRest = rest
                if computedRest < 0 {
                    let bA = scene.bodies[bodyA]
                    let bB = scene.bodies[bodyB]
                    let pA = bA.orientation.act(anchorA) + bA.position
                    let pB = bB.orientation.act(anchorB) + bB.position
                    computedRest = simd_length(pA - pB)
                }
                sp.rest = computedRest
                cpuSprings.append(sp)

            case let .ignoreCollision(bodyA, bodyB):
                let key = Self.pairKey(bodyA, bodyB)
                ignorePairs.insert(key)
            }
        }

        // Allocate GPU buffers with extra capacity for dynamically thrown bodies
        let extra = Self.extraBodyCapacity
        bodyCapacity = bodyCount + extra
        numColors = 1
        // Cap manifold/contact buffers to max collisions per body instead of n*(n-1)/2
        let maxCollisionsPerBody = Int(AVBD_MAX_COLLISIONS_PER_BODY)
        bodyBodyManifoldCapacity = max(bodyCapacity * maxCollisionsPerBody, 1)
        primitiveMeshManifoldCapacity = 1
        meshMeshManifoldCapacity = 1
        maxCollisions = bodyBodyManifoldCapacity + primitiveMeshManifoldCapacity + meshMeshManifoldCapacity
        recentPairCapacity = max(bodyCapacity * (bodyCapacity - 1) / 2, 1)
        collisionMeshCapacity = 1024
        collisionMeshCount = 0
        primitiveMeshPairCapacity = max(bodyCapacity * collisionMeshCapacity, 1)
        meshMeshPairCapacity = max(collisionMeshCapacity * max(collisionMeshCapacity - 1, 0) / 2, 1)
        meshMeshIsoVoxelTrackedPairCapacity = 1

        let bodyBufSize = max(MemoryLayout<AVBDGPUBody>.stride * bodyCapacity, 16)
        let bodyColorBufSize = max(MemoryLayout<Int32>.stride * bodyCapacity, 16)
        let renderColorBufSize = max(MemoryLayout<SIMD4<Float>>.stride * bodyCapacity, 16)
        let renderColorGroupBufSize = max(MemoryLayout<Int32>.stride * bodyCapacity, 16)
        let jointBufSize = max(MemoryLayout<AVBDGPUJoint>.stride * max(cpuJoints.count, 1), 16)
        let springBufSize = max(MemoryLayout<AVBDGPUSpring>.stride * max(cpuSprings.count, 1), 16)
        let exclusionBufSize = max(MemoryLayout<AVBDGPUCollisionExclusion>.stride * bodyCapacity, 16)
        let manifoldBufSize = max(MemoryLayout<AVBDGPUManifold>.stride * maxCollisions, 16)
        let adjBufSize = max(MemoryLayout<AVBDGPUAdjacency>.stride * bodyCapacity, 16)
        let contactBufSize = max(MemoryLayout<AVBDGPUContact>.stride * maxCollisions * SWIFT_AVBD_MAX_CONTACTS_PER_PAIR, 16)
        let contactAllocatorBufSize = max(MemoryLayout<AVBDGPUContactAllocator>.stride, 16)
        let recentPairIndexBufSize = max(MemoryLayout<Int32>.stride * recentPairCapacity, 16)
        let recentPairStateBufSize = max(MemoryLayout<AVBDGPURecentPairCacheState>.stride, 16)
        let recentPairIndirectBufSize = max(MemoryLayout<AVBDGPUIndirectDispatchArgs>.stride, 16)
        let derivedPairCandidateBufSize = max(MemoryLayout<Int32>.stride * bodyBodyManifoldCapacity, 16)
        let derivedPairStateBufSize = max(MemoryLayout<AVBDGPUDerivedPairCandidateState>.stride, 16)
        let derivedPairIndirectBufSize = max(MemoryLayout<AVBDGPUIndirectDispatchArgs>.stride, 16)
        let activeManifoldIndexBufSize = max(MemoryLayout<Int32>.stride * maxCollisions, 16)
        let activeManifoldStateBufSize = max(MemoryLayout<AVBDGPUActiveManifoldListState>.stride, 16)
        let activeManifoldIndirectBufSize = max(MemoryLayout<AVBDGPUIndirectDispatchArgs>.stride, 16)
        let collisionMeshInfoBufSize = max(MemoryLayout<AVBDGPUCollisionMeshInfo>.stride * collisionMeshCapacity, 16)
        let collisionMeshOwnerBodyMaskBufSize = max(MemoryLayout<Int32>.stride * bodyCapacity, 16)
        let collisionMeshVertexBufSize = max(MemoryLayout<SIMD3<Float>>.stride, 16)
        let collisionMeshIndexBufSize = max(MemoryLayout<UInt32>.stride, 16)
        let primitiveMeshPairBufSize = max(MemoryLayout<AVBDGPUPrimitiveMeshPair>.stride * primitiveMeshPairCapacity, 16)
        let primitiveMeshPairStateBufSize = max(MemoryLayout<AVBDGPUPrimitiveMeshPairListState>.stride, 16)
        let primitiveMeshPairIndirectBufSize = max(MemoryLayout<AVBDGPUIndirectDispatchArgs>.stride, 16)
        let meshMeshPairBufSize = max(MemoryLayout<AVBDGPUMeshMeshPair>.stride * meshMeshPairCapacity, 16)
        let meshMeshPairStateBufSize = max(MemoryLayout<AVBDGPUMeshMeshPairListState>.stride, 16)
        let meshMeshPairIndirectBufSize = max(MemoryLayout<AVBDGPUIndirectDispatchArgs>.stride, 16)
        let meshMeshIsoVoxelDebugBufSize = max(MemoryLayout<AVBDGPUMeshMeshIsoVoxelDebug>.stride * meshMeshIsoVoxelTrackedPairCapacity, 16)
        let meshMeshIsoVoxelCoordBufSize = max(
            MemoryLayout<AVBDGPUMeshMeshIsoVoxelCoord>.stride
                * meshMeshIsoVoxelTrackedPairCapacity
                * gpuMeshMeshIsoVoxelCoordsPerPair,
            16
        )

        guard let bb = device.makeBuffer(length: bodyBufSize, options: .storageModeShared),
              let bcb = device.makeBuffer(length: bodyColorBufSize, options: .storageModeShared),
              let rcb = device.makeBuffer(length: renderColorBufSize, options: .storageModeShared),
              let rcgb = device.makeBuffer(length: renderColorGroupBufSize, options: .storageModeShared),
              let jb = device.makeBuffer(length: jointBufSize, options: .storageModeShared),
              let sb = device.makeBuffer(length: springBufSize, options: .storageModeShared),
              let eb = device.makeBuffer(length: exclusionBufSize, options: .storageModeShared),
              let mb = device.makeBuffer(length: manifoldBufSize, options: .storageModeShared),
              let ab = device.makeBuffer(length: adjBufSize, options: .storageModeShared),
              let ctb = device.makeBuffer(length: contactBufSize, options: .storageModeShared),
              let cab = device.makeBuffer(length: contactAllocatorBufSize, options: .storageModeShared),
              let rpi0 = device.makeBuffer(length: recentPairIndexBufSize, options: .storageModeShared),
              let rpi1 = device.makeBuffer(length: recentPairIndexBufSize, options: .storageModeShared),
              let rps0 = device.makeBuffer(length: recentPairStateBufSize, options: .storageModeShared),
              let rps1 = device.makeBuffer(length: recentPairStateBufSize, options: .storageModeShared),
              let rpd0 = device.makeBuffer(length: recentPairIndirectBufSize, options: .storageModeShared),
              let rpd1 = device.makeBuffer(length: recentPairIndirectBufSize, options: .storageModeShared),
              let dpc = device.makeBuffer(length: derivedPairCandidateBufSize, options: .storageModeShared),
              let dps = device.makeBuffer(length: derivedPairStateBufSize, options: .storageModeShared),
              let dpi = device.makeBuffer(length: derivedPairIndirectBufSize, options: .storageModeShared),
              let ami = device.makeBuffer(length: activeManifoldIndexBufSize, options: .storageModeShared),
              let ams = device.makeBuffer(length: activeManifoldStateBufSize, options: .storageModeShared),
              let amd = device.makeBuffer(length: activeManifoldIndirectBufSize, options: .storageModeShared),
              let pmb = device.makeBuffer(length: manifoldBufSize, options: .storageModeShared),
              let pctb = device.makeBuffer(length: contactBufSize, options: .storageModeShared),
              let pami = device.makeBuffer(length: activeManifoldIndexBufSize, options: .storageModeShared),
              let pams = device.makeBuffer(length: activeManifoldStateBufSize, options: .storageModeShared),
              let cmi = device.makeBuffer(length: collisionMeshInfoBufSize, options: .storageModeShared),
              let cmob = device.makeBuffer(length: collisionMeshOwnerBodyMaskBufSize, options: .storageModeShared),
              let cmv = device.makeBuffer(length: collisionMeshVertexBufSize, options: .storageModeShared),
              let cmii = device.makeBuffer(length: collisionMeshIndexBufSize, options: .storageModeShared),
              let pmp = device.makeBuffer(length: primitiveMeshPairBufSize, options: .storageModeShared),
              let pmps = device.makeBuffer(length: primitiveMeshPairStateBufSize, options: .storageModeShared),
              let pmpi = device.makeBuffer(length: primitiveMeshPairIndirectBufSize, options: .storageModeShared),
              let mmp = device.makeBuffer(length: meshMeshPairBufSize, options: .storageModeShared),
              let mmps = device.makeBuffer(length: meshMeshPairStateBufSize, options: .storageModeShared),
              let mmpi = device.makeBuffer(length: meshMeshPairIndirectBufSize, options: .storageModeShared),
              let mmiv = device.makeBuffer(length: meshMeshIsoVoxelDebugBufSize, options: .storageModeShared),
              let mmic = device.makeBuffer(length: meshMeshIsoVoxelCoordBufSize, options: .storageModeShared),
              let pb = device.makeBuffer(length: MemoryLayout<AVBDGPUSolverParams>.stride, options: .storageModeShared)
        else { return nil }

        bodyBuffer = bb
        bodyColorBuffer = bcb
        renderColorBuffer = rcb
        renderColorGroupBuffer = rcgb
        jointBuffer = jb
        springBuffer = sb
        exclusionBuffer = eb
        manifoldBuffer = mb
        adjacencyBuffer = ab
        contactBuffer = ctb
        contactAllocatorBuffer = cab
        recentPairIndexBuffers = [rpi0, rpi1]
        recentPairStateBuffers = [rps0, rps1]
        recentPairIndirectBuffers = [rpd0, rpd1]
        derivedPairCandidateBuffer = dpc
        derivedPairStateBuffer = dps
        derivedPairIndirectBuffer = dpi
        activeManifoldIndexBuffer = ami
        activeManifoldStateBuffer = ams
        activeManifoldIndirectBuffer = amd
        prevManifoldBuffer = pmb
        prevContactBuffer = pctb
        prevActiveManifoldIndexBuffer = pami
        prevActiveManifoldStateBuffer = pams
        collisionMeshInfoBuffer = cmi
        collisionMeshOwnerBodyMaskBuffer = cmob
        collisionMeshVertexBuffer = cmv
        collisionMeshIndexBuffer = cmii
        primitiveMeshPairBuffer = pmp
        primitiveMeshPairStateBuffer = pmps
        primitiveMeshPairIndirectBuffer = pmpi
        meshMeshPairBuffer = mmp
        meshMeshPairStateBuffer = mmps
        meshMeshPairIndirectBuffer = mmpi
        meshMeshIsoVoxelDebugBuffer = mmiv
        meshMeshIsoVoxelCoordBuffer = mmic
        paramsBuffer = pb
        contactCapacity = maxCollisions * SWIFT_AVBD_MAX_CONTACTS_PER_PAIR
        recentPairCacheSlot = 0
        stepsSinceFullBroadphase = .max
        forceFullBroadphase = true

        let allocatorPtr = contactAllocatorBuffer.contents().bindMemory(to: AVBDGPUContactAllocator.self, capacity: 1)
        allocatorPtr.pointee = AVBDGPUContactAllocator(nextContactIndex: Int32(0), contactCapacity: Int32(contactCapacity))

        for stateBuffer in recentPairStateBuffers {
            let statePtr = stateBuffer.contents().bindMemory(to: AVBDGPURecentPairCacheState.self, capacity: 1)
            statePtr.pointee = AVBDGPURecentPairCacheState(count: Int32(0), capacity: Int32(recentPairCapacity))
        }

        for indirectBuffer in recentPairIndirectBuffers {
            let indirectPtr = indirectBuffer.contents().bindMemory(to: AVBDGPUIndirectDispatchArgs.self, capacity: 1)
            indirectPtr.pointee = AVBDGPUIndirectDispatchArgs(threadgroupsPerGrid: (0, 1, 1))
        }

        let derivedPairStatePtr = derivedPairStateBuffer.contents().bindMemory(to: AVBDGPUDerivedPairCandidateState.self, capacity: 1)
        derivedPairStatePtr.pointee = AVBDGPUDerivedPairCandidateState(count: Int32(0), capacity: Int32(bodyBodyManifoldCapacity))

        let derivedPairIndirectPtr = derivedPairIndirectBuffer.contents().bindMemory(to: AVBDGPUIndirectDispatchArgs.self, capacity: 1)
        derivedPairIndirectPtr.pointee = AVBDGPUIndirectDispatchArgs(threadgroupsPerGrid: (0, 1, 1))

        let activeManifoldStatePtr = activeManifoldStateBuffer.contents().bindMemory(to: AVBDGPUActiveManifoldListState.self, capacity: 1)
        activeManifoldStatePtr.pointee = AVBDGPUActiveManifoldListState(count: Int32(0), capacity: Int32(maxCollisions))

        let prevActiveManifoldStatePtr = prevActiveManifoldStateBuffer.contents().bindMemory(to: AVBDGPUActiveManifoldListState.self, capacity: 1)
        prevActiveManifoldStatePtr.pointee = AVBDGPUActiveManifoldListState(count: Int32(0), capacity: Int32(maxCollisions))

        let activeManifoldIndirectPtr = activeManifoldIndirectBuffer.contents().bindMemory(to: AVBDGPUIndirectDispatchArgs.self, capacity: 1)
        activeManifoldIndirectPtr.pointee = AVBDGPUIndirectDispatchArgs(threadgroupsPerGrid: (0, 1, 1))

        let primitiveMeshPairStatePtr = primitiveMeshPairStateBuffer.contents().bindMemory(to: AVBDGPUPrimitiveMeshPairListState.self, capacity: 1)
        primitiveMeshPairStatePtr.pointee = AVBDGPUPrimitiveMeshPairListState(count: Int32(0), capacity: Int32(primitiveMeshPairCapacity))
        collisionMeshOwnerBodyMaskBuffer.contents().initializeMemory(
            as: UInt8.self,
            repeating: 0,
            count: collisionMeshOwnerBodyMaskBuffer.length
        )

        let primitiveMeshPairIndirectPtr = primitiveMeshPairIndirectBuffer.contents().bindMemory(to: AVBDGPUIndirectDispatchArgs.self, capacity: 1)
        primitiveMeshPairIndirectPtr.pointee = AVBDGPUIndirectDispatchArgs(threadgroupsPerGrid: (0, 1, 1))

        let meshMeshPairStatePtr = meshMeshPairStateBuffer.contents().bindMemory(to: AVBDGPUMeshMeshPairListState.self, capacity: 1)
        meshMeshPairStatePtr.pointee = AVBDGPUMeshMeshPairListState(
            count: Int32(0),
            capacity: Int32(meshMeshPairCapacity)
        )

        let meshMeshPairIndirectPtr = meshMeshPairIndirectBuffer.contents().bindMemory(to: AVBDGPUIndirectDispatchArgs.self, capacity: 1)
        meshMeshPairIndirectPtr.pointee = AVBDGPUIndirectDispatchArgs(threadgroupsPerGrid: (0, 1, 1))
        meshMeshIsoVoxelDebugBuffer.contents().initializeMemory(
            as: UInt8.self,
            repeating: 0,
            count: meshMeshIsoVoxelDebugBuffer.length
        )
        meshMeshIsoVoxelCoordBuffer.contents().initializeMemory(
            as: UInt8.self,
            repeating: 0,
            count: meshMeshIsoVoxelCoordBuffer.length
        )

        uploadBodies()
        uploadRenderColors(scene.bodies.map(\.renderColor))
        uploadRenderColorGroups(scene.bodies.map(\.colorGroup))
        uploadJoints()
        uploadSprings()
        computeAndUploadExclusions()
        computeAndUploadBodyColors()
    }

    // MARK: - Dynamic Body Addition

    /// Add a new body at runtime (e.g., for throwing objects into the scene).
    /// Returns the index of the newly added body, or nil if capacity is exceeded.
    func addBody(
        position: SIMD3<Float>,
        velocity: SIMD3<Float>,
        size: SIMD3<Float>,
        density: Float,
        friction: Float,
        renderColor: SIMD4<Float>? = nil,
        colorGroup: Int? = nil,
        renderShape: AVBDRenderShape = .box
    ) -> Int? {
        guard bodyCount < bodyCapacity else { return nil }

        let mass = avbdShapeMass(size: size, density: density, shape: renderShape)
        let moment = avbdShapeMoment(size: size, mass: mass, shape: renderShape)
        let newBody = AVBDGPUBody(
            positionLin: position,
            positionAng: SIMD4<Float>(0, 0, 0, 1),
            initialLin: position,
            initialAng: SIMD4<Float>(0, 0, 0, 1),
            inertialLin: position,
            inertialAng: SIMD4<Float>(0, 0, 0, 1),
            velocityLin: velocity,
            velocityAng: .zero,
            prevVelocityLin: velocity,
            size: size, mass: mass,
            moment: moment, friction: friction,
            renderShape: Int32(renderShape.rawValue)
        )

        let newIndex = bodyCount
        cpuBodies.append(newBody)
        bodyShapes.append(renderShape)
        computedBodyColorGroups.append(0)

        // Write body to GPU buffer
        let bodyPtr = bodyBuffer.contents().bindMemory(to: AVBDGPUBody.self, capacity: bodyCapacity)
        bodyPtr[newIndex] = newBody

        let renderColorPtr = renderColorBuffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: bodyCapacity)
        renderColorPtr[newIndex] = encodeRenderColor(renderColor)

        let renderColorGroupPtr = renderColorGroupBuffer.contents().bindMemory(to: Int32.self, capacity: bodyCapacity)
        renderColorGroupPtr[newIndex] = encodeColorGroup(colorGroup)

        // Initialize exclusion list for new body (no joints/springs, so empty)
        let exclPtr = exclusionBuffer.contents().bindMemory(to: AVBDGPUCollisionExclusion.self, capacity: bodyCapacity)
        exclPtr[newIndex].excludeCount = 0

        bodyCount += 1

        // New body has no joints/springs, assign color 0 (no constraint neighbors)
        let colorPtr = bodyColorBuffer.contents().bindMemory(to: Int32.self, capacity: bodyCapacity)
        colorPtr[newIndex] = 0

        forceFullBroadphase = true
        return newIndex
    }

    func addIgnoredCollisionPair(bodyA: Int, bodyB: Int) {
        addIgnoredCollisionPairs([(bodyA, bodyB)])
    }

    func isDynamicBody(_ bodyIndex: Int) -> Bool {
        guard bodyIndex >= 0, bodyIndex < bodyCount else {
            return false
        }
        return cpuBodies[bodyIndex].mass > 0
    }

    func addIgnoredCollisionPairs<S: Sequence>(_ pairs: S) where S.Element == (Int, Int) {
        var insertedAny = false
        for (bodyA, bodyB) in pairs {
            guard bodyA >= 0, bodyB >= 0, bodyA < bodyCount, bodyB < bodyCount, bodyA != bodyB else {
                continue
            }
            let lower = min(bodyA, bodyB)
            let upper = max(bodyA, bodyB)
            let key = (UInt64(upper) << 32) | UInt64(lower)
            if ignorePairs.insert(key).inserted {
                insertedAny = true
            }
        }

        guard insertedAny else {
            return
        }
        computeAndUploadExclusions()
        forceFullBroadphase = true
    }

    func writeRenderInstances(
        boxInstances: UnsafeMutablePointer<InstanceUniforms>,
        sphereInstances: UnsafeMutablePointer<InstanceUniforms>,
        torusInstances: UnsafeMutablePointer<InstanceUniforms>,
        torusVisualMode: AVBDTorusVisualMode
    ) -> (boxCount: Int, sphereCount: Int, torusCount: Int) {
        downloadBodies()

        var boxCount = 0
        var sphereCount = 0
        var torusCount = 0

        func makeRigidInstance(bodyIndex: Int, body: AVBDGPUBody, shape: AVBDRenderShape) -> InstanceUniforms {
            let modelMatrix = simd_mul(
                matrix4x4_translation(body.positionLin.x, body.positionLin.y, body.positionLin.z),
                simd_mul(
                    matrix4x4_rotation(quaternion: simd_quatf(vector: body.positionAng)),
                    matrix4x4_scale(body.size.x, body.size.y, body.size.z)
                )
            )
            return InstanceUniforms(
                modelMatrix: modelMatrix,
                renderColor: resolvedRenderColor(bodyIndex: bodyIndex),
                shapeParams: SIMD4<Float>(Float(shape.rawValue), 0, 0, 0)
            )
        }

        for i in 0..<bodyCount {
            let body = cpuBodies[i]
            switch bodyShapes[i] {
            case .box:
                boxInstances[boxCount] = makeRigidInstance(bodyIndex: i, body: body, shape: .box)
                boxCount += 1
            case .sphere:
                sphereInstances[sphereCount] = makeRigidInstance(bodyIndex: i, body: body, shape: .sphere)
                sphereCount += 1
            case .torus:
                if torusVisualMode == .solidTorus {
                    let orientation = simd_quatf(vector: body.positionAng)
                    let modelMatrix = simd_mul(
                        matrix4x4_translation(body.positionLin.x, body.positionLin.y, body.positionLin.z),
                        matrix4x4_rotation(quaternion: orientation)
                    )
                    torusInstances[torusCount] = InstanceUniforms(
                        modelMatrix: modelMatrix,
                        renderColor: resolvedRenderColor(bodyIndex: i),
                        shapeParams: SIMD4<Float>(
                            Float(AVBDRenderShape.torus.rawValue),
                            avbdTorusMajorRadius(size: body.size),
                            avbdTorusRenderMinorRadius(size: body.size),
                            0
                        )
                    )
                    torusCount += 1
                } else {
                    let orientation = simd_quatf(vector: body.positionAng)
                    let torusSphereCount = avbdCurrentTorusApproxSphereCount()
                    let sphereDiameter = avbdTorusApproxSphereRadius(size: body.size) * 2.0
                    for torusSphereIndex in 0..<torusSphereCount {
                        let center = orientation.act(
                            avbdTorusApproxSphereLocalCenter(size: body.size, index: torusSphereIndex)
                        ) + body.positionLin
                        let modelMatrix = simd_mul(
                            matrix4x4_translation(center.x, center.y, center.z),
                            matrix4x4_scale(sphereDiameter, sphereDiameter, sphereDiameter)
                        )
                        sphereInstances[sphereCount] = InstanceUniforms(
                            modelMatrix: modelMatrix,
                            renderColor: resolvedRenderColor(bodyIndex: i),
                            shapeParams: SIMD4<Float>(Float(AVBDRenderShape.sphere.rawValue), 0, 0, 0)
                        )
                        sphereCount += 1
                    }
                }
            }
        }

        return (boxCount, sphereCount, torusCount)
    }

    // MARK: - Step

    func step(commandBuffer: MTLCommandBuffer, instanceBuffer: MTLBuffer, instanceOffset: Int) {
        uploadParams()

        let primitiveMeshPairStatePtr = primitiveMeshPairStateBuffer.contents().bindMemory(to: AVBDGPUPrimitiveMeshPairListState.self, capacity: 1)
        primitiveMeshPairStatePtr.pointee = AVBDGPUPrimitiveMeshPairListState(
            count: Int32(0),
            capacity: Int32(primitiveMeshPairCapacity)
        )

        let primitiveMeshPairIndirectPtr = primitiveMeshPairIndirectBuffer.contents().bindMemory(to: AVBDGPUIndirectDispatchArgs.self, capacity: 1)
        primitiveMeshPairIndirectPtr.pointee = AVBDGPUIndirectDispatchArgs(threadgroupsPerGrid: (0, 1, 1))

        let meshMeshPairStatePtr = meshMeshPairStateBuffer.contents().bindMemory(to: AVBDGPUMeshMeshPairListState.self, capacity: 1)
        meshMeshPairStatePtr.pointee = AVBDGPUMeshMeshPairListState(
            count: Int32(0),
            capacity: Int32(meshMeshPairCapacity)
        )

        let meshMeshPairIndirectPtr = meshMeshPairIndirectBuffer.contents().bindMemory(to: AVBDGPUIndirectDispatchArgs.self, capacity: 1)
        meshMeshPairIndirectPtr.pointee = AVBDGPUIndirectDispatchArgs(threadgroupsPerGrid: (0, 1, 1))

        // Blit previous frame's manifold/contact data for warmstarting
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.copy(from: manifoldBuffer, sourceOffset: 0, to: prevManifoldBuffer, destinationOffset: 0, size: manifoldBuffer.length)
            blit.copy(from: contactBuffer, sourceOffset: 0, to: prevContactBuffer, destinationOffset: 0, size: contactBuffer.length)
            blit.copy(from: activeManifoldIndexBuffer, sourceOffset: 0, to: prevActiveManifoldIndexBuffer, destinationOffset: 0, size: activeManifoldIndexBuffer.length)
            blit.copy(from: activeManifoldStateBuffer, sourceOffset: 0, to: prevActiveManifoldStateBuffer, destinationOffset: 0, size: activeManifoldStateBuffer.length)
            blit.endEncoding()
        }

        // GPU compute passes
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        let bodyThreads = MTLSize(width: max(bodyCount, 1), height: 1, depth: 1)
        let jointThreads = MTLSize(width: max(cpuJoints.count, 1), height: 1, depth: 1)
        let totalPairCount = bodyCount * (bodyCount - 1) / 2
        let broadphaseWidth = broadphaseFullPSO.threadExecutionWidth
        let broadphaseFullThreadgroupCount = MTLSize(
            width: max((max(totalPairCount, 1) + broadphaseWidth - 1) / broadphaseWidth, 1),
            height: 1,
            depth: 1
        )
        let adjacencyConstraintThreads = MTLSize(width: max(cpuJoints.count, cpuSprings.count, 1), height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: broadphaseWidth, height: 1, depth: 1)
        let derivedThreadgroupSize = MTLSize(width: buildPrimitiveManifoldsPSO.threadExecutionWidth, height: 1, depth: 1)
        let primitiveMeshBroadphasePairCount = max(bodyCount * collisionMeshCount, 1)
        let primitiveMeshBroadphaseWidth = primitiveMeshBroadphasePSO.threadExecutionWidth
        let primitiveMeshBroadphaseThreads = MTLSize(width: primitiveMeshBroadphasePairCount, height: 1, depth: 1)
        let primitiveMeshCollisionThreadgroupSize = MTLSize(
            width: max(collidePrimitiveMeshPSO.threadExecutionWidth, collidePrimitiveMeshSDFPSO.threadExecutionWidth),
            height: 1,
            depth: 1
        )
        let meshMeshBroadphasePairCount = max(collisionMeshCount * max(collisionMeshCount - 1, 0) / 2, 1)
        let meshMeshBroadphaseWidth = meshMeshBroadphasePSO.threadExecutionWidth
        let meshMeshBroadphaseThreads = MTLSize(width: meshMeshBroadphasePairCount, height: 1, depth: 1)
        let meshMeshCollisionThreadgroupSize = MTLSize(width: collideMeshMeshPSO.threadExecutionWidth, height: 1, depth: 1)
        let currentRecentPairCount = Int(currentRecentPairStateBuffer.contents().bindMemory(to: AVBDGPURecentPairCacheState.self, capacity: 1).pointee.count)
        let shouldRunFullBroadphase = broadphaseFullRefreshStepCount <= 0
            || forceFullBroadphase
            || currentRecentPairCount == 0
            || stepsSinceFullBroadphase >= broadphaseFullRefreshStepCount
        lastBroadphaseUsedCache = broadphaseFullRefreshStepCount > 0 && !shouldRunFullBroadphase

        // Reset and rebuild adjacency entirely on GPU for current active contacts.
        encoder.setComputePipelineState(resetAdjacencyPSO)
        encoder.setBuffer(adjacencyBuffer, offset: 0, index: 0)
        encoder.setBuffer(contactAllocatorBuffer, offset: 0, index: 1)
        encoder.setBuffer(nextRecentPairStateBuffer, offset: 0, index: 2)
        encoder.setBuffer(activeManifoldStateBuffer, offset: 0, index: 3)
        encoder.setBuffer(derivedPairStateBuffer, offset: 0, index: 4)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 5)
        encoder.dispatchThreads(bodyThreads, threadsPerThreadgroup: threadgroupSize)

        if bodyCount > 1 {
            if shouldRunFullBroadphase {
                encoder.setComputePipelineState(broadphaseFullPSO)
                encoder.setBuffer(bodyBuffer, offset: 0, index: 0)
                encoder.setBuffer(exclusionBuffer, offset: 0, index: 1)
                encoder.setBuffer(nextRecentPairIndexBuffer, offset: 0, index: 2)
                encoder.setBuffer(nextRecentPairStateBuffer, offset: 0, index: 3)
                encoder.setBuffer(derivedPairCandidateBuffer, offset: 0, index: 4)
                encoder.setBuffer(derivedPairStateBuffer, offset: 0, index: 5)
                encoder.setBuffer(paramsBuffer, offset: 0, index: 6)
                encoder.dispatchThreadgroups(broadphaseFullThreadgroupCount, threadsPerThreadgroup: threadgroupSize)
            } else {
                encoder.setComputePipelineState(broadphasePartialPSO)
                encoder.setBuffer(bodyBuffer, offset: 0, index: 0)
                encoder.setBuffer(currentRecentPairIndexBuffer, offset: 0, index: 1)
                encoder.setBuffer(currentRecentPairStateBuffer, offset: 0, index: 2)
                encoder.setBuffer(nextRecentPairIndexBuffer, offset: 0, index: 3)
                encoder.setBuffer(nextRecentPairStateBuffer, offset: 0, index: 4)
                encoder.setBuffer(derivedPairCandidateBuffer, offset: 0, index: 5)
                encoder.setBuffer(derivedPairStateBuffer, offset: 0, index: 6)
                encoder.setBuffer(paramsBuffer, offset: 0, index: 7)
                encoder.dispatchThreadgroups(
                    indirectBuffer: currentRecentPairIndirectBuffer,
                    indirectBufferOffset: 0,
                    threadsPerThreadgroup: threadgroupSize
                )
            }
        }


        encoder.setComputePipelineState(prepareBroadphaseIndirectPSO)
        encoder.setBuffer(nextRecentPairStateBuffer, offset: 0, index: 0)
        encoder.setBuffer(nextRecentPairIndirectBuffer, offset: 0, index: 1)
        encoder.setBuffer(derivedPairStateBuffer, offset: 0, index: 2)
        encoder.setBuffer(derivedPairIndirectBuffer, offset: 0, index: 3)
        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

        encoder.setComputePipelineState(buildPrimitiveManifoldsPSO)
        encoder.setBuffer(bodyBuffer, offset: 0, index: 0)
        encoder.setBuffer(derivedPairCandidateBuffer, offset: 0, index: 1)
        encoder.setBuffer(derivedPairStateBuffer, offset: 0, index: 2)
        encoder.setBuffer(manifoldBuffer, offset: 0, index: 3)
        encoder.setBuffer(contactBuffer, offset: 0, index: 4)
        encoder.setBuffer(contactAllocatorBuffer, offset: 0, index: 5)
        encoder.setBuffer(activeManifoldIndexBuffer, offset: 0, index: 6)
        encoder.setBuffer(activeManifoldStateBuffer, offset: 0, index: 7)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 8)
        encoder.dispatchThreadgroups(
            indirectBuffer: derivedPairIndirectBuffer,
            indirectBufferOffset: 0,
            threadsPerThreadgroup: derivedThreadgroupSize
        )

        if enablePrimitiveMeshCollisions && collisionMeshCount > 0 && bodyCount > 0 {
            encoder.setComputePipelineState(primitiveMeshBroadphasePSO)
            encoder.setBuffer(bodyBuffer, offset: 0, index: 0)
            encoder.setBuffer(collisionMeshInfoBuffer, offset: 0, index: 1)
            encoder.setBuffer(primitiveMeshPairBuffer, offset: 0, index: 2)
            encoder.setBuffer(primitiveMeshPairStateBuffer, offset: 0, index: 3)
            encoder.setBuffer(collisionMeshOwnerBodyMaskBuffer, offset: 0, index: 4)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 5)
            encoder.dispatchThreads(
                primitiveMeshBroadphaseThreads,
                threadsPerThreadgroup: MTLSize(width: primitiveMeshBroadphaseWidth, height: 1, depth: 1)
            )

            encoder.setComputePipelineState(preparePrimitiveMeshCollisionIndirectPSO)
            encoder.setBuffer(primitiveMeshPairStateBuffer, offset: 0, index: 0)
            encoder.setBuffer(primitiveMeshPairIndirectBuffer, offset: 0, index: 1)
            encoder.dispatchThreads(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
            )

            let usePrimitiveMeshSDF = collisionMeshSDFArgumentBuffer != nil
                && !collisionMeshSDFResources.isEmpty
            encoder.setComputePipelineState(
                usePrimitiveMeshSDF
                    ? collidePrimitiveMeshSDFPSO
                    : collidePrimitiveMeshPSO
            )
            encoder.setBuffer(bodyBuffer, offset: 0, index: 0)
            encoder.setBuffer(collisionMeshInfoBuffer, offset: 0, index: 1)
            encoder.setBuffer(primitiveMeshPairBuffer, offset: 0, index: 2)
            encoder.setBuffer(primitiveMeshPairStateBuffer, offset: 0, index: 3)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 4)
            encoder.setBuffer(collisionMeshVertexBuffer, offset: 0, index: 5)
            encoder.setBuffer(collisionMeshIndexBuffer, offset: 0, index: 6)
            encoder.setBuffer(manifoldBuffer, offset: 0, index: 7)
            encoder.setBuffer(contactBuffer, offset: 0, index: 8)
            encoder.setBuffer(contactAllocatorBuffer, offset: 0, index: 9)
            encoder.setBuffer(activeManifoldIndexBuffer, offset: 0, index: 10)
            encoder.setBuffer(activeManifoldStateBuffer, offset: 0, index: 11)
            if usePrimitiveMeshSDF,
               let collisionMeshSDFArgumentBuffer {
                encoder.setBuffer(collisionMeshSDFArgumentBuffer, offset: 0, index: 12)
                for resource in collisionMeshSDFResources {
                    encoder.useResource(resource.coarseTexture, usage: .read)
                    encoder.useResource(resource.atlasTexture, usage: .read)
                    encoder.useResource(resource.indirectionTexture, usage: .read)
                }
            }
            encoder.dispatchThreadgroups(
                indirectBuffer: primitiveMeshPairIndirectBuffer,
                indirectBufferOffset: 0,
                threadsPerThreadgroup: primitiveMeshCollisionThreadgroupSize
            )
        }

        if enableMeshMeshCollisions && collisionMeshCount > 1 {
            encoder.setComputePipelineState(meshMeshBroadphasePSO)
            encoder.setBuffer(collisionMeshInfoBuffer, offset: 0, index: 0)
            encoder.setBuffer(meshMeshPairBuffer, offset: 0, index: 1)
            encoder.setBuffer(meshMeshPairStateBuffer, offset: 0, index: 2)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 3)
            encoder.dispatchThreads(
                meshMeshBroadphaseThreads,
                threadsPerThreadgroup: MTLSize(width: meshMeshBroadphaseWidth, height: 1, depth: 1)
            )

            encoder.setComputePipelineState(prepareMeshMeshCollisionIndirectPSO)
            encoder.setBuffer(meshMeshPairStateBuffer, offset: 0, index: 0)
            encoder.setBuffer(meshMeshPairIndirectBuffer, offset: 0, index: 1)
            encoder.dispatchThreads(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
            )

            let useMeshMeshSDF = collisionMeshSDFArgumentBuffer != nil
                && !collisionMeshSDFResources.isEmpty
            if useMeshMeshSDF {
                encoder.setComputePipelineState(collideMeshMeshPSO)
                encoder.setBuffer(bodyBuffer, offset: 0, index: 0)
                encoder.setBuffer(collisionMeshInfoBuffer, offset: 0, index: 1)
                encoder.setBuffer(meshMeshPairBuffer, offset: 0, index: 2)
                encoder.setBuffer(meshMeshPairStateBuffer, offset: 0, index: 3)
                encoder.setBuffer(paramsBuffer, offset: 0, index: 4)
                encoder.setBuffer(manifoldBuffer, offset: 0, index: 5)
                encoder.setBuffer(contactBuffer, offset: 0, index: 6)
                encoder.setBuffer(contactAllocatorBuffer, offset: 0, index: 7)
                encoder.setBuffer(activeManifoldIndexBuffer, offset: 0, index: 8)
                encoder.setBuffer(activeManifoldStateBuffer, offset: 0, index: 9)
                if let collisionMeshSDFArgumentBuffer {
                    encoder.setBuffer(collisionMeshSDFArgumentBuffer, offset: 0, index: 10)
                }
                encoder.setBuffer(meshMeshIsoVoxelDebugBuffer, offset: 0, index: 11)
                encoder.setBuffer(meshMeshIsoVoxelCoordBuffer, offset: 0, index: 12)
                for resource in collisionMeshSDFResources {
                    encoder.useResource(resource.coarseTexture, usage: .read)
                    encoder.useResource(resource.atlasTexture, usage: .read)
                    encoder.useResource(resource.indirectionTexture, usage: .read)
                }
                encoder.dispatchThreadgroups(
                    indirectBuffer: meshMeshPairIndirectBuffer,
                    indirectBufferOffset: 0,
                    threadsPerThreadgroup: meshMeshCollisionThreadgroupSize
                )
            }
        }

        // Prepare active manifold indirect dispatch args AFTER all manifold
        // generation (body-body, primitive-mesh, mesh-mesh) is complete so that
        // the count reflects every active manifold.
        encoder.setComputePipelineState(prepareActiveManifoldsIndirectPSO)
        encoder.setBuffer(activeManifoldStateBuffer, offset: 0, index: 0)
        encoder.setBuffer(activeManifoldIndirectBuffer, offset: 0, index: 1)
        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

        // Warmstart: match new contacts with previous frame and copy penalty/lambda
        if enableContactWarmstart && bodyCount > 1 {
            encoder.setComputePipelineState(warmstartContactsPSO)
            encoder.setBuffer(bodyBuffer, offset: 0, index: 0)
            encoder.setBuffer(manifoldBuffer, offset: 0, index: 1)
            encoder.setBuffer(contactBuffer, offset: 0, index: 2)
            encoder.setBuffer(activeManifoldIndexBuffer, offset: 0, index: 3)
            encoder.setBuffer(activeManifoldStateBuffer, offset: 0, index: 4)
            encoder.setBuffer(prevManifoldBuffer, offset: 0, index: 5)
            encoder.setBuffer(prevContactBuffer, offset: 0, index: 6)
            encoder.setBuffer(prevActiveManifoldIndexBuffer, offset: 0, index: 7)
            encoder.setBuffer(prevActiveManifoldStateBuffer, offset: 0, index: 8)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 9)
            encoder.dispatchThreadgroups(
                indirectBuffer: activeManifoldIndirectBuffer,
                indirectBufferOffset: 0,
                threadsPerThreadgroup: threadgroupSize
            )
        }


        encoder.setComputePipelineState(buildAdjacencyConstraintsPSO)
        encoder.setBuffer(jointBuffer, offset: 0, index: 0)
        encoder.setBuffer(springBuffer, offset: 0, index: 1)
        encoder.setBuffer(adjacencyBuffer, offset: 0, index: 2)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 3)
        encoder.dispatchThreads(adjacencyConstraintThreads, threadsPerThreadgroup: threadgroupSize)

        encoder.setComputePipelineState(buildAdjacencyManifoldsPSO)
        encoder.setBuffer(manifoldBuffer, offset: 0, index: 0)
        encoder.setBuffer(activeManifoldIndexBuffer, offset: 0, index: 1)
        encoder.setBuffer(activeManifoldStateBuffer, offset: 0, index: 2)
        encoder.setBuffer(adjacencyBuffer, offset: 0, index: 3)
        encoder.dispatchThreadgroups(
            indirectBuffer: activeManifoldIndirectBuffer,
            indirectBufferOffset: 0,
            threadsPerThreadgroup: threadgroupSize
        )

        // Forward integrate
        encoder.setComputePipelineState(forwardIntegratePSO)
        encoder.setBuffer(bodyBuffer, offset: 0, index: 0)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 1)
        encoder.dispatchThreads(bodyThreads, threadsPerThreadgroup: threadgroupSize)

        // Init joints
        if cpuJoints.count > 0 {
            encoder.setComputePipelineState(initJointsPSO)
            encoder.setBuffer(bodyBuffer, offset: 0, index: 0)
            encoder.setBuffer(jointBuffer, offset: 0, index: 1)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 2)
            encoder.dispatchThreads(jointThreads, threadsPerThreadgroup: threadgroupSize)
        }

        // Iteration loop
        for _ in 0..<iterations {
            // Body solve — dispatch once per color for Gauss-Seidel-like propagation
            for color in 0..<numColors {
                var currentColor = Int32(color)
                encoder.setComputePipelineState(bodySolvePSO)
                encoder.setBuffer(bodyBuffer, offset: 0, index: 0)
                encoder.setBuffer(jointBuffer, offset: 0, index: 1)
                encoder.setBuffer(springBuffer, offset: 0, index: 2)
                encoder.setBuffer(manifoldBuffer, offset: 0, index: 3)
                encoder.setBuffer(adjacencyBuffer, offset: 0, index: 4)
                encoder.setBuffer(contactBuffer, offset: 0, index: 5)
                encoder.setBuffer(paramsBuffer, offset: 0, index: 6)
                encoder.setBuffer(bodyColorBuffer, offset: 0, index: 7)
                encoder.setBytes(&currentColor, length: MemoryLayout<Int32>.stride, index: 8)
                encoder.dispatchThreads(bodyThreads, threadsPerThreadgroup: threadgroupSize)
            }

            // Dual update joints
            if cpuJoints.count > 0 {
                encoder.setComputePipelineState(dualUpdateJointsPSO)
                encoder.setBuffer(bodyBuffer, offset: 0, index: 0)
                encoder.setBuffer(jointBuffer, offset: 0, index: 1)
                encoder.setBuffer(paramsBuffer, offset: 0, index: 2)
                encoder.dispatchThreads(jointThreads, threadsPerThreadgroup: threadgroupSize)
            }

            // Dual update manifolds
            if bodyCount > 1 {
                encoder.setComputePipelineState(dualUpdateManifoldsPSO)
                encoder.setBuffer(bodyBuffer, offset: 0, index: 0)
                encoder.setBuffer(manifoldBuffer, offset: 0, index: 1)
                encoder.setBuffer(activeManifoldIndexBuffer, offset: 0, index: 2)
                encoder.setBuffer(activeManifoldStateBuffer, offset: 0, index: 3)
                encoder.setBuffer(contactBuffer, offset: 0, index: 4)
                encoder.setBuffer(paramsBuffer, offset: 0, index: 5)
                encoder.dispatchThreadgroups(
                    indirectBuffer: activeManifoldIndirectBuffer,
                    indirectBufferOffset: 0,
                    threadsPerThreadgroup: threadgroupSize
                )
            }
        }

        // Finalize velocities
        encoder.setComputePipelineState(finalizePSO)
        encoder.setBuffer(bodyBuffer, offset: 0, index: 0)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 1)
        encoder.dispatchThreads(bodyThreads, threadsPerThreadgroup: threadgroupSize)

        // Write instance data for rendering
        encoder.setComputePipelineState(writeInstancesPSO)
        encoder.setBuffer(bodyBuffer, offset: 0, index: 0)
        encoder.setBuffer(instanceBuffer, offset: instanceOffset, index: 1)
        encoder.setBuffer(renderColorBuffer, offset: 0, index: 2)
        encoder.setBuffer(renderColorGroupBuffer, offset: 0, index: 3)
        encoder.setBuffer(bodyColorBuffer, offset: 0, index: 4)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 5)
        encoder.dispatchThreads(bodyThreads, threadsPerThreadgroup: threadgroupSize)

        encoder.endEncoding()

        recentPairCacheSlot = 1 - recentPairCacheSlot
        stepsSinceFullBroadphase = shouldRunFullBroadphase ? 0 : (stepsSinceFullBroadphase + 1)
        forceFullBroadphase = false
    }

    // MARK: - Body State Access

    /// Read current body positions back from the GPU buffer.
    func readBodyPositions() -> [SIMD3<Float>] {
        let ptr = bodyBuffer.contents().bindMemory(to: AVBDGPUBody.self, capacity: bodyCount)
        return (0..<bodyCount).map { ptr[$0].positionLin }
    }

    /// Read current body states back from the GPU buffer for verification.
    func readBodies() -> [AVBDGPUBody] {
        let ptr = bodyBuffer.contents().bindMemory(to: AVBDGPUBody.self, capacity: bodyCount)
        return (0..<bodyCount).map { ptr[$0] }
    }

    func readMeshMeshIsoVoxelDebugEntries(includeZeroCandidates: Bool = false) -> [MeshMeshIsoVoxelDebugEntry] {
        let pairCountPtr = meshMeshPairStateBuffer.contents().bindMemory(to: AVBDGPUMeshMeshPairListState.self, capacity: 1)
        let pairCount = max(0, min(Int(pairCountPtr.pointee.count), meshMeshIsoVoxelTrackedPairCapacity))
        guard pairCount > 0 else { return [] }

        let ptr = meshMeshIsoVoxelDebugBuffer.contents().bindMemory(
            to: AVBDGPUMeshMeshIsoVoxelDebug.self,
            capacity: pairCount
        )
        let coordPtr = meshMeshIsoVoxelCoordBuffer.contents().bindMemory(
            to: AVBDGPUMeshMeshIsoVoxelCoord.self,
            capacity: max(meshMeshIsoVoxelTrackedPairCapacity * gpuMeshMeshIsoVoxelCoordsPerPair, 1)
        )

        return (0..<pairCount).compactMap { index in
            let entry = ptr[index]
            guard entry.valid != 0 else { return nil }
            if !includeZeroCandidates && entry.candidateVoxelCount <= 0 {
                return nil
            }
            let compactedCount = max(0, min(Int(entry.compactedVoxelCount), gpuMeshMeshIsoVoxelCoordsPerPair))
            let coordBase = index * gpuMeshMeshIsoVoxelCoordsPerPair
            let voxelCoords = (0..<compactedCount).map { coordIndex -> SIMD3<Int32> in
                let packed = coordPtr[coordBase + coordIndex].voxelCoord
                return SIMD3<Int32>(packed.x, packed.y, packed.z)
            }
            return MeshMeshIsoVoxelDebugEntry(
                meshIndexA: Int(entry.meshIndexA),
                meshIndexB: Int(entry.meshIndexB),
                driverMeshIndex: Int(entry.driverMeshIndex),
                sampledVoxelCount: Int(entry.sampledVoxelCount),
                candidateVoxelCount: Int(entry.candidateVoxelCount),
                compactedVoxelCount: compactedCount,
                sampleStride: Int(entry.sampleStride),
                overflowed: entry.overflowed != 0,
                emittedContactCount: Int(entry.emittedContactCount),
                usedIsoVoxelPath: entry.usedIsoVoxelPath != 0,
                voxelCoords: voxelCoords
            )
        }
    }

    // MARK: - Buffer Transfers

    private func uploadBodies() {
        let ptr = bodyBuffer.contents().bindMemory(to: AVBDGPUBody.self, capacity: bodyCount)
        for i in 0..<bodyCount {
            ptr[i] = cpuBodies[i]
        }
    }

    private func uploadRenderColors(_ renderColors: [SIMD4<Float>?]) {
        let ptr = renderColorBuffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: bodyCapacity)
        for i in 0..<bodyCount {
            ptr[i] = encodeRenderColor(renderColors[i])
        }
    }

    private func uploadRenderColorGroups(_ colorGroups: [Int?]) {
        let ptr = renderColorGroupBuffer.contents().bindMemory(to: Int32.self, capacity: bodyCapacity)
        for i in 0..<bodyCount {
            ptr[i] = encodeColorGroup(colorGroups[i])
        }
    }

    private func downloadBodies() {
        let ptr = bodyBuffer.contents().bindMemory(to: AVBDGPUBody.self, capacity: bodyCount)
        for i in 0..<bodyCount {
            cpuBodies[i] = ptr[i]
        }
    }

    private func uploadJoints() {
        guard !cpuJoints.isEmpty else { return }
        let ptr = jointBuffer.contents().bindMemory(to: AVBDGPUJoint.self, capacity: cpuJoints.count)
        for i in 0..<cpuJoints.count {
            ptr[i] = cpuJoints[i]
        }
    }

    private func uploadSprings() {
        guard !cpuSprings.isEmpty else { return }
        let ptr = springBuffer.contents().bindMemory(to: AVBDGPUSpring.self, capacity: cpuSprings.count)
        for i in 0..<cpuSprings.count {
            ptr[i] = cpuSprings[i]
        }
    }

    /// Build and upload per-body collision exclusion lists from joints, springs,
    /// and explicitly ignored pairs.
    private func computeAndUploadExclusions() {
        let ptr = exclusionBuffer.contents().bindMemory(to: AVBDGPUCollisionExclusion.self, capacity: bodyCapacity)
        let maxExcl = Int(AVBD_MAX_CONSTRAINTS_PER_BODY)

        // Zero out
        for i in 0..<bodyCapacity {
            ptr[i].excludeCount = 0
        }

        // Helper to add exclusion for body at index
        func addExclusion(body: Int, other: Int) {
            guard body >= 0 && body < bodyCount else { return }
            let count = Int(ptr[body].excludeCount)
            guard count < maxExcl else { return }
            withUnsafeMutablePointer(to: &ptr[body].excludeIndices) { idxPtr in
                idxPtr.withMemoryRebound(to: Int32.self, capacity: maxExcl) { arr in
                    arr[count] = Int32(other)
                }
            }
            ptr[body].excludeCount = Int32(count + 1)
        }

        // Joint-connected pairs
        for joint in cpuJoints {
            let a = Int(joint.bodyA)
            let b = Int(joint.bodyB)
            if a >= 0 && b >= 0 && a < bodyCount && b < bodyCount {
                addExclusion(body: a, other: b)
                addExclusion(body: b, other: a)
            }
        }

        // Spring-connected pairs
        for spring in cpuSprings {
            let a = Int(spring.bodyA)
            let b = Int(spring.bodyB)
            if a >= 0 && b >= 0 && a < bodyCount && b < bodyCount {
                addExclusion(body: a, other: b)
                addExclusion(body: b, other: a)
            }
        }

        // Explicitly ignored pairs
        for key in ignorePairs {
            let a = Int(key & 0xFFFFFFFF)
            let b = Int(key >> 32)
            if a >= 0 && a < bodyCount && b >= 0 && b < bodyCount {
                addExclusion(body: a, other: b)
                addExclusion(body: b, other: a)
            }
        }
    }

    /// Greedy graph coloring based on joint/spring adjacency.
    /// Bodies that share a joint or spring get different colors, enabling
    /// Gauss-Seidel-like sequential color-group dispatches on GPU.
    private func computeAndUploadBodyColors() {
        // Build adjacency from constraints
        var adj: [[Int]] = Array(repeating: [], count: bodyCount)
        for joint in cpuJoints {
            let a = Int(joint.bodyA)
            let b = Int(joint.bodyB)
            if a >= 0 && b >= 0 && a < bodyCount && b < bodyCount {
                adj[a].append(b)
                adj[b].append(a)
            }
        }
        for spring in cpuSprings {
            let a = Int(spring.bodyA)
            let b = Int(spring.bodyB)
            if a >= 0 && b >= 0 && a < bodyCount && b < bodyCount {
                adj[a].append(b)
                adj[b].append(a)
            }
        }

        // Greedy coloring
        var colors = Array(repeating: Int32(0), count: bodyCount)
        var maxColor: Int32 = 0
        for i in 0..<bodyCount {
            var usedColors = Set<Int32>()
            for neighbor in adj[i] {
                usedColors.insert(colors[neighbor])
            }
            var c: Int32 = 0
            while usedColors.contains(c) {
                c += 1
            }
            colors[i] = c
            maxColor = max(maxColor, c)
        }
        numColors = Int(maxColor) + 1
        computedBodyColorGroups = colors

        // Upload to GPU buffer
        let ptr = bodyColorBuffer.contents().bindMemory(to: Int32.self, capacity: bodyCapacity)
        for i in 0..<bodyCount {
            ptr[i] = colors[i]
        }
    }

    private func uploadManifolds() {
        guard !cpuManifolds.isEmpty else { return }
        let ptr = manifoldBuffer.contents().bindMemory(to: AVBDGPUManifold.self, capacity: cpuManifolds.count)
        for i in 0..<cpuManifolds.count {
            ptr[i] = cpuManifolds[i]
        }
    }

    private func buildAdjacency() {
        // Reset
        for i in 0..<bodyCount {
            cpuAdjacency[i] = AVBDGPUAdjacency()
        }

        // Joints
        for (ji, joint) in cpuJoints.enumerated() {
            if joint.broken != 0 { continue }
            let a = Int(joint.bodyA)
            let b = Int(joint.bodyB)
            if a >= 0 {
                addToAdjacency(bodyIdx: a, constraintIdx: ji, type: .joint)
            }
            addToAdjacency(bodyIdx: b, constraintIdx: ji, type: .joint)
        }

        // Springs
        for (si, spring) in cpuSprings.enumerated() {
            let a = Int(spring.bodyA)
            let b = Int(spring.bodyB)
            addToAdjacency(bodyIdx: a, constraintIdx: si, type: .spring)
            addToAdjacency(bodyIdx: b, constraintIdx: si, type: .spring)
        }

        // Manifolds
        for (mi, manifold) in cpuManifolds.enumerated() {
            if manifold.active == 0 { continue }
            let a = Int(manifold.bodyA)
            let b = Int(manifold.bodyB)
            addToAdjacency(bodyIdx: a, constraintIdx: mi, type: .manifold)
            addToAdjacency(bodyIdx: b, constraintIdx: mi, type: .manifold)
        }
    }

    private enum ConstraintType { case joint, spring, manifold }

    private func addToAdjacency(bodyIdx: Int, constraintIdx: Int, type: ConstraintType) {
        guard bodyIdx >= 0 && bodyIdx < bodyCount else { return }

        switch type {
        case .joint:
            let count = Int(cpuAdjacency[bodyIdx].jointCount)
            if count < SWIFT_AVBD_MAX_CONSTRAINTS_PER_BODY {
                withUnsafeMutablePointer(to: &cpuAdjacency[bodyIdx].jointIndices) { ptr in
                    ptr.withMemoryRebound(to: Int32.self, capacity: SWIFT_AVBD_MAX_CONSTRAINTS_PER_BODY) { arr in
                        arr[count] = Int32(constraintIdx)
                    }
                }
                cpuAdjacency[bodyIdx].jointCount = Int32(count + 1)
            }
        case .spring:
            let count = Int(cpuAdjacency[bodyIdx].springCount)
            if count < SWIFT_AVBD_MAX_CONSTRAINTS_PER_BODY {
                withUnsafeMutablePointer(to: &cpuAdjacency[bodyIdx].springIndices) { ptr in
                    ptr.withMemoryRebound(to: Int32.self, capacity: SWIFT_AVBD_MAX_CONSTRAINTS_PER_BODY) { arr in
                        arr[count] = Int32(constraintIdx)
                    }
                }
                cpuAdjacency[bodyIdx].springCount = Int32(count + 1)
            }
        case .manifold:
            let count = Int(cpuAdjacency[bodyIdx].manifoldCount)
            if count < SWIFT_AVBD_MAX_CONSTRAINTS_PER_BODY {
                withUnsafeMutablePointer(to: &cpuAdjacency[bodyIdx].manifoldIndices) { ptr in
                    ptr.withMemoryRebound(to: Int32.self, capacity: SWIFT_AVBD_MAX_CONSTRAINTS_PER_BODY) { arr in
                        arr[count] = Int32(constraintIdx)
                    }
                }
                cpuAdjacency[bodyIdx].manifoldCount = Int32(count + 1)
            }
        }
    }

    private func uploadAdjacency() {
        let ptr = adjacencyBuffer.contents().bindMemory(to: AVBDGPUAdjacency.self, capacity: max(bodyCount, 1))
        for i in 0..<bodyCount {
            ptr[i] = cpuAdjacency[i]
        }
    }

    private func uploadParams() {
        let ptr = paramsBuffer.contents().bindMemory(to: AVBDGPUSolverParams.self, capacity: 1)
        let cacheEnabled = broadphaseFullRefreshStepCount > 0
        let cacheTimeHorizon = cacheEnabled ? Float(broadphaseFullRefreshStepCount) * dt : 0.0
        let effectiveCacheMargin: Float = cacheEnabled ? broadphaseCacheMargin : 0.0
        ptr.pointee = AVBDGPUSolverParams(
            dt: dt,
            gravity: gravity,
            alpha: alpha,
            betaLin: betaLin,
            betaAng: betaAng,
            gamma: gamma,
            iterations: Int32(iterations),
            bodyCount: Int32(bodyCount),
            jointCount: Int32(cpuJoints.count),
            springCount: Int32(cpuSprings.count),
            meshCount: Int32(collisionMeshCount),
            primitiveMeshManifoldOffset: Int32(bodyBodyManifoldCapacity),
            meshMeshManifoldOffset: Int32(bodyBodyManifoldCapacity + primitiveMeshManifoldCapacity),
            meshMeshIsoVoxelTrackedPairCapacity: Int32(meshMeshIsoVoxelTrackedPairCapacity),
            meshMeshIsoVoxelCoordsPerPair: Int32(gpuMeshMeshIsoVoxelCoordsPerPair),
            collisionMargin: gpuCollisionMargin,
            cacheMargin: effectiveCacheMargin,
            cacheTimeHorizon: cacheTimeHorizon,
            penaltyMin: gpuPenaltyMin,
            penaltyMax: gpuPenaltyMax,
            stickThreshold: gpuStickThreshold,
            torusApproxSphereCount: Int32(avbdCurrentTorusApproxSphereCount()),
            torusApproxSphereRadiusScale: AVBDTorusApproximationSettings.radiusScale,
            linearDamping: linearDamping,
            angularDamping: angularDamping,
            hydroelasticInteriorWeight: hydroelasticInteriorWeight,
            meshMeshMaxIsoVoxelSamples: Int32(max(8, min(meshMeshMaxIsoVoxelSamples, 4096))),
            meshMeshReduceContacts: meshMeshReduceContacts ? 1 : 0
        )
    }

    func setCollisionMeshes(_ meshes: [AVBDCollisionMeshBroadphaseMesh]) {
        let clampedMeshes = Array(meshes.prefix(min(collisionMeshCapacity, gpuMaxCollisionMeshSDFs)))
        collisionMeshCount = clampedMeshes.count
        ensureCollisionCapacity(forMeshCount: collisionMeshCount)
        collisionMeshSDFResources.removeAll(keepingCapacity: true)
        collisionMeshSDFStatusText = collisionMeshCount > 0 ? "Building..." : "Idle"

        var meshInfos = Array(repeating: AVBDGPUCollisionMeshInfo(), count: max(collisionMeshCapacity, 1))
        var meshVertices: [SIMD3<Float>] = []
        var meshIndices: [UInt32] = []
        var collisionMeshSDFResources: [CollisionMeshSDFResource] = []
        var sdfResourceIndexByKey: [String: Int] = [:]
        var geometryResourceByKey: [String: CollisionMeshGeometryResource] = [:]
        var cacheHitCount = 0
        var uniqueSDFBuildCount = 0
        var totalDenseSDFBytes = 0
        var totalCompactedSDFBytes = 0
        let buildStartTime = CACurrentMediaTime()

        meshVertices.reserveCapacity(clampedMeshes.reduce(0) { $0 + $1.positions.count })
        meshIndices.reserveCapacity(clampedMeshes.reduce(0) { $0 + $1.indices.count })
        collisionMeshSDFResources.reserveCapacity(collisionMeshCount)

        for meshIndex in 0..<collisionMeshCount {
            let mesh = clampedMeshes[meshIndex]
            let sdfPadding = Self.collisionMeshSDFPadding(for: mesh)
            let sdfLocalMinBounds = mesh.localBoundsMin - sdfPadding
            let sdfLocalMaxBounds = mesh.localBoundsMax + sdfPadding
            let sdfResolution = Self.collisionMeshSDFResolution(
                localMinBounds: sdfLocalMinBounds,
                localMaxBounds: sdfLocalMaxBounds
            )

            let worldBounds = Self.transformedBounds(
                localMinBounds: sdfLocalMinBounds,
                localMaxBounds: sdfLocalMaxBounds,
                transform: mesh.transform
            )
            let sdfSize = max(sdfLocalMaxBounds - sdfLocalMinBounds, SIMD3<Float>(repeating: 1.0e-4))
            let sdfVoxelSize = SIMD3<Float>(
                sdfSize.x / Float(max(sdfResolution.x, 1)),
                sdfSize.y / Float(max(sdfResolution.y, 1)),
                sdfSize.z / Float(max(sdfResolution.z, 1))
            )
            let sdfCacheKey = Self.collisionMeshSDFCacheKey(
                mesh: mesh,
                localBoundsMin: sdfLocalMinBounds,
                localBoundsMax: sdfLocalMaxBounds,
                resolution: sdfResolution
            )
            let geometryResource: CollisionMeshGeometryResource
            if let existingGeometryResource = geometryResourceByKey[sdfCacheKey] {
                geometryResource = existingGeometryResource
            } else {
                let vertexOffset = meshVertices.count
                let indexOffset = meshIndices.count
                meshVertices.append(contentsOf: mesh.positions)
                meshIndices.append(contentsOf: mesh.indices.map { UInt32(vertexOffset) + $0 })
                geometryResource = CollisionMeshGeometryResource(
                    vertexOffset: vertexOffset,
                    vertexCount: mesh.positions.count,
                    indexOffset: indexOffset,
                    indexCount: mesh.indices.count
                )
                geometryResourceByKey[sdfCacheKey] = geometryResource
            }
            let sdfResourceIndex: Int
            if let existingResourceIndex = sdfResourceIndexByKey[sdfCacheKey] {
                sdfResourceIndex = existingResourceIndex
            } else {
                let sdfResource: CollisionMeshSDFResource?
                if let cached = Self.cachedCollisionMeshSDF(for: sdfCacheKey) {
                    sdfResource = cached.resource
                    cacheHitCount += 1
                } else {
                    sdfResource = buildCollisionMeshSDFTexture(
                        localVertices: mesh.positions,
                        indices: mesh.indices,
                        localBoundsMin: sdfLocalMinBounds,
                        localBoundsMax: sdfLocalMaxBounds,
                        resolution: sdfResolution
                    )
                    if let sdfResource {
                        Self.storeCollisionMeshSDF(sdfResource, for: sdfCacheKey)
                    }
                }

                guard let sdfResource else {
                    print("Failed to build collision mesh SDF texture for mesh \(meshIndex)")
                    collisionMeshSDFResources.removeAll(keepingCapacity: true)
                    break
                }

                sdfResourceIndex = collisionMeshSDFResources.count
                sdfResourceIndexByKey[sdfCacheKey] = sdfResourceIndex
                collisionMeshSDFResources.append(sdfResource)
                totalDenseSDFBytes += sdfResource.denseByteCount
                totalCompactedSDFBytes += sdfResource.compactedByteCount
                uniqueSDFBuildCount += 1
            }

            meshInfos[meshIndex] = AVBDGPUCollisionMeshInfo(
                vertexOffset: Int32(geometryResource.vertexOffset),
                vertexCount: Int32(geometryResource.vertexCount),
                indexOffset: Int32(geometryResource.indexOffset),
                indexCount: Int32(geometryResource.indexCount),
                ownerBodyIndex: Int32(mesh.ownerBodyIndex),
                sdfResourceIndex: Int32(sdfResourceIndex),
                _reserved0: SIMD2<Int32>(repeating: 0),
                minBounds: SIMD4<Float>(worldBounds.min, 0),
                maxBounds: SIMD4<Float>(worldBounds.max, 0),
                sdfLocalMinBounds: SIMD4<Float>(sdfLocalMinBounds, 0),
                sdfLocalMaxBounds: SIMD4<Float>(sdfLocalMaxBounds, 0),
                sdfVoxelSize: SIMD4<Float>(sdfVoxelSize, 0),
                sdfResolution: SIMD4<Int32>(sdfResolution.x, sdfResolution.y, sdfResolution.z, 0),
                sdfTransform: mesh.transform,
                sdfInvTransform: simd_inverse(mesh.transform)
            )
        }

        collisionMeshGeometryResources = geometryResourceByKey
        collisionMeshSDFResourceIndicesByKey = sdfResourceIndexByKey
        collisionMeshVertexCountUsed = meshVertices.count
        collisionMeshIndexCountUsed = meshIndices.count

        if collisionMeshInfoBuffer.length < MemoryLayout<AVBDGPUCollisionMeshInfo>.stride * max(collisionMeshCapacity, 1) {
            collisionMeshInfoBuffer = device.makeBuffer(
                length: MemoryLayout<AVBDGPUCollisionMeshInfo>.stride * max(collisionMeshCapacity, 1),
                options: .storageModeShared
            )!
        }
        if collisionMeshVertexBuffer.length < MemoryLayout<SIMD3<Float>>.stride * max(meshVertices.count, 1) {
            collisionMeshVertexBuffer = device.makeBuffer(
                length: MemoryLayout<SIMD3<Float>>.stride * max(meshVertices.count, 1),
                options: .storageModeShared
            )!
        }
        if collisionMeshIndexBuffer.length < MemoryLayout<UInt32>.stride * max(meshIndices.count, 1) {
            collisionMeshIndexBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * max(meshIndices.count, 1),
                options: .storageModeShared
            )!
        }

        let infoPtr = collisionMeshInfoBuffer.contents().bindMemory(to: AVBDGPUCollisionMeshInfo.self, capacity: max(collisionMeshCapacity, 1))
        for meshIndex in 0..<max(collisionMeshCapacity, 1) {
            infoPtr[meshIndex] = meshInfos[meshIndex]
        }

        if !meshVertices.isEmpty {
            let vertexPtr = collisionMeshVertexBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: meshVertices.count)
            for vertexIndex in meshVertices.indices {
                vertexPtr[vertexIndex] = meshVertices[vertexIndex]
            }
        }

        if !meshIndices.isEmpty {
            let indexPtr = collisionMeshIndexBuffer.contents().bindMemory(to: UInt32.self, capacity: meshIndices.count)
            for index in meshIndices.indices {
                indexPtr[index] = meshIndices[index]
            }
        }

        self.collisionMeshSDFResources = collisionMeshSDFResources
        updateCollisionMeshSDFArgumentBufferBindings()
        refreshCollisionMeshOwnerBodyMask()

        let buildDurationMS = (CACurrentMediaTime() - buildStartTime) * 1000.0
        if collisionMeshCount == 0 {
            collisionMeshSDFStatusText = "Idle"
        } else if !collisionMeshSDFResources.isEmpty {
            let compactPercent: Int
            if totalDenseSDFBytes > 0 {
                compactPercent = Int((Double(totalCompactedSDFBytes) / Double(totalDenseSDFBytes) * 100.0).rounded())
            } else {
                compactPercent = 100
            }
            if uniqueSDFBuildCount == cacheHitCount && uniqueSDFBuildCount > 0 {
                collisionMeshSDFStatusText = String(format: "Cached %.1f ms %d%%", buildDurationMS, compactPercent)
            } else if cacheHitCount > 0 {
                collisionMeshSDFStatusText = String(
                    format: "Built %d/%d %.1f ms %d%%",
                    uniqueSDFBuildCount - cacheHitCount,
                    uniqueSDFBuildCount,
                    buildDurationMS,
                    compactPercent
                )
            } else {
                collisionMeshSDFStatusText = String(
                    format: "Built %d %.1f ms %d%%",
                    uniqueSDFBuildCount,
                    buildDurationMS,
                    compactPercent
                )
            }
        } else {
            collisionMeshSDFStatusText = "Build Failed"
        }

        forceFullBroadphase = true
    }

    private func refreshCollisionMeshOwnerBodyMask() {
        let ownerMaskPtr = collisionMeshOwnerBodyMaskBuffer.contents().bindMemory(
            to: Int32.self,
            capacity: max(bodyCapacity, 1)
        )
        for bodyIndex in 0..<max(bodyCapacity, 1) {
            ownerMaskPtr[bodyIndex] = 0
        }

        guard collisionMeshCount > 0 else {
            return
        }

        let infoPtr = collisionMeshInfoBuffer.contents().bindMemory(
            to: AVBDGPUCollisionMeshInfo.self,
            capacity: max(collisionMeshCapacity, 1)
        )
        for meshIndex in 0..<collisionMeshCount {
            let ownerBodyIndex = Int(infoPtr[meshIndex].ownerBodyIndex)
            if ownerBodyIndex >= 0 && ownerBodyIndex < bodyCapacity {
                ownerMaskPtr[ownerBodyIndex] = 1
            }
        }
    }

    private func buildCollisionMeshSDFTexture(
        localVertices: [SIMD3<Float>],
        indices: [UInt32],
        localBoundsMin: SIMD3<Float>,
        localBoundsMax: SIMD3<Float>,
        resolution: SIMD3<Int32>
    ) -> CollisionMeshSDFResource? {
        guard let denseBuild = buildDenseCollisionMeshSDFTexture(
            localVertices: localVertices,
            indices: indices,
            localBoundsMin: localBoundsMin,
            localBoundsMax: localBoundsMax,
            resolution: resolution
        ) else {
            return nil
        }

        return compactCollisionMeshSDFTexture(
            denseTexture: denseBuild.texture,
            resolution: resolution,
            voxelSize: denseBuild.voxelSize
        )
    }

    func buildDenseCollisionMeshSDFTexture(
        localVertices: [SIMD3<Float>],
        indices: [UInt32],
        localBoundsMin: SIMD3<Float>,
        localBoundsMax: SIMD3<Float>,
        resolution: SIMD3<Int32>
    ) -> (texture: MTLTexture, voxelSize: SIMD3<Float>)? {
        guard let collisionMeshSDFCommandQueue,
              !localVertices.isEmpty,
              indices.count >= 3,
              indices.count % 3 == 0 else {
            return nil
        }

        let voxelCount = Int(resolution.x * resolution.y * resolution.z)
        guard voxelCount > 0,
              let vertexBuffer = device.makeBuffer(
                bytes: localVertices,
                length: MemoryLayout<SIMD3<Float>>.stride * localVertices.count,
                options: .storageModeShared
              ),
              let indexBuffer = device.makeBuffer(
                bytes: indices,
                length: MemoryLayout<UInt32>.stride * indices.count,
                options: .storageModeShared
              ),
              let insideCountBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * voxelCount,
                options: .storageModeShared
              ) else {
            return nil
        }

        memset(insideCountBuffer.contents(), 0, insideCountBuffer.length)

        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.textureType = .type3D
        textureDescriptor.pixelFormat = .r32Float
        textureDescriptor.width = Int(resolution.x)
        textureDescriptor.height = Int(resolution.y)
        textureDescriptor.depth = Int(resolution.z)
        textureDescriptor.storageMode = .shared
        textureDescriptor.usage = [.shaderRead, .shaderWrite]

        guard let denseTexture = device.makeTexture(descriptor: textureDescriptor),
              let commandBuffer = collisionMeshSDFCommandQueue.makeCommandBuffer() else {
            return nil
        }

        let voxelSize = (localBoundsMax - localBoundsMin) / SIMD3<Float>(
            Float(max(resolution.x, 1)),
            Float(max(resolution.y, 1)),
            Float(max(resolution.z, 1))
        )
        let threadsPerThreadgroup = MTLSize(width: 4, height: 4, depth: 4)
        let threadsPerGrid = MTLSize(
            width: Int(resolution.x),
            height: Int(resolution.y),
            depth: Int(resolution.z)
        )

        if let initializeEncoder = commandBuffer.makeComputeCommandEncoder() {
            initializeEncoder.setComputePipelineState(initializeCollisionMeshSDFPSO)
            initializeEncoder.setTexture(denseTexture, index: 0)
            initializeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            initializeEncoder.endEncoding()
        } else {
            return nil
        }

        let triangleCount = indices.count / 3
        var triangleOffset = 0
        while triangleOffset < triangleCount {
            let triangleChunkCount = min(gpuCollisionMeshSDFTriangleChunkSize, triangleCount - triangleOffset)
            guard let accumulateEncoder = commandBuffer.makeComputeCommandEncoder() else {
                return nil
            }

            var triangleOffsetValue = UInt32(triangleOffset)
            var triangleChunkCountValue = UInt32(triangleChunkCount)
            var sdfOrigin = localBoundsMin
            var voxelSizeValue = voxelSize

            accumulateEncoder.setComputePipelineState(accumulateCollisionMeshSDFPSO)
            accumulateEncoder.setBuffer(vertexBuffer, offset: 0, index: 0)
            accumulateEncoder.setBuffer(indexBuffer, offset: 0, index: 1)
            accumulateEncoder.setTexture(denseTexture, index: 0)
            accumulateEncoder.setBuffer(insideCountBuffer, offset: 0, index: 2)
            accumulateEncoder.setBytes(&triangleOffsetValue, length: MemoryLayout<UInt32>.stride, index: 3)
            accumulateEncoder.setBytes(&triangleChunkCountValue, length: MemoryLayout<UInt32>.stride, index: 4)
            accumulateEncoder.setBytes(&sdfOrigin, length: MemoryLayout<SIMD3<Float>>.stride, index: 5)
            accumulateEncoder.setBytes(&voxelSizeValue, length: MemoryLayout<SIMD3<Float>>.stride, index: 6)
            accumulateEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            accumulateEncoder.endEncoding()

            triangleOffset += triangleChunkCount
        }

        if let finalizeEncoder = commandBuffer.makeComputeCommandEncoder() {
            finalizeEncoder.setComputePipelineState(finalizeCollisionMeshSDFPSO)
            finalizeEncoder.setTexture(denseTexture, index: 0)
            finalizeEncoder.setBuffer(insideCountBuffer, offset: 0, index: 0)
            finalizeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            finalizeEncoder.endEncoding()
        } else {
            return nil
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if commandBuffer.status == .error {
            print("Failed to build collision mesh SDF texture: \(commandBuffer.error?.localizedDescription ?? "unknown error")")
            return nil
        }

        return (denseTexture, voxelSize)
    }

    private func compactCollisionMeshSDFTexture(
        denseTexture: MTLTexture,
        resolution: SIMD3<Int32>,
        voxelSize: SIMD3<Float>
    ) -> CollisionMeshSDFResource? {
        let denseResolution = SIMD3<Int>(Int(resolution.x), Int(resolution.y), Int(resolution.z))
        let denseVoxelCount = denseResolution.x * denseResolution.y * denseResolution.z
        guard denseVoxelCount > 0 else {
            return nil
        }

        var denseData = [Float](repeating: 0, count: denseVoxelCount)
        denseData.withUnsafeMutableBytes { rawBuffer in
            denseTexture.getBytes(
                rawBuffer.baseAddress!,
                bytesPerRow: denseResolution.x * MemoryLayout<Float>.stride,
                bytesPerImage: denseResolution.x * denseResolution.y * MemoryLayout<Float>.stride,
                from: MTLRegionMake3D(0, 0, 0, denseResolution.x, denseResolution.y, denseResolution.z),
                mipmapLevel: 0,
                slice: 0
            )
        }

        guard let compactionData = Self.makeCollisionMeshSDFCompactionData(
            denseData: denseData,
            denseResolution: denseResolution,
            voxelSize: voxelSize
        ) else {
            return nil
        }

        guard let coarseTexture = Self.makeCollisionMeshSDFTexture(
                device: device,
                resolution: compactionData.coarseResolution,
                pixelFormat: .r32Float
              ),
              let atlasTexture = Self.makeCollisionMeshSDFTexture(
                device: device,
                resolution: compactionData.atlasResolution,
                pixelFormat: .r32Float
              ),
              let indirectionTexture = Self.makeCollisionMeshSDFTexture(
                device: device,
                resolution: compactionData.brickGrid,
                pixelFormat: .r32Uint
              ) else {
            return nil
        }

        compactionData.coarseData.withUnsafeBytes { rawBuffer in
            coarseTexture.replace(
                region: MTLRegionMake3D(0, 0, 0, compactionData.coarseResolution.x, compactionData.coarseResolution.y, compactionData.coarseResolution.z),
                mipmapLevel: 0,
                slice: 0,
                withBytes: rawBuffer.baseAddress!,
                bytesPerRow: compactionData.coarseResolution.x * MemoryLayout<Float>.stride,
                bytesPerImage: compactionData.coarseResolution.x * compactionData.coarseResolution.y * MemoryLayout<Float>.stride
            )
        }
        compactionData.atlasData.withUnsafeBytes { rawBuffer in
            atlasTexture.replace(
                region: MTLRegionMake3D(0, 0, 0, compactionData.atlasResolution.x, compactionData.atlasResolution.y, compactionData.atlasResolution.z),
                mipmapLevel: 0,
                slice: 0,
                withBytes: rawBuffer.baseAddress!,
                bytesPerRow: compactionData.atlasResolution.x * MemoryLayout<Float>.stride,
                bytesPerImage: compactionData.atlasResolution.x * compactionData.atlasResolution.y * MemoryLayout<Float>.stride
            )
        }
        compactionData.indirectionData.withUnsafeBytes { rawBuffer in
            indirectionTexture.replace(
                region: MTLRegionMake3D(0, 0, 0, compactionData.brickGrid.x, compactionData.brickGrid.y, compactionData.brickGrid.z),
                mipmapLevel: 0,
                slice: 0,
                withBytes: rawBuffer.baseAddress!,
                bytesPerRow: compactionData.brickGrid.x * MemoryLayout<UInt32>.stride,
                bytesPerImage: compactionData.brickGrid.x * compactionData.brickGrid.y * MemoryLayout<UInt32>.stride
            )
        }

        return CollisionMeshSDFResource(
            coarseTexture: coarseTexture,
            atlasTexture: atlasTexture,
            indirectionTexture: indirectionTexture,
            mappedBrickCount: compactionData.mappedBrickCount,
            denseByteCount: compactionData.denseByteCount,
            compactedByteCount: compactionData.compactedByteCount
        )
    }

    private static func makeCollisionMeshSDFTexture(
        device: MTLDevice,
        resolution: SIMD3<Int>,
        pixelFormat: MTLPixelFormat
    ) -> MTLTexture? {
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = .type3D
        descriptor.pixelFormat = pixelFormat
        descriptor.width = resolution.x
        descriptor.height = resolution.y
        descriptor.depth = resolution.z
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        return device.makeTexture(descriptor: descriptor)
    }

    static func makeCollisionMeshSDFCompactionData(
        denseData: [Float],
        denseResolution: SIMD3<Int>,
        voxelSize: SIMD3<Float>
    ) -> CollisionMeshSDFCompactionData? {
        let brickGrid = collisionMeshSDFBrickGrid(resolution: denseResolution)
        let keepHalfWidth = collisionMeshSDFSparseBandHalfWidth(voxelSize: voxelSize)
        let brickCount = brickGrid.x * brickGrid.y * brickGrid.z
        var keepMask = [Bool](repeating: false, count: brickCount)

        var fallbackBrick = SIMD3<Int>(repeating: 0)
        var fallbackBrickDistance = Float.greatestFiniteMagnitude

        for brickZ in 0..<brickGrid.z {
            for brickY in 0..<brickGrid.y {
                for brickX in 0..<brickGrid.x {
                    let brick = SIMD3<Int>(brickX, brickY, brickZ)
                    let startX = brickX * gpuCollisionMeshSDFBrickDim
                    let startY = brickY * gpuCollisionMeshSDFBrickDim
                    let startZ = brickZ * gpuCollisionMeshSDFBrickDim
                    let endX = min(startX + gpuCollisionMeshSDFBrickDim, denseResolution.x)
                    let endY = min(startY + gpuCollisionMeshSDFBrickDim, denseResolution.y)
                    let endZ = min(startZ + gpuCollisionMeshSDFBrickDim, denseResolution.z)

                    var brickClosestDistance = Float.greatestFiniteMagnitude
                    for z in startZ..<endZ {
                        for y in startY..<endY {
                            for x in startX..<endX {
                                let value = denseData[collisionMeshSDFLinearIndex(x: x, y: y, z: z, resolution: denseResolution)]
                                brickClosestDistance = min(brickClosestDistance, abs(value))
                            }
                        }
                    }

                    if brickClosestDistance < fallbackBrickDistance {
                        fallbackBrickDistance = brickClosestDistance
                        fallbackBrick = brick
                    }

                    if brickClosestDistance <= keepHalfWidth {
                        keepMask[collisionMeshSDFLinearIndex(x: brick.x, y: brick.y, z: brick.z, resolution: brickGrid)] = true
                    }
                }
            }
        }

        if !keepMask.contains(true) {
            keepMask[collisionMeshSDFLinearIndex(x: fallbackBrick.x, y: fallbackBrick.y, z: fallbackBrick.z, resolution: brickGrid)] = true
        }

        if gpuCollisionMeshSDFMappedBrickDilation > 0 {
            var dilatedMask = keepMask
            for brickZ in 0..<brickGrid.z {
                for brickY in 0..<brickGrid.y {
                    for brickX in 0..<brickGrid.x {
                        let brickIndex = collisionMeshSDFLinearIndex(x: brickX, y: brickY, z: brickZ, resolution: brickGrid)
                        guard keepMask[brickIndex] else { continue }
                        for neighborZ in max(0, brickZ - gpuCollisionMeshSDFMappedBrickDilation)...min(brickGrid.z - 1, brickZ + gpuCollisionMeshSDFMappedBrickDilation) {
                            for neighborY in max(0, brickY - gpuCollisionMeshSDFMappedBrickDilation)...min(brickGrid.y - 1, brickY + gpuCollisionMeshSDFMappedBrickDilation) {
                                for neighborX in max(0, brickX - gpuCollisionMeshSDFMappedBrickDilation)...min(brickGrid.x - 1, brickX + gpuCollisionMeshSDFMappedBrickDilation) {
                                    dilatedMask[collisionMeshSDFLinearIndex(x: neighborX, y: neighborY, z: neighborZ, resolution: brickGrid)] = true
                                }
                            }
                        }
                    }
                }
            }
            keepMask = dilatedMask
        }

        var mappedBricks: [SIMD3<Int>] = []
        mappedBricks.reserveCapacity(brickCount)
        for brickZ in 0..<brickGrid.z {
            for brickY in 0..<brickGrid.y {
                for brickX in 0..<brickGrid.x {
                    if keepMask[collisionMeshSDFLinearIndex(x: brickX, y: brickY, z: brickZ, resolution: brickGrid)] {
                        mappedBricks.append(SIMD3<Int>(brickX, brickY, brickZ))
                    }
                }
            }
        }

        let atlasBricks = SIMD3<Int>(
            gpuCollisionMeshSDFAtlasBricksAcross,
            gpuCollisionMeshSDFAtlasBricksAcross,
            max(1, (mappedBricks.count + gpuCollisionMeshSDFAtlasBricksAcross * gpuCollisionMeshSDFAtlasBricksAcross - 1)
                / (gpuCollisionMeshSDFAtlasBricksAcross * gpuCollisionMeshSDFAtlasBricksAcross))
        )
        let atlasResolution = SIMD3<Int>(
            atlasBricks.x * gpuCollisionMeshSDFStoredBrickDim,
            atlasBricks.y * gpuCollisionMeshSDFStoredBrickDim,
            atlasBricks.z * gpuCollisionMeshSDFStoredBrickDim
        )
        let coarseResolution = SIMD3<Int>(
            min(denseResolution.x, max(12, (denseResolution.x + gpuCollisionMeshSDFCoarseDownsampleFactor - 1) / gpuCollisionMeshSDFCoarseDownsampleFactor)),
            min(denseResolution.y, max(12, (denseResolution.y + gpuCollisionMeshSDFCoarseDownsampleFactor - 1) / gpuCollisionMeshSDFCoarseDownsampleFactor)),
            min(denseResolution.z, max(12, (denseResolution.z + gpuCollisionMeshSDFCoarseDownsampleFactor - 1) / gpuCollisionMeshSDFCoarseDownsampleFactor))
        )
        let atlasVoxelCount = atlasResolution.x * atlasResolution.y * atlasResolution.z
        let coarseVoxelCount = coarseResolution.x * coarseResolution.y * coarseResolution.z
        let indirectionVoxelCount = brickGrid.x * brickGrid.y * brickGrid.z
        var atlasData = [Float](repeating: 0, count: atlasVoxelCount)
        var indirectionData = [UInt32](repeating: UInt32.max, count: indirectionVoxelCount)
        var coarseData = [Float](repeating: 0, count: coarseVoxelCount)

        for (mappedIndex, brick) in mappedBricks.enumerated() {
            indirectionData[collisionMeshSDFLinearIndex(x: brick.x, y: brick.y, z: brick.z, resolution: brickGrid)] = UInt32(mappedIndex)

            let atlasBrickZ = mappedIndex / (gpuCollisionMeshSDFAtlasBricksAcross * gpuCollisionMeshSDFAtlasBricksAcross)
            let atlasBrickY = (mappedIndex / gpuCollisionMeshSDFAtlasBricksAcross) % gpuCollisionMeshSDFAtlasBricksAcross
            let atlasBrickX = mappedIndex % gpuCollisionMeshSDFAtlasBricksAcross
            let atlasBaseX = atlasBrickX * gpuCollisionMeshSDFStoredBrickDim
            let atlasBaseY = atlasBrickY * gpuCollisionMeshSDFStoredBrickDim
            let atlasBaseZ = atlasBrickZ * gpuCollisionMeshSDFStoredBrickDim

            for localZ in 0..<gpuCollisionMeshSDFStoredBrickDim {
                let sourceZ = max(0, min(denseResolution.z - 1, brick.z * gpuCollisionMeshSDFBrickDim + localZ - gpuCollisionMeshSDFGuardVoxelCount))
                for localY in 0..<gpuCollisionMeshSDFStoredBrickDim {
                    let sourceY = max(0, min(denseResolution.y - 1, brick.y * gpuCollisionMeshSDFBrickDim + localY - gpuCollisionMeshSDFGuardVoxelCount))
                    for localX in 0..<gpuCollisionMeshSDFStoredBrickDim {
                        let sourceX = max(0, min(denseResolution.x - 1, brick.x * gpuCollisionMeshSDFBrickDim + localX - gpuCollisionMeshSDFGuardVoxelCount))
                        let sourceIndex = collisionMeshSDFLinearIndex(x: sourceX, y: sourceY, z: sourceZ, resolution: denseResolution)
                        let atlasIndex = collisionMeshSDFLinearIndex(
                            x: atlasBaseX + localX,
                            y: atlasBaseY + localY,
                            z: atlasBaseZ + localZ,
                            resolution: atlasResolution
                        )
                        atlasData[atlasIndex] = denseData[sourceIndex]
                    }
                }
            }
        }

        for z in 0..<coarseResolution.z {
            let uz = Float(z) / Float(max(coarseResolution.z - 1, 1))
            for y in 0..<coarseResolution.y {
                let uy = Float(y) / Float(max(coarseResolution.y - 1, 1))
                for x in 0..<coarseResolution.x {
                    let ux = Float(x) / Float(max(coarseResolution.x - 1, 1))
                    coarseData[collisionMeshSDFLinearIndex(x: x, y: y, z: z, resolution: coarseResolution)] =
                        sampleCollisionMeshSDFDenseData(
                            denseData,
                            resolution: denseResolution,
                            uv: SIMD3<Float>(ux, uy, uz)
                        )
                }
            }
        }

        return CollisionMeshSDFCompactionData(
            coarseResolution: coarseResolution,
            atlasResolution: atlasResolution,
            brickGrid: brickGrid,
            coarseData: coarseData,
            atlasData: atlasData,
            indirectionData: indirectionData,
            mappedBrickCount: mappedBricks.count,
            denseByteCount: denseData.count * MemoryLayout<Float>.stride,
            compactedByteCount: (coarseVoxelCount + atlasVoxelCount) * MemoryLayout<Float>.stride
                + indirectionVoxelCount * MemoryLayout<UInt32>.stride
        )
    }

    private static func collisionMeshSDFBrickGrid(resolution: SIMD3<Int>) -> SIMD3<Int> {
        SIMD3<Int>(
            (resolution.x + gpuCollisionMeshSDFBrickDim - 1) / gpuCollisionMeshSDFBrickDim,
            (resolution.y + gpuCollisionMeshSDFBrickDim - 1) / gpuCollisionMeshSDFBrickDim,
            (resolution.z + gpuCollisionMeshSDFBrickDim - 1) / gpuCollisionMeshSDFBrickDim
        )
    }

    static func collisionMeshSDFSparseBandHalfWidth(voxelSize: SIMD3<Float>) -> Float {
        let brickWorldSize = voxelSize * Float(gpuCollisionMeshSDFStoredBrickDim)
        let brickHalfDiagonal = 0.5 * simd_length(brickWorldSize)
        let maxVoxel = max(voxelSize.x, max(voxelSize.y, voxelSize.z))
        return brickHalfDiagonal + max(maxVoxel * 2.0, gpuCollisionMargin * 6.0)
    }

    private static func collisionMeshSDFLinearIndex(
        x: Int,
        y: Int,
        z: Int,
        resolution: SIMD3<Int>
    ) -> Int {
        (z * resolution.y + y) * resolution.x + x
    }

    private static func sampleCollisionMeshSDFDenseData(
        _ denseData: [Float],
        resolution: SIMD3<Int>,
        uv: SIMD3<Float>
    ) -> Float {
        let size = SIMD3<Float>(
            Float(max(resolution.x, 1)),
            Float(max(resolution.y, 1)),
            Float(max(resolution.z, 1))
        )
        let maxCoord = SIMD3<Float>(
            Float(max(resolution.x - 1, 0)),
            Float(max(resolution.y - 1, 0)),
            Float(max(resolution.z - 1, 0))
        )
        let coord = simd_clamp(
            simd_clamp(uv, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1)) * size - 0.5,
            SIMD3<Float>(repeating: 0),
            maxCoord
        )
        let x0 = Int(floor(coord.x))
        let y0 = Int(floor(coord.y))
        let z0 = Int(floor(coord.z))
        let x1 = min(x0 + 1, resolution.x - 1)
        let y1 = min(y0 + 1, resolution.y - 1)
        let z1 = min(z0 + 1, resolution.z - 1)
        let tx = coord.x - Float(x0)
        let ty = coord.y - Float(y0)
        let tz = coord.z - Float(z0)

        func value(_ x: Int, _ y: Int, _ z: Int) -> Float {
            denseData[collisionMeshSDFLinearIndex(x: x, y: y, z: z, resolution: resolution)]
        }

        let c000 = value(x0, y0, z0)
        let c100 = value(x1, y0, z0)
        let c010 = value(x0, y1, z0)
        let c110 = value(x1, y1, z0)
        let c001 = value(x0, y0, z1)
        let c101 = value(x1, y0, z1)
        let c011 = value(x0, y1, z1)
        let c111 = value(x1, y1, z1)

        let c00 = c000 + (c100 - c000) * tx
        let c10 = c010 + (c110 - c010) * tx
        let c01 = c001 + (c101 - c001) * tx
        let c11 = c011 + (c111 - c011) * tx
        let c0 = c00 + (c10 - c00) * ty
        let c1 = c01 + (c11 - c01) * ty
        return c0 + (c1 - c0) * tz
    }

    static func collisionMeshSDFPadding(for mesh: AVBDCollisionMeshBroadphaseMesh) -> SIMD3<Float> {
        let extent = max(mesh.localBoundsMax - mesh.localBoundsMin, SIMD3<Float>(repeating: 1.0e-3))
        return max(extent * 0.05, SIMD3<Float>(repeating: gpuCollisionMargin * 4.0))
    }

    static func collisionMeshSDFResolution(
        localMinBounds: SIMD3<Float>,
        localMaxBounds: SIMD3<Float>
    ) -> SIMD3<Int32> {
        let extent = max(localMaxBounds - localMinBounds, SIMD3<Float>(repeating: 1.0e-3))
        let longestAxis = max(extent.x, max(extent.y, extent.z))
        let voxelScale = Float(gpuCollisionMeshSDFLongestAxisResolution) / max(longestAxis, 1.0e-3)
        let minResolution = SIMD3<Int32>(repeating: 16)
        let maxResolution = SIMD3<Int32>(repeating: gpuCollisionMeshSDFLongestAxisResolution)
        let scaled = SIMD3<Int32>(
            Int32(clamping: Int(ceil(extent.x * voxelScale))),
            Int32(clamping: Int(ceil(extent.y * voxelScale))),
            Int32(clamping: Int(ceil(extent.z * voxelScale)))
        )
        return simd_clamp(max(scaled, minResolution), minResolution, maxResolution)
    }

    private static func transformedBounds(
        localMinBounds: SIMD3<Float>,
        localMaxBounds: SIMD3<Float>,
        transform: simd_float4x4
    ) -> (min: SIMD3<Float>, max: SIMD3<Float>) {
        var minBounds = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maxBounds = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)

        for x in [localMinBounds.x, localMaxBounds.x] {
            for y in [localMinBounds.y, localMaxBounds.y] {
                for z in [localMinBounds.z, localMaxBounds.z] {
                    let transformed = transform * SIMD4<Float>(x, y, z, 1)
                    let point = SIMD3<Float>(transformed.x, transformed.y, transformed.z)
                    minBounds = simd_min(minBounds, point)
                    maxBounds = simd_max(maxBounds, point)
                }
            }
        }

        return (minBounds, maxBounds)
    }

    private static func collisionMeshSDFCacheKey(
        mesh: AVBDCollisionMeshBroadphaseMesh,
        localBoundsMin: SIMD3<Float>,
        localBoundsMax: SIMD3<Float>,
        resolution: SIMD3<Int32>
    ) -> String {
        if !mesh.sdfResourceID.isEmpty {
            return "\(mesh.sdfResourceID)#\(localBoundsMin.x.bitPattern):\(localBoundsMin.y.bitPattern):\(localBoundsMin.z.bitPattern):\(localBoundsMax.x.bitPattern):\(localBoundsMax.y.bitPattern):\(localBoundsMax.z.bitPattern):\(resolution.x):\(resolution.y):\(resolution.z)"
        }

        var hasher = Hasher()
        hasher.combine(mesh.positions.count)
        hasher.combine(mesh.indices.count)
        hasher.combine(localBoundsMin.x.bitPattern)
        hasher.combine(localBoundsMin.y.bitPattern)
        hasher.combine(localBoundsMin.z.bitPattern)
        hasher.combine(localBoundsMax.x.bitPattern)
        hasher.combine(localBoundsMax.y.bitPattern)
        hasher.combine(localBoundsMax.z.bitPattern)
        hasher.combine(resolution.x)
        hasher.combine(resolution.y)
        hasher.combine(resolution.z)

        for position in mesh.positions.prefix(64) {
            hasher.combine(position.x.bitPattern)
            hasher.combine(position.y.bitPattern)
            hasher.combine(position.z.bitPattern)
        }
        for index in mesh.indices.prefix(256) {
            hasher.combine(index)
        }
        return String(hasher.finalize())
    }

    func invalidateBroadphaseCache() {
        forceFullBroadphase = true
    }

    private func ensureCollisionCapacity(forMeshCount meshCount: Int) {
        let requiredPrimitiveMeshManifoldCapacity = max(bodyCapacity * max(meshCount, 1), 1)
        let requiredMeshMeshManifoldCapacity = max(meshCount * max(meshCount - 1, 0) / 2, 1)
        let requiredTrackedPairCapacity = max(min(requiredMeshMeshManifoldCapacity, gpuMeshMeshIsoVoxelTrackedPairLimit), 1)
        let requiredMaxCollisions = bodyBodyManifoldCapacity
            + requiredPrimitiveMeshManifoldCapacity
            + requiredMeshMeshManifoldCapacity
        if requiredMaxCollisions <= maxCollisions {
            primitiveMeshManifoldCapacity = requiredPrimitiveMeshManifoldCapacity
            meshMeshManifoldCapacity = requiredMeshMeshManifoldCapacity
            ensureMeshMeshIsoVoxelDebugCapacity(forTrackedPairCount: requiredTrackedPairCapacity)
            return
        }

        let manifoldBufSize = max(MemoryLayout<AVBDGPUManifold>.stride * requiredMaxCollisions, 16)
        let contactBufSize = max(MemoryLayout<AVBDGPUContact>.stride * requiredMaxCollisions * SWIFT_AVBD_MAX_CONTACTS_PER_PAIR, 16)
        let activeManifoldIndexBufSize = max(MemoryLayout<Int32>.stride * requiredMaxCollisions, 16)
        let activeManifoldStateBufSize = max(MemoryLayout<AVBDGPUActiveManifoldListState>.stride, 16)
        let activeManifoldIndirectBufSize = max(MemoryLayout<AVBDGPUIndirectDispatchArgs>.stride, 16)

        guard let mb = device.makeBuffer(length: manifoldBufSize, options: .storageModeShared),
              let ctb = device.makeBuffer(length: contactBufSize, options: .storageModeShared),
              let ami = device.makeBuffer(length: activeManifoldIndexBufSize, options: .storageModeShared),
              let ams = device.makeBuffer(length: activeManifoldStateBufSize, options: .storageModeShared),
              let amd = device.makeBuffer(length: activeManifoldIndirectBufSize, options: .storageModeShared),
              let pmb = device.makeBuffer(length: manifoldBufSize, options: .storageModeShared),
              let pctb = device.makeBuffer(length: contactBufSize, options: .storageModeShared),
              let pami = device.makeBuffer(length: activeManifoldIndexBufSize, options: .storageModeShared),
              let pams = device.makeBuffer(length: activeManifoldStateBufSize, options: .storageModeShared)
        else {
            return
        }

        manifoldBuffer = mb
        contactBuffer = ctb
        activeManifoldIndexBuffer = ami
        activeManifoldStateBuffer = ams
        activeManifoldIndirectBuffer = amd
        prevManifoldBuffer = pmb
        prevContactBuffer = pctb
        prevActiveManifoldIndexBuffer = pami
        prevActiveManifoldStateBuffer = pams
        primitiveMeshManifoldCapacity = requiredPrimitiveMeshManifoldCapacity
        meshMeshManifoldCapacity = requiredMeshMeshManifoldCapacity
        maxCollisions = requiredMaxCollisions
        contactCapacity = requiredMaxCollisions * SWIFT_AVBD_MAX_CONTACTS_PER_PAIR
        ensureMeshMeshIsoVoxelDebugCapacity(forTrackedPairCount: requiredTrackedPairCapacity)

        let allocatorPtr = contactAllocatorBuffer.contents().bindMemory(to: AVBDGPUContactAllocator.self, capacity: 1)
        allocatorPtr.pointee = AVBDGPUContactAllocator(nextContactIndex: Int32(0), contactCapacity: Int32(contactCapacity))

        let activeManifoldStatePtr = activeManifoldStateBuffer.contents().bindMemory(to: AVBDGPUActiveManifoldListState.self, capacity: 1)
        activeManifoldStatePtr.pointee = AVBDGPUActiveManifoldListState(count: Int32(0), capacity: Int32(maxCollisions))

        let prevActiveManifoldStatePtr = prevActiveManifoldStateBuffer.contents().bindMemory(to: AVBDGPUActiveManifoldListState.self, capacity: 1)
        prevActiveManifoldStatePtr.pointee = AVBDGPUActiveManifoldListState(count: Int32(0), capacity: Int32(maxCollisions))

        let activeManifoldIndirectPtr = activeManifoldIndirectBuffer.contents().bindMemory(to: AVBDGPUIndirectDispatchArgs.self, capacity: 1)
        activeManifoldIndirectPtr.pointee = AVBDGPUIndirectDispatchArgs(threadgroupsPerGrid: (0, 1, 1))
    }

    private func ensureMeshMeshIsoVoxelDebugCapacity(forTrackedPairCount trackedPairCount: Int) {
        let requiredTrackedPairCount = max(trackedPairCount, 1)
        let requiredDebugBufSize = max(
            MemoryLayout<AVBDGPUMeshMeshIsoVoxelDebug>.stride * requiredTrackedPairCount,
            16
        )
        let requiredCoordBufSize = max(
            MemoryLayout<AVBDGPUMeshMeshIsoVoxelCoord>.stride
                * requiredTrackedPairCount
                * gpuMeshMeshIsoVoxelCoordsPerPair,
            16
        )

        if meshMeshIsoVoxelDebugBuffer.length < requiredDebugBufSize {
            guard let newBuffer = device.makeBuffer(length: requiredDebugBufSize, options: .storageModeShared) else {
                return
            }
            newBuffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: newBuffer.length)
            meshMeshIsoVoxelDebugBuffer = newBuffer
        }

        if meshMeshIsoVoxelCoordBuffer.length < requiredCoordBufSize {
            guard let newBuffer = device.makeBuffer(length: requiredCoordBufSize, options: .storageModeShared) else {
                return
            }
            newBuffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: newBuffer.length)
            meshMeshIsoVoxelCoordBuffer = newBuffer
        }

        meshMeshIsoVoxelTrackedPairCapacity = requiredTrackedPairCount
    }

    private static func worldBounds(
        for mesh: AVBDCollisionMeshBroadphaseMesh
    ) -> (min: SIMD3<Float>, max: SIMD3<Float>) {
        let minLocal = mesh.localBoundsMin
        let maxLocal = mesh.localBoundsMax
        let corners: [SIMD3<Float>] = [
            SIMD3<Float>(minLocal.x, minLocal.y, minLocal.z),
            SIMD3<Float>(maxLocal.x, minLocal.y, minLocal.z),
            SIMD3<Float>(minLocal.x, maxLocal.y, minLocal.z),
            SIMD3<Float>(maxLocal.x, maxLocal.y, minLocal.z),
            SIMD3<Float>(minLocal.x, minLocal.y, maxLocal.z),
            SIMD3<Float>(maxLocal.x, minLocal.y, maxLocal.z),
            SIMD3<Float>(minLocal.x, maxLocal.y, maxLocal.z),
            SIMD3<Float>(maxLocal.x, maxLocal.y, maxLocal.z),
        ]

        var worldMin = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var worldMax = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)

        for corner in corners {
            let transformed = mesh.transform * SIMD4<Float>(corner, 1)
            let world = SIMD3<Float>(transformed.x, transformed.y, transformed.z)
            worldMin = simd_min(worldMin, world)
            worldMax = simd_max(worldMax, world)
        }

        return (worldMin, worldMax)
    }

    private func isJointConnected(_ a: Int, _ b: Int) -> Bool {
        Self.isJointConnected(a, b, joints: cpuJoints, springs: cpuSprings)
    }

    private static func isJointConnected(_ a: Int, _ b: Int, joints: [AVBDGPUJoint], springs: [AVBDGPUSpring]) -> Bool {
        for joint in joints {
            let ja = Int(joint.bodyA)
            let jb = Int(joint.bodyB)
            if (ja == a && jb == b) || (ja == b && jb == a) { return true }
        }
        for spring in springs {
            let sa = Int(spring.bodyA)
            let sb = Int(spring.bodyB)
            if (sa == a && sb == b) || (sa == b && sb == a) { return true }
        }
        return false
    }

    private static func pairKey(_ a: Int, _ b: Int) -> UInt64 {
        let lo = UInt64(min(a, b))
        let hi = UInt64(max(a, b))
        return (hi << 32) | lo
    }

    private func collideBodies(_ indexA: Int, _ indexB: Int, basisOut: inout AVBDMat3) -> [AVBDGPUContact] {
        let bodyA = cpuBodies[indexA]
        let bodyB = cpuBodies[indexB]

        let boxA = makeOBB(bodyA)
        let boxB = makeOBB(bodyB)
        let delta = boxB.center - boxA.center

        var bestFace = AVBDGPUSatAxis()
        var bestEdge = AVBDGPUSatAxis()

        for i in 0..<3 {
            if !testAxis(boxA, boxB, delta, boxA.axis[i], .faceA, i, -1, &bestFace) {
                return []
            }
        }
        for i in 0..<3 {
            if !testAxis(boxA, boxB, delta, boxB.axis[i], .faceB, -1, i, &bestFace) {
                return []
            }
        }

        for i in 0..<3 {
            for j in 0..<3 {
                if !testAxis(boxA, boxB, delta, cross(boxA.axis[i], boxB.axis[j]), .edge, i, j, &bestEdge) {
                    return []
                }
            }
        }

        if !bestFace.valid {
            return []
        }

        var best = bestFace
        if bestEdge.valid && 0.95 * bestEdge.separation > bestFace.separation + 0.01 {
            best = bestEdge
        }

        basisOut = orthonormal(-best.normalAB)

        var contacts: [AVBDGPUContact]
        switch best.type {
        case .edge:
            contacts = buildEdgeContact(bodyA, bodyB, boxA, boxB, best.indexA, best.indexB, best.normalAB)
        case .faceA:
            contacts = buildFaceManifold(bodyA, bodyB, boxA, boxB, true, best.indexA, best.normalAB)
        case .faceB:
            contacts = buildFaceManifold(bodyA, bodyB, boxA, boxB, false, best.indexB, best.normalAB)
        }

        for index in contacts.indices {
            let xA = boxA.rotation.act(contacts[index].rA) + bodyA.positionLin
            let xB = boxB.rotation.act(contacts[index].rB) + bodyB.positionLin
            let diff = xA - xB
            contacts[index].C0 = SIMD3<Float>(dot(basisOut.r0, diff), dot(basisOut.r1, diff), dot(basisOut.r2, diff)) + SIMD3<Float>(gpuCollisionMargin, 0, 0)
            contacts[index].penalty = SIMD3<Float>(repeating: gpuContactPenaltyStart)
            contacts[index].active = 1
        }

        return contacts
    }

    private func makeOBB(_ body: AVBDGPUBody) -> AVBDGPUOBB {
        let rotation = simd_quatf(vector: body.positionAng)
        return AVBDGPUOBB(
            center: body.positionLin,
            rotation: rotation,
            half: body.size * 0.5,
            axis: [
                rotation.act(SIMD3<Float>(1, 0, 0)),
                rotation.act(SIMD3<Float>(0, 1, 0)),
                rotation.act(SIMD3<Float>(0, 0, 1))
            ]
        )
    }

    private func absDot(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        abs(dot(a, b))
    }

    private func supportPoint(_ box: AVBDGPUOBB, _ dir: SIMD3<Float>) -> SIMD3<Float> {
        let sx: Float = dot(dir, box.axis[0]) >= 0 ? 1 : -1
        let sy: Float = dot(dir, box.axis[1]) >= 0 ? 1 : -1
        let sz: Float = dot(dir, box.axis[2]) >= 0 ? 1 : -1

        return box.center
            + box.axis[0] * (box.half.x * sx)
            + box.axis[1] * (box.half.y * sy)
            + box.axis[2] * (box.half.z * sz)
    }

    private func getFaceAxes(_ box: AVBDGPUOBB, _ axisIndex: Int) -> (SIMD3<Float>, SIMD3<Float>, Float, Float) {
        if axisIndex == 0 {
            return (box.axis[1], box.axis[2], box.half.y, box.half.z)
        } else if axisIndex == 1 {
            return (box.axis[0], box.axis[2], box.half.x, box.half.z)
        } else {
            return (box.axis[0], box.axis[1], box.half.x, box.half.y)
        }
    }

    private func buildFaceFrame(_ box: AVBDGPUOBB, _ axisIndex: Int, _ outwardNormal: SIMD3<Float>) -> AVBDGPUFaceFrame {
        let faceSign: Float = dot(outwardNormal, box.axis[axisIndex]) >= 0 ? 1 : -1
        let normal = box.axis[axisIndex] * faceSign
        let (u, v, extentU, extentV) = getFaceAxes(box, axisIndex)
        return AVBDGPUFaceFrame(
            normal: normal,
            center: box.center + normal * box.half[axisIndex],
            u: u,
            v: v,
            extentU: extentU,
            extentV: extentV
        )
    }

    private func chooseIncidentFaceAxis(_ box: AVBDGPUOBB, _ referenceNormal: SIMD3<Float>) -> Int {
        var axis = 0
        var best = -Float.greatestFiniteMagnitude
        for i in 0..<3 {
            let d = absDot(box.axis[i], referenceNormal)
            if d > best {
                best = d
                axis = i
            }
        }
        return axis
    }

    private func buildIncidentFace(_ box: AVBDGPUOBB, _ axisIndex: Int, _ referenceNormal: SIMD3<Float>) -> [SIMD3<Float>] {
        let faceSign: Float = dot(box.axis[axisIndex], referenceNormal) > 0 ? -1 : 1
        let faceNormal = box.axis[axisIndex] * faceSign
        let faceCenter = box.center + faceNormal * box.half[axisIndex]
        let (u, v, extentU, extentV) = getFaceAxes(box, axisIndex)

        return [
            faceCenter + u * extentU + v * extentV,
            faceCenter - u * extentU + v * extentV,
            faceCenter - u * extentU - v * extentV,
            faceCenter + u * extentU - v * extentV
        ]
    }

    private func clipPolygonAgainstPlane(_ inVerts: [SIMD3<Float>], _ planeNormal: SIMD3<Float>, _ planeOffset: Float) -> [SIMD3<Float>] {
        if inVerts.isEmpty {
            return []
        }

        var outVerts: [SIMD3<Float>] = []
        var a = inVerts[inVerts.count - 1]
        var da = dot(planeNormal, a) - planeOffset

        for b in inVerts {
            let db = dot(planeNormal, b) - planeOffset
            let aInside = da <= gpuPlaneEpsilon
            let bInside = db <= gpuPlaneEpsilon

            if aInside != bInside {
                var t: Float = 0
                let denom = da - db
                if abs(denom) > gpuSatAxisEpsilon {
                    t = min(max(da / denom, 0.0), 1.0)
                }
                if outVerts.count < gpuMaxPolyVerts {
                    outVerts.append(a + (b - a) * t)
                }
            }

            if bInside && outVerts.count < gpuMaxPolyVerts {
                outVerts.append(b)
            }

            a = b
            da = db
        }

        return outVerts
    }

    private func makeContact(bodyA: AVBDGPUBody, bodyB: AVBDGPUBody, xA: SIMD3<Float>, xB: SIMD3<Float>, featureKey: Int) -> AVBDGPUContact {
        let quatA = simd_quatf(vector: bodyA.positionAng)
        let quatB = simd_quatf(vector: bodyB.positionAng)
        var contact = AVBDGPUContact()
        contact.featureKey = Int32(featureKey)
        contact.rA = quatA.inverse.act(xA - bodyA.positionLin)
        contact.rB = quatB.inverse.act(xB - bodyB.positionLin)
        return contact
    }

    private func appendContact(
        _ contacts: inout [AVBDGPUContact],
        _ contactMidpoints: inout [SIMD3<Float>],
        bodyA: AVBDGPUBody,
        bodyB: AVBDGPUBody,
        xA: SIMD3<Float>,
        xB: SIMD3<Float>,
        featureKey: Int
    ) {
        let midpoint = (xA + xB) * 0.5
        for existing in contactMidpoints where simd_length_squared(midpoint - existing) < gpuContactMergeDistSq {
            return
        }
        if contacts.count >= SWIFT_AVBD_MAX_CONTACTS_PER_PAIR_BURST {
            return
        }
        contacts.append(makeContact(bodyA: bodyA, bodyB: bodyB, xA: xA, xB: xB, featureKey: featureKey))
        contactMidpoints.append(midpoint)
    }

    private func testAxis(
        _ boxA: AVBDGPUOBB,
        _ boxB: AVBDGPUOBB,
        _ delta: SIMD3<Float>,
        _ axis: SIMD3<Float>,
        _ type: AVBDAxisType,
        _ indexA: Int,
        _ indexB: Int,
        _ best: inout AVBDGPUSatAxis
    ) -> Bool {
        let lenSq = simd_length_squared(axis)
        if lenSq < gpuSatAxisEpsilon {
            return true
        }

        var n = axis / sqrt(lenSq)
        if dot(n, delta) < 0 {
            n = -n
        }

        let distance = abs(dot(delta, n))
        let rA = boxA.half.x * absDot(n, boxA.axis[0])
            + boxA.half.y * absDot(n, boxA.axis[1])
            + boxA.half.z * absDot(n, boxA.axis[2])
        let rB = boxB.half.x * absDot(n, boxB.axis[0])
            + boxB.half.y * absDot(n, boxB.axis[1])
            + boxB.half.z * absDot(n, boxB.axis[2])

        let separation = distance - (rA + rB)
        if separation > 0 {
            return false
        }

        if !best.valid || separation > best.separation {
            best.valid = true
            best.type = type
            best.indexA = indexA
            best.indexB = indexB
            best.separation = separation
            best.normalAB = n
        }

        return true
    }

    private func supportEdge(_ box: AVBDGPUOBB, _ axisIndex: Int, _ dir: SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) {
        let axis1 = (axisIndex + 1) % 3
        let axis2 = (axisIndex + 2) % 3
        let sign1: Float = dot(dir, box.axis[axis1]) >= 0 ? 1 : -1
        let sign2: Float = dot(dir, box.axis[axis2]) >= 0 ? 1 : -1
        let edgeCenter = box.center
            + box.axis[axis1] * (box.half[axis1] * sign1)
            + box.axis[axis2] * (box.half[axis2] * sign2)
        return (
            edgeCenter - box.axis[axisIndex] * box.half[axisIndex],
            edgeCenter + box.axis[axisIndex] * box.half[axisIndex]
        )
    }

    private func closestPointsOnSegments(
        _ p0: SIMD3<Float>,
        _ p1: SIMD3<Float>,
        _ q0: SIMD3<Float>,
        _ q1: SIMD3<Float>
    ) -> (SIMD3<Float>, SIMD3<Float>) {
        let d1 = p1 - p0
        let d2 = q1 - q0
        let r = p0 - q0
        let a = dot(d1, d1)
        let e = dot(d2, d2)
        let f = dot(d2, r)
        var s: Float = 0
        var t: Float = 0

        if a <= gpuSatAxisEpsilon && e <= gpuSatAxisEpsilon {
            return (p0, q0)
        }

        if a <= gpuSatAxisEpsilon {
            t = min(max(f / e, 0), 1)
        } else {
            let c = dot(d1, r)
            if e <= gpuSatAxisEpsilon {
                s = min(max(-c / a, 0), 1)
            } else {
                let b = dot(d1, d2)
                let denom = a * e - b * b
                if abs(denom) > gpuSatAxisEpsilon {
                    s = min(max((b * f - c * e) / denom, 0), 1)
                }
                t = (b * s + f) / e
                if t < 0 {
                    t = 0
                    s = min(max(-c / a, 0), 1)
                } else if t > 1 {
                    t = 1
                    s = min(max((b - c) / a, 0), 1)
                }
            }
        }

        return (p0 + d1 * s, q0 + d2 * t)
    }

    private func buildFaceManifold(
        _ bodyA: AVBDGPUBody,
        _ bodyB: AVBDGPUBody,
        _ boxA: AVBDGPUOBB,
        _ boxB: AVBDGPUOBB,
        _ referenceIsA: Bool,
        _ referenceAxis: Int,
        _ normalAB: SIMD3<Float>
    ) -> [AVBDGPUContact] {
        let referenceBox = referenceIsA ? boxA : boxB
        let incidentBox = referenceIsA ? boxB : boxA
        let referenceOutward = referenceIsA ? normalAB : -normalAB
        let referenceFace = buildFaceFrame(referenceBox, referenceAxis, referenceOutward)
        let incidentAxis = chooseIncidentFaceAxis(incidentBox, referenceFace.normal)

        var clipped = buildIncidentFace(incidentBox, incidentAxis, referenceFace.normal)
        clipped = clipPolygonAgainstPlane(clipped, referenceFace.u, dot(referenceFace.u, referenceFace.center) + referenceFace.extentU)
        if clipped.isEmpty { return [] }
        clipped = clipPolygonAgainstPlane(clipped, -referenceFace.u, dot(-referenceFace.u, referenceFace.center) + referenceFace.extentU)
        if clipped.isEmpty { return [] }
        clipped = clipPolygonAgainstPlane(clipped, referenceFace.v, dot(referenceFace.v, referenceFace.center) + referenceFace.extentV)
        if clipped.isEmpty { return [] }
        clipped = clipPolygonAgainstPlane(clipped, -referenceFace.v, dot(-referenceFace.v, referenceFace.center) + referenceFace.extentV)
        if clipped.isEmpty { return [] }

        var contacts: [AVBDGPUContact] = []
        var contactMidpoints: [SIMD3<Float>] = []
        var featurePrefix = Int(referenceIsA ? AVBDAxisType.faceA.rawValue : AVBDAxisType.faceB.rawValue) << 24
        featurePrefix |= (referenceAxis & 0xFF) << 16
        featurePrefix |= (incidentAxis & 0xFF) << 8

        for i in clipped.indices where contacts.count < SWIFT_AVBD_MAX_CONTACTS_PER_PAIR_BURST {
            let pIncident = clipped[i]
            let distance = dot(pIncident - referenceFace.center, referenceFace.normal)
            if distance > gpuPlaneEpsilon {
                continue
            }

            let pReference = pIncident - referenceFace.normal * distance
            let xA = referenceIsA ? pReference : pIncident
            let xB = referenceIsA ? pIncident : pReference
            appendContact(&contacts, &contactMidpoints, bodyA: bodyA, bodyB: bodyB, xA: xA, xB: xB, featureKey: featurePrefix | (i & 0xFF))
        }

        if contacts.isEmpty {
            appendContact(
                &contacts,
                &contactMidpoints,
                bodyA: bodyA,
                bodyB: bodyB,
                xA: supportPoint(boxA, normalAB),
                xB: supportPoint(boxB, -normalAB),
                featureKey: featurePrefix
            )
        }

        return contacts
    }

    private func buildEdgeContact(
        _ bodyA: AVBDGPUBody,
        _ bodyB: AVBDGPUBody,
        _ boxA: AVBDGPUOBB,
        _ boxB: AVBDGPUOBB,
        _ axisA: Int,
        _ axisB: Int,
        _ normalAB: SIMD3<Float>
    ) -> [AVBDGPUContact] {
        let (a0, a1) = supportEdge(boxA, axisA, normalAB)
        let (b0, b1) = supportEdge(boxB, axisB, -normalAB)
        var (xA, xB) = closestPointsOnSegments(a0, a1, b0, b1)

        var contacts: [AVBDGPUContact] = []
        var contactMidpoints: [SIMD3<Float>] = []
        let featureKey = (Int(AVBDAxisType.edge.rawValue) << 24) | ((axisA & 0xFF) << 8) | (axisB & 0xFF)
        appendContact(&contacts, &contactMidpoints, bodyA: bodyA, bodyB: bodyB, xA: xA, xB: xB, featureKey: featureKey)

        if contacts.isEmpty {
            xA = supportPoint(boxA, normalAB)
            xB = supportPoint(boxB, -normalAB)
            appendContact(&contacts, &contactMidpoints, bodyA: bodyA, bodyB: bodyB, xA: xA, xB: xB, featureKey: featureKey)
        }

        return contacts
    }

    private func orthonormal(_ normal: SIMD3<Float>) -> AVBDMat3 {
        var t1: SIMD3<Float>
        if abs(normal.x) > abs(normal.z) {
            t1 = SIMD3<Float>(-normal.y, normal.x, 0)
        } else {
            t1 = SIMD3<Float>(0, -normal.z, normal.y)
        }
        let len = simd_length(t1)
        t1 = len > 1.0e-8 ? t1 / len : SIMD3<Float>(1, 0, 0)
        let t2 = cross(normal, t1)
        return AVBDMat3(normal, t1, t2)
    }
}
