//
//  AVBDGPUSolver.swift
//  MetalAVBD
//
//  GPU-accelerated AVBD solver using Metal compute shaders.
//  Mirrors the CPU AVBDSolver but runs the iteration loop on GPU.
//  Broadphase collision detection and contact generation also run on GPU.
//

import Metal
import simd

private let SWIFT_AVBD_MAX_CONTACTS_PER_PAIR = Int(AVBD_MAX_CONTACTS_PER_PAIR)
private let SWIFT_AVBD_MAX_CONTACTS_PER_PAIR_BURST = SWIFT_AVBD_MAX_CONTACTS_PER_PAIR * 2
private let SWIFT_AVBD_MAX_CONSTRAINTS_PER_BODY = Int(AVBD_MAX_CONSTRAINTS_PER_BODY)
private let SWIFT_AVBD_BROADPHASE_THREADGROUP_SIZE = Int(AVBD_BROADPHASE_THREADGROUP_SIZE)
private let SWIFT_AVBD_DERIVED_THREADGROUP_SIZE = Int(AVBD_DERIVED_THREADGROUP_SIZE)
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

final class AVBDGPUSolver {
    let device: MTLDevice

    // Compute pipelines
    private let forwardIntegratePSO: MTLComputePipelineState
    private let resetAdjacencyPSO: MTLComputePipelineState
    private let broadphaseFullPSO: MTLComputePipelineState
    private let broadphasePartialPSO: MTLComputePipelineState
    private let prepareBroadphaseIndirectPSO: MTLComputePipelineState
    private let processBroadphaseDerivedPSO: MTLComputePipelineState
    private let prepareDerivedPairsIndirectPSO: MTLComputePipelineState
    private let prepareActiveManifoldsIndirectPSO: MTLComputePipelineState
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
    private var manifoldAllocatorBuffer: MTLBuffer
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
    private var paramsBuffer: MTLBuffer
    private var maxCollisions: Int
    private var recentPairCapacity: Int
    private var contactCapacity: Int
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

    private func resolvedRenderColor(bodyIndex: Int) -> SIMD4<Float> {
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

    init?(device: MTLDevice, scene: AVBDScene) {
        self.device = device

        guard let library = device.makeDefaultLibrary() else { return nil }

        func makePSO(_ name: String) -> MTLComputePipelineState? {
            guard let fn = library.makeFunction(name: name) else {
                print("Missing compute function: \(name)")
                return nil
            }
            return try? device.makeComputePipelineState(function: fn)
        }

        guard let fwd = makePSO("avbd_forward_integrate"),
              let ra  = makePSO("avbd_reset_adjacency"),
              let bpf = makePSO("avbd_broadphase_full"),
              let bpp = makePSO("avbd_broadphase_partial"),
              let pbi = makePSO("avbd_prepare_broadphase_indirect"),
              let pbd = makePSO("avbd_process_broadphase_pair_derived"),
              let pdi = makePSO("avbd_prepare_derived_pairs_indirect"),
              let pai = makePSO("avbd_prepare_active_manifolds_indirect"),
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
        processBroadphaseDerivedPSO = pbd
        prepareDerivedPairsIndirectPSO = pdi
        prepareActiveManifoldsIndirectPSO = pai
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
        maxCollisions = max(bodyCapacity * maxCollisionsPerBody, 1)
        recentPairCapacity = max(bodyCapacity * (bodyCapacity - 1) / 2, 1)

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
        let manifoldAllocatorBufSize = max(MemoryLayout<AVBDGPUManifoldAllocator>.stride, 16)
        let recentPairIndexBufSize = max(MemoryLayout<Int32>.stride * recentPairCapacity, 16)
        let recentPairStateBufSize = max(MemoryLayout<AVBDGPURecentPairCacheState>.stride, 16)
        let recentPairIndirectBufSize = max(MemoryLayout<AVBDGPUIndirectDispatchArgs>.stride, 16)
        let derivedPairCandidateBufSize = max(MemoryLayout<Int32>.stride * maxCollisions, 16)
        let derivedPairStateBufSize = max(MemoryLayout<AVBDGPUDerivedPairCandidateState>.stride, 16)
        let derivedPairIndirectBufSize = max(MemoryLayout<AVBDGPUIndirectDispatchArgs>.stride, 16)
        let activeManifoldIndexBufSize = max(MemoryLayout<Int32>.stride * maxCollisions, 16)
        let activeManifoldStateBufSize = max(MemoryLayout<AVBDGPUActiveManifoldListState>.stride, 16)
        let activeManifoldIndirectBufSize = max(MemoryLayout<AVBDGPUIndirectDispatchArgs>.stride, 16)

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
              let mab = device.makeBuffer(length: manifoldAllocatorBufSize, options: .storageModeShared),
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
        manifoldAllocatorBuffer = mab
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
        paramsBuffer = pb
        contactCapacity = maxCollisions * SWIFT_AVBD_MAX_CONTACTS_PER_PAIR
        recentPairCacheSlot = 0
        stepsSinceFullBroadphase = .max
        forceFullBroadphase = true

        let allocatorPtr = contactAllocatorBuffer.contents().bindMemory(to: AVBDGPUContactAllocator.self, capacity: 1)
        allocatorPtr.pointee = AVBDGPUContactAllocator(nextContactIndex: Int32(0), contactCapacity: Int32(contactCapacity))

        let manifoldAllocatorPtr = manifoldAllocatorBuffer.contents().bindMemory(to: AVBDGPUManifoldAllocator.self, capacity: 1)
        manifoldAllocatorPtr.pointee = AVBDGPUManifoldAllocator(nextManifoldIndex: Int32(0), manifoldCapacity: Int32(maxCollisions))

        for stateBuffer in recentPairStateBuffers {
            let statePtr = stateBuffer.contents().bindMemory(to: AVBDGPURecentPairCacheState.self, capacity: 1)
            statePtr.pointee = AVBDGPURecentPairCacheState(count: Int32(0), capacity: Int32(recentPairCapacity))
        }

        for indirectBuffer in recentPairIndirectBuffers {
            let indirectPtr = indirectBuffer.contents().bindMemory(to: AVBDGPUIndirectDispatchArgs.self, capacity: 1)
            indirectPtr.pointee = AVBDGPUIndirectDispatchArgs(threadgroupsPerGrid: (0, 1, 1))
        }

        let derivedPairStatePtr = derivedPairStateBuffer.contents().bindMemory(to: AVBDGPUDerivedPairCandidateState.self, capacity: 1)
        derivedPairStatePtr.pointee = AVBDGPUDerivedPairCandidateState(count: Int32(0), capacity: Int32(maxCollisions))

        let derivedPairIndirectPtr = derivedPairIndirectBuffer.contents().bindMemory(to: AVBDGPUIndirectDispatchArgs.self, capacity: 1)
        derivedPairIndirectPtr.pointee = AVBDGPUIndirectDispatchArgs(threadgroupsPerGrid: (0, 1, 1))

        let activeManifoldStatePtr = activeManifoldStateBuffer.contents().bindMemory(to: AVBDGPUActiveManifoldListState.self, capacity: 1)
        activeManifoldStatePtr.pointee = AVBDGPUActiveManifoldListState(count: Int32(0), capacity: Int32(maxCollisions))

        let prevActiveManifoldStatePtr = prevActiveManifoldStateBuffer.contents().bindMemory(to: AVBDGPUActiveManifoldListState.self, capacity: 1)
        prevActiveManifoldStatePtr.pointee = AVBDGPUActiveManifoldListState(count: Int32(0), capacity: Int32(maxCollisions))

        let activeManifoldIndirectPtr = activeManifoldIndirectBuffer.contents().bindMemory(to: AVBDGPUIndirectDispatchArgs.self, capacity: 1)
        activeManifoldIndirectPtr.pointee = AVBDGPUIndirectDispatchArgs(threadgroupsPerGrid: (0, 1, 1))

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
        let broadphaseFullThreadgroupCount = MTLSize(
            width: max((max(totalPairCount, 1) + SWIFT_AVBD_BROADPHASE_THREADGROUP_SIZE - 1) / SWIFT_AVBD_BROADPHASE_THREADGROUP_SIZE, 1),
            height: 1,
            depth: 1
        )
        let adjacencyConstraintThreads = MTLSize(width: max(cpuJoints.count, cpuSprings.count, 1), height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: SWIFT_AVBD_BROADPHASE_THREADGROUP_SIZE, height: 1, depth: 1)
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
        encoder.setBuffer(manifoldAllocatorBuffer, offset: 0, index: 4)
        encoder.setBuffer(derivedPairStateBuffer, offset: 0, index: 5)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 6)
        encoder.dispatchThreads(bodyThreads, threadsPerThreadgroup: threadgroupSize)

        if bodyCount > 1 {
            if shouldRunFullBroadphase {
                encoder.setComputePipelineState(broadphaseFullPSO)
                encoder.setBuffer(bodyBuffer, offset: 0, index: 0)
                encoder.setBuffer(exclusionBuffer, offset: 0, index: 1)
                encoder.setBuffer(manifoldAllocatorBuffer, offset: 0, index: 2)
                encoder.setBuffer(manifoldBuffer, offset: 0, index: 3)
                encoder.setBuffer(contactBuffer, offset: 0, index: 4)
                encoder.setBuffer(contactAllocatorBuffer, offset: 0, index: 5)
                encoder.setBuffer(nextRecentPairIndexBuffer, offset: 0, index: 6)
                encoder.setBuffer(nextRecentPairStateBuffer, offset: 0, index: 7)
                encoder.setBuffer(activeManifoldIndexBuffer, offset: 0, index: 8)
                encoder.setBuffer(activeManifoldStateBuffer, offset: 0, index: 9)
                encoder.setBuffer(paramsBuffer, offset: 0, index: 10)
                encoder.dispatchThreadgroups(broadphaseFullThreadgroupCount, threadsPerThreadgroup: threadgroupSize)
            } else {
                encoder.setComputePipelineState(broadphasePartialPSO)
                encoder.setBuffer(bodyBuffer, offset: 0, index: 0)
                encoder.setBuffer(currentRecentPairIndexBuffer, offset: 0, index: 1)
                encoder.setBuffer(currentRecentPairStateBuffer, offset: 0, index: 2)
                encoder.setBuffer(manifoldAllocatorBuffer, offset: 0, index: 3)
                encoder.setBuffer(manifoldBuffer, offset: 0, index: 4)
                encoder.setBuffer(contactBuffer, offset: 0, index: 5)
                encoder.setBuffer(contactAllocatorBuffer, offset: 0, index: 6)
                encoder.setBuffer(nextRecentPairIndexBuffer, offset: 0, index: 7)
                encoder.setBuffer(nextRecentPairStateBuffer, offset: 0, index: 8)
                encoder.setBuffer(activeManifoldIndexBuffer, offset: 0, index: 9)
                encoder.setBuffer(activeManifoldStateBuffer, offset: 0, index: 10)
                encoder.setBuffer(paramsBuffer, offset: 0, index: 11)
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
        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

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
            manifoldCapacity: Int32(maxCollisions),
            collisionMargin: gpuCollisionMargin,
            cacheMargin: effectiveCacheMargin,
            cacheTimeHorizon: cacheTimeHorizon,
            penaltyMin: gpuPenaltyMin,
            penaltyMax: gpuPenaltyMax,
            stickThreshold: gpuStickThreshold,
            torusApproxSphereCount: Int32(avbdCurrentTorusApproxSphereCount()),
            torusApproxSphereRadiusScale: AVBDTorusApproximationSettings.radiusScale
        )
    }

    func invalidateBroadphaseCache() {
        forceFullBroadphase = true
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
