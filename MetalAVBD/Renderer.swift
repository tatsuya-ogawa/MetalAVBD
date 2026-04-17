//
//  Renderer.swift
//  MetalAVBD
//
//  Created by Tatsuya Ogawa on 2026/04/07.
//

// Our platform independent renderer class

import Dispatch
import Foundation
import Metal
import MetalKit
import QuartzCore
import simd

// The 256 byte aligned size of our uniform structure
let alignedUniformsSize = (MemoryLayout<Uniforms>.size + 0xFF) & -0x100

let maxBuffersInFlight = 3

nonisolated enum RendererError: Error {
    case badVertexDescriptor
}

nonisolated enum AVBDSolverMode: Int {
    case gpu
    case cpu

    var displayName: String {
        switch self {
        case .gpu: return "GPU"
        case .cpu: return "CPU"
        }
    }
}

nonisolated enum AVBDSimulationRunMode: Int {
    case manual
    case auto

    var displayName: String {
        switch self {
        case .manual: return "Manual"
        case .auto: return "Auto"
        }
    }
}

nonisolated enum AVBDTorusVisualMode: Int {
    case proxySpheres
    case solidTorus

    var displayName: String {
        switch self {
        case .proxySpheres: return "Proxy"
        case .solidTorus: return "Torus"
        }
    }
}

nonisolated enum AVBDProjectileKind: Int {
    case box
    case sphere
    case torus
    case armadillo

    var displayName: String {
        switch self {
        case .box: return "Cube"
        case .sphere: return "Sphere"
        case .torus: return "Torus"
        case .armadillo: return "Armadillo"
        }
    }
}

nonisolated struct StaticMesh {
    var positions: MTLBuffer
    var normals: MTLBuffer
    var indexBuffer: MTLBuffer
    var indexCount: Int
}

struct RendererDebugStats: Sendable {
    var fps: Double
    var frameTimeMS: Double
    var stepCount: Int
    var bodyCount: Int
    var solverModeName: String
    var simulationModeName: String
    var broadphaseModeName: String
    var warmstartModeName: String
    var collisionSDFStatusName: String
}

private struct ArmadilloRenderProxy {
    var bodyIndex: Int
    var baseColor: SIMD4<Float>
}

private struct SceneArmadilloPlacement {
    var modelMatrix: simd_float4x4
    var position: SIMD3<Float>
    var orientation: simd_quatf
    var targetSize: Float
    var color: SIMD4<Float>
}

class Renderer: NSObject, MTKViewDelegate {
    // Set `AVBD_ARMADILLO_PRELOAD=0` in the scheme environment to skip the
    // shared Armadillo SDF preload path.
    private static let enableArmadilloCollisionPreload =
        ProcessInfo.processInfo.environment["AVBD_ARMADILLO_PRELOAD"] != "0"
    private static let minProjectileSize: Float = 0.25
    private static let maxProjectileSize: Float = 20.0
    private static let defaultProjectileSize: Float = 1.5
    private static let minProjectileMass: Float = 0.1
    private static let maxProjectileMass: Float = 25.0
    private static let defaultProjectileMass: Float = 3.375
    private static let minProjectileSpeed: Float = 1.0
    private static let maxProjectileSpeed: Float = 200.0
    private static let defaultProjectileSpeed: Float = 60.0
    private static let minProjectileFriction: Float = 0.0
    private static let maxProjectileFriction: Float = 2.0
    private static let defaultProjectileFriction: Float = 0.5
    private static let centerSceneArmadilloTargetSize: Float = 18.0
    private static let centerSceneArmadilloColor = SIMD4<Float>(0.63, 0.66, 0.61, 1.0)
    private static let minLinearDamping: Float = 0.0
    private static let maxLinearDamping: Float = 2.0
    private static let defaultLinearDamping: Float = 0.15
    private static let minAngularDamping: Float = 0.0
    private static let maxAngularDamping: Float = 4.0
    private static let defaultAngularDamping: Float = 0.75

    public let device: MTLDevice

#if !targetEnvironment(simulator)
    let commandQueue: MTL4CommandQueue
    let commandBuffer: MTL4CommandBuffer
    let commandAllocators: [MTL4CommandAllocator]
    let commandQueueResidencySet: MTLResidencySet
    let vertexArgumentTable: MTL4ArgumentTable
    let fragmentArgumentTable: MTL4ArgumentTable
#endif

    let endFrameEvent: MTLSharedEvent
    var frameIndex = 0

    var dynamicUniformBuffer: MTLBuffer
    var instanceBuffer: MTLBuffer
    var sphereInstanceBuffer: MTLBuffer
    var torusInstanceBuffer: MTLBuffer
    var armadilloInstanceBuffer: MTLBuffer
    var sdfDebugInstanceBuffer: MTLBuffer
    var pipelineState: MTLRenderPipelineState
    var sdfDebugPipelineState: MTLRenderPipelineState
    var depthState: MTLDepthStencilState
    var debugDepthState: MTLDepthStencilState

    var uniformBufferOffset = 0
    var instanceBufferOffset = 0

    var uniformBufferIndex = 0

    var uniforms: UnsafeMutablePointer<Uniforms>

    var projectionMatrix: matrix_float4x4 = matrix_float4x4()
    var viewMatrix: matrix_float4x4 = matrix_float4x4()

    var boxMesh: MTKMesh
    var sphereMesh: MTKMesh
    var torusMesh: StaticMesh
    var armadilloMesh: StaticMesh?
    private var collisionMeshBroadphaseMeshes: [AVBDCollisionMeshBroadphaseMesh] = []
    private var armadilloTestAsset: AVBDTriangleMeshAsset?
    private var armadilloTestASBuilder: AVBDTriangleMeshAccelerationStructureBuilder?
    private var armadilloTestAS: AVBDTriangleMeshAccelerationStructureBuildResult?
    private var armadilloDynamicRenderProxies: [ArmadilloRenderProxy] = []
    var solver: AVBDSolver
    var gpuSolver: AVBDGPUSolver?
    var computeCommandQueue: MTLCommandQueue?
    private(set) var currentSceneID: AVBDSceneID = .empty
    private(set) var currentSolverMode: AVBDSolverMode = .gpu
    private(set) var currentSimulationRunMode: AVBDSimulationRunMode = .manual
    private(set) var currentTorusVisualMode: AVBDTorusVisualMode = .solidTorus
    var maxInstanceCount: Int
    var instanceBufferFrameStride: Int
    var boxInstanceCount = 0
    var sphereInstanceCount = 0
    var torusInstanceCount = 0
    var armadilloInstanceCount = 0
    var sdfDebugInstanceCount = 0
    private(set) var currentProjectileKind: AVBDProjectileKind = .box
    private(set) var currentProjectileSize: Float = Renderer.defaultProjectileSize
    private(set) var currentProjectileMass: Float = Renderer.defaultProjectileMass
    private(set) var currentProjectileSpeed: Float = Renderer.defaultProjectileSpeed
    private(set) var currentProjectileFriction: Float = Renderer.defaultProjectileFriction
    private(set) var currentLinearDamping: Float = Renderer.defaultLinearDamping
    private(set) var currentAngularDamping: Float = Renderer.defaultAngularDamping
    private var pendingManualSteps = 0
    private let armadilloCollisionPreloadQueue = DispatchQueue(
        label: "MetalAVBD.Renderer.ArmadilloCollisionPreload",
        qos: .utility
    )
    private var armadilloCollisionPreloadRequestID: UInt64 = 0
    private var collisionSDFStatusOverride: String?
    private let cameraTarget = SIMD3<Float>(0, 0, 5.0)
    private var cameraDistance: Float = 50.0
    private var cameraAzimuth: Float = radians_from_degrees(90.0)
    private var cameraElevation: Float = 0.35
    private let minCameraDistance: Float = 0.2
    private let maxCameraDistance: Float = 1000.0
    private let minCameraElevation: Float = radians_from_degrees(-89.0)
    private let maxCameraElevation: Float = radians_from_degrees(89.0)
    private let orbitGestureSpeed: Float = 0.005
    private let debugStatsPublishInterval: CFTimeInterval = 0.2
    private var lastFrameTimestamp: CFTimeInterval?
    private var accumulatedFrameTime: CFTimeInterval = 0
    private var accumulatedFrameCount = 0
    private var lastDebugStatsPublishTimestamp: CFTimeInterval = 0
    private var completedSimulationStepCount = 0
    private var sceneDefaults = AVBDSceneDefaults()
    private(set) var showCollisionMeshSDFBounds = false
    var onDebugStatsUpdated: ((RendererDebugStats) -> Void)?

    var currentBroadphaseFullRefreshStepCount: Int { sceneDefaults.broadphaseFullRefreshStepCount }
    var enableContactWarmstart: Bool { sceneDefaults.enableContactWarmstart }

    @MainActor
    init?(metalKitView: MTKView) {
#if targetEnvironment(simulator)
        return nil
#else
        let device = metalKitView.device!
        self.device = device

        self.commandQueue = device.makeMTL4CommandQueue()!
        self.commandBuffer = device.makeCommandBuffer()!
        self.commandAllocators = (0...maxBuffersInFlight).map { _ in device.makeCommandAllocator()! }

        let argTableDesc = MTL4ArgumentTableDescriptor()
        argTableDesc.maxBufferBindCount = 4
        self.vertexArgumentTable = try! device.makeArgumentTable(descriptor: argTableDesc)
        let fragmentArgTableDesc = MTL4ArgumentTableDescriptor()
        fragmentArgTableDesc.maxBufferBindCount = 3
        self.fragmentArgumentTable = try! device.makeArgumentTable(descriptor: fragmentArgTableDesc)

        self.endFrameEvent = device.makeSharedEvent()!
        frameIndex = maxBuffersInFlight
        self.endFrameEvent.signaledValue = UInt64(frameIndex - 1)

        let uniformBufferSize = alignedUniformsSize * maxBuffersInFlight

        guard let uniformBuffer = self.device.makeBuffer(length: uniformBufferSize, options: [MTLResourceOptions.storageModeShared]) else { return nil }
        dynamicUniformBuffer = uniformBuffer

        self.dynamicUniformBuffer.label = "UniformBuffer"

        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents()).bindMemory(to: Uniforms.self, capacity: 1)

        let scene = AVBDSceneFactory.makeDefaultScene()
        currentSceneID = scene.id
        solver = AVBDSolver(scene: scene)
        maxInstanceCount = Renderer.computeMaxInstanceCount()
        instanceBufferFrameStride = MemoryLayout<InstanceUniforms>.stride * maxInstanceCount

        guard let computeCommandQueue = device.makeCommandQueue() else {
            fatalError("Failed to create compute command queue")
        }
        self.computeCommandQueue = computeCommandQueue
        
        guard let gpuSolver = AVBDGPUSolver(device: device, scene: scene) else {
            print("Failed to initialize GPU AVBD Solver. Falling back to CPU solver.")
            self.gpuSolver = nil
            currentSolverMode = .cpu
            guard let instances = self.device.makeBuffer(
                length: instanceBufferFrameStride * maxBuffersInFlight,
                options: [MTLResourceOptions.storageModeShared]
            ) else { return nil }
            instanceBuffer = instances
            self.instanceBuffer.label = "AVBDInstanceBuffer"

            guard let sphereInstances = self.device.makeBuffer(
                length: instanceBufferFrameStride * maxBuffersInFlight,
                options: [MTLResourceOptions.storageModeShared]
            ) else { return nil }
            sphereInstanceBuffer = sphereInstances
            self.sphereInstanceBuffer.label = "AVBDSphereInstanceBuffer"

            guard let torusInstances = self.device.makeBuffer(
                length: instanceBufferFrameStride * maxBuffersInFlight,
                options: [MTLResourceOptions.storageModeShared]
            ) else { return nil }
            torusInstanceBuffer = torusInstances
            self.torusInstanceBuffer.label = "AVBDTorusInstanceBuffer"

            guard let armadilloInstances = self.device.makeBuffer(
                length: instanceBufferFrameStride * maxBuffersInFlight,
                options: [MTLResourceOptions.storageModeShared]
            ) else { return nil }
            armadilloInstanceBuffer = armadilloInstances
            self.armadilloInstanceBuffer.label = "AVBDArmadilloInstanceBuffer"

            guard let sdfDebugInstances = self.device.makeBuffer(
                length: instanceBufferFrameStride * maxBuffersInFlight,
                options: [MTLResourceOptions.storageModeShared]
            ) else { return nil }
            sdfDebugInstanceBuffer = sdfDebugInstances
            self.sdfDebugInstanceBuffer.label = "AVBDSDFDebugInstanceBuffer"

            metalKitView.depthStencilPixelFormat = MTLPixelFormat.depth32Float_stencil8
            metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
            metalKitView.sampleCount = 1

            let meshVertexDescriptor = Renderer.buildMeshVertexDescriptor()
            let pipelineVertexDescriptor = Renderer.buildPipelineVertexDescriptor()

            do {
                pipelineState = try Renderer.buildRenderPipelineWithDevice(device: device,
                                                                           metalKitView: metalKitView,
                                                                           mtlVertexDescriptor: pipelineVertexDescriptor)
                sdfDebugPipelineState = try Renderer.buildSDFDebugRenderPipelineWithDevice(device: device,
                                                                                           metalKitView: metalKitView,
                                                                                           mtlVertexDescriptor: pipelineVertexDescriptor)
            } catch {
                print("Unable to compile render pipeline state.  Error info: \(error)")
                return nil
            }

            let depthStateDescriptor = MTLDepthStencilDescriptor()
            depthStateDescriptor.depthCompareFunction = MTLCompareFunction.less
            depthStateDescriptor.isDepthWriteEnabled = true
            guard let state = device.makeDepthStencilState(descriptor: depthStateDescriptor) else { return nil }
            depthState = state
            let debugDepthStateDescriptor = MTLDepthStencilDescriptor()
            debugDepthStateDescriptor.depthCompareFunction = .lessEqual
            debugDepthStateDescriptor.isDepthWriteEnabled = false
            guard let debugState = device.makeDepthStencilState(descriptor: debugDepthStateDescriptor) else { return nil }
            debugDepthState = debugState

            do {
                boxMesh = try Renderer.buildMesh(device: device, mtlVertexDescriptor: meshVertexDescriptor, shape: .box)
                sphereMesh = try Renderer.buildMesh(device: device, mtlVertexDescriptor: meshVertexDescriptor, shape: .sphere)
                guard let builtTorusMesh = Renderer.buildTorusMesh(device: device) else {
                    return nil
                }
                torusMesh = builtTorusMesh
            } catch {
                print("Unable to build MetalKit Mesh. Error info: \(error)")
                return nil
            }

            let residencySetDesc = MTLResidencySetDescriptor()
            residencySetDesc.initialCapacity = boxMesh.vertexBuffers.count + boxMesh.submeshes.count + sphereMesh.vertexBuffers.count + sphereMesh.submeshes.count + 6
            let residencySet = try! self.device.makeResidencySet(descriptor: residencySetDesc)
            residencySet.addAllocations(boxMesh.vertexBuffers.map { $0.buffer })
            residencySet.addAllocations(boxMesh.submeshes.map { $0.indexBuffer.buffer })
            residencySet.addAllocations(sphereMesh.vertexBuffers.map { $0.buffer })
            residencySet.addAllocations(sphereMesh.submeshes.map { $0.indexBuffer.buffer })
            residencySet.addAllocations([torusMesh.positions, torusMesh.normals, torusMesh.indexBuffer])
            residencySet.addAllocations([dynamicUniformBuffer, instanceBuffer, sphereInstanceBuffer, torusInstanceBuffer, armadilloInstanceBuffer, sdfDebugInstanceBuffer])
            residencySet.commit()
            commandQueue.addResidencySet(residencySet)
            commandQueueResidencySet = residencySet

            super.init()
            sceneDefaults = scene.defaults
            applySimulationParameters()
            updateViewMatrix()
            populateRenderInstances()
            beginStanfordArmadilloTestLoad()
            return
        }
        self.gpuSolver = gpuSolver
        currentSolverMode = .gpu

        guard let instances = self.device.makeBuffer(
            length: instanceBufferFrameStride * maxBuffersInFlight,
            options: [MTLResourceOptions.storageModeShared]
        ) else { return nil }
        instanceBuffer = instances
        self.instanceBuffer.label = "AVBDInstanceBuffer"

        guard let sphereInstances = self.device.makeBuffer(
            length: instanceBufferFrameStride * maxBuffersInFlight,
            options: [MTLResourceOptions.storageModeShared]
        ) else { return nil }
        sphereInstanceBuffer = sphereInstances
        self.sphereInstanceBuffer.label = "AVBDSphereInstanceBuffer"

        guard let torusInstances = self.device.makeBuffer(
            length: instanceBufferFrameStride * maxBuffersInFlight,
            options: [MTLResourceOptions.storageModeShared]
        ) else { return nil }
        torusInstanceBuffer = torusInstances
        self.torusInstanceBuffer.label = "AVBDTorusInstanceBuffer"

        guard let armadilloInstances = self.device.makeBuffer(
            length: instanceBufferFrameStride * maxBuffersInFlight,
            options: [MTLResourceOptions.storageModeShared]
        ) else { return nil }
        armadilloInstanceBuffer = armadilloInstances
        self.armadilloInstanceBuffer.label = "AVBDArmadilloInstanceBuffer"

        guard let sdfDebugInstances = self.device.makeBuffer(
            length: instanceBufferFrameStride * maxBuffersInFlight,
            options: [MTLResourceOptions.storageModeShared]
        ) else { return nil }
        sdfDebugInstanceBuffer = sdfDebugInstances
        self.sdfDebugInstanceBuffer.label = "AVBDSDFDebugInstanceBuffer"

        metalKitView.depthStencilPixelFormat = MTLPixelFormat.depth32Float_stencil8
        metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
        metalKitView.sampleCount = 1

        let meshVertexDescriptor = Renderer.buildMeshVertexDescriptor()
        let pipelineVertexDescriptor = Renderer.buildPipelineVertexDescriptor()

        do {
            pipelineState = try Renderer.buildRenderPipelineWithDevice(device: device,
                                                                       metalKitView: metalKitView,
                                                                       mtlVertexDescriptor: pipelineVertexDescriptor)
            sdfDebugPipelineState = try Renderer.buildSDFDebugRenderPipelineWithDevice(device: device,
                                                                                       metalKitView: metalKitView,
                                                                                       mtlVertexDescriptor: pipelineVertexDescriptor)
        } catch {
            print("Unable to compile render pipeline state.  Error info: \(error)")
            return nil
        }

        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.less
        depthStateDescriptor.isDepthWriteEnabled = true
        guard let state = device.makeDepthStencilState(descriptor: depthStateDescriptor) else { return nil }
        depthState = state
        let debugDepthStateDescriptor = MTLDepthStencilDescriptor()
        debugDepthStateDescriptor.depthCompareFunction = .lessEqual
        debugDepthStateDescriptor.isDepthWriteEnabled = false
        guard let debugState = device.makeDepthStencilState(descriptor: debugDepthStateDescriptor) else { return nil }
        debugDepthState = debugState

        do {
            boxMesh = try Renderer.buildMesh(device: device, mtlVertexDescriptor: meshVertexDescriptor, shape: .box)
            sphereMesh = try Renderer.buildMesh(device: device, mtlVertexDescriptor: meshVertexDescriptor, shape: .sphere)
            guard let builtTorusMesh = Renderer.buildTorusMesh(device: device) else {
                return nil
            }
            torusMesh = builtTorusMesh
        } catch {
            print("Unable to build MetalKit Mesh. Error info: \(error)")
            return nil
        }

        let residencySetDesc = MTLResidencySetDescriptor()
        residencySetDesc.initialCapacity = boxMesh.vertexBuffers.count + boxMesh.submeshes.count + sphereMesh.vertexBuffers.count + sphereMesh.submeshes.count + 6
        let residencySet = try! self.device.makeResidencySet(descriptor: residencySetDesc)
        residencySet.addAllocations(boxMesh.vertexBuffers.map { $0.buffer })
        residencySet.addAllocations(boxMesh.submeshes.map { $0.indexBuffer.buffer })
        residencySet.addAllocations(sphereMesh.vertexBuffers.map { $0.buffer })
        residencySet.addAllocations(sphereMesh.submeshes.map { $0.indexBuffer.buffer })
        residencySet.addAllocations([torusMesh.positions, torusMesh.normals, torusMesh.indexBuffer])
        residencySet.addAllocations([dynamicUniformBuffer, instanceBuffer, sphereInstanceBuffer, torusInstanceBuffer, armadilloInstanceBuffer, sdfDebugInstanceBuffer])
        residencySet.commit()
        commandQueue.addResidencySet(residencySet)
        commandQueueResidencySet = residencySet

        super.init()
        sceneDefaults = scene.defaults
        applySimulationParameters()
        self.gpuSolver?.broadphaseFullRefreshStepCount = sceneDefaults.broadphaseFullRefreshStepCount
        self.gpuSolver?.enableContactWarmstart = enableContactWarmstart
        updateViewMatrix()
        populateRenderInstances()
        beginStanfordArmadilloTestLoad()
#endif
    }

    private func beginStanfordArmadilloTestLoad() {
        guard let computeCommandQueue else {
            return
        }

        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self else {
                return
            }

            do {
                let loader = AVBDStanfordArmadilloLoader(device: self.device)
                let asset = try loader.load()

                guard let builder = AVBDTriangleMeshAccelerationStructureBuilder(
                    device: self.device,
                    commandQueue: computeCommandQueue
                ) else {
                    print("Failed to create Stanford Armadillo AS builder.")
                    return
                }

                let result = try builder.build(
                    geometries: [asset.makeGeometry(label: "StanfordArmadillo")],
                    instances: [.identity(geometryIndex: 0)]
                )

                DispatchQueue.main.async { [weak self] in
                    self?.armadilloTestAsset = asset
                    self?.armadilloMesh = asset.makeStaticMesh()
                    self?.armadilloTestASBuilder = builder
                    self?.armadilloTestAS = result
                    self?.addArmadilloResourcesToResidencySetIfNeeded()
                    self?.rebuildSceneCollisionMeshes()
                    self?.applyCollisionMeshesToGPUSolver(fullRebuild: true)
                    self?.scheduleArmadilloCollisionResourcePreloadIfNeeded()
                    self?.populateRenderInstances()
                }

                print(
                    """
                    Stanford Armadillo loaded from \(asset.sourceURL.absoluteString)
                    Cache: \(asset.cachePLYURL.path)
                    Vertices: \(asset.vertexCount), Triangles: \(asset.indexCount / 3)
                    """
                )
            } catch {
                print("Failed to load Stanford Armadillo test asset: \(error)")
            }
        }
    }

    private func addArmadilloResourcesToResidencySetIfNeeded() {
#if !targetEnvironment(simulator)
        guard let armadilloMesh else {
            return
        }
        commandQueueResidencySet.addAllocations([
            armadilloMesh.positions,
            armadilloMesh.normals,
            armadilloMesh.indexBuffer
        ])
        commandQueueResidencySet.commit()
#endif
    }

    private func applyCollisionMeshesToGPUSolver(fullRebuild: Bool = false) {
        guard let gpuSolver else {
            return
        }

        if fullRebuild || !gpuSolver.updateCollisionMeshInstances(collisionMeshBroadphaseMeshes) {
            gpuSolver.setCollisionMeshes(collisionMeshBroadphaseMeshes)
            addCollisionMeshSDFResourcesToResidencySetIfNeeded()
        }
    }

    private func armadilloCollisionResourceTemplate() -> AVBDCollisionMeshBroadphaseMesh? {
        guard let armadilloTestAsset else {
            return nil
        }
        return AVBDCollisionMeshBroadphaseMesh(
            sdfResourceID: "stanford-armadillo",
            ownerBodyIndex: -1,
            localBoundsMin: armadilloTestAsset.boundsMin,
            localBoundsMax: armadilloTestAsset.boundsMax,
            transform: matrix_identity_float4x4,
            positions: armadilloTestAsset.positionData,
            indices: armadilloTestAsset.indexData
        )
    }

    private func scheduleArmadilloCollisionResourcePreloadIfNeeded() {
        guard Renderer.enableArmadilloCollisionPreload,
              let preloadSolver = gpuSolver,
              let preloadMesh = armadilloCollisionResourceTemplate() else {
            collisionSDFStatusOverride = nil
            return
        }

        armadilloCollisionPreloadRequestID &+= 1
        let requestID = armadilloCollisionPreloadRequestID
        collisionSDFStatusOverride = "Preloading..."

        armadilloCollisionPreloadQueue.async { [weak self] in
            let didPrepare = preloadSolver.prepareCollisionMeshSDFCache(preloadMesh)
            DispatchQueue.main.async { [weak self] in
                guard let self else {
                    return
                }
                guard requestID == self.armadilloCollisionPreloadRequestID else {
                    return
                }

                defer {
                    self.collisionSDFStatusOverride = nil
                }

                guard didPrepare, let gpuSolver = self.gpuSolver else {
                    return
                }

                _ = gpuSolver.prewarmCollisionMeshResource(preloadMesh)
                self.addCollisionMeshSDFResourcesToResidencySetIfNeeded()
            }
        }
    }

    private func armadilloModelMatrix(
        position: SIMD3<Float>,
        orientation: simd_quatf,
        targetSize: Float
    ) -> simd_float4x4? {
        guard let armadilloTestAsset else {
            return nil
        }

        let span = armadilloTestAsset.boundsMax - armadilloTestAsset.boundsMin
        let maxExtent = max(span.x, max(span.y, span.z))
        let scale = maxExtent > 0 ? targetSize / maxExtent : 1.0
        return simd_mul(
            matrix4x4_translation(position.x, position.y, position.z),
            simd_mul(
                matrix4x4_rotation(quaternion: orientation),
                simd_mul(
                    matrix4x4_scale(scale, scale, scale),
                    matrix4x4_translation(-armadilloTestAsset.boundsCenter.x, -armadilloTestAsset.boundsCenter.y, -armadilloTestAsset.boundsCenter.z)
                )
            )
        )
    }

    private func rebuildCollisionMeshes(dynamicBodies: [AVBDGPUBody] = []) {
        collisionMeshBroadphaseMeshes.removeAll(keepingCapacity: true)

        if currentSceneID == .armadilloCenter,
           let armadilloTestAsset,
           let placement = sceneArmadilloPlacement() {
            collisionMeshBroadphaseMeshes.append(
                AVBDCollisionMeshBroadphaseMesh(
                    sdfResourceID: "stanford-armadillo",
                    ownerBodyIndex: -1,
                    localBoundsMin: armadilloTestAsset.boundsMin,
                    localBoundsMax: armadilloTestAsset.boundsMax,
                    transform: placement.modelMatrix,
                    positions: armadilloTestAsset.positionData,
                    indices: armadilloTestAsset.indexData
                )
            )
        }

        guard let armadilloTestAsset else {
            return
        }

        let proxies = filteredArmadilloRenderProxies(bodyCount: dynamicBodies.count)
        for proxy in proxies {
            guard proxy.bodyIndex >= 0, proxy.bodyIndex < dynamicBodies.count else {
                continue
            }
            let body = dynamicBodies[proxy.bodyIndex]
            let orientation = simd_quatf(vector: body.positionAng)
            guard let modelMatrix = armadilloModelMatrix(
                position: body.positionLin,
                orientation: orientation,
                targetSize: projectileRenderSize(for: body.size, projectileKind: .armadillo)
            ) else {
                continue
            }
            collisionMeshBroadphaseMeshes.append(
                AVBDCollisionMeshBroadphaseMesh(
                    sdfResourceID: "stanford-armadillo",
                    ownerBodyIndex: proxy.bodyIndex,
                    localBoundsMin: armadilloTestAsset.boundsMin,
                    localBoundsMax: armadilloTestAsset.boundsMax,
                    transform: modelMatrix,
                    positions: armadilloTestAsset.positionData,
                    indices: armadilloTestAsset.indexData
                )
            )
        }
    }

    private func appendDynamicArmadilloCollisionMesh(
        bodyIndex: Int,
        position: SIMD3<Float>,
        orientation: simd_quatf,
        size: SIMD3<Float>
    ) {
        guard let armadilloTestAsset,
              let modelMatrix = armadilloModelMatrix(
                position: position,
                orientation: orientation,
                targetSize: projectileRenderSize(for: size, projectileKind: .armadillo)
              ) else {
            return
        }

        collisionMeshBroadphaseMeshes.append(
            AVBDCollisionMeshBroadphaseMesh(
                sdfResourceID: "stanford-armadillo",
                ownerBodyIndex: bodyIndex,
                localBoundsMin: armadilloTestAsset.boundsMin,
                localBoundsMax: armadilloTestAsset.boundsMax,
                transform: modelMatrix,
                positions: armadilloTestAsset.positionData,
                indices: armadilloTestAsset.indexData
            )
        )
    }

    private func addCollisionMeshSDFResourcesToResidencySetIfNeeded() {
#if !targetEnvironment(simulator)
        guard let gpuSolver else {
            return
        }

        var allocations: [any MTLAllocation] = []
        if let infoBuffer = gpuSolver.collisionMeshDebugInfoBuffer {
            allocations.append(infoBuffer)
        }
        if let sdfArgumentBuffer = gpuSolver.collisionMeshDebugSDFArgumentBuffer {
            allocations.append(sdfArgumentBuffer)
        }
        allocations.append(contentsOf: gpuSolver.collisionMeshDebugTextures)

        guard !allocations.isEmpty else {
            return
        }

        commandQueueResidencySet.addAllocations(allocations)
        commandQueueResidencySet.commit()
#endif
    }

    private func sceneArmadilloPlacement() -> SceneArmadilloPlacement? {
        guard currentSceneID == .armadilloCenter,
              let armadilloTestAsset else {
            return nil
        }

        let targetSize = Self.centerSceneArmadilloTargetSize
        let span = armadilloTestAsset.boundsMax - armadilloTestAsset.boundsMin
        let maxExtent = max(span.x, max(span.y, span.z))
        let scale = maxExtent > 0 ? targetSize / maxExtent : 1.0
        let groundHeight: Float = 0.5
        let orientation = simd_quatf(angle: .pi * 0.5, axis: SIMD3<Float>(1, 0, 0))
        let centeredLocalTransform = simd_mul(
            matrix4x4_rotation(quaternion: orientation),
            simd_mul(
                matrix4x4_scale(scale, scale, scale),
                matrix4x4_translation(-armadilloTestAsset.boundsCenter.x, -armadilloTestAsset.boundsCenter.y, -armadilloTestAsset.boundsCenter.z)
            )
        )

        let localBoundsMin = armadilloTestAsset.boundsMin
        let localBoundsMax = armadilloTestAsset.boundsMax
        let corners: [SIMD3<Float>] = [
            SIMD3<Float>(localBoundsMin.x, localBoundsMin.y, localBoundsMin.z),
            SIMD3<Float>(localBoundsMax.x, localBoundsMin.y, localBoundsMin.z),
            SIMD3<Float>(localBoundsMin.x, localBoundsMax.y, localBoundsMin.z),
            SIMD3<Float>(localBoundsMax.x, localBoundsMax.y, localBoundsMin.z),
            SIMD3<Float>(localBoundsMin.x, localBoundsMin.y, localBoundsMax.z),
            SIMD3<Float>(localBoundsMax.x, localBoundsMin.y, localBoundsMax.z),
            SIMD3<Float>(localBoundsMin.x, localBoundsMax.y, localBoundsMax.z),
            SIMD3<Float>(localBoundsMax.x, localBoundsMax.y, localBoundsMax.z),
        ]

        var transformedMinZ = Float.greatestFiniteMagnitude
        for corner in corners {
            let transformed = centeredLocalTransform * SIMD4<Float>(corner, 1)
            transformedMinZ = min(transformedMinZ, transformed.z)
        }

        let position = SIMD3<Float>(0, 0, groundHeight - transformedMinZ)
        let modelMatrix = simd_mul(
            matrix4x4_translation(position.x, position.y, position.z),
            centeredLocalTransform
        )

        return SceneArmadilloPlacement(
            modelMatrix: modelMatrix,
            position: position,
            orientation: orientation,
            targetSize: targetSize,
            color: Self.centerSceneArmadilloColor
        )
    }

    private func rebuildSceneCollisionMeshes() {
        rebuildCollisionMeshes()
    }

    class func buildMeshVertexDescriptor() -> MTLVertexDescriptor {
        let mtlVertexDescriptor = MTLVertexDescriptor()

        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].bufferIndex = BufferIndex.meshPositions.rawValue

        mtlVertexDescriptor.attributes[VertexAttribute.normal.rawValue].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[VertexAttribute.normal.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.normal.rawValue].bufferIndex = BufferIndex.meshNormals.rawValue

        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stride = MemoryLayout<SIMD3<Float>>.stride
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction = MTLVertexStepFunction.perVertex

        mtlVertexDescriptor.layouts[BufferIndex.meshNormals.rawValue].stride = MemoryLayout<SIMD3<Float>>.stride
        mtlVertexDescriptor.layouts[BufferIndex.meshNormals.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshNormals.rawValue].stepFunction = MTLVertexStepFunction.perVertex

        return mtlVertexDescriptor
    }

    class func buildPipelineVertexDescriptor() -> MTLVertexDescriptor {
        let mtlVertexDescriptor = buildMeshVertexDescriptor()
        let instanceBufferIndex = BufferIndex.instances.rawValue
        let vectorStride = MemoryLayout<SIMD4<Float>>.stride

        mtlVertexDescriptor.attributes[VertexAttribute.modelColumn0.rawValue].format = MTLVertexFormat.float4
        mtlVertexDescriptor.attributes[VertexAttribute.modelColumn0.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.modelColumn0.rawValue].bufferIndex = instanceBufferIndex

        mtlVertexDescriptor.attributes[VertexAttribute.modelColumn1.rawValue].format = MTLVertexFormat.float4
        mtlVertexDescriptor.attributes[VertexAttribute.modelColumn1.rawValue].offset = vectorStride
        mtlVertexDescriptor.attributes[VertexAttribute.modelColumn1.rawValue].bufferIndex = instanceBufferIndex

        mtlVertexDescriptor.attributes[VertexAttribute.modelColumn2.rawValue].format = MTLVertexFormat.float4
        mtlVertexDescriptor.attributes[VertexAttribute.modelColumn2.rawValue].offset = vectorStride * 2
        mtlVertexDescriptor.attributes[VertexAttribute.modelColumn2.rawValue].bufferIndex = instanceBufferIndex

        mtlVertexDescriptor.attributes[VertexAttribute.modelColumn3.rawValue].format = MTLVertexFormat.float4
        mtlVertexDescriptor.attributes[VertexAttribute.modelColumn3.rawValue].offset = vectorStride * 3
        mtlVertexDescriptor.attributes[VertexAttribute.modelColumn3.rawValue].bufferIndex = instanceBufferIndex

        mtlVertexDescriptor.attributes[VertexAttribute.color.rawValue].format = MTLVertexFormat.float4
        mtlVertexDescriptor.attributes[VertexAttribute.color.rawValue].offset = MemoryLayout<matrix_float4x4>.stride
        mtlVertexDescriptor.attributes[VertexAttribute.color.rawValue].bufferIndex = instanceBufferIndex

        mtlVertexDescriptor.attributes[VertexAttribute.shapeParams.rawValue].format = MTLVertexFormat.float4
        mtlVertexDescriptor.attributes[VertexAttribute.shapeParams.rawValue].offset = MemoryLayout<matrix_float4x4>.stride + MemoryLayout<SIMD4<Float>>.stride
        mtlVertexDescriptor.attributes[VertexAttribute.shapeParams.rawValue].bufferIndex = instanceBufferIndex

        mtlVertexDescriptor.layouts[instanceBufferIndex].stride = MemoryLayout<InstanceUniforms>.stride
        mtlVertexDescriptor.layouts[instanceBufferIndex].stepRate = 1
        mtlVertexDescriptor.layouts[instanceBufferIndex].stepFunction = MTLVertexStepFunction.perInstance

        return mtlVertexDescriptor
    }

#if !targetEnvironment(simulator)

    @MainActor
    class func buildRenderPipelineWithDevice(device: MTLDevice,
                                             metalKitView: MTKView,
                                             mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTLRenderPipelineState {
        /// Build a render state pipeline object

        let library = device.makeDefaultLibrary()
        let compiler = try device.makeCompiler(descriptor: MTL4CompilerDescriptor())

        let vertexFunctionDescriptor = MTL4LibraryFunctionDescriptor()
        vertexFunctionDescriptor.library = library
        vertexFunctionDescriptor.name = "vertexShader"
        let fragmentFunctionDescriptor = MTL4LibraryFunctionDescriptor()
        fragmentFunctionDescriptor.library = library
        fragmentFunctionDescriptor.name = "fragmentShader"

        let pipelineDescriptor = MTL4RenderPipelineDescriptor()
        pipelineDescriptor.label = "AVBDRenderPipeline"
        pipelineDescriptor.rasterSampleCount = metalKitView.sampleCount
        pipelineDescriptor.vertexFunctionDescriptor = vertexFunctionDescriptor
        pipelineDescriptor.fragmentFunctionDescriptor = fragmentFunctionDescriptor
        pipelineDescriptor.vertexDescriptor = mtlVertexDescriptor

        pipelineDescriptor.colorAttachments[0].pixelFormat = metalKitView.colorPixelFormat

        return try compiler.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    @MainActor
    class func buildSDFDebugRenderPipelineWithDevice(device: MTLDevice,
                                                     metalKitView: MTKView,
                                                     mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTLRenderPipelineState {
        let library = device.makeDefaultLibrary()
        let compiler = try device.makeCompiler(descriptor: MTL4CompilerDescriptor())

        let vertexFunctionDescriptor = MTL4LibraryFunctionDescriptor()
        vertexFunctionDescriptor.library = library
        vertexFunctionDescriptor.name = "sdfDebugVertexShader"
        let fragmentFunctionDescriptor = MTL4LibraryFunctionDescriptor()
        fragmentFunctionDescriptor.library = library
        fragmentFunctionDescriptor.name = "sdfDebugFragmentShader"

        let pipelineDescriptor = MTL4RenderPipelineDescriptor()
        pipelineDescriptor.label = "AVBDSDFDebugRenderPipeline"
        pipelineDescriptor.rasterSampleCount = metalKitView.sampleCount
        pipelineDescriptor.vertexFunctionDescriptor = vertexFunctionDescriptor
        pipelineDescriptor.fragmentFunctionDescriptor = fragmentFunctionDescriptor
        pipelineDescriptor.vertexDescriptor = mtlVertexDescriptor
        pipelineDescriptor.colorAttachments[0].pixelFormat = metalKitView.colorPixelFormat
        pipelineDescriptor.colorAttachments[0].blendingState = .enabled
        pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha

        return try compiler.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

#endif

    class func buildMesh(
        device: MTLDevice,
        mtlVertexDescriptor: MTLVertexDescriptor,
        shape: AVBDRenderShape
    ) throws -> MTKMesh {
        /// Create and condition mesh data to feed into a pipeline using the given vertex descriptor

        let metalAllocator = MTKMeshBufferAllocator(device: device)

        let mdlMesh: MDLMesh
        switch shape {
        case .box:
            mdlMesh = MDLMesh.newBox(
                withDimensions: SIMD3<Float>(1, 1, 1),
                segments: SIMD3<UInt32>(1, 1, 1),
                geometryType: MDLGeometryType.triangles,
                inwardNormals: false,
                allocator: metalAllocator
            )
        case .sphere:
            mdlMesh = MDLMesh.newEllipsoid(
                withRadii: SIMD3<Float>(0.5, 0.5, 0.5),
                radialSegments: 24,
                verticalSegments: 16,
                geometryType: MDLGeometryType.triangles,
                inwardNormals: false,
                hemisphere: false,
                allocator: metalAllocator
            )
        case .torus:
            mdlMesh = MDLMesh.newEllipsoid(
                withRadii: SIMD3<Float>(0.5, 0.5, 0.5),
                radialSegments: 24,
                verticalSegments: 16,
                geometryType: MDLGeometryType.triangles,
                inwardNormals: false,
                hemisphere: false,
                allocator: metalAllocator
            )
        }

        let mdlVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor)

        guard let attributes = mdlVertexDescriptor.attributes as? [MDLVertexAttribute] else {
            throw RendererError.badVertexDescriptor
        }
        attributes[VertexAttribute.position.rawValue].name = MDLVertexAttributePosition
        attributes[VertexAttribute.normal.rawValue].name = MDLVertexAttributeNormal

        mdlMesh.vertexDescriptor = mdlVertexDescriptor

        return try MTKMesh(mesh: mdlMesh, device: device)
    }

    class func buildTorusMesh(device: MTLDevice, radialSegments: Int = 48, tubeSegments: Int = 24) -> StaticMesh? {
        let vertexCount = radialSegments * tubeSegments
        let indexCount = radialSegments * tubeSegments * 6

        var positions = Array(repeating: SIMD3<Float>.zero, count: vertexCount)
        let normals = Array(repeating: SIMD3<Float>.zero, count: vertexCount)
        var indices = Array(repeating: UInt32(0), count: indexCount)

        for radial in 0..<radialSegments {
            let u = (2.0 * Float.pi * Float(radial)) / Float(radialSegments)
            for tube in 0..<tubeSegments {
                let v = (2.0 * Float.pi * Float(tube)) / Float(tubeSegments)
                let vertexIndex = radial * tubeSegments + tube
                positions[vertexIndex] = SIMD3<Float>(u, v, 0)
            }
        }

        var indexCursor = 0
        for radial in 0..<radialSegments {
            let nextRadial = (radial + 1) % radialSegments
            for tube in 0..<tubeSegments {
                let nextTube = (tube + 1) % tubeSegments

                let i0 = UInt32(radial * tubeSegments + tube)
                let i1 = UInt32(nextRadial * tubeSegments + tube)
                let i2 = UInt32(nextRadial * tubeSegments + nextTube)
                let i3 = UInt32(radial * tubeSegments + nextTube)

                indices[indexCursor + 0] = i0
                indices[indexCursor + 1] = i1
                indices[indexCursor + 2] = i2
                indices[indexCursor + 3] = i0
                indices[indexCursor + 4] = i2
                indices[indexCursor + 5] = i3
                indexCursor += 6
            }
        }

        guard let positionBuffer = device.makeBuffer(bytes: positions,
                                                     length: MemoryLayout<SIMD3<Float>>.stride * positions.count,
                                                     options: [MTLResourceOptions.storageModeShared]),
              let normalBuffer = device.makeBuffer(bytes: normals,
                                                   length: MemoryLayout<SIMD3<Float>>.stride * normals.count,
                                                   options: [MTLResourceOptions.storageModeShared]),
              let indexBuffer = device.makeBuffer(bytes: indices,
                                                  length: MemoryLayout<UInt32>.stride * indices.count,
                                                  options: [MTLResourceOptions.storageModeShared]) else {
            return nil
        }

        positionBuffer.label = "AVBDTorusMeshPositions"
        normalBuffer.label = "AVBDTorusMeshNormals"
        indexBuffer.label = "AVBDTorusMeshIndices"

        return StaticMesh(
            positions: positionBuffer,
            normals: normalBuffer,
            indexBuffer: indexBuffer,
            indexCount: indexCount
        )
    }

    private func updateDynamicBufferState() {
        /// Update the state of our uniform buffers before rendering

        uniformBufferIndex = (uniformBufferIndex + 1) % maxBuffersInFlight

        uniformBufferOffset = alignedUniformsSize * uniformBufferIndex
        instanceBufferOffset = instanceBufferFrameStride * uniformBufferIndex

        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset).bindMemory(to: Uniforms.self, capacity: 1)
    }

    private func updateGameState() {
        /// Update any game state before rendering

        updateViewMatrix()
        let cameraEye = orbitEye(distance: cameraDistance, azimuth: cameraAzimuth, elevation: cameraElevation, target: cameraTarget)
        let viewProjectionMatrix = simd_mul(projectionMatrix, viewMatrix)
        uniforms[0].viewProjectionMatrix = viewProjectionMatrix
        uniforms[0].inverseViewProjectionMatrix = viewProjectionMatrix.inverse
        uniforms[0].cameraWorldPosition = cameraEye
        uniforms[0].padding0 = 0
    }

    private var currentBodyCount: Int {
        switch currentSolverMode {
        case .gpu:
            return gpuSolver?.bodyCount ?? solver.bodies.count
        case .cpu:
            return solver.bodies.count
        }
    }

    private var currentBroadphaseModeName: String {
        guard currentSolverMode == .gpu, let gpuSolver else {
            return "CPU Solver"
        }
        guard gpuSolver.bodyCount > 1 else {
            return sceneDefaults.broadphaseFullRefreshStepCount > 0 ? "Cached" : "Full"
        }
        if sceneDefaults.broadphaseFullRefreshStepCount <= 0 {
            return "Full"
        }
        return gpuSolver.lastBroadphaseUsedCache ? "Cached" : "Refresh"
    }

    private var currentWarmstartModeName: String {
        guard currentSolverMode == .gpu else {
            return "CPU Solver"
        }
        return enableContactWarmstart ? "On" : "Off"
    }

    private var currentCollisionSDFStatusName: String {
        if let collisionSDFStatusOverride {
            return collisionSDFStatusOverride
        }
        guard currentSolverMode == .gpu, let gpuSolver else {
            return collisionMeshBroadphaseMeshes.isEmpty ? "Idle" : "CPU"
        }
        return gpuSolver.collisionMeshSDFStatusText
    }

    private func publishDebugStatsIfNeeded(frameTimestamp: CFTimeInterval) {
        if let lastFrameTimestamp {
            accumulatedFrameTime += frameTimestamp - lastFrameTimestamp
            accumulatedFrameCount += 1
        }
        self.lastFrameTimestamp = frameTimestamp

        guard accumulatedFrameCount > 0 else { return }
        guard frameTimestamp - lastDebugStatsPublishTimestamp >= debugStatsPublishInterval else { return }

        let averageFrameTime = accumulatedFrameTime / Double(accumulatedFrameCount)
        let stats = RendererDebugStats(
            fps: averageFrameTime > 0 ? 1.0 / averageFrameTime : 0,
            frameTimeMS: averageFrameTime * 1000.0,
            stepCount: completedSimulationStepCount,
            bodyCount: currentBodyCount,
            solverModeName: currentSolverMode.displayName,
            simulationModeName: currentSimulationRunMode.displayName,
            broadphaseModeName: currentBroadphaseModeName,
            warmstartModeName: currentWarmstartModeName,
            collisionSDFStatusName: currentCollisionSDFStatusName
        )

        accumulatedFrameTime = 0
        accumulatedFrameCount = 0
        lastDebugStatsPublishTimestamp = frameTimestamp

        if let onDebugStatsUpdated {
            DispatchQueue.main.async {
                onDebugStatsUpdated(stats)
            }
        }
    }

    @MainActor
    func setBroadphaseFullRefreshStepCount(_ count: Int) {
        sceneDefaults.setBroadphaseFullRefreshStepCount(count)
        gpuSolver?.broadphaseFullRefreshStepCount = sceneDefaults.broadphaseFullRefreshStepCount
    }

    @MainActor
    func setContactWarmstartEnabled(_ enabled: Bool) {
        sceneDefaults.setEnableContactWarmstart(enabled)
        gpuSolver?.enableContactWarmstart = enabled
    }

    func currentDebugStats() -> RendererDebugStats {
        RendererDebugStats(
            fps: 0,
            frameTimeMS: 0,
            stepCount: completedSimulationStepCount,
            bodyCount: currentBodyCount,
            solverModeName: currentSolverMode.displayName,
            simulationModeName: currentSimulationRunMode.displayName,
            broadphaseModeName: currentBroadphaseModeName,
            warmstartModeName: currentWarmstartModeName,
            collisionSDFStatusName: currentCollisionSDFStatusName
        )
    }

    @MainActor
    func setShowCollisionMeshSDFBounds(_ enabled: Bool) {
        showCollisionMeshSDFBounds = enabled
        populateRenderInstances()
    }

    func setProjectileKind(_ kind: AVBDProjectileKind) {
        currentProjectileKind = kind
    }

    func setProjectileSize(_ size: Float) {
        currentProjectileSize = clamp(size, min: Self.minProjectileSize, max: Self.maxProjectileSize)
    }

    func setProjectileMass(_ mass: Float) {
        currentProjectileMass = clamp(mass, min: Self.minProjectileMass, max: Self.maxProjectileMass)
    }

    func setProjectileSpeed(_ speed: Float) {
        currentProjectileSpeed = clamp(speed, min: Self.minProjectileSpeed, max: Self.maxProjectileSpeed)
    }

    func setProjectileFriction(_ friction: Float) {
        currentProjectileFriction = clamp(friction, min: Self.minProjectileFriction, max: Self.maxProjectileFriction)
    }

    @MainActor
    func setLinearDamping(_ damping: Float) {
        currentLinearDamping = clamp(damping, min: Self.minLinearDamping, max: Self.maxLinearDamping)
        applySimulationParameters()
    }

    @MainActor
    func setAngularDamping(_ damping: Float) {
        currentAngularDamping = clamp(damping, min: Self.minAngularDamping, max: Self.maxAngularDamping)
        applySimulationParameters()
    }

    func setTorusVisualMode(_ mode: AVBDTorusVisualMode) {
        currentTorusVisualMode = mode
        populateRenderInstances()
    }

    var currentTorusApproxSphereCount: Int {
        avbdCurrentTorusApproxSphereCount()
    }

    var currentTorusApproxSphereRadiusScale: Float {
        AVBDTorusApproximationSettings.radiusScale
    }

    func setTorusApproximation(sphereCount: Int, radiusScale: Float) {
        AVBDTorusApproximationSettings.update(sphereCount: sphereCount, radiusScale: radiusScale)
        gpuSolver?.invalidateBroadphaseCache()
        populateRenderInstances()
    }

    private func makeRigidInstance(
        position: SIMD3<Float>,
        orientation: simd_quatf,
        size: SIMD3<Float>,
        color: SIMD4<Float>,
        shape: AVBDRenderShape
    ) -> InstanceUniforms {
        let modelMatrix = simd_mul(
            matrix4x4_translation(position.x, position.y, position.z),
            simd_mul(matrix4x4_rotation(quaternion: orientation), matrix4x4_scale(size.x, size.y, size.z))
        )
        return InstanceUniforms(
            modelMatrix: modelMatrix,
            renderColor: color,
            shapeParams: SIMD4<Float>(Float(shape.rawValue), 0, 0, 0)
        )
    }

    private func makeProxySphereInstance(center: SIMD3<Float>, diameter: Float, color: SIMD4<Float>) -> InstanceUniforms {
        InstanceUniforms(
            modelMatrix: simd_mul(
                matrix4x4_translation(center.x, center.y, center.z),
                matrix4x4_scale(diameter, diameter, diameter)
            ),
            renderColor: color,
            shapeParams: SIMD4<Float>(Float(AVBDRenderShape.sphere.rawValue), 0, 0, 0)
        )
    }

    private func makeTorusInstance(
        position: SIMD3<Float>,
        orientation: simd_quatf,
        size: SIMD3<Float>,
        color: SIMD4<Float>
    ) -> InstanceUniforms {
        InstanceUniforms(
            modelMatrix: simd_mul(
                matrix4x4_translation(position.x, position.y, position.z),
                matrix4x4_rotation(quaternion: orientation)
            ),
            renderColor: color,
            shapeParams: SIMD4<Float>(
                Float(AVBDRenderShape.torus.rawValue),
                avbdTorusMajorRadius(size: size),
                avbdTorusRenderMinorRadius(size: size),
                0
            )
        )
    }

    private func makeArmadilloInstance(
        position: SIMD3<Float>,
        orientation: simd_quatf,
        targetSize: Float,
        color: SIMD4<Float>
    ) -> InstanceUniforms? {
        guard let modelMatrix = armadilloModelMatrix(
            position: position,
            orientation: orientation,
            targetSize: targetSize
        ) else {
            return nil
        }

        return InstanceUniforms(
            modelMatrix: modelMatrix,
            renderColor: color,
            shapeParams: SIMD4<Float>(Float(AVBDRenderShape.box.rawValue), 0, 0, 0)
        )
    }

    @MainActor
    func orbitCamera(delta: CGPoint) {
        cameraAzimuth -= Float(delta.x) * orbitGestureSpeed
        cameraElevation += Float(delta.y) * orbitGestureSpeed
        cameraElevation = clamp(cameraElevation, min: minCameraElevation, max: maxCameraElevation)
        updateViewMatrix()
    }

    @MainActor
    func zoomCamera(scaleDelta: CGFloat) {
        let scale = Float(scaleDelta)
        guard scale > 0 else { return }
        cameraDistance = clamp(cameraDistance / scale, min: minCameraDistance, max: maxCameraDistance)
        updateViewMatrix()
    }

    @discardableResult
    @MainActor
    private func rebuildScene(_ id: AVBDSceneID, solverMode: AVBDSolverMode, shouldApplySceneDefaults: Bool) -> AVBDSolverMode {
        let scene = AVBDSceneFactory.make(id)
        currentSceneID = scene.id
        if shouldApplySceneDefaults {
            sceneDefaults = scene.defaults
        }
        solver = AVBDSolver(scene: scene)
        gpuSolver = AVBDGPUSolver(device: device, scene: scene)
        rebuildSceneCollisionMeshes()
        applyCollisionMeshesToGPUSolver(fullRebuild: true)
        scheduleArmadilloCollisionResourcePreloadIfNeeded()
        armadilloDynamicRenderProxies.removeAll()
        applySimulationParameters()
        gpuSolver?.broadphaseFullRefreshStepCount = sceneDefaults.broadphaseFullRefreshStepCount
        gpuSolver?.enableContactWarmstart = enableContactWarmstart

        if solverMode == .gpu && gpuSolver == nil {
            print("Failed to switch to GPU solver for scene: \(id.displayName)")
            currentSolverMode = .cpu
        } else {
            currentSolverMode = solverMode
        }

        boxInstanceCount = 0
        sphereInstanceCount = 0
        torusInstanceCount = 0
        pendingManualSteps = 0
        completedSimulationStepCount = 0
        return currentSolverMode
    }

    private func applySimulationParameters() {
        solver.dt = sceneDefaults.simulationStepDeltaTime
        solver.iterations = sceneDefaults.solverIterationCount
        solver.linearDamping = currentLinearDamping
        solver.angularDamping = currentAngularDamping
        gpuSolver?.dt = sceneDefaults.simulationStepDeltaTime
        gpuSolver?.iterations = sceneDefaults.solverIterationCount
        gpuSolver?.linearDamping = currentLinearDamping
        gpuSolver?.angularDamping = currentAngularDamping
    }

    @MainActor
    func setScene(_ id: AVBDSceneID) {
#if targetEnvironment(simulator)
        return
#else
        _ = rebuildScene(id, solverMode: currentSolverMode, shouldApplySceneDefaults: true)
        populateRenderInstances()
#endif
    }

    @MainActor
    func setSolverMode(_ mode: AVBDSolverMode) {
#if targetEnvironment(simulator)
        return
#else
        _ = rebuildScene(currentSceneID, solverMode: mode, shouldApplySceneDefaults: false)
        populateRenderInstances()
#endif
    }

    @MainActor
    func resetSimulation() {
#if targetEnvironment(simulator)
        return
#else
        _ = rebuildScene(currentSceneID, solverMode: currentSolverMode, shouldApplySceneDefaults: false)
        populateRenderInstances()
#endif
    }

    @MainActor
    func setSimulationRunMode(_ mode: AVBDSimulationRunMode) {
        currentSimulationRunMode = mode
        if mode == .auto {
            pendingManualSteps = 0
        }
    }

    var currentSimulationStepDeltaTime: Float {
        sceneDefaults.simulationStepDeltaTime
    }

    var currentSolverIterationCount: Int {
        sceneDefaults.solverIterationCount
    }

    var currentSimulationStepsPerFrame: Int {
        sceneDefaults.simulationStepsPerFrame
    }

    @MainActor
    func setSimulationStepDeltaTime(_ dt: Float) {
        sceneDefaults.setSimulationStepDeltaTime(dt)
        applySimulationParameters()
    }

    @MainActor
    func setSolverIterationCount(_ iterations: Int) {
        sceneDefaults.setSolverIterationCount(iterations)
        applySimulationParameters()
    }

    @MainActor
    func setSimulationStepsPerFrame(_ stepsPerFrame: Int) {
        sceneDefaults.setSimulationStepsPerFrame(stepsPerFrame)
    }

    @MainActor
    func requestManualStep() {
        guard currentSimulationRunMode == .manual else { return }
        pendingManualSteps += 1
    }

    private func shouldAdvanceSimulationThisFrame() -> Bool {
        switch currentSimulationRunMode {
        case .auto:
            return true
        case .manual:
            guard pendingManualSteps > 0 else { return false }
            pendingManualSteps -= 1
            return true
        }
    }

    /// Throw a projectile into the scene from the camera toward the tapped screen point.
    func throwBody(at screenPoint: CGPoint, viewSize: CGSize) {
        updateViewMatrix()

        // Convert screen point to NDC (-1..1)
        let ndcX = Float(screenPoint.x / viewSize.width) * 2.0 - 1.0
        let ndcY = 1.0 - Float(screenPoint.y / viewSize.height) * 2.0

        // Unproject from NDC to world space using inverse VP
        let vpMatrix = simd_mul(projectionMatrix, viewMatrix)
        let invVP = vpMatrix.inverse

        let nearClip = SIMD4<Float>(ndcX, ndcY, 0.0, 1.0)
        let farClip = SIMD4<Float>(ndcX, ndcY, 1.0, 1.0)

        var worldNear = invVP * nearClip
        worldNear /= worldNear.w
        var worldFar = invVP * farClip
        worldFar /= worldFar.w

        let rayOrigin = SIMD3<Float>(worldNear.x, worldNear.y, worldNear.z)
        let rayDir = normalize(SIMD3<Float>(worldFar.x - worldNear.x, worldFar.y - worldNear.y, worldFar.z - worldNear.z))

        // Spawn position slightly in front of the camera
        let spawnPos = rayOrigin + rayDir * 2.0
        let throwVelocity = rayDir * currentProjectileSpeed

        let projectilePhysicsShape = physicsRenderShape(for: currentProjectileKind)
        let projectileSize = projectileSizeVector(for: currentProjectileKind)
        let projectileDensity = projectileDensity(for: projectileSize, shape: projectilePhysicsShape)
        let colors: [SIMD4<Float>] = [
            SIMD4<Float>(1.0, 0.3, 0.2, 1.0),
            SIMD4<Float>(0.2, 0.8, 1.0, 1.0),
            SIMD4<Float>(1.0, 0.9, 0.1, 1.0),
            SIMD4<Float>(0.3, 1.0, 0.4, 1.0),
            SIMD4<Float>(1.0, 0.5, 0.0, 1.0),
        ]
        switch currentSolverMode {
        case .gpu:
            guard let gpuSolver else { return }
            let existingBodyCount = gpuSolver.bodyCount
            let color = colors[gpuSolver.bodyCount % colors.count]
            let bodyIndex = gpuSolver.addBody(
                position: spawnPos,
                velocity: throwVelocity,
                size: projectileSize,
                density: projectileDensity,
                friction: currentProjectileFriction,
                renderColor: color,
                renderShape: projectilePhysicsShape
            )
            if currentProjectileKind == .armadillo, let bodyIndex {
                armadilloDynamicRenderProxies.append(ArmadilloRenderProxy(bodyIndex: bodyIndex, baseColor: color))
            }
            if let newBodyIndex = bodyIndex {
                if currentProjectileKind == .armadillo {
                    // Keep SAT enabled against static bodies such as the ground.
                    gpuSolver.addIgnoredCollisionPairs(
                        (0..<existingBodyCount)
                            .filter { gpuSolver.isDynamicBody($0) }
                            .map { (newBodyIndex, $0) }
                    )
                    appendDynamicArmadilloCollisionMesh(
                        bodyIndex: newBodyIndex,
                        position: spawnPos,
                        orientation: simd_quatf(),
                        size: projectileSize
                    )
                } else {
                    gpuSolver.addIgnoredCollisionPairs(
                        armadilloDynamicRenderProxies
                            .filter { $0.bodyIndex != newBodyIndex }
                            .map { (newBodyIndex, $0.bodyIndex) }
                    )
                }
                if currentProjectileKind == .armadillo,
                   let collisionMeshInstance = collisionMeshBroadphaseMeshes.last,
                   gpuSolver.appendCollisionMeshInstance(collisionMeshInstance) {
                    addCollisionMeshSDFResourcesToResidencySetIfNeeded()
                } else {
                    applyCollisionMeshesToGPUSolver(fullRebuild: true)
                }
            }
        case .cpu:
            let color = colors[solver.bodies.count % colors.count]
            let bodyIndex = solver.addBody(
                position: spawnPos,
                velocity: throwVelocity,
                size: projectileSize,
                density: projectileDensity,
                friction: currentProjectileFriction,
                renderColor: color,
                renderShape: projectilePhysicsShape
            )
            if currentProjectileKind == .armadillo {
                armadilloDynamicRenderProxies.append(ArmadilloRenderProxy(bodyIndex: bodyIndex, baseColor: color))
            }
        }
    }

    private func physicsRenderShape(for projectileKind: AVBDProjectileKind) -> AVBDRenderShape {
        switch projectileKind {
        case .box:
            return .box
        case .sphere, .armadillo:
            return .sphere
        case .torus:
            return .torus
        }
    }

    private func projectileSizeVector(for projectileKind: AVBDProjectileKind) -> SIMD3<Float> {
        switch projectileKind {
        case .box, .sphere, .armadillo:
            return SIMD3<Float>(repeating: currentProjectileSize)
        case .torus:
            return SIMD3<Float>(currentProjectileSize, currentProjectileSize, currentProjectileSize * 0.35)
        }
    }

    private func projectileRenderSize(for bodySize: SIMD3<Float>, projectileKind: AVBDProjectileKind) -> Float {
        switch projectileKind {
        case .armadillo:
            return max(bodySize.x, max(bodySize.y, bodySize.z))
        case .box, .sphere, .torus:
            return max(bodySize.x, max(bodySize.y, bodySize.z))
        }
    }

    private func projectileDensity(for size: SIMD3<Float>, shape: AVBDRenderShape) -> Float {
        let unitDensityMass = avbdShapeMass(size: size, density: 1.0, shape: shape)
        guard unitDensityMass > 0 else { return 1.0 }
        return currentProjectileMass / unitDensityMass
    }

    private func filteredArmadilloRenderProxies(bodyCount: Int) -> [ArmadilloRenderProxy] {
        armadilloDynamicRenderProxies = armadilloDynamicRenderProxies.filter { $0.bodyIndex < bodyCount }
        return armadilloDynamicRenderProxies
    }

    private func projectileKind(for renderShape: AVBDRenderShape, bodyIndex: Int, armadilloBodyIndices: Set<Int>) -> AVBDProjectileKind {
        if armadilloBodyIndices.contains(bodyIndex) {
            return .armadillo
        }

        switch renderShape {
        case .box, .sphere:
            return renderShape == .box ? .box : .sphere
        case .torus:
            return .torus
        }
    }

    private func appendSceneStaticRenderInstances(
        armadilloInstances: UnsafeMutablePointer<InstanceUniforms>
    ) {
        guard let placement = sceneArmadilloPlacement(),
              let armadilloInstance = makeArmadilloInstance(
                position: placement.position,
                orientation: placement.orientation,
                targetSize: placement.targetSize,
                color: placement.color
              ) else {
            return
        }

        armadilloInstances[armadilloInstanceCount] = armadilloInstance
        armadilloInstanceCount += 1
    }

    private func appendCollisionMeshDebugBounds(
        sdfDebugInstances: UnsafeMutablePointer<InstanceUniforms>
    ) {
        guard showCollisionMeshSDFBounds,
              let gpuSolver else {
            return
        }

        let meshInfos = gpuSolver.collisionMeshInfoSnapshot()
        for meshIndex in meshInfos.indices {
            let meshInfo = meshInfos[meshIndex]
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
            let localCenter = 0.5 * (localMinBounds + localMaxBounds)
            let localExtent = max(localMaxBounds - localMinBounds, SIMD3<Float>(repeating: 0.01))
            let modelMatrix = simd_mul(
                meshInfo.sdfTransform,
                simd_mul(
                    matrix4x4_translation(localCenter.x, localCenter.y, localCenter.z),
                    matrix4x4_scale(localExtent.x, localExtent.y, localExtent.z)
                )
            )
            sdfDebugInstances[sdfDebugInstanceCount] = InstanceUniforms(
                modelMatrix: modelMatrix,
                renderColor: SIMD4<Float>(0.18, 0.92, 1.0, 0.38),
                shapeParams: SIMD4<Float>(Float(AVBDRenderShape.box.rawValue), Float(meshIndex), 0, 0)
            )
            sdfDebugInstanceCount += 1
        }
    }

    private func populateRenderInstances() {
        let boxInstances = UnsafeMutableRawPointer(instanceBuffer.contents() + instanceBufferOffset)
            .bindMemory(to: InstanceUniforms.self, capacity: maxInstanceCount)
        let sphereInstances = UnsafeMutableRawPointer(sphereInstanceBuffer.contents() + instanceBufferOffset)
            .bindMemory(to: InstanceUniforms.self, capacity: maxInstanceCount)
        let torusInstances = UnsafeMutableRawPointer(torusInstanceBuffer.contents() + instanceBufferOffset)
            .bindMemory(to: InstanceUniforms.self, capacity: maxInstanceCount)
        let armadilloInstances = UnsafeMutableRawPointer(armadilloInstanceBuffer.contents() + instanceBufferOffset)
            .bindMemory(to: InstanceUniforms.self, capacity: maxInstanceCount)
        let sdfDebugInstances = UnsafeMutableRawPointer(sdfDebugInstanceBuffer.contents() + instanceBufferOffset)
            .bindMemory(to: InstanceUniforms.self, capacity: maxInstanceCount)

        boxInstanceCount = 0
        sphereInstanceCount = 0
        torusInstanceCount = 0
        armadilloInstanceCount = 0
        sdfDebugInstanceCount = 0

        var gpuBodies: [AVBDGPUBody] = []
        let armadilloBodyIndices: Set<Int>
        if currentSolverMode == .gpu, let gpuSolver {
            gpuBodies = gpuSolver.readBodies()
            armadilloBodyIndices = Set(filteredArmadilloRenderProxies(bodyCount: gpuBodies.count).map(\.bodyIndex))
        } else {
            armadilloBodyIndices = Set(filteredArmadilloRenderProxies(bodyCount: solver.bodies.count).map(\.bodyIndex))
        }

        if currentSolverMode == .gpu, let gpuSolver {
            for bodyIndex in gpuBodies.indices {
                let body = gpuBodies[bodyIndex]
                let position = body.positionLin
                let orientation = simd_quatf(vector: body.positionAng)
                let color = gpuSolver.resolvedRenderColor(bodyIndex: bodyIndex)
                let projectileKind = projectileKind(
                    for: AVBDRenderShape(rawValue: Int(body.renderShape)) ?? .box,
                    bodyIndex: bodyIndex,
                    armadilloBodyIndices: armadilloBodyIndices
                )
                appendRenderInstance(
                    projectileKind: projectileKind,
                    renderShape: AVBDRenderShape(rawValue: Int(body.renderShape)) ?? .box,
                    position: position,
                    orientation: orientation,
                    size: body.size,
                    color: color,
                    boxInstances: boxInstances,
                    sphereInstances: sphereInstances,
                    torusInstances: torusInstances,
                    armadilloInstances: armadilloInstances
                )
            }
            rebuildCollisionMeshes(dynamicBodies: gpuBodies)
            applyCollisionMeshesToGPUSolver()
            appendSceneStaticRenderInstances(armadilloInstances: armadilloInstances)
            appendCollisionMeshDebugBounds(sdfDebugInstances: sdfDebugInstances)
            return
        }

        for bodyIndex in solver.bodies.indices {
            let body = solver.bodies[bodyIndex]
            let projectileKind = projectileKind(
                for: body.renderShape,
                bodyIndex: bodyIndex,
                armadilloBodyIndices: armadilloBodyIndices
            )
            appendRenderInstance(
                projectileKind: projectileKind,
                renderShape: body.renderShape,
                position: body.positionLin,
                orientation: body.positionAng,
                size: body.size,
                color: body.color,
                boxInstances: boxInstances,
                sphereInstances: sphereInstances,
                torusInstances: torusInstances,
                armadilloInstances: armadilloInstances
            )
        }
        appendSceneStaticRenderInstances(armadilloInstances: armadilloInstances)
        appendCollisionMeshDebugBounds(sdfDebugInstances: sdfDebugInstances)
    }

    private func appendRenderInstance(
        projectileKind: AVBDProjectileKind,
        renderShape: AVBDRenderShape,
        position: SIMD3<Float>,
        orientation: simd_quatf,
        size: SIMD3<Float>,
        color: SIMD4<Float>,
        boxInstances: UnsafeMutablePointer<InstanceUniforms>,
        sphereInstances: UnsafeMutablePointer<InstanceUniforms>,
        torusInstances: UnsafeMutablePointer<InstanceUniforms>,
        armadilloInstances: UnsafeMutablePointer<InstanceUniforms>
    ) {
        if projectileKind == .armadillo {
            if let armadilloInstance = makeArmadilloInstance(
                position: position,
                orientation: orientation,
                targetSize: projectileRenderSize(for: size, projectileKind: projectileKind),
                color: color
            ) {
                armadilloInstances[armadilloInstanceCount] = armadilloInstance
                armadilloInstanceCount += 1
            }
            return
        }

        switch renderShape {
        case .box:
            boxInstances[boxInstanceCount] = makeRigidInstance(
                position: position,
                orientation: orientation,
                size: size,
                color: color,
                shape: .box
            )
            boxInstanceCount += 1
        case .sphere:
            sphereInstances[sphereInstanceCount] = makeRigidInstance(
                position: position,
                orientation: orientation,
                size: size,
                color: color,
                shape: .sphere
            )
            sphereInstanceCount += 1
        case .torus:
            if currentTorusVisualMode == .solidTorus {
                torusInstances[torusInstanceCount] = makeTorusInstance(
                    position: position,
                    orientation: orientation,
                    size: size,
                    color: color
                )
                torusInstanceCount += 1
            } else {
                let torusSphereCount = avbdCurrentTorusApproxSphereCount()
                let sphereDiameter = avbdTorusApproxSphereRadius(size: size) * 2.0
                for torusSphereIndex in 0..<torusSphereCount {
                    let center = orientation.act(
                        avbdTorusApproxSphereLocalCenter(size: size, index: torusSphereIndex)
                    ) + position
                    sphereInstances[sphereInstanceCount] = makeProxySphereInstance(
                        center: center,
                        diameter: sphereDiameter,
                        color: color
                    )
                    sphereInstanceCount += 1
                }
            }
        }
    }

    private func runSolverStep() {
        guard currentSolverMode == .gpu else {
            solver.step()
            applyApproximateMeshCollisionsToCPUSolver()
            completedSimulationStepCount += 1
            populateRenderInstances()
            return
        }

        guard let gpuSolver = gpuSolver,
              let computeQueue = computeCommandQueue,
              let cmdBuf = computeQueue.makeCommandBuffer() else {
            currentSolverMode = .cpu
            solver.step()
            applyApproximateMeshCollisionsToCPUSolver()
            completedSimulationStepCount += 1
            populateRenderInstances()
            return
        }

        gpuSolver.step(commandBuffer: cmdBuf, instanceBuffer: instanceBuffer, instanceOffset: instanceBufferOffset)
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        completedSimulationStepCount += 1
        populateRenderInstances()
    }

    private func applyApproximateMeshCollisionsToCPUSolver() {
        guard !collisionMeshBroadphaseMeshes.isEmpty else {
            return
        }

        for body in solver.bodies where body.mass > 0 {
            let radius = avbdShapeRadius(size: body.size, shape: body.renderShape)
            var center = body.positionLin
            var velocity = body.velocityLin

            for mesh in collisionMeshBroadphaseMeshes {
                let worldBounds = approximateWorldBounds(for: mesh)
                let correction = sphereAABBCorrection(
                    center: center,
                    radius: radius,
                    minBounds: worldBounds.min,
                    maxBounds: worldBounds.max
                )
                guard let correction else {
                    continue
                }

                center += correction.normal * correction.penetration
                let inwardVelocity = simd_dot(velocity, correction.normal)
                if inwardVelocity < 0 {
                    velocity -= correction.normal * inwardVelocity
                }
            }

            body.positionLin = center
            body.velocityLin = velocity
            body.prevVelocityLin = velocity
        }
    }

    private func approximateWorldBounds(
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

    private func sphereAABBCorrection(
        center: SIMD3<Float>,
        radius: Float,
        minBounds: SIMD3<Float>,
        maxBounds: SIMD3<Float>
    ) -> (normal: SIMD3<Float>, penetration: Float)? {
        let closest = simd_clamp(center, minBounds, maxBounds)
        let delta = center - closest
        let distanceSquared = simd_length_squared(delta)

        if distanceSquared > 1.0e-8 {
            let distance = sqrt(distanceSquared)
            guard distance < radius else { return nil }
            return (delta / distance, radius - distance + 0.001)
        }

        let distancesToFaces = [
            (center.x - minBounds.x, SIMD3<Float>(1, 0, 0)),
            (maxBounds.x - center.x, SIMD3<Float>(-1, 0, 0)),
            (center.y - minBounds.y, SIMD3<Float>(0, 1, 0)),
            (maxBounds.y - center.y, SIMD3<Float>(0, -1, 0)),
            (center.z - minBounds.z, SIMD3<Float>(0, 0, 1)),
            (maxBounds.z - center.z, SIMD3<Float>(0, 0, -1)),
        ]

        guard let nearestFace = distancesToFaces.min(by: { $0.0 < $1.0 }) else {
            return nil
        }

        return (nearestFace.1, radius + nearestFace.0 + 0.001)
    }

    private func bindMesh(_ mesh: MTKMesh, instanceBuffer: MTLBuffer) {
        vertexArgumentTable.setAddress(instanceBuffer.gpuAddress + UInt64(instanceBufferOffset), index: BufferIndex.instances.rawValue)

        for (index, element) in mesh.vertexDescriptor.layouts.enumerated() {
            guard let layout = element as? MDLVertexBufferLayout else {
                return
            }

            if layout.stride != 0 {
                let buffer = mesh.vertexBuffers[index]
                vertexArgumentTable.setAddress(buffer.buffer.gpuAddress + UInt64(buffer.offset), index: index)
            }
        }
    }

    private func bindMesh(_ mesh: StaticMesh, instanceBuffer: MTLBuffer) {
        vertexArgumentTable.setAddress(instanceBuffer.gpuAddress + UInt64(instanceBufferOffset), index: BufferIndex.instances.rawValue)
        vertexArgumentTable.setAddress(mesh.positions.gpuAddress, index: BufferIndex.meshPositions.rawValue)
        vertexArgumentTable.setAddress(mesh.normals.gpuAddress, index: BufferIndex.meshNormals.rawValue)
    }

    private func drawMesh(_ mesh: MTKMesh, instanceCount: Int, encoder: any MTL4RenderCommandEncoder) {
        guard instanceCount > 0 else { return }

        for submesh in mesh.submeshes {
            encoder.drawIndexedPrimitives(primitiveType: submesh.primitiveType,
                                          indexCount: submesh.indexCount,
                                          indexType: submesh.indexType,
                                          indexBuffer: submesh.indexBuffer.buffer.gpuAddress + UInt64(submesh.indexBuffer.offset),
                                          indexBufferLength: submesh.indexBuffer.buffer.length,
                                          instanceCount: instanceCount)
        }
    }

    private func drawMesh(_ mesh: StaticMesh, instanceCount: Int, encoder: any MTL4RenderCommandEncoder) {
        guard instanceCount > 0 else { return }

        encoder.drawIndexedPrimitives(
            primitiveType: .triangle,
            indexCount: mesh.indexCount,
            indexType: .uint32,
            indexBuffer: mesh.indexBuffer.gpuAddress,
            indexBufferLength: mesh.indexBuffer.length,
            instanceCount: instanceCount
        )
    }

    func draw(in view: MTKView) {
        /// Per frame updates are here

#if !targetEnvironment(simulator)
        let frameTimestamp = CACurrentMediaTime()

        guard let drawable = view.currentDrawable else { return }

        /// Delay getting the currentMTL4RenderPassDescriptor until we absolutely need it to avoid
        ///   holding onto the drawable and blocking the display pipeline any longer than necessary
        guard let renderPassDescriptor = view.currentMTL4RenderPassDescriptor else { return }

        let previousValueToWaitFor = self.frameIndex - maxBuffersInFlight
        self.endFrameEvent.wait(untilSignaledValue: UInt64(previousValueToWaitFor), timeoutMS: 10)

        self.updateDynamicBufferState()

        self.updateGameState()
        if shouldAdvanceSimulationThisFrame() {
            let stepCount = currentSimulationRunMode == .auto ? sceneDefaults.simulationStepsPerFrame : 1
            for _ in 0..<stepCount {
                self.runSolverStep()
            }
        } else {
            self.populateRenderInstances()
        }
        self.publishDebugStatsIfNeeded(frameTimestamp: frameTimestamp)

        let commandAllocator = self.commandAllocators[uniformBufferIndex]
        commandAllocator.reset()
        commandBuffer.beginCommandBuffer(allocator: commandAllocator)

        guard let renderEncoder = self.commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            fatalError("Failed to create render command encoder")
        }

        /// Final pass rendering code here
        renderEncoder.label = "Primary Render Encoder"

        renderEncoder.pushDebugGroup("Draw AVBD Scene")

        renderEncoder.setCullMode(.none)

        renderEncoder.setFrontFacing(.counterClockwise)

        renderEncoder.setRenderPipelineState(pipelineState)

        renderEncoder.setDepthStencilState(depthState)

        renderEncoder.setArgumentTable(self.vertexArgumentTable, stages: .vertex)

        self.vertexArgumentTable.setAddress(dynamicUniformBuffer.gpuAddress + UInt64(uniformBufferOffset), index: BufferIndex.uniforms.rawValue)
        bindMesh(boxMesh, instanceBuffer: instanceBuffer)
        drawMesh(boxMesh, instanceCount: boxInstanceCount, encoder: renderEncoder)
        bindMesh(sphereMesh, instanceBuffer: sphereInstanceBuffer)
        drawMesh(sphereMesh, instanceCount: sphereInstanceCount, encoder: renderEncoder)
        bindMesh(torusMesh, instanceBuffer: torusInstanceBuffer)
        drawMesh(torusMesh, instanceCount: torusInstanceCount, encoder: renderEncoder)
        if let armadilloMesh {
            bindMesh(armadilloMesh, instanceBuffer: armadilloInstanceBuffer)
            drawMesh(armadilloMesh, instanceCount: armadilloInstanceCount, encoder: renderEncoder)
        }

        if showCollisionMeshSDFBounds,
           sdfDebugInstanceCount > 0,
           let gpuSolver,
           let collisionMeshInfoBuffer = gpuSolver.collisionMeshDebugInfoBuffer,
           let collisionMeshSDFArgumentBuffer = gpuSolver.collisionMeshDebugSDFArgumentBuffer {
            renderEncoder.setRenderPipelineState(sdfDebugPipelineState)
            renderEncoder.setDepthStencilState(debugDepthState)
            renderEncoder.setArgumentTable(self.fragmentArgumentTable, stages: .fragment)
            fragmentArgumentTable.setAddress(dynamicUniformBuffer.gpuAddress + UInt64(uniformBufferOffset), index: 0)
            fragmentArgumentTable.setAddress(collisionMeshInfoBuffer.gpuAddress, index: 1)
            fragmentArgumentTable.setAddress(collisionMeshSDFArgumentBuffer.gpuAddress, index: 2)
            bindMesh(boxMesh, instanceBuffer: sdfDebugInstanceBuffer)
            drawMesh(boxMesh, instanceCount: sdfDebugInstanceCount, encoder: renderEncoder)
            renderEncoder.setDepthStencilState(depthState)
        }

        renderEncoder.popDebugGroup()

        renderEncoder.endEncoding()

        commandBuffer.useResidencySet((view.layer as! CAMetalLayer).residencySet)
        commandBuffer.endCommandBuffer()

        commandQueue.waitForDrawable(drawable)
        commandQueue.commit([commandBuffer])
        commandQueue.signalDrawable(drawable)
        commandQueue.signalEvent(self.endFrameEvent, value: UInt64(self.frameIndex))
        self.frameIndex += 1
        drawable.present()
#endif
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        /// Respond to drawable size or orientation changes here

        let aspect = Float(size.width) / Float(size.height)
        projectionMatrix = matrix_perspective_right_hand(fovyRadians: radians_from_degrees(45), aspectRatio: aspect, nearZ: 0.1, farZ: 2000.0)
    }

    private static func computeMaxInstanceCount(extraBodies: Int = 50) -> Int {
        let maxSceneInstances = AVBDSceneID.allCases.reduce(1) { partialMax, sceneID in
            let instanceCount = AVBDSceneFactory.make(sceneID).bodies.reduce(0) { partialCount, body in
                partialCount + avbdRenderInstanceMultiplier(shape: body.renderShape)
            }
            return max(partialMax, instanceCount)
        }
        return max(maxSceneInstances + extraBodies * avbdTorusApproxSphereCountMax, 1)
    }

    private func updateViewMatrix() {
        let cameraEye = orbitEye(distance: cameraDistance, azimuth: cameraAzimuth, elevation: cameraElevation, target: cameraTarget)
        viewMatrix = matrix4x4_look_at_right_hand(eye: cameraEye, target: cameraTarget, up: SIMD3<Float>(0, 0, 1))
    }
}

// Generic matrix math utility functions
func matrix4x4_rotation(radians: Float, axis: SIMD3<Float>) -> matrix_float4x4 {
    let unitAxis = normalize(axis)
    let ct = cosf(radians)
    let st = sinf(radians)
    let ci = 1 - ct
    let x = unitAxis.x, y = unitAxis.y, z = unitAxis.z
    return matrix_float4x4.init(columns: (vector_float4(    ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
                                          vector_float4(x * y * ci - z * st,     ct + y * y * ci, z * y * ci + x * st, 0),
                                          vector_float4(x * z * ci + y * st, y * z * ci - x * st,     ct + z * z * ci, 0),
                                          vector_float4(                  0,                   0,                   0, 1)))
}

func matrix4x4_rotation(quaternion: simd_quatf) -> matrix_float4x4 {
    let q = quaternion.normalized.vector
    let x = q.x
    let y = q.y
    let z = q.z
    let w = q.w
    let xx = x * x
    let yy = y * y
    let zz = z * z
    let xy = x * y
    let xz = x * z
    let yz = y * z
    let wx = w * x
    let wy = w * y
    let wz = w * z

    return matrix_float4x4.init(columns: (
        vector_float4(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy), 0.0),
        vector_float4(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx), 0.0),
        vector_float4(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy), 0.0),
        vector_float4(0.0, 0.0, 0.0, 1.0)
    ))
}

func matrix4x4_translation(_ translationX: Float, _ translationY: Float, _ translationZ: Float) -> matrix_float4x4 {
    return matrix_float4x4.init(columns: (vector_float4(1, 0, 0, 0),
                                          vector_float4(0, 1, 0, 0),
                                          vector_float4(0, 0, 1, 0),
                                          vector_float4(translationX, translationY, translationZ, 1)))
}

func matrix4x4_scale(_ scaleX: Float, _ scaleY: Float, _ scaleZ: Float) -> matrix_float4x4 {
    return matrix_float4x4.init(columns: (vector_float4(scaleX, 0, 0, 0),
                                          vector_float4(0, scaleY, 0, 0),
                                          vector_float4(0, 0, scaleZ, 0),
                                          vector_float4(0, 0, 0, 1)))
}

func matrix4x4_look_at_right_hand(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) -> matrix_float4x4 {
    let zAxis = normalize(eye - target)
    let xAxis = normalize(cross(up, zAxis))
    let yAxis = cross(zAxis, xAxis)

    return matrix_float4x4.init(columns: (
        vector_float4(xAxis.x, yAxis.x, zAxis.x, 0),
        vector_float4(xAxis.y, yAxis.y, zAxis.y, 0),
        vector_float4(xAxis.z, yAxis.z, zAxis.z, 0),
        vector_float4(-dot(xAxis, eye), -dot(yAxis, eye), -dot(zAxis, eye), 1)
    ))
}

func orbitEye(distance: Float, azimuth: Float, elevation: Float, target: SIMD3<Float>) -> SIMD3<Float> {
    let ce = cosf(elevation)
    let se = sinf(elevation)
    let ca = cosf(azimuth)
    let sa = sinf(azimuth)
    let offset = SIMD3<Float>(distance * ce * ca, distance * ce * sa, distance * se)
    return target + offset
}

func matrix_perspective_right_hand(fovyRadians fovy: Float, aspectRatio: Float, nearZ: Float, farZ: Float) -> matrix_float4x4 {
    let ys = 1 / tanf(fovy * 0.5)
    let xs = ys / aspectRatio
    let zs = farZ / (nearZ - farZ)
    return matrix_float4x4.init(columns: (vector_float4(xs,  0, 0,   0),
                                          vector_float4( 0, ys, 0,   0),
                                          vector_float4( 0,  0, zs, -1),
                                          vector_float4( 0,  0, zs * nearZ, 0)))
}

func radians_from_degrees(_ degrees: Float) -> Float {
    return (degrees / 180) * .pi
}

private func clamp(_ value: Float, min lowerBound: Float, max upperBound: Float) -> Float {
    Swift.max(lowerBound, Swift.min(value, upperBound))
}
