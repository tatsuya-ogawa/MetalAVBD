//
//  Renderer.swift
//  MetalAVBD
//
//  Created by Tatsuya Ogawa on 2026/04/07.
//

// Our platform independent renderer class

import Dispatch
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
}

class Renderer: NSObject, MTKViewDelegate {
    private static let minProjectileSize: Float = 0.25
    private static let maxProjectileSize: Float = 20.0
    private static let defaultProjectileSize: Float = 1.5
    private static let minProjectileMass: Float = 0.1
    private static let maxProjectileMass: Float = 25.0
    private static let defaultProjectileMass: Float = 3.375
    private static let minProjectileSpeed: Float = 1.0
    private static let maxProjectileSpeed: Float = 200.0
    private static let defaultProjectileSpeed: Float = 60.0

    public let device: MTLDevice

#if !targetEnvironment(simulator)
    let commandQueue: MTL4CommandQueue
    let commandBuffer: MTL4CommandBuffer
    let commandAllocators: [MTL4CommandAllocator]
    let commandQueueResidencySet: MTLResidencySet
    let vertexArgumentTable: MTL4ArgumentTable
#endif

    let endFrameEvent: MTLSharedEvent
    var frameIndex = 0

    var dynamicUniformBuffer: MTLBuffer
    var instanceBuffer: MTLBuffer
    var sphereInstanceBuffer: MTLBuffer
    var torusInstanceBuffer: MTLBuffer
    var pipelineState: MTLRenderPipelineState
    var depthState: MTLDepthStencilState

    var uniformBufferOffset = 0
    var instanceBufferOffset = 0

    var uniformBufferIndex = 0

    var uniforms: UnsafeMutablePointer<Uniforms>

    var projectionMatrix: matrix_float4x4 = matrix_float4x4()
    var viewMatrix: matrix_float4x4 = matrix_float4x4()

    var boxMesh: MTKMesh
    var sphereMesh: MTKMesh
    var torusMesh: StaticMesh
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
    var projectileRenderShape: AVBDRenderShape = .box
    private(set) var currentProjectileSize: Float = Renderer.defaultProjectileSize
    private(set) var currentProjectileMass: Float = Renderer.defaultProjectileMass
    private(set) var currentProjectileSpeed: Float = Renderer.defaultProjectileSpeed
    private var pendingManualSteps = 0
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

            metalKitView.depthStencilPixelFormat = MTLPixelFormat.depth32Float_stencil8
            metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
            metalKitView.sampleCount = 1

            let meshVertexDescriptor = Renderer.buildMeshVertexDescriptor()
            let pipelineVertexDescriptor = Renderer.buildPipelineVertexDescriptor()

            do {
                pipelineState = try Renderer.buildRenderPipelineWithDevice(device: device,
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
            residencySet.addAllocations([dynamicUniformBuffer, instanceBuffer, sphereInstanceBuffer, torusInstanceBuffer])
            residencySet.commit()
            commandQueue.addResidencySet(residencySet)
            commandQueueResidencySet = residencySet

            super.init()
            sceneDefaults = scene.defaults
            applySimulationParameters()
            updateViewMatrix()
            populateRenderInstances()
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

        metalKitView.depthStencilPixelFormat = MTLPixelFormat.depth32Float_stencil8
        metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
        metalKitView.sampleCount = 1

        let meshVertexDescriptor = Renderer.buildMeshVertexDescriptor()
        let pipelineVertexDescriptor = Renderer.buildPipelineVertexDescriptor()

        do {
            pipelineState = try Renderer.buildRenderPipelineWithDevice(device: device,
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
        residencySet.addAllocations([dynamicUniformBuffer, instanceBuffer, sphereInstanceBuffer, torusInstanceBuffer])
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
#endif
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
        uniforms[0].viewProjectionMatrix = simd_mul(projectionMatrix, viewMatrix)
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
            warmstartModeName: currentWarmstartModeName
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
            warmstartModeName: currentWarmstartModeName
        )
    }

    func setProjectileRenderShape(_ shape: AVBDRenderShape) {
        projectileRenderShape = shape
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
        gpuSolver?.dt = sceneDefaults.simulationStepDeltaTime
        gpuSolver?.iterations = sceneDefaults.solverIterationCount
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

        let projectileSize = projectileSizeVector(for: projectileRenderShape)
        let projectileDensity = projectileDensity(for: projectileSize, shape: projectileRenderShape)
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
            let color = colors[gpuSolver.bodyCount % colors.count]
            _ = gpuSolver.addBody(
                position: spawnPos,
                velocity: throwVelocity,
                size: projectileSize,
                density: projectileDensity,
                friction: 0.5,
                renderColor: color,
                renderShape: projectileRenderShape
            )
        case .cpu:
            let color = colors[solver.bodies.count % colors.count]
            _ = solver.addBody(
                position: spawnPos,
                velocity: throwVelocity,
                size: projectileSize,
                density: projectileDensity,
                friction: 0.5,
                renderColor: color,
                renderShape: projectileRenderShape
            )
        }
    }

    private func projectileSizeVector(for shape: AVBDRenderShape) -> SIMD3<Float> {
        switch shape {
        case .box, .sphere:
            return SIMD3<Float>(repeating: currentProjectileSize)
        case .torus:
            return SIMD3<Float>(currentProjectileSize, currentProjectileSize, currentProjectileSize * 0.35)
        }
    }

    private func projectileDensity(for size: SIMD3<Float>, shape: AVBDRenderShape) -> Float {
        let unitDensityMass = avbdShapeMass(size: size, density: 1.0, shape: shape)
        guard unitDensityMass > 0 else { return 1.0 }
        return currentProjectileMass / unitDensityMass
    }

    private func populateRenderInstances() {
        let boxInstances = UnsafeMutableRawPointer(instanceBuffer.contents() + instanceBufferOffset)
            .bindMemory(to: InstanceUniforms.self, capacity: maxInstanceCount)
        let sphereInstances = UnsafeMutableRawPointer(sphereInstanceBuffer.contents() + instanceBufferOffset)
            .bindMemory(to: InstanceUniforms.self, capacity: maxInstanceCount)
        let torusInstances = UnsafeMutableRawPointer(torusInstanceBuffer.contents() + instanceBufferOffset)
            .bindMemory(to: InstanceUniforms.self, capacity: maxInstanceCount)

        if currentSolverMode == .gpu, let gpuSolver = gpuSolver {
            let counts = gpuSolver.writeRenderInstances(
                boxInstances: boxInstances,
                sphereInstances: sphereInstances,
                torusInstances: torusInstances,
                torusVisualMode: currentTorusVisualMode
            )
            boxInstanceCount = counts.boxCount
            sphereInstanceCount = counts.sphereCount
            torusInstanceCount = counts.torusCount
            return
        }

        boxInstanceCount = 0
        sphereInstanceCount = 0
        torusInstanceCount = 0

        for body in solver.bodies {
            switch body.renderShape {
            case .box:
                boxInstances[boxInstanceCount] = makeRigidInstance(
                    position: body.positionLin,
                    orientation: body.positionAng,
                    size: body.size,
                    color: body.color,
                    shape: .box
                )
                boxInstanceCount += 1
            case .sphere:
                sphereInstances[sphereInstanceCount] = makeRigidInstance(
                    position: body.positionLin,
                    orientation: body.positionAng,
                    size: body.size,
                    color: body.color,
                    shape: .sphere
                )
                sphereInstanceCount += 1
            case .torus:
                if currentTorusVisualMode == .solidTorus {
                    torusInstances[torusInstanceCount] = makeTorusInstance(
                        position: body.positionLin,
                        orientation: body.positionAng,
                        size: body.size,
                        color: body.color
                    )
                    torusInstanceCount += 1
                } else {
                    let torusSphereCount = avbdCurrentTorusApproxSphereCount()
                    let sphereDiameter = avbdTorusApproxSphereRadius(size: body.size) * 2.0
                    for torusSphereIndex in 0..<torusSphereCount {
                        let center = body.positionAng.act(
                            avbdTorusApproxSphereLocalCenter(size: body.size, index: torusSphereIndex)
                        ) + body.positionLin
                        sphereInstances[sphereInstanceCount] = makeProxySphereInstance(
                            center: center,
                            diameter: sphereDiameter,
                            color: body.color
                        )
                        sphereInstanceCount += 1
                    }
                }
            }
        }
    }

    private func runSolverStep() {
        guard currentSolverMode == .gpu else {
            solver.step()
            completedSimulationStepCount += 1
            populateRenderInstances()
            return
        }

        guard let gpuSolver = gpuSolver,
              let computeQueue = computeCommandQueue,
              let cmdBuf = computeQueue.makeCommandBuffer() else {
            currentSolverMode = .cpu
            solver.step()
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
