//
//  AVBDTriangleMeshAccelerationStructure.swift
//  MetalAVBD
//

import Metal
import simd

enum AVBDTriangleMeshAccelerationStructureError: Error {
    case failedToCreateCommandBuffer
    case failedToCreateCommandEncoder
    case failedToCreateBuffer(String)
    case failedToCreateAccelerationStructure(String)
    case emptyGeometry
    case emptyInstances
    case invalidVertexStride(Int)
    case invalidIndexCount(Int)
    case invalidGeometryIndex(Int)
    case buildFailed(String)
}

struct AVBDTriangleMeshGeometry {
    var vertexBuffer: MTLBuffer
    var vertexBufferOffset: Int = 0
    var vertexStride: Int = MemoryLayout<SIMD3<Float>>.stride
    var indexBuffer: MTLBuffer
    var indexBufferOffset: Int = 0
    var indexType: MTLIndexType = .uint32
    var indexCount: Int
    var isOpaque: Bool = true
    var label: String?

    var triangleCount: Int {
        indexCount / 3
    }
}

struct AVBDTriangleMeshInstance {
    var geometryIndex: Int
    var transform: simd_float4x4
    var mask: UInt32 = 0xFF
    var options: MTLAccelerationStructureInstanceOptions = []

    static func identity(geometryIndex: Int) -> Self {
        Self(geometryIndex: geometryIndex, transform: matrix_identity_float4x4)
    }
}

struct AVBDTriangleMeshAccelerationStructureBuildResult {
    var primitiveAccelerationStructures: [MTLAccelerationStructure]
    var instanceAccelerationStructure: MTLAccelerationStructure
    var instanceDescriptorBuffer: MTLBuffer
}

final class AVBDTriangleMeshAccelerationStructureBuilder {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    private(set) var geometries: [AVBDTriangleMeshGeometry] = []
    private(set) var instances: [AVBDTriangleMeshInstance] = []
    private(set) var primitiveAccelerationStructures: [MTLAccelerationStructure] = []
    private(set) var instanceAccelerationStructure: MTLAccelerationStructure?
    private(set) var instanceDescriptorBuffer: MTLBuffer?

    init?(device: MTLDevice, commandQueue: MTLCommandQueue? = nil) {
        guard let resolvedCommandQueue = commandQueue ?? device.makeCommandQueue() else {
            return nil
        }

        self.device = device
        self.commandQueue = resolvedCommandQueue
    }

    @discardableResult
    func build(
        geometries: [AVBDTriangleMeshGeometry],
        instances: [AVBDTriangleMeshInstance]
    ) throws -> AVBDTriangleMeshAccelerationStructureBuildResult {
        guard !geometries.isEmpty else {
            throw AVBDTriangleMeshAccelerationStructureError.emptyGeometry
        }

        for geometry in geometries {
            try validate(geometry: geometry)
        }

        let primitiveDescriptors = geometries.enumerated().map { index, geometry in
            makePrimitiveDescriptor(for: geometry, index: index)
        }

        let primitiveAccelerationStructures = try buildCompactedAccelerationStructures(for: primitiveDescriptors)

        self.geometries = geometries
        self.primitiveAccelerationStructures = primitiveAccelerationStructures

        return try rebuildTopLevel(instances: instances)
    }

    @discardableResult
    func rebuildTopLevel(
        instances: [AVBDTriangleMeshInstance]
    ) throws -> AVBDTriangleMeshAccelerationStructureBuildResult {
        guard !primitiveAccelerationStructures.isEmpty else {
            throw AVBDTriangleMeshAccelerationStructureError.buildFailed(
                "Primitive acceleration structures must be built before the top-level AS."
            )
        }
        guard !instances.isEmpty else {
            throw AVBDTriangleMeshAccelerationStructureError.emptyInstances
        }

        let instanceDescriptorBuffer = try makeInstanceDescriptorBuffer(instances: instances)

        let instanceDescriptor = MTLInstanceAccelerationStructureDescriptor()
        instanceDescriptor.instanceCount = instances.count
        instanceDescriptor.instanceDescriptorType = .indirect
        instanceDescriptor.instanceDescriptorBuffer = instanceDescriptorBuffer

        let instanceAccelerationStructure = try buildCompactedAccelerationStructure(with: instanceDescriptor)
        instanceAccelerationStructure.label = "AVBDTriangleMeshTLAS"

        self.instances = instances
        self.instanceDescriptorBuffer = instanceDescriptorBuffer
        self.instanceAccelerationStructure = instanceAccelerationStructure

        return AVBDTriangleMeshAccelerationStructureBuildResult(
            primitiveAccelerationStructures: primitiveAccelerationStructures,
            instanceAccelerationStructure: instanceAccelerationStructure,
            instanceDescriptorBuffer: instanceDescriptorBuffer
        )
    }

    private func validate(geometry: AVBDTriangleMeshGeometry) throws {
        if geometry.vertexStride <= 0 {
            throw AVBDTriangleMeshAccelerationStructureError.invalidVertexStride(geometry.vertexStride)
        }
        if geometry.indexCount <= 0 || geometry.indexCount % 3 != 0 {
            throw AVBDTriangleMeshAccelerationStructureError.invalidIndexCount(geometry.indexCount)
        }
    }

    private func makePrimitiveDescriptor(
        for geometry: AVBDTriangleMeshGeometry,
        index: Int
    ) -> MTLPrimitiveAccelerationStructureDescriptor {
        let triangleDescriptor = MTLAccelerationStructureTriangleGeometryDescriptor()
        triangleDescriptor.vertexBuffer = geometry.vertexBuffer
        triangleDescriptor.vertexBufferOffset = geometry.vertexBufferOffset
        triangleDescriptor.vertexStride = geometry.vertexStride
        triangleDescriptor.indexBuffer = geometry.indexBuffer
        triangleDescriptor.indexBufferOffset = geometry.indexBufferOffset
        triangleDescriptor.indexType = geometry.indexType
        triangleDescriptor.triangleCount = geometry.triangleCount
        triangleDescriptor.opaque = geometry.isOpaque

        let primitiveDescriptor = MTLPrimitiveAccelerationStructureDescriptor()
        primitiveDescriptor.geometryDescriptors = [triangleDescriptor]
        return primitiveDescriptor
    }

    private func makeInstanceDescriptorBuffer(
        instances: [AVBDTriangleMeshInstance]
    ) throws -> MTLBuffer {
        var descriptors: [MTLIndirectAccelerationStructureInstanceDescriptor] = []
        descriptors.reserveCapacity(instances.count)

        for instance in instances {
            guard primitiveAccelerationStructures.indices.contains(instance.geometryIndex) else {
                throw AVBDTriangleMeshAccelerationStructureError.invalidGeometryIndex(instance.geometryIndex)
            }

            var descriptor = MTLIndirectAccelerationStructureInstanceDescriptor()
            descriptor.accelerationStructureID = primitiveAccelerationStructures[instance.geometryIndex].gpuResourceID
            descriptor.mask = instance.mask
            descriptor.options = instance.options
            descriptor.transformationMatrix = packedFloat4x3(from: instance.transform)
            descriptors.append(descriptor)
        }

        let descriptorStride = MemoryLayout<MTLIndirectAccelerationStructureInstanceDescriptor>.stride
        let descriptorLength = descriptorStride * descriptors.count

        let buffer = descriptors.withUnsafeBytes { rawBuffer -> MTLBuffer? in
            guard let baseAddress = rawBuffer.baseAddress else { return nil }
            return device.makeBuffer(bytes: baseAddress, length: descriptorLength, options: .storageModeShared)
        }

        guard let buffer else {
            throw AVBDTriangleMeshAccelerationStructureError.failedToCreateBuffer(
                "Failed to allocate the indirect AS instance descriptor buffer."
            )
        }

        buffer.label = "AVBDTriangleMeshASInstances"
        return buffer
    }

    private func buildCompactedAccelerationStructure(
        with descriptor: MTLAccelerationStructureDescriptor
    ) throws -> MTLAccelerationStructure {
        guard let accelerationStructure = try buildCompactedAccelerationStructures(for: [descriptor]).first else {
            throw AVBDTriangleMeshAccelerationStructureError.failedToCreateAccelerationStructure(
                "Failed to build a compacted acceleration structure."
            )
        }
        return accelerationStructure
    }

    private func buildCompactedAccelerationStructures(
        for descriptors: [MTLAccelerationStructureDescriptor]
    ) throws -> [MTLAccelerationStructure] {
        let descriptorsAndSizes = descriptors.map { descriptor in
            (descriptor, device.accelerationStructureSizes(descriptor: descriptor))
        }

        let scratchBufferSize = max(descriptorsAndSizes.map(\.1.buildScratchBufferSize).max() ?? 0, 1)
        let compactedSizesBufferLength = max(MemoryLayout<UInt32>.stride * descriptors.count, MemoryLayout<UInt32>.stride)

        guard let scratchBuffer = device.makeBuffer(length: scratchBufferSize, options: .storageModePrivate) else {
            throw AVBDTriangleMeshAccelerationStructureError.failedToCreateBuffer(
                "Failed to allocate the AS build scratch buffer."
            )
        }
        guard let compactedSizesBuffer = device.makeBuffer(length: compactedSizesBufferLength, options: .storageModeShared) else {
            throw AVBDTriangleMeshAccelerationStructureError.failedToCreateBuffer(
                "Failed to allocate the AS compacted-size buffer."
            )
        }

        scratchBuffer.label = "AVBDTriangleMeshASScratch"
        compactedSizesBuffer.label = "AVBDTriangleMeshASCompactedSizes"

        guard let buildCommandBuffer = commandQueue.makeCommandBuffer() else {
            throw AVBDTriangleMeshAccelerationStructureError.failedToCreateCommandBuffer
        }
        guard let buildEncoder = buildCommandBuffer.makeAccelerationStructureCommandEncoder() else {
            throw AVBDTriangleMeshAccelerationStructureError.failedToCreateCommandEncoder
        }

        buildCommandBuffer.label = "AVBDTriangleMeshASBuild"
        buildEncoder.label = "AVBDTriangleMeshASBuild"

        let accelerationStructures: [MTLAccelerationStructure] = try descriptorsAndSizes.enumerated().map { index, descriptorAndSizes in
            let (descriptor, sizes) = descriptorAndSizes
            guard let accelerationStructure = device.makeAccelerationStructure(size: sizes.accelerationStructureSize) else {
                throw AVBDTriangleMeshAccelerationStructureError.failedToCreateAccelerationStructure(
                    "Failed to allocate primitive AS \(index)."
                )
            }
            accelerationStructure.label = "AVBDTriangleMeshBLAS_\(index)"
            buildEncoder.build(
                accelerationStructure: accelerationStructure,
                descriptor: descriptor,
                scratchBuffer: scratchBuffer,
                scratchBufferOffset: 0
            )
            buildEncoder.writeCompactedSize(
                accelerationStructure: accelerationStructure,
                buffer: compactedSizesBuffer,
                offset: MemoryLayout<UInt32>.stride * index
            )
            return accelerationStructure
        }

        buildEncoder.endEncoding()
        buildCommandBuffer.commit()
        buildCommandBuffer.waitUntilCompleted()

        if let error = buildCommandBuffer.error {
            throw AVBDTriangleMeshAccelerationStructureError.buildFailed(error.localizedDescription)
        }

        let compactedSizes = compactedSizesBuffer.contents().bindMemory(to: UInt32.self, capacity: descriptors.count)

        guard let compactCommandBuffer = commandQueue.makeCommandBuffer() else {
            throw AVBDTriangleMeshAccelerationStructureError.failedToCreateCommandBuffer
        }
        guard let compactEncoder = compactCommandBuffer.makeAccelerationStructureCommandEncoder() else {
            throw AVBDTriangleMeshAccelerationStructureError.failedToCreateCommandEncoder
        }

        compactCommandBuffer.label = "AVBDTriangleMeshASCompact"
        compactEncoder.label = "AVBDTriangleMeshASCompact"

        let compactedAccelerationStructures: [MTLAccelerationStructure] = try accelerationStructures.enumerated().map { index, accelerationStructure in
            let compactedSize = Int(compactedSizes[index])
            guard let compactedAccelerationStructure = device.makeAccelerationStructure(size: compactedSize) else {
                throw AVBDTriangleMeshAccelerationStructureError.failedToCreateAccelerationStructure(
                    "Failed to allocate compacted AS \(index)."
                )
            }
            compactedAccelerationStructure.label = "AVBDTriangleMeshCompactedAS_\(index)"
            compactEncoder.copyAndCompact(
                sourceAccelerationStructure: accelerationStructure,
                destinationAccelerationStructure: compactedAccelerationStructure
            )
            return compactedAccelerationStructure
        }

        compactEncoder.endEncoding()
        compactCommandBuffer.commit()
        compactCommandBuffer.waitUntilCompleted()

        if let error = compactCommandBuffer.error {
            throw AVBDTriangleMeshAccelerationStructureError.buildFailed(error.localizedDescription)
        }

        return compactedAccelerationStructures
    }

    private func packedFloat4x3(from transform: simd_float4x4) -> MTLPackedFloat4x3 {
        var packed = MTLPackedFloat4x3()
        packed.columns.0.x = transform.columns.0.x
        packed.columns.0.y = transform.columns.0.y
        packed.columns.0.z = transform.columns.0.z
        packed.columns.1.x = transform.columns.1.x
        packed.columns.1.y = transform.columns.1.y
        packed.columns.1.z = transform.columns.1.z
        packed.columns.2.x = transform.columns.2.x
        packed.columns.2.y = transform.columns.2.y
        packed.columns.2.z = transform.columns.2.z
        packed.columns.3.x = transform.columns.3.x
        packed.columns.3.y = transform.columns.3.y
        packed.columns.3.z = transform.columns.3.z
        return packed
    }
}
