//
//  MetalAVBDTests.swift
//  MetalAVBDTests
//
//  Created by Tatsuya Ogawa on 2026/04/12.
//

import Metal
import Testing
import simd
@testable import MetalAVBD

struct MetalAVBDTests {

    @Test func avbdGPUSolverInitializes() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let scene = AVBDSceneFactory.makeDefaultScene()
        let gpuSolver = AVBDGPUSolver(device: device, scene: scene)
        #expect(gpuSolver != nil)
    }

    @Test func stepMatchesExpectedFreeFallMotion() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let scene = makeFreeFallScene()
        let gpuSolver = try #require(AVBDGPUSolver(device: device, scene: scene))
        let instanceBuffer = try makeInstanceBuffer(device: device, bodyCount: scene.bodies.count)

        let stepCount = 3
        for _ in 0..<stepCount {
            try runStep(gpuSolver: gpuSolver, queue: queue, instanceBuffer: instanceBuffer)
        }

        let body = try #require(gpuSolver.readBodies().first)
        let dt = gpuSolver.dt
        let gravity = gpuSolver.gravity
        let initialPosition = scene.bodies[0].position
        let initialVelocity = scene.bodies[0].velocity
        let steps = Float(stepCount)

        let expectedPosition = initialPosition
            + initialVelocity * (dt * steps)
            + SIMD3<Float>(0, 0, gravity) * (dt * dt * (steps * (steps + 1)) * 0.5)
        let expectedVelocity = initialVelocity
            + SIMD3<Float>(0, 0, gravity) * (dt * steps)

        expectNearlyEqual(body.positionLin, expectedPosition, tolerance: 5.0e-5)
        expectNearlyEqual(body.velocityLin, expectedVelocity, tolerance: 5.0e-5)
        expectNearlyEqual(body.positionAng, scene.bodies[0].orientation.vector, tolerance: 1.0e-6)
    }

    @Test func stepMatchesCPUSolverForSpringScene() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let scene = makeSpringScene()

        let cpuSolver = AVBDSolver(scene: scene)
        cpuSolver.dt = 1.0 / 120.0
        cpuSolver.gravity = -9.8
        cpuSolver.iterations = 8

        let gpuSolver = try #require(AVBDGPUSolver(device: device, scene: scene))
        gpuSolver.dt = cpuSolver.dt
        gpuSolver.gravity = cpuSolver.gravity
        gpuSolver.iterations = cpuSolver.iterations

        let instanceBuffer = try makeInstanceBuffer(device: device, bodyCount: scene.bodies.count)

        for _ in 0..<4 {
            cpuSolver.step()
            try runStep(gpuSolver: gpuSolver, queue: queue, instanceBuffer: instanceBuffer)
        }

        let gpuBodies = gpuSolver.readBodies()
        #expect(gpuBodies.count == cpuSolver.bodies.count)

        for index in gpuBodies.indices {
            let gpuBody = gpuBodies[index]
            let cpuBody = cpuSolver.bodies[index]
            expectNearlyEqual(gpuBody.positionLin, cpuBody.positionLin, tolerance: 1.0e-3)
            expectNearlyEqual(gpuBody.velocityLin, cpuBody.velocityLin, tolerance: 1.0e-3)

            let gpuOrientation = simd_quatf(vector: gpuBody.positionAng)
            let cpuOrientation = cpuBody.positionAng
            let alignment = abs(simd_dot(gpuOrientation.vector, cpuOrientation.vector))
            #expect(abs(alignment - 1.0) <= 5.0e-4)
        }
    }

    @Test func compactedCollisionMeshSDFPreservesNearSurfaceValues() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let solver = try #require(AVBDGPUSolver(device: device, scene: AVBDSceneFactory.makeDefaultScene()))
        let mesh = makeUnitCubeCollisionMesh()
        let stats = try #require(solver.debugCompareCollisionMeshSDF(mesh, sampleGridResolution: 24))

        if stats.nearSurfaceMaxAbsoluteError > 0.08 {
            print(
                """
                SDF compare debug:
                overall max=\(stats.maxAbsoluteError) uv=\(stats.worstSampleUV) dense=\(stats.worstSampleDenseValue) compact=\(stats.worstSampleCompactValue) brick=\(stats.worstSampleBrick) coarse=\(stats.worstSampleUsedCoarseFallback)
                near max=\(stats.nearSurfaceMaxAbsoluteError) uv=\(stats.nearSurfaceWorstSampleUV) dense=\(stats.nearSurfaceWorstSampleDenseValue) compact=\(stats.nearSurfaceWorstSampleCompactValue) brick=\(stats.nearSurfaceWorstSampleBrick) coarse=\(stats.nearSurfaceWorstSampleUsedCoarseFallback)
                """
            )
        }

        #expect(stats.mappedBrickCount > 0)
        #expect(stats.mappedBrickCount < stats.totalBrickCount)
        #expect(stats.nearSurfaceSampleCount > 0)
        #expect(stats.meanAbsoluteError <= 0.08)
        #expect(stats.maxAbsoluteError <= 0.30)
        #expect(stats.nearSurfaceMeanAbsoluteError <= 0.025)
        #expect(stats.nearSurfaceMaxAbsoluteError <= 0.08)
    }

    private func runStep(
        gpuSolver: AVBDGPUSolver,
        queue: MTLCommandQueue,
        instanceBuffer: MTLBuffer
    ) throws {
        let commandBuffer = try #require(queue.makeCommandBuffer())
        gpuSolver.step(commandBuffer: commandBuffer, instanceBuffer: instanceBuffer, instanceOffset: 0)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)
        #expect(commandBuffer.error == nil)
    }

    private func makeInstanceBuffer(device: MTLDevice, bodyCount: Int) throws -> MTLBuffer {
        let length = max(bodyCount, 1) * MemoryLayout<InstanceUniforms>.stride
        return try #require(device.makeBuffer(length: length, options: .storageModeShared))
    }

    private func makeFreeFallScene() -> AVBDScene {
        AVBDScene(
            id: .empty,
            name: "Free Fall",
            bodies: [
                AVBDRigidBody(
                    renderShape: .box,
                    size: SIMD3<Float>(1, 1, 1),
                    density: 2.0,
                    friction: 0.5,
                    position: SIMD3<Float>(1.5, -2.0, 5.0),
                    orientation: simd_quatf(angle: 0, axis: SIMD3<Float>(0, 0, 1)),
                    velocity: SIMD3<Float>(2.0, -1.0, 3.5),
                    renderColor: nil,
                    colorGroup: nil
                )
            ],
            constraints: []
        )
    }

    private func makeSpringScene() -> AVBDScene {
        let bodies = [
            AVBDRigidBody(
                renderShape: .box,
                size: SIMD3<Float>(0.8, 0.8, 0.8),
                density: 1.0,
                friction: 0.4,
                position: SIMD3<Float>(-1.25, 0.0, 4.5),
                orientation: simd_quatf(angle: 0, axis: SIMD3<Float>(0, 0, 1)),
                velocity: SIMD3<Float>(0.2, 0.0, 0.0),
                renderColor: nil,
                colorGroup: nil
            ),
            AVBDRigidBody(
                renderShape: .box,
                size: SIMD3<Float>(0.8, 0.8, 0.8),
                density: 1.0,
                friction: 0.4,
                position: SIMD3<Float>(1.25, 0.0, 4.5),
                orientation: simd_quatf(angle: 0, axis: SIMD3<Float>(0, 0, 1)),
                velocity: SIMD3<Float>(-0.2, 0.0, 0.0),
                renderColor: nil,
                colorGroup: nil
            )
        ]

        return AVBDScene(
            id: .empty,
            name: "Spring Pair",
            bodies: bodies,
            constraints: [
                .spring(
                    bodyA: 0,
                    bodyB: 1,
                    anchorA: .zero,
                    anchorB: .zero,
                    stiffness: 60.0,
                    rest: 1.5
                )
            ]
        )
    }

    private func makeUnitCubeCollisionMesh() -> AVBDCollisionMeshBroadphaseMesh {
        let halfExtent: Float = 0.5
        let positions: [SIMD3<Float>] = [
            SIMD3<Float>(-halfExtent, -halfExtent, -halfExtent),
            SIMD3<Float>( halfExtent, -halfExtent, -halfExtent),
            SIMD3<Float>( halfExtent,  halfExtent, -halfExtent),
            SIMD3<Float>(-halfExtent,  halfExtent, -halfExtent),
            SIMD3<Float>(-halfExtent, -halfExtent,  halfExtent),
            SIMD3<Float>( halfExtent, -halfExtent,  halfExtent),
            SIMD3<Float>( halfExtent,  halfExtent,  halfExtent),
            SIMD3<Float>(-halfExtent,  halfExtent,  halfExtent),
        ]
        let indices: [UInt32] = [
            0, 1, 2, 0, 2, 3,
            4, 6, 5, 4, 7, 6,
            0, 4, 5, 0, 5, 1,
            1, 5, 6, 1, 6, 2,
            2, 6, 7, 2, 7, 3,
            3, 7, 4, 3, 4, 0,
        ]

        return AVBDCollisionMeshBroadphaseMesh(
            sdfResourceID: "test-cube",
            ownerBodyIndex: -1,
            localBoundsMin: SIMD3<Float>(repeating: -halfExtent),
            localBoundsMax: SIMD3<Float>(repeating: halfExtent),
            transform: matrix_identity_float4x4,
            positions: positions,
            indices: indices
        )
    }

    private func expectNearlyEqual(_ lhs: SIMD3<Float>, _ rhs: SIMD3<Float>, tolerance: Float) {
        #expect(simd_length(lhs - rhs) <= tolerance)
    }

    private func expectNearlyEqual(_ lhs: SIMD4<Float>, _ rhs: SIMD4<Float>, tolerance: Float) {
        #expect(simd_length(lhs - rhs) <= tolerance)
    }
}

fileprivate struct CollisionMeshSDFDebugComparisonStats {
    var sampleCount: Int
    var mappedBrickCount: Int
    var totalBrickCount: Int
    var maxAbsoluteError: Float
    var meanAbsoluteError: Float
    var worstSampleUV: SIMD3<Float>
    var worstSampleDenseValue: Float
    var worstSampleCompactValue: Float
    var worstSampleBrick: SIMD3<Int>
    var worstSampleUsedCoarseFallback: Bool
    var nearSurfaceSampleCount: Int
    var nearSurfaceMaxAbsoluteError: Float
    var nearSurfaceMeanAbsoluteError: Float
    var nearSurfaceWorstSampleUV: SIMD3<Float>
    var nearSurfaceWorstSampleDenseValue: Float
    var nearSurfaceWorstSampleCompactValue: Float
    var nearSurfaceWorstSampleBrick: SIMD3<Int>
    var nearSurfaceWorstSampleUsedCoarseFallback: Bool
}

fileprivate struct CollisionMeshSDFCompactedSample {
    var value: Float
    var brick: SIMD3<Int>
    var usedCoarseFallback: Bool
}

private let testCollisionMeshSDFBrickDim = 8
private let testCollisionMeshSDFGuardVoxelCount = 1
private let testCollisionMeshSDFStoredBrickDim = testCollisionMeshSDFBrickDim + testCollisionMeshSDFGuardVoxelCount * 2

extension AVBDGPUSolver {
    fileprivate func debugCompareCollisionMeshSDF(
        _ mesh: AVBDCollisionMeshBroadphaseMesh,
        sampleGridResolution: Int = 20
    ) -> CollisionMeshSDFDebugComparisonStats? {
        let sdfPadding = Self.collisionMeshSDFPadding(for: mesh)
        let sdfLocalMinBounds = mesh.localBoundsMin - sdfPadding
        let sdfLocalMaxBounds = mesh.localBoundsMax + sdfPadding
        let sdfResolution = Self.collisionMeshSDFResolution(
            localMinBounds: sdfLocalMinBounds,
            localMaxBounds: sdfLocalMaxBounds
        )

        guard let denseBuild = buildDenseCollisionMeshSDFTexture(
            localVertices: mesh.positions,
            indices: mesh.indices,
            localBoundsMin: sdfLocalMinBounds,
            localBoundsMax: sdfLocalMaxBounds,
            resolution: sdfResolution
        ) else {
            return nil
        }

        let denseResolution = SIMD3<Int>(Int(sdfResolution.x), Int(sdfResolution.y), Int(sdfResolution.z))
        let denseVoxelCount = denseResolution.x * denseResolution.y * denseResolution.z
        var denseData = [Float](repeating: 0, count: denseVoxelCount)
        denseData.withUnsafeMutableBytes { rawBuffer in
            denseBuild.texture.getBytes(
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
            voxelSize: denseBuild.voxelSize
        ) else {
            return nil
        }

        let sampleAxisCount = max(sampleGridResolution, 2)
        let totalSampleCount = sampleAxisCount * sampleAxisCount * sampleAxisCount
        let nearSurfaceThreshold = Self.collisionMeshSDFSparseBandHalfWidth(voxelSize: denseBuild.voxelSize)
        var sumAbsoluteError: Float = 0
        var maxAbsoluteError: Float = 0
        var worstSampleUV = SIMD3<Float>(repeating: 0)
        var worstSampleDenseValue: Float = 0
        var worstSampleCompactValue: Float = 0
        var worstSampleBrick = SIMD3<Int>(repeating: -1)
        var worstSampleUsedCoarseFallback = false
        var nearSurfaceSumAbsoluteError: Float = 0
        var nearSurfaceMaxAbsoluteError: Float = 0
        var nearSurfaceSampleCount = 0
        var nearSurfaceWorstSampleUV = SIMD3<Float>(repeating: 0)
        var nearSurfaceWorstSampleDenseValue: Float = 0
        var nearSurfaceWorstSampleCompactValue: Float = 0
        var nearSurfaceWorstSampleBrick = SIMD3<Int>(repeating: -1)
        var nearSurfaceWorstSampleUsedCoarseFallback = false

        for z in 0..<sampleAxisCount {
            let uz = Float(z) / Float(sampleAxisCount - 1)
            for y in 0..<sampleAxisCount {
                let uy = Float(y) / Float(sampleAxisCount - 1)
                for x in 0..<sampleAxisCount {
                    let ux = Float(x) / Float(sampleAxisCount - 1)
                    let uv = SIMD3<Float>(ux, uy, uz)
                    let denseSample = sampleCollisionMeshSDFDenseData(
                        denseData,
                        resolution: denseResolution,
                        uv: uv
                    )
                    let compactSample = sampleCollisionMeshSDFCompactedData(
                        compactionData: compactionData,
                        denseResolution: denseResolution,
                        uv: uv
                    )
                    let absoluteError = abs(compactSample.value - denseSample)
                    sumAbsoluteError += absoluteError
                    if absoluteError > maxAbsoluteError {
                        maxAbsoluteError = absoluteError
                        worstSampleUV = uv
                        worstSampleDenseValue = denseSample
                        worstSampleCompactValue = compactSample.value
                        worstSampleBrick = compactSample.brick
                        worstSampleUsedCoarseFallback = compactSample.usedCoarseFallback
                    }

                    if abs(denseSample) <= nearSurfaceThreshold {
                        nearSurfaceSampleCount += 1
                        nearSurfaceSumAbsoluteError += absoluteError
                        if absoluteError > nearSurfaceMaxAbsoluteError {
                            nearSurfaceMaxAbsoluteError = absoluteError
                            nearSurfaceWorstSampleUV = uv
                            nearSurfaceWorstSampleDenseValue = denseSample
                            nearSurfaceWorstSampleCompactValue = compactSample.value
                            nearSurfaceWorstSampleBrick = compactSample.brick
                            nearSurfaceWorstSampleUsedCoarseFallback = compactSample.usedCoarseFallback
                        }
                    }
                }
            }
        }

        return CollisionMeshSDFDebugComparisonStats(
            sampleCount: totalSampleCount,
            mappedBrickCount: compactionData.mappedBrickCount,
            totalBrickCount: compactionData.brickGrid.x * compactionData.brickGrid.y * compactionData.brickGrid.z,
            maxAbsoluteError: maxAbsoluteError,
            meanAbsoluteError: sumAbsoluteError / Float(max(totalSampleCount, 1)),
            worstSampleUV: worstSampleUV,
            worstSampleDenseValue: worstSampleDenseValue,
            worstSampleCompactValue: worstSampleCompactValue,
            worstSampleBrick: worstSampleBrick,
            worstSampleUsedCoarseFallback: worstSampleUsedCoarseFallback,
            nearSurfaceSampleCount: nearSurfaceSampleCount,
            nearSurfaceMaxAbsoluteError: nearSurfaceMaxAbsoluteError,
            nearSurfaceMeanAbsoluteError: nearSurfaceSumAbsoluteError / Float(max(nearSurfaceSampleCount, 1)),
            nearSurfaceWorstSampleUV: nearSurfaceWorstSampleUV,
            nearSurfaceWorstSampleDenseValue: nearSurfaceWorstSampleDenseValue,
            nearSurfaceWorstSampleCompactValue: nearSurfaceWorstSampleCompactValue,
            nearSurfaceWorstSampleBrick: nearSurfaceWorstSampleBrick,
            nearSurfaceWorstSampleUsedCoarseFallback: nearSurfaceWorstSampleUsedCoarseFallback
        )
    }

    private func sampleCollisionMeshSDFDenseData(
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

    private func sampleCollisionMeshSDFCompactedData(
        compactionData: CollisionMeshSDFCompactionData,
        denseResolution: SIMD3<Int>,
        uv: SIMD3<Float>
    ) -> CollisionMeshSDFCompactedSample {
        let clampedUV = simd_clamp(uv, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1))
        let sampleCoord = simd_clamp(
            clampedUV * SIMD3<Float>(Float(denseResolution.x), Float(denseResolution.y), Float(denseResolution.z)) - 0.5,
            SIMD3<Float>(repeating: 0),
            SIMD3<Float>(
                Float(max(denseResolution.x - 1, 0)),
                Float(max(denseResolution.y - 1, 0)),
                Float(max(denseResolution.z - 1, 0))
            )
        )
        let brickIdx = SIMD3<Int>(
            Int(floor(sampleCoord.x / Float(testCollisionMeshSDFBrickDim))),
            Int(floor(sampleCoord.y / Float(testCollisionMeshSDFBrickDim))),
            Int(floor(sampleCoord.z / Float(testCollisionMeshSDFBrickDim)))
        )

        if brickIdx.x < 0 || brickIdx.y < 0 || brickIdx.z < 0
            || brickIdx.x >= compactionData.brickGrid.x
            || brickIdx.y >= compactionData.brickGrid.y
            || brickIdx.z >= compactionData.brickGrid.z {
            return CollisionMeshSDFCompactedSample(
                value: sampleCollisionMeshSDFDenseData(
                    compactionData.coarseData,
                    resolution: compactionData.coarseResolution,
                    uv: clampedUV
                ),
                brick: brickIdx,
                usedCoarseFallback: true
            )
        }

        let brickLinearIndex = collisionMeshSDFLinearIndex(
            x: brickIdx.x,
            y: brickIdx.y,
            z: brickIdx.z,
            resolution: compactionData.brickGrid
        )
        let brickAtlasIndex = compactionData.indirectionData[brickLinearIndex]
        if brickAtlasIndex == UInt32.max {
            return CollisionMeshSDFCompactedSample(
                value: sampleCollisionMeshSDFDenseData(
                    compactionData.coarseData,
                    resolution: compactionData.coarseResolution,
                    uv: clampedUV
                ),
                brick: brickIdx,
                usedCoarseFallback: true
            )
        }

        let localPos = SIMD3<Float>(
            sampleCoord.x - Float(brickIdx.x * testCollisionMeshSDFBrickDim),
            sampleCoord.y - Float(brickIdx.y * testCollisionMeshSDFBrickDim),
            sampleCoord.z - Float(brickIdx.z * testCollisionMeshSDFBrickDim)
        )
        let atlasBricksAcross = max(compactionData.atlasResolution.x / testCollisionMeshSDFStoredBrickDim, 1)
        let atlasBricksDown = max(compactionData.atlasResolution.y / testCollisionMeshSDFStoredBrickDim, 1)
        let atlasLayerStride = atlasBricksAcross * atlasBricksDown
        let atlasIndex = Int(brickAtlasIndex)
        let atlasBrickZ = atlasIndex / atlasLayerStride
        let atlasBrickY = (atlasIndex / atlasBricksAcross) % atlasBricksDown
        let atlasBrickX = atlasIndex % atlasBricksAcross
        let atlasUV = SIMD3<Float>(
            (Float(atlasBrickX * testCollisionMeshSDFStoredBrickDim) + Float(testCollisionMeshSDFGuardVoxelCount) + localPos.x + 0.5) / Float(max(compactionData.atlasResolution.x, 1)),
            (Float(atlasBrickY * testCollisionMeshSDFStoredBrickDim) + Float(testCollisionMeshSDFGuardVoxelCount) + localPos.y + 0.5) / Float(max(compactionData.atlasResolution.y, 1)),
            (Float(atlasBrickZ * testCollisionMeshSDFStoredBrickDim) + Float(testCollisionMeshSDFGuardVoxelCount) + localPos.z + 0.5) / Float(max(compactionData.atlasResolution.z, 1))
        )
        return CollisionMeshSDFCompactedSample(
            value: sampleCollisionMeshSDFDenseData(
                compactionData.atlasData,
                resolution: compactionData.atlasResolution,
                uv: atlasUV
            ),
            brick: brickIdx,
            usedCoarseFallback: false
        )
    }

    private func collisionMeshSDFLinearIndex(
        x: Int,
        y: Int,
        z: Int,
        resolution: SIMD3<Int>
    ) -> Int {
        (z * resolution.y + y) * resolution.x + x
    }
}
