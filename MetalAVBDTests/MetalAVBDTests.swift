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

    private func expectNearlyEqual(_ lhs: SIMD3<Float>, _ rhs: SIMD3<Float>, tolerance: Float) {
        #expect(simd_length(lhs - rhs) <= tolerance)
    }

    private func expectNearlyEqual(_ lhs: SIMD4<Float>, _ rhs: SIMD4<Float>, tolerance: Float) {
        #expect(simd_length(lhs - rhs) <= tolerance)
    }
}
