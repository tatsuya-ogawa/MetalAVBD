//
//  AVBDScene.swift
//  MetalAVBD
//

import Foundation
import simd

nonisolated enum AVBDSceneID: CaseIterable {
    case empty
    case ground
    case dynamicFriction
    case staticFriction
    case tower
    case pyramid
    case ring
    case chainMail
    case rope
    case heavyRope
    case spring
    case springsRatio
    case stack
    case stackRatio
    case softBody
    case bridge
    case breakable
}

nonisolated let avbdTorusApproxSphereCountDefault = Int(AVBD_TORUS_APPROX_SPHERE_COUNT_DEFAULT)
nonisolated let avbdTorusApproxSphereCountMax = Int(AVBD_TORUS_APPROX_SPHERE_COUNT_MAX)
nonisolated let avbdTorusApproxSphereRadiusScaleDefault: Float = Float(AVBD_TORUS_APPROX_SPHERE_RADIUS_SCALE_DEFAULT)

nonisolated enum AVBDTorusApproximationSettings {
    private static let lock = NSLock()
    private static var _sphereCount = avbdTorusApproxSphereCountDefault
    private static var _radiusScale = avbdTorusApproxSphereRadiusScaleDefault

    static var sphereCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return _sphereCount
    }

    static var radiusScale: Float {
        lock.lock()
        defer { lock.unlock() }
        return _radiusScale
    }

    static func update(sphereCount: Int? = nil, radiusScale: Float? = nil) {
        lock.lock()
        defer { lock.unlock() }
        if let sphereCount {
            _sphereCount = min(max(sphereCount, Int(AVBD_TORUS_APPROX_SPHERE_COUNT_MIN)), avbdTorusApproxSphereCountMax)
        }
        if let radiusScale {
            _radiusScale = min(max(radiusScale, 0.25), 4.0)
        }
    }
}

nonisolated func avbdCurrentTorusApproxSphereCount() -> Int {
    AVBDTorusApproximationSettings.sphereCount
}

nonisolated func avbdTorusOuterRadius(size: SIMD3<Float>) -> Float {
    max(min(size.x, size.y), 0.0) * 0.5
}

nonisolated func avbdTorusMinorRadius(size: SIMD3<Float>) -> Float {
    let outerDiameter = avbdTorusOuterRadius(size: size) * 2.0
    let tubeDiameter = max(min(size.z, outerDiameter), 0.0)
    return tubeDiameter * 0.5
}

nonisolated func avbdTorusMajorRadius(size: SIMD3<Float>) -> Float {
    max(avbdTorusOuterRadius(size: size) - avbdTorusMinorRadius(size: size), 0.0)
}

nonisolated func avbdTorusApproxSphereRadius(size: SIMD3<Float>) -> Float {
    avbdTorusMinorRadius(size: size) * AVBDTorusApproximationSettings.radiusScale
}

nonisolated func avbdTorusRenderMinorRadius(size: SIMD3<Float>) -> Float {
    avbdTorusApproxSphereRadius(size: size)
}

nonisolated func avbdTorusApproxSphereLocalCenter(size: SIMD3<Float>, index: Int) -> SIMD3<Float> {
    let majorRadius = avbdTorusMajorRadius(size: size)
    let angle = (2.0 * .pi * Float(index)) / Float(avbdCurrentTorusApproxSphereCount())
    return SIMD3<Float>(cos(angle) * majorRadius, sin(angle) * majorRadius, 0.0)
}

nonisolated let avbdStaticBodyRenderColor = SIMD4<Float>(0.47, 0.50, 0.47, 1.0)
nonisolated let avbdDynamicBodyRenderColor = SIMD4<Float>(0.72, 0.80, 0.92, 1.0)
nonisolated let avbdColorGroupRenderPalette: [SIMD4<Float>] = [
    SIMD4<Float>(0.93, 0.42, 0.39, 1.0),
    SIMD4<Float>(0.97, 0.71, 0.30, 1.0),
    SIMD4<Float>(0.48, 0.79, 0.51, 1.0),
    SIMD4<Float>(0.35, 0.73, 0.92, 1.0),
    SIMD4<Float>(0.63, 0.52, 0.90, 1.0),
]

nonisolated func avbdDefaultRenderColor(isStatic: Bool) -> SIMD4<Float> {
    isStatic ? avbdStaticBodyRenderColor : avbdDynamicBodyRenderColor
}

nonisolated func avbdRenderColor(forColorGroup colorGroup: Int) -> SIMD4<Float> {
    guard !avbdColorGroupRenderPalette.isEmpty else {
        return avbdDynamicBodyRenderColor
    }
    let paletteIndex = ((colorGroup % avbdColorGroupRenderPalette.count) + avbdColorGroupRenderPalette.count) % avbdColorGroupRenderPalette.count
    return avbdColorGroupRenderPalette[paletteIndex]
}

nonisolated func avbdResolveRenderColor(
    renderColor: SIMD4<Float>?,
    colorGroup: Int?,
    isStatic: Bool
) -> SIMD4<Float> {
    if let renderColor {
        return renderColor
    }
    if let colorGroup {
        return avbdRenderColor(forColorGroup: colorGroup)
    }
    return avbdDefaultRenderColor(isStatic: isStatic)
}

nonisolated func avbdRenderInstanceMultiplier(shape: AVBDRenderShape) -> Int {
    switch shape {
    case .box, .sphere:
        return 1
    case .torus:
        return Int(AVBD_TORUS_APPROX_SPHERE_COUNT_MAX)
    }
}

nonisolated func avbdShapeRadius(size: SIMD3<Float>, shape: AVBDRenderShape) -> Float {
    switch shape {
    case .box:
        return simd_length(size * 0.5)
    case .sphere:
        return max(size.x, max(size.y, size.z)) * 0.5
    case .torus:
        return avbdTorusMajorRadius(size: size) + avbdTorusApproxSphereRadius(size: size)
    }
}

nonisolated func avbdShapeMass(size: SIMD3<Float>, density: Float, shape: AVBDRenderShape) -> Float {
    switch shape {
    case .box:
        return size.x * size.y * size.z * density
    case .sphere:
        let radius = avbdShapeRadius(size: size, shape: shape)
        return (4.0 / 3.0) * .pi * radius * radius * radius * density
    case .torus:
        let majorRadius = avbdTorusMajorRadius(size: size)
        let minorRadius = avbdTorusMinorRadius(size: size)
        return 2.0 * .pi * .pi * majorRadius * minorRadius * minorRadius * density
    }
}

nonisolated func avbdShapeMoment(size: SIMD3<Float>, mass: Float, shape: AVBDRenderShape) -> SIMD3<Float> {
    switch shape {
    case .box:
        return SIMD3<Float>(
            (size.y * size.y + size.z * size.z) / 12.0 * mass,
            (size.x * size.x + size.z * size.z) / 12.0 * mass,
            (size.x * size.x + size.y * size.y) / 12.0 * mass
        )
    case .sphere:
        let radius = avbdShapeRadius(size: size, shape: shape)
        let inertia = 0.4 * mass * radius * radius
        return SIMD3<Float>(repeating: inertia)
    case .torus:
        let majorRadius = avbdTorusMajorRadius(size: size)
        let minorRadius = avbdTorusMinorRadius(size: size)
        let planarInertia = mass * (0.5 * majorRadius * majorRadius + 0.625 * minorRadius * minorRadius)
        let axialInertia = mass * (majorRadius * majorRadius + 0.75 * minorRadius * minorRadius)
        return SIMD3<Float>(planarInertia, planarInertia, axialInertia)
    }
}

nonisolated struct AVBDRigidBody {
    var renderShape: AVBDRenderShape = .box
    var size: SIMD3<Float>
    var density: Float
    var friction: Float
    var position: SIMD3<Float>
    var orientation: simd_quatf
    var velocity: SIMD3<Float>
    var renderColor: SIMD4<Float>?
    var colorGroup: Int?

    var mass: Float {
        avbdShapeMass(size: size, density: density, shape: renderShape)
    }

    var isStatic: Bool {
        mass <= 0.0
    }

    var resolvedRenderColor: SIMD4<Float> {
        avbdResolveRenderColor(renderColor: renderColor, colorGroup: colorGroup, isStatic: isStatic)
    }
}

nonisolated enum AVBDConstraint {
    case joint(bodyA: Int, bodyB: Int, anchorA: SIMD3<Float>, anchorB: SIMD3<Float>, stiffnessLin: Float, stiffnessAng: Float, fracture: Float?)
    case spring(bodyA: Int, bodyB: Int, anchorA: SIMD3<Float>, anchorB: SIMD3<Float>, stiffness: Float, rest: Float)
    case ignoreCollision(bodyA: Int, bodyB: Int)
}

nonisolated struct AVBDScene {
    var id: AVBDSceneID
    var name: String
    var bodies: [AVBDRigidBody]
    var constraints: [AVBDConstraint]
    var defaults: AVBDSceneDefaults = AVBDSceneDefaults()
}

nonisolated struct AVBDSceneDefaults {
    static let minSimulationStepDeltaTime: Float = 1.0 / 240.0
    static let maxSimulationStepDeltaTime: Float = 1.0 / 15.0
    static let defaultSimulationStepDeltaTime: Float = 1.0 / 60.0
    static let minSolverIterationCount = 1
    static let maxSolverIterationCount = 64
    static let defaultSolverIterationCount = 10
    static let minSimulationStepsPerFrame = 1
    static let maxSimulationStepsPerFrame = 16
    static let defaultSimulationStepsPerFrame = 1
    static let minBroadphaseFullRefreshStepCount = 0
    static let maxBroadphaseFullRefreshStepCount = 300
    static let defaultBroadphaseFullRefreshStepCount = 0
    static let defaultEnableContactWarmstart = false

    var simulationStepDeltaTime: Float
    var solverIterationCount: Int
    var simulationStepsPerFrame: Int
    var broadphaseFullRefreshStepCount: Int
    var enableContactWarmstart: Bool

    init(
        simulationStepDeltaTime: Float? = nil,
        solverIterationCount: Int? = nil,
        simulationStepsPerFrame: Int? = nil,
        broadphaseFullRefreshStepCount: Int? = nil,
        enableContactWarmstart: Bool? = nil
    ) {
        self.simulationStepDeltaTime = Self.clampedSimulationStepDeltaTime(
            simulationStepDeltaTime ?? Self.defaultSimulationStepDeltaTime
        )
        self.solverIterationCount = Self.clampedSolverIterationCount(
            solverIterationCount ?? Self.defaultSolverIterationCount
        )
        self.simulationStepsPerFrame = Self.clampedSimulationStepsPerFrame(
            simulationStepsPerFrame ?? Self.defaultSimulationStepsPerFrame
        )
        self.broadphaseFullRefreshStepCount = Self.clampedBroadphaseFullRefreshStepCount(
            broadphaseFullRefreshStepCount ?? Self.defaultBroadphaseFullRefreshStepCount
        )
        self.enableContactWarmstart = enableContactWarmstart ?? Self.defaultEnableContactWarmstart
    }

    static func clampedSimulationStepDeltaTime(_ dt: Float) -> Float {
        Swift.max(minSimulationStepDeltaTime, Swift.min(dt, maxSimulationStepDeltaTime))
    }

    static func clampedSolverIterationCount(_ count: Int) -> Int {
        Swift.max(minSolverIterationCount, Swift.min(count, maxSolverIterationCount))
    }

    static func clampedSimulationStepsPerFrame(_ count: Int) -> Int {
        Swift.max(minSimulationStepsPerFrame, Swift.min(count, maxSimulationStepsPerFrame))
    }

    static func clampedBroadphaseFullRefreshStepCount(_ count: Int) -> Int {
        Swift.max(minBroadphaseFullRefreshStepCount, Swift.min(count, maxBroadphaseFullRefreshStepCount))
    }

    mutating func setSimulationStepDeltaTime(_ dt: Float) {
        simulationStepDeltaTime = Self.clampedSimulationStepDeltaTime(dt)
    }

    mutating func setSolverIterationCount(_ count: Int) {
        solverIterationCount = Self.clampedSolverIterationCount(count)
    }

    mutating func setSimulationStepsPerFrame(_ count: Int) {
        simulationStepsPerFrame = Self.clampedSimulationStepsPerFrame(count)
    }

    mutating func setBroadphaseFullRefreshStepCount(_ count: Int) {
        broadphaseFullRefreshStepCount = Self.clampedBroadphaseFullRefreshStepCount(count)
    }

    mutating func setEnableContactWarmstart(_ enabled: Bool) {
        enableContactWarmstart = enabled
    }
}

nonisolated private final class AVBDSceneBuilder {
    private(set) var bodies: [AVBDRigidBody] = []
    private(set) var constraints: [AVBDConstraint] = []

    func rigid(
        renderShape: AVBDRenderShape = .box,
        size: SIMD3<Float>,
        density: Float,
        friction: Float,
        position: SIMD3<Float>,
        orientation: simd_quatf = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 0, 1)),
        velocity: SIMD3<Float> = .zero,
        renderColor: SIMD4<Float>? = nil,
        colorGroup: Int? = nil
    ) -> Int {
        bodies.append(AVBDRigidBody(
            renderShape: renderShape,
            size: size,
            density: density,
            friction: friction,
            position: position,
            orientation: orientation,
            velocity: velocity,
            renderColor: renderColor,
            colorGroup: colorGroup
        ))
        return bodies.count - 1
    }

    func joint(
        _ bodyA: Int,
        _ bodyB: Int,
        _ anchorA: SIMD3<Float>,
        _ anchorB: SIMD3<Float>,
        stiffnessLin: Float = .infinity,
        stiffnessAng: Float = 0.0,
        fracture: Float? = nil
    ) {
        constraints.append(.joint(
            bodyA: bodyA,
            bodyB: bodyB,
            anchorA: anchorA,
            anchorB: anchorB,
            stiffnessLin: stiffnessLin,
            stiffnessAng: stiffnessAng,
            fracture: fracture
        ))
    }

    func spring(
        _ bodyA: Int,
        _ bodyB: Int,
        _ anchorA: SIMD3<Float>,
        _ anchorB: SIMD3<Float>,
        stiffness: Float,
        rest: Float = -1.0
    ) {
        constraints.append(.spring(
            bodyA: bodyA,
            bodyB: bodyB,
            anchorA: anchorA,
            anchorB: anchorB,
            stiffness: stiffness,
            rest: rest
        ))
    }

    func ignoreCollision(_ bodyA: Int, _ bodyB: Int) {
        constraints.append(.ignoreCollision(bodyA: bodyA, bodyB: bodyB))
    }
}

nonisolated enum AVBDSceneFactory {
    private enum RingTopSupportMode {
        case singlePoint
        case twoPoint
        case fullyFixed
    }

    // Build-time switch for the hanging ring scenes.
    private static let ringTopSupportMode: RingTopSupportMode = .twoPoint
    private static let ringBridgeTopSupportMode: RingTopSupportMode = .twoPoint

    static func makeDefaultScene() -> AVBDScene {
        make(.chainMail)
    }

    static func make(_ id: AVBDSceneID) -> AVBDScene {
        let builder = AVBDSceneBuilder()

        switch id {
        case .empty:
            break

        case .ground:
            addGround(builder, z: 0)
            _ = builder.rigid(size: SIMD3<Float>(1, 1, 1), density: 1.0, friction: 0.5, position: SIMD3<Float>(0, 0, 4))

        case .dynamicFriction:
            addGround(builder, z: 0)
            for x in 0...10 {
                let friction = 5.0 - (Float(x) / 10.0 * 5.0)
                _ = builder.rigid(
                    size: SIMD3<Float>(1, 1, 0.5),
                    density: 1.0,
                    friction: friction,
                    position: SIMD3<Float>(0, -30.0 + Float(x) * 2.0, 0.75),
                    velocity: SIMD3<Float>(10, 0, 0)
                )
            }

        case .staticFriction:
            addGround(builder, z: 0)
            let angle = radians(30.0)
            let rampOrientation = simd_quatf(angle: angle, axis: SIMD3<Float>(0, 1, 0))
            let ramp = builder.rigid(
                size: SIMD3<Float>(40, 24, 1),
                density: 0.0,
                friction: 1.0,
                position: SIMD3<Float>(0, 0, 3),
                orientation: rampOrientation,
                renderColor: SIMD4<Float>(0.56, 0.56, 0.50, 1.0)
            )
            let rampBody = builder.bodies[ramp]
            let rampTangent = simd_normalize(rampBody.orientation.act(SIMD3<Float>(1, 0, 0)))
            let rampNormal = simd_normalize(rampBody.orientation.act(SIMD3<Float>(0, 0, 1)))

            for i in 0...10 {
                let friction = Float(i) / 10.0 * 0.25 + 0.25
                let y = -10.0 + Float(i) * 2.0
                let position = rampBody.position + rampTangent * -12.0 + SIMD3<Float>(0, y, 0) + rampNormal * 1.05
                _ = builder.rigid(size: SIMD3<Float>(1, 1, 1), density: 1.0, friction: friction, position: position)
            }

        case .tower: // Cylindrical tower structure (staggered)
            let numRows = 15
            let radius: Float = 12.0
            let hSpace: Float = 0.5
            let vSpace: Float = 0.5
            addGround(builder, z: -0.5)
            
            // Constant number of blocks for the cylindrical wall
            let circumference = 2.0 * .pi * radius
            let numBlocks = Int(floor(circumference / hSpace))
            let angleStep = (2.0 * .pi) / Float(numBlocks)
            let colors: [SIMD4<Float>] = [
                SIMD4<Float>(1.0, 0.0, 0.0, 1.0),
                SIMD4<Float>(0.0, 1.0, 0.0, 1.0),
                SIMD4<Float>(0.0, 0.0, 1.0, 1.0),
                SIMD4<Float>(1.0, 1.0, 0.0, 1.0),
                SIMD4<Float>(1.0, 0.0, 1.0, 1.0),
                SIMD4<Float>(0.0, 1.0, 1.0, 1.0),
            ]
            for row in 0..<numRows {
                let z = Float(row) * vSpace + 0.26
                
                // Every other row is shifted by half a block for masonry-style staggering
                let angleOffset = (row % 2 == 0) ? 0.0 : angleStep * 0.5
                
                for i in 0..<numBlocks {
                    let angle = Float(i) * angleStep + angleOffset
                    let x = radius * cos(angle)
                    let y = radius * sin(angle)
                    let orientation = simd_quatf(angle: angle, axis: SIMD3<Float>(0, 0, 1))
                    
                    _ = builder.rigid(
                        size: SIMD3<Float>(1, 0.5, 0.5),
                        density: 1.0,
                        friction: 0.8,
                        position: SIMD3<Float>(x, y, z),
                        orientation: orientation,
                        renderColor: colors[(row + i) % colors.count],
                        colorGroup: (row + i) % 2,
                    )
                }
            }

        case .rope:
            addGround(builder, z: -20)
            var previous: Int?
            for i in 0..<20 {
                let current = builder.rigid(
                    size: SIMD3<Float>(1, 0.5, 0.5),
                    density: i == 0 ? 0.0 : 1.0,
                    friction: 0.5,
                    position: SIMD3<Float>(Float(i), 0.0, 10.0)
                )
                if let previous {
                    builder.joint(previous, current, SIMD3<Float>(0.5, 0, 0), SIMD3<Float>(-0.5, 0, 0))
                }
                previous = current
            }

        case .heavyRope:
            let count = 20
            let heavySize: Float = 5.0
            addGround(builder, z: -20)
            var previous: Int?
            for i in 0..<count {
                let isHeavyEnd = i == count - 1
                let current = builder.rigid(
                    size: isHeavyEnd ? SIMD3<Float>(heavySize, heavySize, heavySize) : SIMD3<Float>(1, 0.5, 0.5),
                    density: i == 0 ? 0.0 : 1.0,
                    friction: 0.5,
                    position: SIMD3<Float>(Float(i) + (isHeavyEnd ? heavySize / 2.0 : 0.0), 0.0, 10.0)
                )
                if let previous {
                    builder.joint(
                        previous,
                        current,
                        SIMD3<Float>(0.5, 0, 0),
                        isHeavyEnd ? SIMD3<Float>(-heavySize / 2.0, 0, 0) : SIMD3<Float>(-0.5, 0, 0)
                    )
                }
                previous = current
            }

        case .spring:
            addGround(builder, z: 0)
            let anchor = builder.rigid(size: SIMD3<Float>(1, 1, 1), density: 0.0, friction: 0.5, position: SIMD3<Float>(0, 0, 14.0), renderColor: SIMD4<Float>(0.95, 0.64, 0.28, 1.0))
            let block = builder.rigid(size: SIMD3<Float>(2, 2, 2), density: 1.0, friction: 0.5, position: SIMD3<Float>(0, 0, 8.0))
            builder.spring(anchor, block, .zero, .zero, stiffness: 100.0, rest: 4.0)

        case .springsRatio:
            let count = 8
            addGround(builder, z: -10)
            var previous: Int?
            for i in 0..<count {
                let x = (Float(i) - Float(count - 1) * 0.5) * 3.0
                let current = builder.rigid(
                    size: SIMD3<Float>(1, 0.75, 0.75),
                    density: i == 0 || i == count - 1 ? 0.0 : 1.0,
                    friction: 0.5,
                    position: SIMD3<Float>(x, 0.0, 12.0)
                )
                if let previous {
                    builder.spring(previous, current, SIMD3<Float>(0.5, 0, 0), SIMD3<Float>(-0.5, 0, 0), stiffness: i % 2 == 0 ? 10.0 : 10_000.0, rest: 3.0)
                }
                previous = current
            }

        case .stack:
            addGround(builder, z: 0)
            for i in 0..<10 {
                _ = builder.rigid(size: SIMD3<Float>(1, 1, 1), density: 1.0, friction: 0.5, position: SIMD3<Float>(0, 0, Float(i) * 1.5 + 1.0))
            }

        case .stackRatio:
            let groundThickness: Float = 1.0
            _ = builder.rigid(size: SIMD3<Float>(100, 100, groundThickness), density: 0.0, friction: 0.5, position: .zero)
            var topZ = groundThickness * 0.5
            var size: Float = 1.0
            for _ in 0..<4 {
                let half = size * 0.5
                let centerZ = topZ + half
                _ = builder.rigid(size: SIMD3<Float>(size, size, size), density: 1.0, friction: 0.5, position: SIMD3<Float>(0, 0, centerZ))
                topZ = centerZ + half
                size *= 2.0
            }

        case .softBody:
            addSoftBody(to: builder)

        case .bridge:
            addBridge(to: builder)

        case .breakable:
            addBreakable(to: builder)

        case .pyramid:
            addPyramid(builder)

        case .ring:
            addRing(builder)

        case .chainMail:
            addChainMail(builder)
        }

        return AVBDScene(
            id: id,
            name: id.displayName,
            bodies: builder.bodies,
            constraints: builder.constraints,
            defaults: defaults(for: id)
        )
    }

    private static func defaults(for id: AVBDSceneID) -> AVBDSceneDefaults {
        switch id {
        case .empty,
             .ground,
             .dynamicFriction,
             .staticFriction,
             .tower,
             .pyramid,
             .ring,
             .rope,
             .heavyRope,
             .spring,
             .springsRatio,
             .stack,
             .stackRatio,
             .softBody,
             .bridge,
             .breakable:
            return AVBDSceneDefaults()
        case .chainMail:
            return AVBDSceneDefaults(
                simulationStepDeltaTime: 1.0 / 120.0,
                simulationStepsPerFrame: 4
            )
        }
    }

    private static func addGround(_ builder: AVBDSceneBuilder, z: Float, colorGroup: Int? = nil) {
        _ = builder.rigid(size: SIMD3<Float>(100, 100, 1), density: 0.0, friction: 0.5, position: SIMD3<Float>(0, 0, z),colorGroup: colorGroup)
    }

    private static func addSoftBody(to builder: AVBDSceneBuilder) {
        addGround(builder, z: 0)

        let stiffnessLin: Float = 1000.0
        let stiffnessAng: Float = 250.0
        let width = 4
        let depth = 4
        let height = 4
        let count = 3
        let size: Float = 0.8
        let half = size * 0.5
        let baseZ: Float = 8.0
        let stackGap: Float = 2.0

        for i in 0..<count {
            var grid = Array(repeating: Array(repeating: Array(repeating: 0, count: height), count: depth), count: width)
            let stackZ = Float(i) * (Float(height) * size + stackGap)

            for x in 0..<width {
                for y in 0..<depth {
                    for z in 0..<height {
                        let px = (Float(x) - Float(width - 1) * 0.5) * size
                        let py = (Float(y) - Float(depth - 1) * 0.5) * size
                        let pz = baseZ + stackZ + Float(z) * size
                        grid[x][y][z] = builder.rigid(size: SIMD3<Float>(size, size, size), density: 1.0, friction: 0.5, position: SIMD3<Float>(px, py, pz))
                    }
                }
            }

            for x in 1..<width {
                for y in 0..<depth {
                    for z in 0..<height {
                        builder.joint(grid[x - 1][y][z], grid[x][y][z], SIMD3<Float>(half, 0, 0), SIMD3<Float>(-half, 0, 0), stiffnessLin: stiffnessLin, stiffnessAng: stiffnessAng)
                    }
                }
            }

            for x in 0..<width {
                for y in 1..<depth {
                    for z in 0..<height {
                        builder.joint(grid[x][y - 1][z], grid[x][y][z], SIMD3<Float>(0, half, 0), SIMD3<Float>(0, -half, 0), stiffnessLin: stiffnessLin, stiffnessAng: stiffnessAng)
                    }
                }
            }

            for x in 0..<width {
                for y in 0..<depth {
                    for z in 1..<height {
                        builder.joint(grid[x][y][z - 1], grid[x][y][z], SIMD3<Float>(0, 0, half), SIMD3<Float>(0, 0, -half), stiffnessLin: stiffnessLin, stiffnessAng: stiffnessAng)
                    }
                }
            }
        }
    }

    private static func addBridge(to builder: AVBDSceneBuilder) {
        let count = 40
        let plankLength: Float = 1.0
        let plankWidth: Float = 4.0
        let plankHeight: Float = 0.5
        let halfLength = plankLength * 0.5
        let halfWidth = plankWidth * 0.5
        addGround(builder, z: 0)

        var previous: Int?
        for i in 0..<count {
            let current = builder.rigid(
                size: SIMD3<Float>(plankLength, plankWidth, plankHeight),
                density: i == 0 || i == count - 1 ? 0.0 : 1.0,
                friction: 0.5,
                position: SIMD3<Float>(Float(i) - Float(count) / 2.0, 0.0, 10.0)
            )
            if let previous {
                builder.joint(previous, current, SIMD3<Float>(halfLength, halfWidth, 0), SIMD3<Float>(-halfLength, halfWidth, 0))
                builder.joint(previous, current, SIMD3<Float>(halfLength, -halfWidth, 0), SIMD3<Float>(-halfLength, -halfWidth, 0))
            }
            previous = current
        }

        for x in 0..<(count / 4) {
            for y in 0..<(count / 8) {
                _ = builder.rigid(size: SIMD3<Float>(1, 1, 1), density: 1.0, friction: 0.5, position: SIMD3<Float>(Float(x) - Float(count) / 8.0, 0.0, Float(y) + 12.0))
            }
        }
    }

    private static func addBreakable(to builder: AVBDSceneBuilder) {
        let count = 10
        let stackCount = 5
        let breakForce: Float = 90.0
        addGround(builder, z: 0)

        var previous: Int?
        for i in 0...count {
            let current = builder.rigid(size: SIMD3<Float>(1, 1, 0.5), density: 1.0, friction: 0.5, position: SIMD3<Float>(Float(i) - Float(count) / 2.0, 0.0, 6.0))
            if let previous {
                builder.joint(previous, current, SIMD3<Float>(0.5, 0, 0), SIMD3<Float>(-0.5, 0, 0), stiffnessLin: .infinity, stiffnessAng: .infinity, fracture: breakForce)
            }
            previous = current
        }

        _ = builder.rigid(size: SIMD3<Float>(1, 1, 5), density: 0.0, friction: 0.5, position: SIMD3<Float>(-Float(count) / 2.0, 0, 2.5))
        _ = builder.rigid(size: SIMD3<Float>(1, 1, 5), density: 0.0, friction: 0.5, position: SIMD3<Float>(Float(count) / 2.0, 0, 2.5))

        for i in 0..<stackCount {
            _ = builder.rigid(size: SIMD3<Float>(2, 1, 1), density: 1.0, friction: 0.5, position: SIMD3<Float>(0, 0, Float(i) * 2.0 + 8.0))
        }
    }

    private static func addPyramid(_ builder: AVBDSceneBuilder) {
        let size = 16
        addGround(builder, z: -0.5)
        for y in 0..<size {
            for x in 0..<(size - y) {
                let px = Float(x) * 1.01 + Float(y) * 0.5 - Float(size) / 2.0
                let py: Float = 0.0
                let pz = Float(y) * 0.85 + 0.5
                _ = builder.rigid(
                    size: SIMD3<Float>(1, 0.5, 0.5),
                    density: 1.0,
                    friction: 0.5,
                    position: SIMD3<Float>(px, py, pz)
                )
            }
        }
    }
    private static func addRing(_ builder: AVBDSceneBuilder) {
//        addGround(builder, z: -50)

        let anchorReferencePos = SIMD3<Float>(0, 0, 12)

        let segmentCount = 12
        let majorRadius: Float = 3.0
        let tubeDiameter: Float = 0.8
        let tubeRadius = tubeDiameter * 0.5
        let torusSize = SIMD3<Float>(
            repeating: 2.0 * (majorRadius + tubeRadius)
        )
        let resolvedTorusSize = SIMD3<Float>(torusSize.x, torusSize.y, tubeDiameter)

        let alignRot = simd_quatf(angle: .pi / Float(segmentCount), axis: SIMD3<Float>(0, 0, 1))
        let baseRingOrientation = simd_quatf(angle: -.pi / 2.0, axis: SIMD3<Float>(0, 1, 0)) * alignRot

        let topSegmentIndex = segmentCount - 1
        let localAttachPoint = torusLoopLocalPoint(
            majorRadius: majorRadius,
            segmentCount: segmentCount,
            segmentIndex: topSegmentIndex
        )

        let density: Float = 10.0
        let ringCount = 20
        let ringColors: [SIMD4<Float>] = [
            SIMD4<Float>(0.4, 0.7, 1.0, 1.0),
            SIMD4<Float>(0.93, 0.42, 0.39, 1.0)
        ]

        let attachOffset = baseRingOrientation.act(localAttachPoint)
        let topRingCenter = anchorReferencePos - attachOffset

        var rings: [Int] = []
        rings.reserveCapacity(ringCount)

        for index in 0..<ringCount {
            let rotationStep = simd_quatf(angle: Float(index) * (.pi / 2.0), axis: SIMD3<Float>(0, 0, 1))
            let orientation = rotationStep * baseRingOrientation
            let center = topRingCenter + SIMD3<Float>(0, 0, -majorRadius * Float(index) * 1.1)
            let ring = builder.rigid(
                renderShape: .torus,
                size: resolvedTorusSize,
                density: density,
                friction: 0.5,
                position: center,
                orientation: orientation,
                renderColor: ringColors[index % ringColors.count],
                colorGroup: index % 2 + 1,
            )
            rings.append(ring)
        }

        guard let topRing = rings.first else {
            return
        }

        applyTopTorusSupport(
            builder,
            topBody: topRing,
            topBodyCenter: topRingCenter,
            topBodyOrientation: baseRingOrientation,
            majorRadius: majorRadius,
            primarySegmentIndex: topSegmentIndex,
            segmentCount: segmentCount,
            mode: ringTopSupportMode
        )
    }

    private static func addChainMail(_ builder: AVBDSceneBuilder) {
        addGround(builder, z: -100.0,colorGroup: 0)

        let rows = 6
        let cols = 8
        let majorRadius: Float = 1.8
        let tubeDiameter: Float = 0.42
        let tubeRadius = tubeDiameter * 0.5
        let torusSize = SIMD3<Float>(
            2.0 * (majorRadius + tubeRadius),
            2.0 * (majorRadius + tubeRadius),
            tubeDiameter
        )
        let baseHeight: Float = 15.0
        let density: Float = 1.0
        let friction: Float = 0.7
        let baseSpacingX = majorRadius * 2.55
        let baseSpacingY = majorRadius * 2.55
        let connectorLift: Float = tubeRadius * 0.45
        let jointStiffnessLin: Float = 1_200.0
        let jointStiffnessAng: Float = 60.0

        let flatOrientation = simd_quatf(angle: 0.0, axis: SIMD3<Float>(0, 0, 1))
        let horizontalConnectorOrientation = simd_quatf(angle: -.pi / 2.0, axis: SIMD3<Float>(0, 1, 0))
        let verticalConnectorOrientation = simd_quatf(angle: .pi / 2.0, axis: SIMD3<Float>(1, 0, 0))

        let width = Float(cols - 1) * baseSpacingX
        let depth = Float(rows - 1) * baseSpacingY
        let origin = SIMD3<Float>(-width * 0.5, -depth * 0.5, baseHeight)

        var baseRings = Array(repeating: Array(repeating: 0, count: cols), count: rows)
        var baseCenters = Array(repeating: Array(repeating: SIMD3<Float>.zero, count: cols), count: rows)

        for row in 0..<rows {
            for col in 0..<cols {
                let center = origin + SIMD3<Float>(Float(col) * baseSpacingX, Float(row) * baseSpacingY, 0.0)
                let ring = builder.rigid(
                    renderShape: .torus,
                    size: torusSize,
                    density: density,
                    friction: friction,
                    position: center,
                    orientation: flatOrientation,
                    renderColor: torusColor(row: row, col: col),
                    colorGroup: (row + col) % 2,
                )
                baseRings[row][col] = ring
                baseCenters[row][col] = center
            }
        }

        for row in 0..<(rows-1) {
            for col in 0..<cols {
                let leftCenter = baseCenters[row][col]
                let connectorCenter = leftCenter + SIMD3<Float>(0, 0, connectorLift)
                + SIMD3<Float>( 0,torusSize.y*0.5,0)
                let connector = builder.rigid(
                    renderShape: .torus,
                    size: torusSize,
                    density: density,
                    friction: friction,
                    position: connectorCenter,
                    orientation: horizontalConnectorOrientation,
                    renderColor: SIMD4<Float>(0.86, 0.88, 0.92, 1.0),
                    colorGroup: (row + col) % 2 + 2,
                )
            }
        }

        for row in 0..<rows {
            for col in 0..<(cols - 1) {
                let topCenter = baseCenters[row][col]
                let connectorCenter = topCenter + SIMD3<Float>(0, 0, connectorLift)
                + SIMD3<Float>(torusSize.x * 0.5, 0, 0)
                let connector = builder.rigid(
                    renderShape: .torus,
                    size: torusSize,
                    density: density,
                    friction: friction,
                    position: connectorCenter,
                    orientation: verticalConnectorOrientation,
                    renderColor: SIMD4<Float>(0.78, 0.82, 0.87, 1.0),
                    colorGroup: (row + col) % 2 + 2,
                )
            }
        }

        let cornerSupportOutwardDirections: [(row: Int, col: Int, outward: SIMD3<Float>)] = [
            (0, 0, SIMD3<Float>(-1, -1, 0)),
            (0, cols - 1, SIMD3<Float>(1, -1, 0)),
            (rows - 1, 0, SIMD3<Float>(-1, 1, 0)),
            (rows - 1, cols - 1, SIMD3<Float>(1, 1, 0))
        ]

        for support in cornerSupportOutwardDirections {
            attachChainMailCornerSupport(
                builder,
                body: baseRings[support.row][support.col],
                center: baseCenters[support.row][support.col],
                orientation: flatOrientation,
                majorRadius: majorRadius,
                outwardDirection: support.outward,
                colorGroup: 2,
            )
        }
    }

    private static func radians(_ degrees: Float) -> Float {
        degrees / 180.0 * .pi
    }

    private static func makePolygonRing(
        in builder: AVBDSceneBuilder,
        center: SIMD3<Float>,
        orientation ringOrientation: simd_quatf = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 0, 1)),
        segmentCount: Int,
        segmentSize: SIMD3<Float>,
        density: Float,
        friction: Float,
        color: SIMD4<Float>,
        stiffnessLin: Float,
        stiffnessAng: Float,
        diameterSpringStiffness: Float = 0.0,
        ignoreSelfCollisions: Bool = false
    ) -> [Int] {
        let step = 2.0 * Float.pi / Float(segmentCount)
        let vertexRadius = segmentSize.x / (2.0 * sin(.pi / Float(segmentCount)))
        let edgeMidRadius = vertexRadius * cos(.pi / Float(segmentCount))
        let halfLength = segmentSize.x * 0.5

        var ring: [Int] = []
        ring.reserveCapacity(segmentCount)

        for i in 0..<segmentCount {
            let midpointAngle = (Float(i) + 0.5) * step
            let localPos = SIMD3<Float>(
                edgeMidRadius * cos(midpointAngle),
                edgeMidRadius * sin(midpointAngle),
                0
            )
            let position = center + ringOrientation.act(localPos)
            let localRot = simd_quatf(angle: midpointAngle + .pi * 0.5, axis: SIMD3<Float>(0, 0, 1))
            let orientation = ringOrientation * localRot
            let body = builder.rigid(
                size: segmentSize,
                density: density,
                friction: friction,
                position: position,
                orientation: orientation,
                renderColor: color
            )
            ring.append(body)
        }

        for i in 0..<segmentCount {
            let current = ring[i]
            let next = ring[(i + 1) % segmentCount]
            builder.joint(
                current,
                next,
                SIMD3<Float>(halfLength, 0, 0),
                SIMD3<Float>(-halfLength, 0, 0),
                stiffnessLin: stiffnessLin,
                stiffnessAng: stiffnessAng
            )
        }

        if diameterSpringStiffness > 0, segmentCount % 2 == 0 {
            let halfCount = segmentCount / 2
            for i in 0..<halfCount {
                builder.spring(
                    ring[i],
                    ring[i + halfCount],
                    .zero,
                    .zero,
                    stiffness: diameterSpringStiffness
                )
            }
        }

        if ignoreSelfCollisions {
            ignoreCollisionsBetweenAdjacentBodiesInLoop(ring, in: builder)
        }

        return ring
    }


    private static func makeSquaredRing(
        in builder: AVBDSceneBuilder,
        center: SIMD3<Float>,
        orientation ringOrientation: simd_quatf = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 0, 1)),
        segmentCount: Int,
        segmentSize: SIMD3<Float>,
        density: Float,
        friction: Float,
        color: SIMD4<Float>,
        stiffnessLin: Float,
        stiffnessAng: Float,
        diameterSpringStiffness: Float = 0.0,
        ignoreSelfCollisions: Bool = false
    ) -> [Int] {
        let segmentsPerSide = segmentCount / 4
        let dx = segmentSize.x
        let sideLength = Float(segmentsPerSide) * dx
        let halfSide = sideLength * 0.5
        let halfLength = dx * 0.5

        var ring: [Int] = []
        ring.reserveCapacity(segmentCount)

        // 4 sides: 0:+X(Right), 1:+Y(Top), 2:-X(Left), 3:-Y(Bottom)
        for s in 0..<4 {
            let sideAngle = Float(s) * (.pi * 0.5)
            let sideRot = simd_quatf(angle: sideAngle, axis: SIMD3<Float>(0, 0, 1))

            for i in 0..<segmentsPerSide {
                // local offset along the side - centered
                let offset = (Float(i) + 0.5 - Float(segmentsPerSide) * 0.5) * dx

                // Position in ring local space: for side 0, we start with side at +X facing +Y
                let pLocal = SIMD3<Float>(halfSide, offset, 0)
                let position = center + ringOrientation.act(sideRot.act(pLocal))

                // Orientation: sideRot rotated 90 degrees to face along the side
                let localRot = sideRot * simd_quatf(angle: .pi * 0.5, axis: SIMD3<Float>(0, 0, 1))
                let orientation = ringOrientation * localRot

                let body = builder.rigid(
                    size: segmentSize,
                    density: density,
                    friction: friction,
                    position: position,
                    orientation: orientation,
                    renderColor: color
                )
                ring.append(body)
            }
        }

        // Connect everything in a loop
        for i in 0..<segmentCount {
            let current = ring[i]
            let next = ring[(i + 1) % segmentCount]
            builder.joint(
                current,
                next,
                SIMD3<Float>(halfLength, 0, 0),
                SIMD3<Float>(-halfLength, 0, 0),
                stiffnessLin: stiffnessLin,
                stiffnessAng: stiffnessAng
            )
        }

        if diameterSpringStiffness > 0, segmentCount % 2 == 0 {
            let halfCount = segmentCount / 2
            for i in 0..<halfCount {
                builder.spring(
                    ring[i],
                    ring[i + halfCount],
                    .zero,
                    .zero,
                    stiffness: diameterSpringStiffness
                )
            }
        }

        if ignoreSelfCollisions {
            ignoreCollisionsBetweenAdjacentBodiesInLoop(ring, in: builder)
        }

        return ring
    }

    private static func ignoreCollisionsBetweenAdjacentBodiesInLoop(_ bodies: [Int], in builder: AVBDSceneBuilder) {
        guard bodies.count > 1 else { return }
        for i in 0..<bodies.count {
            builder.ignoreCollision(bodies[i], bodies[(i + 1) % bodies.count])
        }
    }

    private static func applyTopRingSupport(
        _ builder: AVBDSceneBuilder,
        topRing: [Int],
        primarySegmentIndex: Int,
        mode: RingTopSupportMode
    ) {
        let supportedSegments: [Int]
        switch mode {
        case .singlePoint:
            supportedSegments = [topRing[primarySegmentIndex]]
        case .twoPoint:
            supportedSegments = [
                topRing[primarySegmentIndex],
                topRing[(primarySegmentIndex + 1) % topRing.count]
            ]
        case .fullyFixed:
            supportedSegments = topRing
        }

        let anchorColor = SIMD4<Float>(0.95, 0.64, 0.28, 1.0)
        var anchors: [Int] = []
        anchors.reserveCapacity(supportedSegments.count)

        for segment in supportedSegments {
            let segmentBody = builder.bodies[segment]
            let anchor = builder.rigid(
                size: SIMD3<Float>(0.2, 0.2, 0.2),
                density: 0.0,
                friction: 0.5,
                position: segmentBody.position,
                orientation: segmentBody.orientation,
                renderColor: anchorColor
            )
            anchors.append(anchor)
            builder.joint(anchor, segment, .zero, .zero, stiffnessLin: .infinity, stiffnessAng: .infinity)
        }

        // Ignore collisions between support anchors and the top ring to prevent constraint fighting.
        for anchor in anchors {
            for segment in topRing {
                builder.ignoreCollision(anchor, segment)
            }
        }
    }

    private static func torusLoopLocalPoint(
        majorRadius: Float,
        segmentCount: Int,
        segmentIndex: Int
    ) -> SIMD3<Float> {
        let step = 2.0 * Float.pi / Float(segmentCount)
        let midpointAngle = (Float(segmentIndex) + 0.5) * step
        return SIMD3<Float>(
            majorRadius * cos(midpointAngle),
            majorRadius * sin(midpointAngle),
            0
        )
    }

    private static func applyTopTorusSupport(
        _ builder: AVBDSceneBuilder,
        topBody: Int,
        topBodyCenter: SIMD3<Float>,
        topBodyOrientation: simd_quatf,
        majorRadius: Float,
        primarySegmentIndex: Int,
        segmentCount: Int,
        mode: RingTopSupportMode
    ) {
        let anchorColor = SIMD4<Float>(0.95, 0.64, 0.28, 1.0)
        let stiffnessAng: Float = mode == .fullyFixed ? .infinity : 0.0

        let localAnchorPoints: [SIMD3<Float>]
        switch mode {
        case .singlePoint:
            localAnchorPoints = [
                torusLoopLocalPoint(majorRadius: majorRadius, segmentCount: segmentCount, segmentIndex: primarySegmentIndex)
            ]
        case .twoPoint:
            localAnchorPoints = [
                torusLoopLocalPoint(majorRadius: majorRadius, segmentCount: segmentCount, segmentIndex: primarySegmentIndex),
                torusLoopLocalPoint(majorRadius: majorRadius, segmentCount: segmentCount, segmentIndex: (primarySegmentIndex + 1) % segmentCount)
            ]
        case .fullyFixed:
            localAnchorPoints = [.zero]
        }

        var anchors: [Int] = []
        anchors.reserveCapacity(localAnchorPoints.count)

        for localAnchorPoint in localAnchorPoints {
            let anchorWorldPosition = topBodyCenter + topBodyOrientation.act(localAnchorPoint)
            let anchor = builder.rigid(
                renderShape: .torus,
                size: SIMD3<Float>(0.0, 0.0, 0.0),
                density: 0.0,
                friction: 0.5,
                position: anchorWorldPosition,
                orientation: topBodyOrientation,
                renderColor: anchorColor,
                colorGroup: 0
            )
            anchors.append(anchor)
            builder.joint(
                anchor,
                topBody,
                .zero,
                localAnchorPoint,
                stiffnessLin: .infinity,
                stiffnessAng: stiffnessAng
            )
        }

        for anchor in anchors {
            builder.ignoreCollision(anchor, topBody)
        }
    }

    private static func torusLocalPoint(
        majorRadius: Float,
        orientation: simd_quatf,
        towardWorldDirection direction: SIMD3<Float>
    ) -> SIMD3<Float> {
        let normalizedDirection = simd_length_squared(direction) > 0.0
            ? simd_normalize(direction)
            : SIMD3<Float>(1, 0, 0)
        let localDirection = orientation.inverse.act(normalizedDirection)
        let planar = SIMD2<Float>(localDirection.x, localDirection.y)
        guard simd_length_squared(planar) > 1e-6 else {
            return SIMD3<Float>(majorRadius, 0, 0)
        }

        let localPlanarDirection = simd_normalize(planar)
        return SIMD3<Float>(localPlanarDirection.x * majorRadius, localPlanarDirection.y * majorRadius, 0)
    }

    private static func connectTorusBodies(
        _ builder: AVBDSceneBuilder,
        bodyA: Int,
        centerA: SIMD3<Float>,
        orientationA: simd_quatf,
        bodyB: Int,
        centerB: SIMD3<Float>,
        orientationB: simd_quatf,
        majorRadius: Float,
        stiffnessLin: Float,
        stiffnessAng: Float,
        verticalBias: Float = 0.0
    ) {
        let lateralVector = centerB - centerA
        guard simd_length_squared(lateralVector) > 1e-6 else {
            return
        }

        let lateralDirection = simd_normalize(lateralVector)
        let verticalDirection = SIMD3<Float>(0, 0, 1)

        let directionA = simd_normalize(lateralDirection + verticalDirection * verticalBias)
        let directionB = simd_normalize(-lateralDirection + verticalDirection * verticalBias)

        let anchorA = torusLocalPoint(
            majorRadius: majorRadius,
            orientation: orientationA,
            towardWorldDirection: directionA
        )
        let anchorB = torusLocalPoint(
            majorRadius: majorRadius,
            orientation: orientationB,
            towardWorldDirection: directionB
        )

        builder.joint(
            bodyA,
            bodyB,
            anchorA,
            anchorB,
            stiffnessLin: stiffnessLin,
            stiffnessAng: stiffnessAng
        )
        builder.ignoreCollision(bodyA, bodyB)
    }

    private static func attachChainMailCornerSupport(
        _ builder: AVBDSceneBuilder,
        body: Int,
        center: SIMD3<Float>,
        orientation: simd_quatf,
        majorRadius: Float,
        outwardDirection: SIMD3<Float>,
        colorGroup: Int,
    ) {
        let anchorColor = SIMD4<Float>(0.95, 0.64, 0.28, 1.0)
        let outward = simd_normalize(outwardDirection)
        let supportDirections = [
            simd_normalize(SIMD3<Float>(0, 0, 1) + outward * 0.35),
            simd_normalize(SIMD3<Float>(0, 0, 1) - outward * 0.35)
        ]

        for supportDirection in supportDirections {
            let localAnchorPoint = torusLocalPoint(
                majorRadius: majorRadius,
                orientation: orientation,
                towardWorldDirection: supportDirection
            )
            let anchorWorldPosition = center + orientation.act(localAnchorPoint)
            let anchor = builder.rigid(
                renderShape: .torus,
                size: SIMD3<Float>(repeating: 0.0),
                density: 0.0,
                friction: 0.5,
                position: anchorWorldPosition,
                orientation: orientation,
                renderColor: anchorColor,
                colorGroup: colorGroup
            )
            builder.joint(anchor, body, .zero, localAnchorPoint, stiffnessLin: .infinity, stiffnessAng: 0.0)
            builder.ignoreCollision(anchor, body)
        }
    }

    private static func ignoreCollisionsWithinBodyGroup(_ bodies: [Int], in builder: AVBDSceneBuilder) {
        guard bodies.count > 1 else { return }
        for i in 0..<(bodies.count - 1) {
            for j in (i + 1)..<bodies.count {
                builder.ignoreCollision(bodies[i], bodies[j])
            }
        }
    }

    private static func connectHorizontalRings(
        _ builder: AVBDSceneBuilder,
        left: [Int],
        right: [Int],
        halfLength: Float,
        stiffnessLin: Float,
        stiffnessAng: Float
    ) {
        builder.joint(
            left[0],
            right[2],
            SIMD3<Float>(-halfLength, 0, 0),
            SIMD3<Float>(halfLength, 0, 0),
            stiffnessLin: stiffnessLin,
            stiffnessAng: stiffnessAng
        )
        builder.joint(
            left[5],
            right[3],
            SIMD3<Float>(halfLength, 0, 0),
            SIMD3<Float>(-halfLength, 0, 0),
            stiffnessLin: stiffnessLin,
            stiffnessAng: stiffnessAng
        )
    }

    private static func connectVerticalRings(
        _ builder: AVBDSceneBuilder,
        top: [Int],
        bottom: [Int],
        halfLength: Float,
        stiffnessLin: Float,
        stiffnessAng: Float
    ) {
        builder.joint(
            top[1],
            bottom[4],
            SIMD3<Float>(halfLength, 0, 0),
            SIMD3<Float>(-halfLength, 0, 0),
            stiffnessLin: stiffnessLin,
            stiffnessAng: stiffnessAng
        )
        builder.joint(
            top[2],
            bottom[5],
            SIMD3<Float>(-halfLength, 0, 0),
            SIMD3<Float>(halfLength, 0, 0),
            stiffnessLin: stiffnessLin,
            stiffnessAng: stiffnessAng
        )
    }

    private static func torusColor(row: Int, col: Int) -> SIMD4<Float> {
        let palette: [SIMD4<Float>] = [
            SIMD4<Float>(0.93, 0.42, 0.39, 1.0),
            SIMD4<Float>(0.97, 0.71, 0.30, 1.0),
            SIMD4<Float>(0.48, 0.79, 0.51, 1.0),
            SIMD4<Float>(0.35, 0.73, 0.92, 1.0),
            SIMD4<Float>(0.63, 0.52, 0.90, 1.0)
        ]
        return palette[(row + col) % palette.count]
    }
}

nonisolated extension AVBDSceneID {
    var displayName: String {
        switch self {
        case .empty: return "Empty"
        case .ground: return "Ground"
        case .dynamicFriction: return "Dynamic Friction"
        case .staticFriction: return "Static Friction"
        case .tower: return "Tower"
        case .pyramid: return "Pyramid"
        case .ring: return "Ring"
        case .chainMail: return "Chain Mail"
        case .rope: return "Rope"
        case .heavyRope: return "Heavy Rope"
        case .spring: return "Spring"
        case .springsRatio: return "Spring Ratio"
        case .stack: return "Stack"
        case .stackRatio: return "Stack Ratio"
        case .softBody: return "Soft Body"
        case .bridge: return "Bridge"
        case .breakable: return "Breakable"
        }
    }
}
