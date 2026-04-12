//
//  AVBDSolver.swift
//  MetalAVBD
//

import Foundation
import simd

nonisolated private let penaltyMin: Float = 1.0
nonisolated private let penaltyMax: Float = 10_000_000_000.0
nonisolated private let collisionMargin: Float = 0.01
nonisolated private let stickThreshold: Float = 0.00001

nonisolated private let maxContacts = Int(AVBD_MAX_CONTACTS_PER_PAIR_BURST)
nonisolated private let maxPolyVerts = Int(AVBD_COLLISION_MAX_POLY_VERTS)
nonisolated private let satAxisEpsilon: Float = Float(AVBD_COLLISION_SAT_AXIS_EPSILON)
nonisolated private let planeEpsilon: Float = Float(AVBD_COLLISION_PLANE_EPSILON)
nonisolated private let contactMergeDistSq: Float = Float(AVBD_COLLISION_CONTACT_MERGE_DIST_SQ)

nonisolated struct AVBDMat3 {
    var r0: SIMD3<Float>
    var r1: SIMD3<Float>
    var r2: SIMD3<Float>

    init(_ r0: SIMD3<Float>, _ r1: SIMD3<Float>, _ r2: SIMD3<Float>) {
        self.r0 = r0
        self.r1 = r1
        self.r2 = r2
    }

    init(
        _ m00: Float, _ m01: Float, _ m02: Float,
        _ m10: Float, _ m11: Float, _ m12: Float,
        _ m20: Float, _ m21: Float, _ m22: Float
    ) {
        self.r0 = SIMD3<Float>(m00, m01, m02)
        self.r1 = SIMD3<Float>(m10, m11, m12)
        self.r2 = SIMD3<Float>(m20, m21, m22)
    }

    subscript(row: Int) -> SIMD3<Float> {
        get {
            switch row {
            case 0: return r0
            case 1: return r1
            default: return r2
            }
        }
        set {
            switch row {
            case 0: r0 = newValue
            case 1: r1 = newValue
            default: r2 = newValue
            }
        }
    }

    func col(_ column: Int) -> SIMD3<Float> {
        SIMD3<Float>(r0[column], r1[column], r2[column])
    }

    static let zero = AVBDMat3(.zero, .zero, .zero)
    static let identity = AVBDMat3(1, 0, 0, 0, 1, 0, 0, 0, 1)
}

nonisolated private func + (lhs: AVBDMat3, rhs: AVBDMat3) -> AVBDMat3 {
    AVBDMat3(lhs.r0 + rhs.r0, lhs.r1 + rhs.r1, lhs.r2 + rhs.r2)
}

nonisolated private func += (lhs: inout AVBDMat3, rhs: AVBDMat3) {
    lhs.r0 += rhs.r0
    lhs.r1 += rhs.r1
    lhs.r2 += rhs.r2
}

nonisolated private func - (lhs: AVBDMat3, rhs: AVBDMat3) -> AVBDMat3 {
    AVBDMat3(lhs.r0 - rhs.r0, lhs.r1 - rhs.r1, lhs.r2 - rhs.r2)
}

nonisolated private prefix func - (value: AVBDMat3) -> AVBDMat3 {
    AVBDMat3(-value.r0, -value.r1, -value.r2)
}

nonisolated private func * (lhs: AVBDMat3, rhs: Float) -> AVBDMat3 {
    AVBDMat3(lhs.r0 * rhs, lhs.r1 * rhs, lhs.r2 * rhs)
}

nonisolated private func * (lhs: Float, rhs: AVBDMat3) -> AVBDMat3 {
    rhs * lhs
}

nonisolated private func / (lhs: AVBDMat3, rhs: Float) -> AVBDMat3 {
    AVBDMat3(lhs.r0 / rhs, lhs.r1 / rhs, lhs.r2 / rhs)
}

nonisolated private func * (lhs: AVBDMat3, rhs: SIMD3<Float>) -> SIMD3<Float> {
    SIMD3<Float>(dot(lhs.r0, rhs), dot(lhs.r1, rhs), dot(lhs.r2, rhs))
}

nonisolated private func * (lhs: AVBDMat3, rhs: AVBDMat3) -> AVBDMat3 {
    AVBDMat3(
        SIMD3<Float>(dot(lhs.r0, rhs.col(0)), dot(lhs.r0, rhs.col(1)), dot(lhs.r0, rhs.col(2))),
        SIMD3<Float>(dot(lhs.r1, rhs.col(0)), dot(lhs.r1, rhs.col(1)), dot(lhs.r1, rhs.col(2))),
        SIMD3<Float>(dot(lhs.r2, rhs.col(0)), dot(lhs.r2, rhs.col(1)), dot(lhs.r2, rhs.col(2)))
    )
}

nonisolated private func diagonal(_ m00: Float, _ m11: Float, _ m22: Float) -> AVBDMat3 {
    AVBDMat3(m00, 0, 0, 0, m11, 0, 0, 0, m22)
}

nonisolated private func transpose(_ m: AVBDMat3) -> AVBDMat3 {
    AVBDMat3(m.col(0), m.col(1), m.col(2))
}

nonisolated private func skew(_ r: SIMD3<Float>) -> AVBDMat3 {
    AVBDMat3(
        0, -r.z, r.y,
        r.z, 0, -r.x,
        -r.y, r.x, 0
    )
}

nonisolated private func outer(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> AVBDMat3 {
    AVBDMat3(b * a.x, b * a.y, b * a.z)
}

nonisolated private func diagonalize(_ m: AVBDMat3) -> AVBDMat3 {
    diagonal(simd_length(m.col(0)), simd_length(m.col(1)), simd_length(m.col(2)))
}

nonisolated private func absVec(_ v: SIMD3<Float>) -> SIMD3<Float> {
    SIMD3<Float>(abs(v.x), abs(v.y), abs(v.z))
}

nonisolated private func clampVec(_ v: SIMD3<Float>, _ lower: Float, _ upper: Float) -> SIMD3<Float> {
    SIMD3<Float>(
        min(max(v.x, lower), upper),
        min(max(v.y, lower), upper),
        min(max(v.z, lower), upper)
    )
}

nonisolated private func minVec(_ v: SIMD3<Float>, _ scalar: Float) -> SIMD3<Float> {
    SIMD3<Float>(min(v.x, scalar), min(v.y, scalar), min(v.z, scalar))
}

nonisolated private func safeNormalize(_ v: SIMD3<Float>) -> SIMD3<Float> {
    let len = simd_length(v)
    return len > 1.0e-8 ? v / len : SIMD3<Float>(1, 0, 0)
}

nonisolated private func quatDelta(_ a: simd_quatf, _ b: simd_quatf) -> SIMD3<Float> {
    (a * b.inverse).imag * 2.0
}

nonisolated private func quatAddAngular(_ q: simd_quatf, _ v: SIMD3<Float>) -> simd_quatf {
    let dq = simd_quatf(ix: v.x, iy: v.y, iz: v.z, r: 0)
    return simd_quatf(vector: q.vector + (dq * q).vector * 0.5).normalized
}

nonisolated private func transform(_ position: SIMD3<Float>, _ orientation: simd_quatf, _ local: SIMD3<Float>) -> SIMD3<Float> {
    orientation.act(local) + position
}

nonisolated private func sign(_ x: Float) -> Float {
    x < 0 ? -1.0 : (x > 0 ? 1.0 : 0.0)
}

nonisolated private func solveSystem(
    aLin: AVBDMat3,
    aAng: AVBDMat3,
    aCross: AVBDMat3,
    bLin: SIMD3<Float>,
    bAng: SIMD3<Float>
) -> (SIMD3<Float>, SIMD3<Float>) {
    let A11 = aLin[0][0]
    let A21 = aLin[1][0], A22 = aLin[1][1]
    let A31 = aLin[2][0], A32 = aLin[2][1], A33 = aLin[2][2]
    let A41 = aCross[0][0], A42 = aCross[0][1], A43 = aCross[0][2], A44 = aAng[0][0]
    let A51 = aCross[1][0], A52 = aCross[1][1], A53 = aCross[1][2], A54 = aAng[1][0], A55 = aAng[1][1]
    let A61 = aCross[2][0], A62 = aCross[2][1], A63 = aCross[2][2], A64 = aAng[2][0], A65 = aAng[2][1], A66 = aAng[2][2]

    let L21 = A21 / A11
    let L31 = A31 / A11
    let L41 = A41 / A11
    let L51 = A51 / A11
    let L61 = A61 / A11

    let D1 = A11
    let D2 = A22 - L21 * L21 * D1

    let L32 = (A32 - L21 * L31 * D1) / D2
    let L42 = (A42 - L21 * L41 * D1) / D2
    let L52 = (A52 - L21 * L51 * D1) / D2
    let L62 = (A62 - L21 * L61 * D1) / D2

    let D3 = A33 - (L31 * L31 * D1 + L32 * L32 * D2)

    let L43 = (A43 - L31 * L41 * D1 - L32 * L42 * D2) / D3
    let L53 = (A53 - L31 * L51 * D1 - L32 * L52 * D2) / D3
    let L63 = (A63 - L31 * L61 * D1 - L32 * L62 * D2) / D3

    let D4 = A44 - (L41 * L41 * D1 + L42 * L42 * D2 + L43 * L43 * D3)

    let L54 = (A54 - L41 * L51 * D1 - L42 * L52 * D2 - L43 * L53 * D3) / D4
    let L64 = (A64 - L41 * L61 * D1 - L42 * L62 * D2 - L43 * L63 * D3) / D4

    let D5 = A55 - (L51 * L51 * D1 + L52 * L52 * D2 + L53 * L53 * D3 + L54 * L54 * D4)
    let L65 = (A65 - L51 * L61 * D1 - L52 * L62 * D2 - L53 * L63 * D3 - L54 * L64 * D4) / D5

    let D6 = A66 - (L61 * L61 * D1 + L62 * L62 * D2 + L63 * L63 * D3 + L64 * L64 * D4 + L65 * L65 * D5)

    let y1 = bLin[0]
    let y2 = bLin[1] - L21 * y1
    let y3 = bLin[2] - L31 * y1 - L32 * y2
    let y4 = bAng[0] - L41 * y1 - L42 * y2 - L43 * y3
    let y5 = bAng[1] - L51 * y1 - L52 * y2 - L53 * y3 - L54 * y4
    let y6 = bAng[2] - L61 * y1 - L62 * y2 - L63 * y3 - L64 * y4 - L65 * y5

    let z1 = y1 / D1
    let z2 = y2 / D2
    let z3 = y3 / D3
    let z4 = y4 / D4
    let z5 = y5 / D5
    let z6 = y6 / D6

    var xLin = SIMD3<Float>.zero
    var xAng = SIMD3<Float>.zero
    xAng[2] = z6
    xAng[1] = z5 - L65 * xAng[2]
    xAng[0] = z4 - L54 * xAng[1] - L64 * xAng[2]
    xLin[2] = z3 - L43 * xAng[0] - L53 * xAng[1] - L63 * xAng[2]
    xLin[1] = z2 - L32 * xLin[2] - L42 * xAng[0] - L52 * xAng[1] - L62 * xAng[2]
    xLin[0] = z1 - L21 * xLin[1] - L31 * xLin[2] - L41 * xAng[0] - L51 * xAng[1] - L61 * xAng[2]

    return (xLin, xAng)
}

nonisolated final class AVBDPhysicsBody {
    var positionLin: SIMD3<Float>
    var positionAng: simd_quatf
    var initialLin: SIMD3<Float>
    var initialAng: simd_quatf
    var inertialLin: SIMD3<Float>
    var inertialAng: simd_quatf
    var velocityLin: SIMD3<Float>
    var velocityAng: SIMD3<Float>
    var prevVelocityLin: SIMD3<Float>
    let size: SIMD3<Float>
    let mass: Float
    let moment: SIMD3<Float>
    let friction: Float
    let renderShape: AVBDRenderShape
    let color: SIMD4<Float>

    var radius: Float {
        avbdShapeRadius(size: size, shape: renderShape)
    }

    init(_ body: AVBDRigidBody) {
        positionLin = body.position
        positionAng = body.orientation
        initialLin = body.position
        initialAng = body.orientation
        inertialLin = body.position
        inertialAng = body.orientation
        velocityLin = body.velocity
        velocityAng = .zero
        prevVelocityLin = body.velocity
        size = body.size
        mass = avbdShapeMass(size: body.size, density: body.density, shape: body.renderShape)
        moment = avbdShapeMoment(size: body.size, mass: mass, shape: body.renderShape)
        friction = body.friction
        renderShape = body.renderShape
        color = body.resolvedRenderColor
    }
}

nonisolated class AVBDForce {
    var bodyA: AVBDPhysicsBody?
    var bodyB: AVBDPhysicsBody

    init(bodyA: AVBDPhysicsBody?, bodyB: AVBDPhysicsBody) {
        self.bodyA = bodyA
        self.bodyB = bodyB
    }

    func initialize(solver: AVBDSolver) -> Bool {
        true
    }

    func updatePrimal(
        body: AVBDPhysicsBody,
        alpha: Float,
        lhsLin: inout AVBDMat3,
        lhsAng: inout AVBDMat3,
        lhsCross: inout AVBDMat3,
        rhsLin: inout SIMD3<Float>,
        rhsAng: inout SIMD3<Float>
    ) {
    }

    func updateDual(solver: AVBDSolver, alpha: Float) {
    }

    func connects(_ a: AVBDPhysicsBody, _ b: AVBDPhysicsBody) -> Bool {
        (bodyA === a && bodyB === b) || (bodyA === b && bodyB === a)
    }

    func actsOn(_ body: AVBDPhysicsBody) -> Bool {
        bodyA === body || bodyB === body
    }
}

nonisolated final class AVBDIgnoreCollisionForce: AVBDForce {
}

nonisolated final class AVBDJointForce: AVBDForce {
    var rA: SIMD3<Float>
    var rB: SIMD3<Float>
    var C0Lin = SIMD3<Float>.zero
    var C0Ang = SIMD3<Float>.zero
    var penaltyLin = SIMD3<Float>.zero
    var penaltyAng = SIMD3<Float>.zero
    var lambdaLin = SIMD3<Float>.zero
    var lambdaAng = SIMD3<Float>.zero
    var stiffnessLin: Float
    var stiffnessAng: Float
    var fracture: Float
    var torqueArm: Float
    var broken = false

    init(
        bodyA: AVBDPhysicsBody?,
        bodyB: AVBDPhysicsBody,
        rA: SIMD3<Float>,
        rB: SIMD3<Float>,
        stiffnessLin: Float,
        stiffnessAng: Float,
        fracture: Float
    ) {
        self.rA = rA
        self.rB = rB
        self.stiffnessLin = stiffnessLin
        self.stiffnessAng = stiffnessAng
        self.fracture = fracture
        torqueArm = simd_length_squared((bodyA?.size ?? .zero) + bodyB.size)
        super.init(bodyA: bodyA, bodyB: bodyB)
    }

    override func initialize(solver: AVBDSolver) -> Bool {
        let pA = bodyA.map { transform($0.positionLin, $0.positionAng, rA) } ?? rA
        let pB = transform(bodyB.positionLin, bodyB.positionAng, rB)
        C0Lin = pA - pB
        C0Ang = quatDelta(bodyA?.positionAng ?? simd_quatf(angle: 0, axis: SIMD3<Float>(0, 0, 1)), bodyB.positionAng) * torqueArm

        lambdaLin *= solver.alpha * solver.gamma
        lambdaAng *= solver.alpha * solver.gamma
        penaltyLin = clampVec(penaltyLin * solver.gamma, penaltyMin, penaltyMax)
        penaltyAng = clampVec(penaltyAng * solver.gamma, penaltyMin, penaltyMax)

        penaltyLin = minVec(penaltyLin, stiffnessLin)
        penaltyAng = minVec(penaltyAng, stiffnessAng)

        return !broken
    }

    override func updatePrimal(
        body: AVBDPhysicsBody,
        alpha: Float,
        lhsLin: inout AVBDMat3,
        lhsAng: inout AVBDMat3,
        lhsCross: inout AVBDMat3,
        rhsLin: inout SIMD3<Float>,
        rhsAng: inout SIMD3<Float>
    ) {
        if simd_length_squared(penaltyLin) > 0 {
            let K = diagonal(penaltyLin.x, penaltyLin.y, penaltyLin.z)
            let pA = bodyA.map { transform($0.positionLin, $0.positionAng, rA) } ?? rA
            let pB = transform(bodyB.positionLin, bodyB.positionAng, rB)
            var C = pA - pB

            if stiffnessLin.isInfinite {
                C -= C0Lin * alpha
            }

            let F = K * C + lambdaLin
            let isA = bodyA === body
            let jLin = isA ? AVBDMat3.identity : AVBDMat3.identity * -1.0
            let jAng = isA
                ? skew(-bodyA!.positionAng.act(rA))
                : skew(bodyB.positionAng.act(rB))

            let jLinT = transpose(jLin)
            let jAngT = transpose(jAng)
            let jAngTk = jAngT * K

            lhsLin += jLinT * K * jLin
            lhsAng += jAngTk * jAng
            lhsCross += jAngTk * jLin

            let r = isA ? bodyA!.positionAng.act(rA) : -bodyB.positionAng.act(rB)
            let H = geometricStiffnessBallSocket(0, r) * F[0]
                + geometricStiffnessBallSocket(1, r) * F[1]
                + geometricStiffnessBallSocket(2, r) * F[2]
            lhsAng += diagonalize(H)

            rhsLin += jLinT * F
            rhsAng += jAngT * F
        }

        if simd_length_squared(penaltyAng) > 0 {
            let K = diagonal(penaltyAng.x, penaltyAng.y, penaltyAng.z)
            var C = quatDelta(bodyA?.positionAng ?? simd_quatf(angle: 0, axis: SIMD3<Float>(0, 0, 1)), bodyB.positionAng) * torqueArm

            if stiffnessAng.isInfinite {
                C -= C0Ang * alpha
            }

            let F = K * C + lambdaAng
            let jAng = (bodyA === body ? AVBDMat3.identity : AVBDMat3.identity * -1.0) * torqueArm

            lhsAng += transpose(jAng) * K * jAng
            rhsAng += transpose(jAng) * F
        }
    }

    override func updateDual(solver: AVBDSolver, alpha: Float) {
        if simd_length_squared(penaltyLin) > 0 {
            let K = diagonal(penaltyLin.x, penaltyLin.y, penaltyLin.z)
            let pA = bodyA.map { transform($0.positionLin, $0.positionAng, rA) } ?? rA
            let pB = transform(bodyB.positionLin, bodyB.positionAng, rB)
            var C = pA - pB

            if stiffnessLin.isInfinite {
                C -= C0Lin * alpha
                lambdaLin = K * C + lambdaLin
            }

            penaltyLin = minVec(penaltyLin + absVec(C) * solver.betaLin, min(stiffnessLin, penaltyMax))
        }

        if simd_length_squared(penaltyAng) > 0 {
            let K = diagonal(penaltyAng.x, penaltyAng.y, penaltyAng.z)
            var C = quatDelta(bodyA?.positionAng ?? simd_quatf(angle: 0, axis: SIMD3<Float>(0, 0, 1)), bodyB.positionAng) * torqueArm

            if stiffnessAng.isInfinite {
                C -= C0Ang * alpha
                lambdaAng = K * C + lambdaAng
            }

            penaltyAng = minVec(penaltyAng + absVec(C) * solver.betaAng, min(stiffnessAng, penaltyMax))
        }

        if simd_length_squared(lambdaAng) > fracture * fracture {
            penaltyLin = .zero
            penaltyAng = .zero
            lambdaLin = .zero
            lambdaAng = .zero
            broken = true
        }
    }

    private func geometricStiffnessBallSocket(_ k: Int, _ v: SIMD3<Float>) -> AVBDMat3 {
        var m = diagonal(-v[k], -v[k], -v[k])
        var row0 = m[0]
        var row1 = m[1]
        var row2 = m[2]
        row0[k] += v[0]
        row1[k] += v[1]
        row2[k] += v[2]
        m[0] = row0
        m[1] = row1
        m[2] = row2
        return m
    }
}

nonisolated final class AVBDSpringForce: AVBDForce {
    var rA: SIMD3<Float>
    var rB: SIMD3<Float>
    var rest: Float
    var stiffness: Float

    init(bodyA: AVBDPhysicsBody, bodyB: AVBDPhysicsBody, rA: SIMD3<Float>, rB: SIMD3<Float>, stiffness: Float, rest: Float) {
        self.rA = rA
        self.rB = rB
        self.stiffness = stiffness
        self.rest = rest
        super.init(bodyA: bodyA, bodyB: bodyB)

        if self.rest < 0 {
            let pA = transform(bodyA.positionLin, bodyA.positionAng, rA)
            let pB = transform(bodyB.positionLin, bodyB.positionAng, rB)
            self.rest = simd_length(pA - pB)
        }
    }

    override func updatePrimal(
        body: AVBDPhysicsBody,
        alpha: Float,
        lhsLin: inout AVBDMat3,
        lhsAng: inout AVBDMat3,
        lhsCross: inout AVBDMat3,
        rhsLin: inout SIMD3<Float>,
        rhsAng: inout SIMD3<Float>
    ) {
        guard let bodyA else { return }

        let pA = transform(bodyA.positionLin, bodyA.positionAng, rA)
        let pB = transform(bodyB.positionLin, bodyB.positionAng, rB)
        let d = pA - pB
        let dLen = simd_length(d)
        if dLen <= 1.0e-6 {
            return
        }

        let n = d / dLen
        let C = dLen - rest
        let f = stiffness * C

        let rWorld: SIMD3<Float>
        let jLin: SIMD3<Float>
        let jAng: SIMD3<Float>
        if body === bodyA {
            rWorld = bodyA.positionAng.act(rA)
            jLin = n
            jAng = cross(rWorld, n)
        } else {
            rWorld = bodyB.positionAng.act(rB)
            jLin = -n
            jAng = -cross(rWorld, n)
        }

        lhsLin += outer(jLin, jLin) * stiffness
        lhsAng += outer(jAng, jAng) * stiffness
        lhsCross += outer(jAng, jLin) * stiffness
        rhsLin += jLin * f
        rhsAng += jAng * f
    }
}

nonisolated private struct AVBDContact {
    var featureKey = 0
    var rA = SIMD3<Float>.zero
    var rB = SIMD3<Float>.zero
    var C0 = SIMD3<Float>.zero
    var penalty = SIMD3<Float>.zero
    var lambda = SIMD3<Float>.zero
    var stick = false
}

nonisolated final class AVBDManifoldForce: AVBDForce {
    private var contacts: [AVBDContact] = []
    var basis = AVBDMat3.identity
    var friction: Float = 0.0

    override func initialize(solver: AVBDSolver) -> Bool {
        guard let bodyA else { return false }
        friction = sqrt(bodyA.friction * bodyB.friction)
        var newContacts = AVBDCollider.collide(bodyA: bodyA, bodyB: bodyB, basisOut: &basis)

        for index in newContacts.indices {
            if let previous = contacts.first(where: { $0.featureKey == newContacts[index].featureKey }) {
                let newRA = newContacts[index].rA
                let newRB = newContacts[index].rB
                newContacts[index] = previous
                if !previous.stick {
                    newContacts[index].rA = newRA
                    newContacts[index].rB = newRB
                }
            }
        }

        contacts = newContacts

        for index in contacts.indices {
            let xA = transform(bodyA.positionLin, bodyA.positionAng, contacts[index].rA)
            let xB = transform(bodyB.positionLin, bodyB.positionAng, contacts[index].rB)
            contacts[index].C0 = basis * (xA - xB) + SIMD3<Float>(collisionMargin, 0, 0)
            contacts[index].lambda *= solver.alpha * solver.gamma
            contacts[index].penalty = clampVec(contacts[index].penalty * solver.gamma, penaltyMin, penaltyMax)
        }

        return !contacts.isEmpty
    }

    override func updatePrimal(
        body: AVBDPhysicsBody,
        alpha: Float,
        lhsLin: inout AVBDMat3,
        lhsAng: inout AVBDMat3,
        lhsCross: inout AVBDMat3,
        rhsLin: inout SIMD3<Float>,
        rhsAng: inout SIMD3<Float>
    ) {
        guard let bodyA else { return }

        let dqALin = bodyA.positionLin - bodyA.initialLin
        let dqAAng = quatDelta(bodyA.positionAng, bodyA.initialAng)
        let dqBLin = bodyB.positionLin - bodyB.initialLin
        let dqBAng = quatDelta(bodyB.positionAng, bodyB.initialAng)

        for contact in contacts {
            let rAWorld = bodyA.positionAng.act(contact.rA)
            let rBWorld = bodyB.positionAng.act(contact.rB)

            let jALin = basis
            let jBLin = -basis
            let jAAng = AVBDMat3(cross(rAWorld, jALin[0]), cross(rAWorld, jALin[1]), cross(rAWorld, jALin[2]))
            let jBAng = AVBDMat3(cross(rBWorld, jBLin[0]), cross(rBWorld, jBLin[1]), cross(rBWorld, jBLin[2]))

            let K = diagonal(contact.penalty.x, contact.penalty.y, contact.penalty.z)
            let C = contact.C0 * (1 - alpha) + jALin * dqALin + jBLin * dqBLin + jAAng * dqAAng + jBAng * dqBAng
            var F = K * C + contact.lambda

            F[0] = min(F[0], 0.0)
            let bounds = abs(F[0]) * friction
            let frictionScale = simd_length(SIMD2<Float>(F[1], F[2]))
            if frictionScale > bounds && frictionScale > 0 {
                F[1] *= bounds / frictionScale
                F[2] *= bounds / frictionScale
            }

            let jLin = body === bodyA ? jALin : jBLin
            let jAng = body === bodyA ? jAAng : jBAng

            let jLinT = transpose(jLin)
            let jAngT = transpose(jAng)
            let jAngTk = jAngT * K

            lhsLin += jLinT * K * jLin
            lhsAng += jAngTk * jAng
            lhsCross += jAngTk * jLin
            rhsLin += jLinT * F
            rhsAng += jAngT * F
        }
    }

    override func updateDual(solver: AVBDSolver, alpha: Float) {
        guard let bodyA else { return }

        let dqALin = bodyA.positionLin - bodyA.initialLin
        let dqAAng = quatDelta(bodyA.positionAng, bodyA.initialAng)
        let dqBLin = bodyB.positionLin - bodyB.initialLin
        let dqBAng = quatDelta(bodyB.positionAng, bodyB.initialAng)

        for index in contacts.indices {
            let rAWorld = bodyA.positionAng.act(contacts[index].rA)
            let rBWorld = bodyB.positionAng.act(contacts[index].rB)

            let jALin = basis
            let jBLin = -basis
            let jAAng = AVBDMat3(cross(rAWorld, jALin[0]), cross(rAWorld, jALin[1]), cross(rAWorld, jALin[2]))
            let jBAng = AVBDMat3(cross(rBWorld, jBLin[0]), cross(rBWorld, jBLin[1]), cross(rBWorld, jBLin[2]))

            let K = diagonal(contacts[index].penalty.x, contacts[index].penalty.y, contacts[index].penalty.z)
            let C = contacts[index].C0 * (1 - alpha) + jALin * dqALin + jBLin * dqBLin + jAAng * dqAAng + jBAng * dqBAng
            var F = K * C + contacts[index].lambda

            F[0] = min(F[0], 0.0)
            let bounds = abs(F[0]) * friction
            let frictionScale = simd_length(SIMD2<Float>(F[1], F[2]))
            if frictionScale > bounds && frictionScale > 0 {
                F[1] *= bounds / frictionScale
                F[2] *= bounds / frictionScale
            }

            contacts[index].lambda = F

            if F[0] < 0 {
                contacts[index].penalty[0] = min(contacts[index].penalty[0] + solver.betaLin * abs(C[0]), penaltyMax)
            }
            if frictionScale <= bounds {
                contacts[index].penalty[1] = min(contacts[index].penalty[1] + solver.betaLin * abs(C[1]), penaltyMax)
                contacts[index].penalty[2] = min(contacts[index].penalty[2] + solver.betaLin * abs(C[2]), penaltyMax)
                contacts[index].stick = simd_length(SIMD2<Float>(C[1], C[2])) < stickThreshold
            }
        }
    }
}

nonisolated final class AVBDSolver {
    var dt: Float = 1.0 / 60.0
    var gravity: Float = -10.0
    var iterations = 20
    var alpha: Float = 0.9
    var betaLin: Float = 100_000.0
    var betaAng: Float = 100.0
    var gamma: Float = 0.999

    private(set) var bodies: [AVBDPhysicsBody]
    private var forces: [AVBDForce]

    init(scene: AVBDScene) {
        bodies = scene.bodies.map(AVBDPhysicsBody.init)
        forces = []

        for constraint in scene.constraints {
            switch constraint {
            case let .joint(bodyA, bodyB, anchorA, anchorB, stiffnessLin, stiffnessAng, fracture):
                forces.append(AVBDJointForce(
                    bodyA: bodies[bodyA],
                    bodyB: bodies[bodyB],
                    rA: anchorA,
                    rB: anchorB,
                    stiffnessLin: stiffnessLin,
                    stiffnessAng: stiffnessAng,
                    fracture: fracture ?? .infinity
                ))
            case let .spring(bodyA, bodyB, anchorA, anchorB, stiffness, rest):
                forces.append(AVBDSpringForce(bodyA: bodies[bodyA], bodyB: bodies[bodyB], rA: anchorA, rB: anchorB, stiffness: stiffness, rest: rest))
            case let .ignoreCollision(bodyA, bodyB):
                forces.append(AVBDIgnoreCollisionForce(bodyA: bodies[bodyA], bodyB: bodies[bodyB]))
            }
        }
    }

    @discardableResult
    func addBody(
        position: SIMD3<Float>,
        velocity: SIMD3<Float>,
        size: SIMD3<Float>,
        density: Float,
        friction: Float,
        renderColor: SIMD4<Float>,
        renderShape: AVBDRenderShape = .box
    ) -> Int {
        let body = AVBDRigidBody(
            renderShape: renderShape,
            size: size,
            density: density,
            friction: friction,
            position: position,
            orientation: simd_quatf(angle: 0, axis: SIMD3<Float>(0, 0, 1)),
            velocity: velocity,
            renderColor: renderColor,
            colorGroup: nil
        )
        bodies.append(AVBDPhysicsBody(body))
        return bodies.count - 1
    }

    func step() {
        broadphase()

        forces = forces.filter { $0.initialize(solver: self) }

        for body in bodies {
            body.inertialLin = body.positionLin + body.velocityLin * dt
            if body.mass > 0 {
                body.inertialLin += SIMD3<Float>(0, 0, gravity) * (dt * dt)
            }
            body.inertialAng = quatAddAngular(body.positionAng, body.velocityAng * dt)

            let accel = (body.velocityLin - body.prevVelocityLin) / dt
            let accelExt = accel.z * sign(gravity)
            var accelWeight = min(max(accelExt / abs(gravity), 0.0), 1.0)
            if !accelWeight.isFinite {
                accelWeight = 0.0
            }

            body.initialLin = body.positionLin
            body.initialAng = body.positionAng
            if body.mass > 0 {
                body.positionLin = body.positionLin + body.velocityLin * dt + SIMD3<Float>(0, 0, gravity) * (accelWeight * dt * dt)
                body.positionAng = quatAddAngular(body.positionAng, body.velocityAng * dt)
            }
        }

        for _ in 0..<iterations {
            for body in bodies where body.mass > 0 {
                let mLin = diagonal(body.mass, body.mass, body.mass)
                let mAng = diagonal(body.moment.x, body.moment.y, body.moment.z)

                var lhsLin = mLin / (dt * dt)
                var lhsAng = mAng / (dt * dt)
                var lhsCross = AVBDMat3.zero
                var rhsLin = (mLin / (dt * dt)) * (body.positionLin - body.inertialLin)
                var rhsAng = (mAng / (dt * dt)) * quatDelta(body.positionAng, body.inertialAng)

                for force in forces where force.actsOn(body) {
                    force.updatePrimal(body: body, alpha: alpha, lhsLin: &lhsLin, lhsAng: &lhsAng, lhsCross: &lhsCross, rhsLin: &rhsLin, rhsAng: &rhsAng)
                }

                let (dxLin, dxAng) = solveSystem(aLin: lhsLin, aAng: lhsAng, aCross: lhsCross, bLin: -rhsLin, bAng: -rhsAng)
                if dxLin.x.isFinite && dxLin.y.isFinite && dxLin.z.isFinite && dxAng.x.isFinite && dxAng.y.isFinite && dxAng.z.isFinite {
                    body.positionLin += dxLin
                    body.positionAng = quatAddAngular(body.positionAng, dxAng)
                }
            }

            for force in forces {
                force.updateDual(solver: self, alpha: alpha)
            }
        }

        for body in bodies {
            body.prevVelocityLin = body.velocityLin
            if body.mass > 0 {
                body.velocityLin = (body.positionLin - body.initialLin) / dt
                body.velocityAng = quatDelta(body.positionAng, body.initialAng) / dt
            }
        }
    }

    private func broadphase() {
        for i in bodies.indices {
            let bodyA = bodies[i]
            for j in bodies.index(after: i)..<bodies.count {
                let bodyB = bodies[j]
                if bodyA.mass <= 0 && bodyB.mass <= 0 {
                    continue
                }

                let dp = bodyA.positionLin - bodyB.positionLin
                let r = bodyA.radius + bodyB.radius
                if simd_length_squared(dp) <= r * r && !constrained(bodyA, bodyB) {
                    forces.append(AVBDManifoldForce(bodyA: bodyA, bodyB: bodyB))
                }
            }
        }
    }

    private func constrained(_ bodyA: AVBDPhysicsBody, _ bodyB: AVBDPhysicsBody) -> Bool {
        forces.contains { $0.connects(bodyA, bodyB) }
    }
}

nonisolated private let sphereSphereFeatureKey = Int(AVBDCollisionFeaturePrefix.sphereSphere.rawValue)
nonisolated private let sphereBoxFeatureKey = Int(AVBDCollisionFeaturePrefix.sphereBox.rawValue)
nonisolated private let torusSphereFeatureKey = Int(AVBDCollisionFeaturePrefix.torusSphere.rawValue)
nonisolated private let torusBoxFeatureKey = Int(AVBDCollisionFeaturePrefix.torusBox.rawValue)
nonisolated private let torusTorusFeatureKey = Int(AVBDCollisionFeaturePrefix.torusTorus.rawValue)

nonisolated private struct AVBDOBB {
    var center: SIMD3<Float>
    var rotation: simd_quatf
    var half: SIMD3<Float>
    var axis: [SIMD3<Float>]
}

nonisolated private struct AVBDSatAxis {
    var type: AVBDAxisType = .faceA
    var indexA = -1
    var indexB = -1
    var separation = -Float.greatestFiniteMagnitude
    var normalAB = SIMD3<Float>.zero
    var valid = false
}

nonisolated private struct AVBDFaceFrame {
    var axisIndex = 0
    var normal = SIMD3<Float>.zero
    var center = SIMD3<Float>.zero
    var u = SIMD3<Float>.zero
    var v = SIMD3<Float>.zero
    var extentU: Float = 0
    var extentV: Float = 0
}

nonisolated private enum AVBDCollider {
    static func collide(bodyA: AVBDPhysicsBody, bodyB: AVBDPhysicsBody, basisOut: inout AVBDMat3) -> [AVBDContact] {
        switch (bodyA.renderShape, bodyB.renderShape) {
        case (.sphere, .sphere):
            return collideSphereSphere(bodyA: bodyA, bodyB: bodyB, basisOut: &basisOut)
        case (.sphere, .box):
            return collideSphereBox(sphereBody: bodyA, boxBody: bodyB, sphereIsBodyA: true, basisOut: &basisOut)
        case (.box, .sphere):
            return collideSphereBox(sphereBody: bodyB, boxBody: bodyA, sphereIsBodyA: false, basisOut: &basisOut)
        case (.torus, _):
            return collideTorusBody(torusBody: bodyA, otherBody: bodyB, torusIsBodyA: true, basisOut: &basisOut)
        case (_, .torus):
            return collideTorusBody(torusBody: bodyB, otherBody: bodyA, torusIsBodyA: false, basisOut: &basisOut)
        case (.box, .box):
            break
        }

        let boxA = makeOBB(bodyA)
        let boxB = makeOBB(bodyB)
        let delta = boxB.center - boxA.center

        var bestFace = AVBDSatAxis()
        var bestEdge = AVBDSatAxis()

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
        if bestEdge.valid {
            let edgeRelTol: Float = 0.95
            let edgeAbsTol: Float = 0.01
            if edgeRelTol * bestEdge.separation > bestFace.separation + edgeAbsTol {
                best = bestEdge
            }
        }

        basisOut = orthonormal(-best.normalAB)

        switch best.type {
        case .edge:
            return buildEdgeContact(bodyA, bodyB, boxA, boxB, best.indexA, best.indexB, best.normalAB)
        case .faceA:
            return buildFaceManifold(bodyA, bodyB, boxA, boxB, true, best.indexA, best.normalAB)
        case .faceB:
            return buildFaceManifold(bodyA, bodyB, boxA, boxB, false, best.indexB, best.normalAB)
        }
    }

    private static func makeOBB(_ body: AVBDPhysicsBody) -> AVBDOBB {
        AVBDOBB(
            center: body.positionLin,
            rotation: body.positionAng,
            half: body.size * 0.5,
            axis: [
                body.positionAng.act(SIMD3<Float>(1, 0, 0)),
                body.positionAng.act(SIMD3<Float>(0, 1, 0)),
                body.positionAng.act(SIMD3<Float>(0, 0, 1))
            ]
        )
    }

    private static func closestPointOnBox(_ point: SIMD3<Float>, _ box: AVBDOBB) -> (closest: SIMD3<Float>, normalToBox: SIMD3<Float>) {
        let localPoint = box.rotation.inverse.act(point - box.center)
        let clamped = SIMD3<Float>(
            min(max(localPoint.x, -box.half.x), box.half.x),
            min(max(localPoint.y, -box.half.y), box.half.y),
            min(max(localPoint.z, -box.half.z), box.half.z)
        )
        let deltaLocal = clamped - localPoint
        let deltaLenSq = simd_length_squared(deltaLocal)
        if deltaLenSq > satAxisEpsilon {
            return (box.rotation.act(clamped) + box.center, box.rotation.act(deltaLocal / sqrt(deltaLenSq)))
        }

        let dx = box.half.x - abs(localPoint.x)
        let dy = box.half.y - abs(localPoint.y)
        let dz = box.half.z - abs(localPoint.z)
        let axis: Int
        if dx <= dy && dx <= dz {
            axis = 0
        } else if dy <= dz {
            axis = 1
        } else {
            axis = 2
        }

        var closestLocal = localPoint
        var normalLocal = SIMD3<Float>.zero
        let signToFace: Float
        switch axis {
        case 0:
            signToFace = localPoint.x >= 0 ? 1 : -1
            closestLocal.x = box.half.x * signToFace
            normalLocal.x = signToFace
        case 1:
            signToFace = localPoint.y >= 0 ? 1 : -1
            closestLocal.y = box.half.y * signToFace
            normalLocal.y = signToFace
        default:
            signToFace = localPoint.z >= 0 ? 1 : -1
            closestLocal.z = box.half.z * signToFace
            normalLocal.z = signToFace
        }

        return (box.rotation.act(closestLocal) + box.center, box.rotation.act(normalLocal))
    }

    private static func collideSphereSphere(
        bodyA: AVBDPhysicsBody,
        bodyB: AVBDPhysicsBody,
        basisOut: inout AVBDMat3
    ) -> [AVBDContact] {
        let delta = bodyB.positionLin - bodyA.positionLin
        let distSq = simd_length_squared(delta)
        let radius = bodyA.radius + bodyB.radius
        if distSq > radius * radius {
            return []
        }

        let normalAB = distSq > satAxisEpsilon ? (delta / sqrt(distSq)) : SIMD3<Float>(1, 0, 0)
        basisOut = orthonormal(-normalAB)

        let xA = bodyA.positionLin + normalAB * bodyA.radius
        let xB = bodyB.positionLin - normalAB * bodyB.radius

        return [makeContact(bodyA: bodyA, bodyB: bodyB, xA: xA, xB: xB, featureKey: sphereSphereFeatureKey)]
    }

    private static func collideSphereBox(
        sphereBody: AVBDPhysicsBody,
        boxBody: AVBDPhysicsBody,
        sphereIsBodyA: Bool,
        basisOut: inout AVBDMat3
    ) -> [AVBDContact] {
        let box = makeOBB(boxBody)
        let (closestPoint, sphereToBoxNormal) = closestPointOnBox(sphereBody.positionLin, box)
        let offset = closestPoint - sphereBody.positionLin
        let distSq = simd_length_squared(offset)
        if distSq > sphereBody.radius * sphereBody.radius {
            return []
        }

        let normalAB = sphereIsBodyA ? sphereToBoxNormal : -sphereToBoxNormal
        basisOut = orthonormal(-normalAB)

        let spherePoint = sphereBody.positionLin + sphereToBoxNormal * sphereBody.radius
        let xA = sphereIsBodyA ? spherePoint : closestPoint
        let xB = sphereIsBodyA ? closestPoint : spherePoint

        return [makeContact(
            bodyA: sphereIsBodyA ? sphereBody : boxBody,
            bodyB: sphereIsBodyA ? boxBody : sphereBody,
            xA: xA,
            xB: xB,
            featureKey: sphereBoxFeatureKey
        )]
    }

    private static func collideTorusBody(
        torusBody: AVBDPhysicsBody,
        otherBody: AVBDPhysicsBody,
        torusIsBodyA: Bool,
        basisOut: inout AVBDMat3
    ) -> [AVBDContact] {
        let torusSphereRadius = avbdTorusApproxSphereRadius(size: torusBody.size)
        let torusSphereCount = avbdCurrentTorusApproxSphereCount()
        var contacts: [AVBDContact] = []
        var contactMidpoints: [SIMD3<Float>] = []
        var bestNormalAB = SIMD3<Float>(1, 0, 0)
        var bestPenetration = -Float.greatestFiniteMagnitude

        let appendCandidate: (_ normalFromTorusToOther: SIMD3<Float>, _ xOnTorus: SIMD3<Float>, _ xOnOther: SIMD3<Float>, _ featureKey: Int, _ penetration: Float) -> Void = {
            normalFromTorusToOther, xOnTorus, xOnOther, featureKey, penetration in
            let normalAB = torusIsBodyA ? normalFromTorusToOther : -normalFromTorusToOther
            let xA = torusIsBodyA ? xOnTorus : xOnOther
            let xB = torusIsBodyA ? xOnOther : xOnTorus
            appendContact(
                &contacts,
                &contactMidpoints,
                bodyA: torusIsBodyA ? torusBody : otherBody,
                bodyB: torusIsBodyA ? otherBody : torusBody,
                xA: xA,
                xB: xB,
                featureKey: featureKey
            )
            if penetration > bestPenetration {
                bestPenetration = penetration
                bestNormalAB = normalAB
            }
        }

        switch otherBody.renderShape {
        case .sphere:
            for torusSphereIndex in 0..<torusSphereCount {
                let torusSphereCenter = torusBody.positionAng.act(
                    avbdTorusApproxSphereLocalCenter(size: torusBody.size, index: torusSphereIndex)
                ) + torusBody.positionLin
                guard let contact = sphereSphereContact(
                    centerA: torusSphereCenter,
                    radiusA: torusSphereRadius,
                    centerB: otherBody.positionLin,
                    radiusB: otherBody.radius
                ) else {
                    continue
                }
                appendCandidate(
                    contact.normalAB,
                    contact.xA,
                    contact.xB,
                    torusSphereFeatureKey | (torusSphereIndex & 0xFF),
                    contact.penetration
                )
            }
        case .box:
            let box = makeOBB(otherBody)
            for torusSphereIndex in 0..<torusSphereCount {
                let torusSphereCenter = torusBody.positionAng.act(
                    avbdTorusApproxSphereLocalCenter(size: torusBody.size, index: torusSphereIndex)
                ) + torusBody.positionLin
                guard let contact = sphereBoxContact(
                    sphereCenter: torusSphereCenter,
                    sphereRadius: torusSphereRadius,
                    box: box
                ) else {
                    continue
                }
                appendCandidate(
                    contact.normalFromSphereToBox,
                    contact.spherePoint,
                    contact.boxPoint,
                    torusBoxFeatureKey | (torusSphereIndex & 0xFF),
                    contact.penetration
                )
            }
        case .torus:
            let otherSphereRadius = avbdTorusApproxSphereRadius(size: otherBody.size)
            for torusSphereIndex in 0..<torusSphereCount {
                let torusSphereCenter = torusBody.positionAng.act(
                    avbdTorusApproxSphereLocalCenter(size: torusBody.size, index: torusSphereIndex)
                ) + torusBody.positionLin
                for otherSphereIndex in 0..<torusSphereCount {
                    let otherSphereCenter = otherBody.positionAng.act(
                        avbdTorusApproxSphereLocalCenter(size: otherBody.size, index: otherSphereIndex)
                    ) + otherBody.positionLin
                    guard let contact = sphereSphereContact(
                        centerA: torusSphereCenter,
                        radiusA: torusSphereRadius,
                        centerB: otherSphereCenter,
                        radiusB: otherSphereRadius
                    ) else {
                        continue
                    }
                    appendCandidate(
                        contact.normalAB,
                        contact.xA,
                        contact.xB,
                        torusTorusFeatureKey | ((torusSphereIndex & 0xFF) << 8) | (otherSphereIndex & 0xFF),
                        contact.penetration
                    )
                }
            }
        }

        guard !contacts.isEmpty else {
            return []
        }
        basisOut = orthonormal(-bestNormalAB)
        return contacts
    }

    private static func sphereSphereContact(
        centerA: SIMD3<Float>,
        radiusA: Float,
        centerB: SIMD3<Float>,
        radiusB: Float
    ) -> (normalAB: SIMD3<Float>, xA: SIMD3<Float>, xB: SIMD3<Float>, penetration: Float)? {
        let delta = centerB - centerA
        let distSq = simd_length_squared(delta)
        let radius = radiusA + radiusB
        guard distSq <= radius * radius else {
            return nil
        }

        let dist = sqrt(max(distSq, 0.0))
        let normalAB = dist > satAxisEpsilon ? (delta / dist) : SIMD3<Float>(1, 0, 0)
        return (
            normalAB,
            centerA + normalAB * radiusA,
            centerB - normalAB * radiusB,
            radius - dist
        )
    }

    private static func sphereBoxContact(
        sphereCenter: SIMD3<Float>,
        sphereRadius: Float,
        box: AVBDOBB
    ) -> (normalFromSphereToBox: SIMD3<Float>, spherePoint: SIMD3<Float>, boxPoint: SIMD3<Float>, penetration: Float)? {
        let (closestPoint, normalFromSphereToBox) = closestPointOnBox(sphereCenter, box)
        let offset = closestPoint - sphereCenter
        let distSq = simd_length_squared(offset)
        guard distSq <= sphereRadius * sphereRadius else {
            return nil
        }

        return (
            normalFromSphereToBox,
            sphereCenter + normalFromSphereToBox * sphereRadius,
            closestPoint,
            sphereRadius - sqrt(max(distSq, 0.0))
        )
    }

    private static func absDot(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        abs(dot(a, b))
    }

    private static func supportPoint(_ box: AVBDOBB, _ dir: SIMD3<Float>) -> SIMD3<Float> {
        let sx: Float = dot(dir, box.axis[0]) >= 0 ? 1 : -1
        let sy: Float = dot(dir, box.axis[1]) >= 0 ? 1 : -1
        let sz: Float = dot(dir, box.axis[2]) >= 0 ? 1 : -1

        return box.center
            + box.axis[0] * (box.half.x * sx)
            + box.axis[1] * (box.half.y * sy)
            + box.axis[2] * (box.half.z * sz)
    }

    private static func getFaceAxes(_ box: AVBDOBB, _ axisIndex: Int) -> (SIMD3<Float>, SIMD3<Float>, Float, Float) {
        if axisIndex == 0 {
            return (box.axis[1], box.axis[2], box.half.y, box.half.z)
        } else if axisIndex == 1 {
            return (box.axis[0], box.axis[2], box.half.x, box.half.z)
        } else {
            return (box.axis[0], box.axis[1], box.half.x, box.half.y)
        }
    }

    private static func buildFaceFrame(_ box: AVBDOBB, _ axisIndex: Int, _ outwardNormal: SIMD3<Float>) -> AVBDFaceFrame {
        let faceSign: Float = dot(outwardNormal, box.axis[axisIndex]) >= 0 ? 1 : -1
        let normal = box.axis[axisIndex] * faceSign
        let (u, v, extentU, extentV) = getFaceAxes(box, axisIndex)
        return AVBDFaceFrame(
            axisIndex: axisIndex,
            normal: normal,
            center: box.center + normal * box.half[axisIndex],
            u: u,
            v: v,
            extentU: extentU,
            extentV: extentV
        )
    }

    private static func chooseIncidentFaceAxis(_ box: AVBDOBB, _ referenceNormal: SIMD3<Float>) -> Int {
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

    private static func buildIncidentFace(_ box: AVBDOBB, _ axisIndex: Int, _ referenceNormal: SIMD3<Float>) -> [SIMD3<Float>] {
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

    private static func clipPolygonAgainstPlane(_ inVerts: [SIMD3<Float>], _ planeNormal: SIMD3<Float>, _ planeOffset: Float) -> [SIMD3<Float>] {
        if inVerts.isEmpty {
            return []
        }

        var outVerts: [SIMD3<Float>] = []
        var a = inVerts[inVerts.count - 1]
        var da = dot(planeNormal, a) - planeOffset

        for b in inVerts {
            let db = dot(planeNormal, b) - planeOffset
            let aInside = da <= planeEpsilon
            let bInside = db <= planeEpsilon

            if aInside != bInside {
                var t: Float = 0
                let denom = da - db
                if abs(denom) > satAxisEpsilon {
                    t = min(max(da / denom, 0.0), 1.0)
                }
                if outVerts.count < maxPolyVerts {
                    outVerts.append(a + (b - a) * t)
                }
            }

            if bInside && outVerts.count < maxPolyVerts {
                outVerts.append(b)
            }

            a = b
            da = db
        }

        return outVerts
    }

    private static func makeContact(
        bodyA: AVBDPhysicsBody,
        bodyB: AVBDPhysicsBody,
        xA: SIMD3<Float>,
        xB: SIMD3<Float>,
        featureKey: Int
    ) -> AVBDContact {
        AVBDContact(
            featureKey: featureKey,
            rA: bodyA.positionAng.inverse.act(xA - bodyA.positionLin),
            rB: bodyB.positionAng.inverse.act(xB - bodyB.positionLin),
            C0: .zero,
            penalty: .zero,
            lambda: .zero,
            stick: false
        )
    }

    private static func appendContact(
        _ contacts: inout [AVBDContact],
        _ contactMidpoints: inout [SIMD3<Float>],
        bodyA: AVBDPhysicsBody,
        bodyB: AVBDPhysicsBody,
        xA: SIMD3<Float>,
        xB: SIMD3<Float>,
        featureKey: Int
    ) {
        let midpoint = (xA + xB) * 0.5
        for existing in contactMidpoints where simd_length_squared(midpoint - existing) < contactMergeDistSq {
            return
        }
        if contacts.count >= maxContacts {
            return
        }
        contacts.append(makeContact(bodyA: bodyA, bodyB: bodyB, xA: xA, xB: xB, featureKey: featureKey))
        contactMidpoints.append(midpoint)
    }

    private static func testAxis(
        _ boxA: AVBDOBB,
        _ boxB: AVBDOBB,
        _ delta: SIMD3<Float>,
        _ axis: SIMD3<Float>,
        _ type: AVBDAxisType,
        _ indexA: Int,
        _ indexB: Int,
        _ best: inout AVBDSatAxis
    ) -> Bool {
        let lenSq = simd_length_squared(axis)
        if lenSq < satAxisEpsilon {
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

    private static func supportEdge(_ box: AVBDOBB, _ axisIndex: Int, _ dir: SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) {
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

    private static func closestPointsOnSegments(
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

        if a <= satAxisEpsilon && e <= satAxisEpsilon {
            return (p0, q0)
        }

        if a <= satAxisEpsilon {
            t = min(max(f / e, 0), 1)
        } else {
            let c = dot(d1, r)
            if e <= satAxisEpsilon {
                s = min(max(-c / a, 0), 1)
            } else {
                let b = dot(d1, d2)
                let denom = a * e - b * b
                if abs(denom) > satAxisEpsilon {
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

    private static func buildFaceManifold(
        _ bodyA: AVBDPhysicsBody,
        _ bodyB: AVBDPhysicsBody,
        _ boxA: AVBDOBB,
        _ boxB: AVBDOBB,
        _ referenceIsA: Bool,
        _ referenceAxis: Int,
        _ normalAB: SIMD3<Float>
    ) -> [AVBDContact] {
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

        var contacts: [AVBDContact] = []
        var contactMidpoints: [SIMD3<Float>] = []
        var featurePrefix = Int(referenceIsA ? AVBDAxisType.faceA.rawValue : AVBDAxisType.faceB.rawValue) << 24
        featurePrefix |= (referenceAxis & 0xFF) << 16
        featurePrefix |= (incidentAxis & 0xFF) << 8

        for i in clipped.indices where contacts.count < maxContacts {
            let pIncident = clipped[i]
            let distance = dot(pIncident - referenceFace.center, referenceFace.normal)
            if distance > planeEpsilon {
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

    private static func buildEdgeContact(
        _ bodyA: AVBDPhysicsBody,
        _ bodyB: AVBDPhysicsBody,
        _ boxA: AVBDOBB,
        _ boxB: AVBDOBB,
        _ axisA: Int,
        _ axisB: Int,
        _ normalAB: SIMD3<Float>
    ) -> [AVBDContact] {
        let (a0, a1) = supportEdge(boxA, axisA, normalAB)
        let (b0, b1) = supportEdge(boxB, axisB, -normalAB)
        var (xA, xB) = closestPointsOnSegments(a0, a1, b0, b1)

        var contacts: [AVBDContact] = []
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

    private static func orthonormal(_ normal: SIMD3<Float>) -> AVBDMat3 {
        var t1 = abs(normal.x) > abs(normal.z)
            ? SIMD3<Float>(-normal.y, normal.x, 0)
            : SIMD3<Float>(0, -normal.z, normal.y)
        t1 = safeNormalize(t1)
        let t2 = cross(normal, t1)
        return AVBDMat3(normal, t1, t2)
    }
}
