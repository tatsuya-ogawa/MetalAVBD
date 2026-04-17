//
//  AVBDStanfordArmadilloLoader.swift
//  MetalAVBD
//

import Foundation
import Metal
import simd
import zlib

enum AVBDStanfordArmadilloLoaderError: Error {
    case invalidResponse
    case unexpectedStatusCode(Int)
    case failedToDownload
    case failedToCreateDirectory(String)
    case failedToReadHeader
    case unsupportedPLYFormat(String)
    case malformedHeader(String)
    case malformedPayload(String)
    case failedToCreateBuffer(String)
    case failedToOpenArchive(String)
    case failedToWriteDecompressedFile(String)
    case decompressionFailed(String)
}

struct AVBDTriangleMeshAsset {
    var positionData: [SIMD3<Float>]
    var normalData: [SIMD3<Float>]
    var indexData: [UInt32]
    var positions: MTLBuffer
    var normals: MTLBuffer
    var indexBuffer: MTLBuffer
    var vertexCount: Int
    var indexCount: Int
    var boundsMin: SIMD3<Float>
    var boundsMax: SIMD3<Float>
    var boundsCenter: SIMD3<Float>
    var boundingSphereRadius: Float
    var sourceURL: URL
    var cacheDirectoryURL: URL
    var cacheArchiveURL: URL
    var cachePLYURL: URL

    func makeGeometry(label: String? = nil) -> AVBDTriangleMeshGeometry {
        AVBDTriangleMeshGeometry(
            vertexBuffer: positions,
            vertexStride: MemoryLayout<SIMD3<Float>>.stride,
            indexBuffer: indexBuffer,
            indexCount: indexCount,
            label: label
        )
    }

    func makeStaticMesh() -> StaticMesh {
        StaticMesh(
            positions: positions,
            normals: normals,
            indexBuffer: indexBuffer,
            indexCount: indexCount
        )
    }
}

final class AVBDStanfordArmadilloLoader {
    static let remoteURL = URL(string: "http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz")!
    static let archiveFilename = "Armadillo.ply.gz"
    static let plyFilename = "Armadillo.ply"

    private let device: MTLDevice
    private let session: URLSession
    private let fileManager: FileManager

    init(
        device: MTLDevice,
        session: URLSession = .shared,
        fileManager: FileManager = .default
    ) {
        self.device = device
        self.session = session
        self.fileManager = fileManager
    }

    func load() throws -> AVBDTriangleMeshAsset {
        let cacheDirectoryURL = Self.cacheDirectoryURL()
        let archiveURL = cacheDirectoryURL.appendingPathComponent(Self.archiveFilename)
        let plyURL = cacheDirectoryURL.appendingPathComponent(Self.plyFilename)

        try ensureCacheDirectory(at: cacheDirectoryURL)
        try ensureArchive(at: archiveURL)
        try ensureDecompressedPLY(archiveURL: archiveURL, plyURL: plyURL)

        let data = try Data(contentsOf: plyURL, options: [.mappedIfSafe])
        let parsedMesh = try parsePLY(data: data)

        let positionLength = MemoryLayout<SIMD3<Float>>.stride * parsedMesh.positions.count
        let normalLength = MemoryLayout<SIMD3<Float>>.stride * parsedMesh.normals.count
        let indexLength = MemoryLayout<UInt32>.stride * parsedMesh.indices.count

        guard let positionsBuffer = parsedMesh.positions.withUnsafeBytes({ rawBuffer -> MTLBuffer? in
            guard let baseAddress = rawBuffer.baseAddress else { return nil }
            return device.makeBuffer(bytes: baseAddress, length: positionLength, options: .storageModeShared)
        }) else {
            throw AVBDStanfordArmadilloLoaderError.failedToCreateBuffer("Failed to create the Armadillo vertex buffer.")
        }

        guard let normalsBuffer = parsedMesh.normals.withUnsafeBytes({ rawBuffer -> MTLBuffer? in
            guard let baseAddress = rawBuffer.baseAddress else { return nil }
            return device.makeBuffer(bytes: baseAddress, length: normalLength, options: .storageModeShared)
        }) else {
            throw AVBDStanfordArmadilloLoaderError.failedToCreateBuffer("Failed to create the Armadillo normal buffer.")
        }

        guard let indexBuffer = parsedMesh.indices.withUnsafeBytes({ rawBuffer -> MTLBuffer? in
            guard let baseAddress = rawBuffer.baseAddress else { return nil }
            return device.makeBuffer(bytes: baseAddress, length: indexLength, options: .storageModeShared)
        }) else {
            throw AVBDStanfordArmadilloLoaderError.failedToCreateBuffer("Failed to create the Armadillo index buffer.")
        }

        positionsBuffer.label = "StanfordArmadilloPositions"
        normalsBuffer.label = "StanfordArmadilloNormals"
        indexBuffer.label = "StanfordArmadilloIndices"

        return AVBDTriangleMeshAsset(
            positionData: parsedMesh.positions,
            normalData: parsedMesh.normals,
            indexData: parsedMesh.indices,
            positions: positionsBuffer,
            normals: normalsBuffer,
            indexBuffer: indexBuffer,
            vertexCount: parsedMesh.positions.count,
            indexCount: parsedMesh.indices.count,
            boundsMin: parsedMesh.boundsMin,
            boundsMax: parsedMesh.boundsMax,
            boundsCenter: (parsedMesh.boundsMin + parsedMesh.boundsMax) * 0.5,
            boundingSphereRadius: parsedMesh.boundingSphereRadius,
            sourceURL: Self.remoteURL,
            cacheDirectoryURL: cacheDirectoryURL,
            cacheArchiveURL: archiveURL,
            cachePLYURL: plyURL
        )
    }

    static func cacheDirectoryURL() -> URL {
        URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("MetalAVBD", isDirectory: true)
            .appendingPathComponent("StanfordArmadillo", isDirectory: true)
    }

    private func ensureCacheDirectory(at directoryURL: URL) throws {
        do {
            try fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        } catch {
            throw AVBDStanfordArmadilloLoaderError.failedToCreateDirectory(directoryURL.path)
        }
    }

    private func ensureArchive(at archiveURL: URL) throws {
        guard !fileManager.fileExists(atPath: archiveURL.path) else {
            return
        }

        let semaphore = DispatchSemaphore(value: 0)
        var downloadError: Error?

        let task = session.downloadTask(with: Self.remoteURL) { temporaryURL, response, error in
            defer { semaphore.signal() }

            if let error {
                downloadError = error
                return
            }

            guard let httpResponse = response as? HTTPURLResponse else {
                downloadError = AVBDStanfordArmadilloLoaderError.invalidResponse
                return
            }

            guard (200...299).contains(httpResponse.statusCode) else {
                downloadError = AVBDStanfordArmadilloLoaderError.unexpectedStatusCode(httpResponse.statusCode)
                return
            }

            guard let temporaryURL else {
                downloadError = AVBDStanfordArmadilloLoaderError.failedToDownload
                return
            }

            do {
                if self.fileManager.fileExists(atPath: archiveURL.path) {
                    try self.fileManager.removeItem(at: archiveURL)
                }
                try self.fileManager.moveItem(at: temporaryURL, to: archiveURL)
            } catch {
                downloadError = error
            }
        }

        task.resume()
        semaphore.wait()

        if let downloadError {
            throw downloadError
        }
    }

    private func ensureDecompressedPLY(archiveURL: URL, plyURL: URL) throws {
        if fileManager.fileExists(atPath: plyURL.path) {
            return
        }

        try gunzip(archiveURL: archiveURL, destinationURL: plyURL)
    }

    private func gunzip(archiveURL: URL, destinationURL: URL) throws {
        guard let archive = gzopen(archiveURL.path, "rb") else {
            throw AVBDStanfordArmadilloLoaderError.failedToOpenArchive(archiveURL.path)
        }
        defer { gzclose(archive) }

        fileManager.createFile(atPath: destinationURL.path, contents: nil)

        guard let outputHandle = FileHandle(forWritingAtPath: destinationURL.path) else {
            throw AVBDStanfordArmadilloLoaderError.failedToWriteDecompressedFile(destinationURL.path)
        }

        defer {
            try? outputHandle.close()
        }

        let chunkSize = 64 * 1024
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: chunkSize)
        defer { buffer.deallocate() }

        while true {
            let readCount = gzread(archive, buffer, UInt32(chunkSize))

            if readCount > 0 {
                let data = Data(bytes: buffer, count: Int(readCount))
                try outputHandle.write(contentsOf: data)
                continue
            }

            if readCount == 0 {
                break
            }

            var zlibErrorCode: Int32 = 0
            let errorCString = gzerror(archive, &zlibErrorCode)
            let errorMessage = errorCString.map { String(cString: $0) } ?? "Unknown gzip error"
            try? fileManager.removeItem(at: destinationURL)
            throw AVBDStanfordArmadilloLoaderError.decompressionFailed(errorMessage)
        }
    }

    private func parsePLY(data: Data) throws -> (
        positions: [SIMD3<Float>],
        normals: [SIMD3<Float>],
        indices: [UInt32],
        boundsMin: SIMD3<Float>,
        boundsMax: SIMD3<Float>,
        boundingSphereRadius: Float
    ) {
        let header = try parseHeader(data: data)
        var offset = header.payloadOffset

        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity(header.vertexCount)

        var boundsMin = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var boundsMax = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)

        for _ in 0..<header.vertexCount {
            let x = try readFloat32BigEndian(data: data, offset: &offset)
            let y = try readFloat32BigEndian(data: data, offset: &offset)
            let z = try readFloat32BigEndian(data: data, offset: &offset)
            let position = SIMD3<Float>(x, y, z)
            positions.append(position)
            boundsMin = simd_min(boundsMin, position)
            boundsMax = simd_max(boundsMax, position)
        }

        var indices: [UInt32] = []
        indices.reserveCapacity(header.faceCount * 3)

        for _ in 0..<header.faceCount {
            _ = try readUInt8(data: data, offset: &offset)
            let vertexCount = Int(try readUInt8(data: data, offset: &offset))

            var faceIndices: [UInt32] = []
            faceIndices.reserveCapacity(vertexCount)

            for _ in 0..<vertexCount {
                let rawIndex = try readUInt32BigEndian(data: data, offset: &offset)
                faceIndices.append(rawIndex)
            }

            guard faceIndices.count >= 3 else {
                continue
            }

            if faceIndices.count == 3 {
                indices.append(contentsOf: faceIndices)
                continue
            }

            for triangleIndex in 1..<(faceIndices.count - 1) {
                indices.append(faceIndices[0])
                indices.append(faceIndices[triangleIndex])
                indices.append(faceIndices[triangleIndex + 1])
            }
        }

        var normals = Array(repeating: SIMD3<Float>.zero, count: positions.count)
        if !indices.isEmpty {
            for triangleStart in stride(from: 0, to: indices.count, by: 3) {
                let i0 = Int(indices[triangleStart])
                let i1 = Int(indices[triangleStart + 1])
                let i2 = Int(indices[triangleStart + 2])
                let p0 = positions[i0]
                let p1 = positions[i1]
                let p2 = positions[i2]
                let faceNormal = simd_cross(p1 - p0, p2 - p0)
                normals[i0] += faceNormal
                normals[i1] += faceNormal
                normals[i2] += faceNormal
            }
        }

        for index in normals.indices {
            let normal = normals[index]
            let lengthSquared = simd_length_squared(normal)
            normals[index] = lengthSquared > 1.0e-12 ? normal / sqrt(lengthSquared) : SIMD3<Float>(0, 0, 1)
        }

        let boundsCenter = (boundsMin + boundsMax) * 0.5
        var boundingSphereRadius: Float = 0
        for position in positions {
            boundingSphereRadius = max(boundingSphereRadius, simd_length(position - boundsCenter))
        }

        return (positions, normals, indices, boundsMin, boundsMax, boundingSphereRadius)
    }

    private func parseHeader(data: Data) throws -> (vertexCount: Int, faceCount: Int, payloadOffset: Int) {
        let endMarker = Data("end_header\n".utf8)
        guard let endRange = data.range(of: endMarker) else {
            throw AVBDStanfordArmadilloLoaderError.failedToReadHeader
        }

        let headerData = data[..<endRange.upperBound]
        let headerText = String(decoding: headerData, as: UTF8.self)

        var vertexCount: Int?
        var faceCount: Int?
        var format: String?

        for line in headerText.split(separator: "\n", omittingEmptySubsequences: false) {
            let trimmedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)

            if trimmedLine.hasPrefix("format ") {
                format = String(trimmedLine.dropFirst("format ".count))
            } else if trimmedLine.hasPrefix("element vertex ") {
                vertexCount = Int(trimmedLine.dropFirst("element vertex ".count))
            } else if trimmedLine.hasPrefix("element face ") {
                faceCount = Int(trimmedLine.dropFirst("element face ".count))
            }
        }

        guard let format else {
            throw AVBDStanfordArmadilloLoaderError.malformedHeader("Missing PLY format line.")
        }

        guard format == "binary_big_endian 1.0" else {
            throw AVBDStanfordArmadilloLoaderError.unsupportedPLYFormat(format)
        }

        guard let vertexCount, let faceCount else {
            throw AVBDStanfordArmadilloLoaderError.malformedHeader("Missing vertex or face count.")
        }

        return (vertexCount, faceCount, endRange.upperBound)
    }

    private func readUInt8(data: Data, offset: inout Int) throws -> UInt8 {
        guard offset + 1 <= data.count else {
            throw AVBDStanfordArmadilloLoaderError.malformedPayload("Unexpected end of file while reading UInt8.")
        }
        defer { offset += 1 }
        return data[offset]
    }

    private func readUInt32BigEndian(data: Data, offset: inout Int) throws -> UInt32 {
        guard offset + 4 <= data.count else {
            throw AVBDStanfordArmadilloLoaderError.malformedPayload("Unexpected end of file while reading UInt32.")
        }

        let value =
            (UInt32(data[offset]) << 24) |
            (UInt32(data[offset + 1]) << 16) |
            (UInt32(data[offset + 2]) << 8) |
            UInt32(data[offset + 3])
        offset += 4
        return value
    }

    private func readFloat32BigEndian(data: Data, offset: inout Int) throws -> Float {
        let bitPattern = try readUInt32BigEndian(data: data, offset: &offset)
        return Float(bitPattern: bitPattern)
    }
}
