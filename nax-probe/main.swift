// NAX Layout Probe — Swift Harness
// Compiles probe.metal, dispatches on M5 Air, dumps results as JSON.
// Usage: swift main.swift [--test 0|1|2|3|4|5|all] [--iterations N]

import Metal
import Foundation

// MARK: - Setup

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("No Metal device")
}
print("Device: \(device.name)")
print("GPU family: Apple\(device.supportsFamily(.apple9) ? "9+" : device.supportsFamily(.apple8) ? "8" : "≤7")")

// Compile Metal source
let metalURL = URL(fileURLWithPath: "probe.metal", relativeTo: URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
let metalSource = try String(contentsOf: metalURL, encoding: .utf8)

let options = MTLCompileOptions()
options.languageVersion = .version3_2  // Metal 4 / MSL 3.2

let library: MTLLibrary
do {
    library = try device.makeLibrary(source: metalSource, options: options)
} catch {
    print("Metal compilation failed: \(error)")
    print("Note: Requires macOS 26+ with Metal 4 support")
    exit(1)
}

let queue = device.makeCommandQueue()!

// MARK: - Helper

func makeBuffer<T>(_ data: [T]) -> MTLBuffer {
    return device.makeBuffer(bytes: data, length: data.count * MemoryLayout<T>.stride, options: .storageModeShared)!
}

func readBuffer<T>(_ buffer: MTLBuffer, count: Int, as: T.Type) -> [T] {
    let ptr = buffer.contents().bindMemory(to: T.self, capacity: count)
    return Array(UnsafeBufferPointer(start: ptr, count: count))
}

typealias Half = UInt16
func floatToHalf(_ f: Float) -> Half {
    var f = f
    var h: Half = 0
    withUnsafePointer(to: &f) { fp in
        withUnsafeMutablePointer(to: &h) { hp in
            // Use vImageConvert
            var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: fp), height: 1, width: 1, rowBytes: 4)
            var dst = vImage_Buffer(data: UnsafeMutableRawPointer(hp), height: 1, width: 1, rowBytes: 2)
            vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0)
        }
    }
    return h
}

func halfToFloat(_ h: Half) -> Float {
    var h = h
    var f: Float = 0
    withUnsafePointer(to: &h) { hp in
        withUnsafeMutablePointer(to: &f) { fp in
            var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: hp), height: 1, width: 1, rowBytes: 2)
            var dst = vImage_Buffer(data: UnsafeMutableRawPointer(fp), height: 1, width: 1, rowBytes: 4)
            vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
        }
    }
    return f
}

import Accelerate

func makeHalfBuffer(_ floats: [Float]) -> MTLBuffer {
    let halfs = floats.map { floatToHalf($0) }
    return makeBuffer(halfs)
}

func readHalfBuffer(_ buffer: MTLBuffer, count: Int) -> [Float] {
    let halfs = readBuffer(buffer, count: count, as: Half.self)
    return halfs.map { halfToFloat($0) }
}

func dispatch1SIMD(_ name: String, buffers: [MTLBuffer]) {
    let fn = library.makeFunction(name: name)!
    let pso = try! device.makeComputePipelineState(function: fn)
    let cmd = queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    for (i, buf) in buffers.enumerated() {
        enc.setBuffer(buf, offset: 0, index: i)
    }
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    if let err = cmd.error {
        print("  GPU error: \(err)")
    }
}

func timeDispatch(_ name: String, buffers: [MTLBuffer], iterations: Int = 100) -> Double {
    let fn = library.makeFunction(name: name)!
    let pso = try! device.makeComputePipelineState(function: fn)

    // Warmup
    for _ in 0..<5 {
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        for (i, buf) in buffers.enumerated() { enc.setBuffer(buf, offset: 0, index: i) }
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        for (i, buf) in buffers.enumerated() { enc.setBuffer(buf, offset: 0, index: i) }
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let elapsed = CFAbsoluteTimeGetCurrent() - start
    return elapsed / Double(iterations)
}

// MARK: - Matrix helpers

func identity16() -> [Float] {
    var m = [Float](repeating: 0, count: 256)
    for i in 0..<16 { m[i * 16 + i] = 1.0 }
    return m
}

func sequential16() -> [Float] {
    return (0..<256).map { Float($0) }
}

func ones16() -> [Float] {
    return [Float](repeating: 1.0, count: 256)
}

func oneHot16(position: Int) -> [Float] {
    var m = [Float](repeating: 0, count: 256)
    m[position] = 1.0
    return m
}

func printMatrix(_ name: String, _ data: [Float], rows: Int = 16, cols: Int = 16) {
    print("  \(name):")
    for r in 0..<min(rows, 16) {
        let row = (0..<min(cols, 16)).map { c in
            String(format: "%6.1f", data[r * cols + c])
        }.joined(separator: " ")
        print("    [\(row)]")
    }
}

// MARK: - Tests

func test0_coord_map() {
    print("\n== Test 0: get_coord() Thread-to-Element Mapping ==")
    let outBuf = device.makeBuffer(length: 32 * 12 * 4, options: .storageModeShared)!
    dispatch1SIMD("probe_coord_map", buffers: [outBuf])

    let out = readBuffer(outBuf, count: 32 * 12, as: Float.self)

    print("  lane | row0 | col_base | row1 | elements (row×16+col)")
    print("  " + String(repeating: "-", count: 65))
    for lane in 0..<32 {
        let base = lane * 12
        let laneId = Int(out[base + 0])
        let row0   = Int(out[base + 1])
        let colB   = Int(out[base + 2])
        let row1   = Int(out[base + 3])
        let elems = (0..<8).map { Int(out[base + 4 + $0]) }
        print(String(format: "  %4d | %4d | %8d | %4d | %@",
                     laneId, row0, colB, row1,
                     elems.map { String($0) }.joined(separator: ", ")))
    }

    // Build the full 16×16 ownership map
    print("\n  Ownership map (which lane owns each element):")
    var ownership = [Int](repeating: -1, count: 256)
    for lane in 0..<32 {
        let base = lane * 12
        for i in 0..<8 {
            let idx = Int(out[base + 4 + i])
            if idx >= 0 && idx < 256 { ownership[idx] = lane }
        }
    }
    print("       col: ", (0..<16).map { String(format: "%3d", $0) }.joined(separator: " "))
    for r in 0..<16 {
        let row = (0..<16).map { c in
            String(format: "%3d", ownership[r * 16 + c])
        }.joined(separator: " ")
        print("    r\(String(format: "%02d", r)): \(row)")
    }
}

func test1_identity() {
    print("\n== Test 1: Identity Matmul (I×B = B) ==")
    let A = makeHalfBuffer(identity16())
    let B = makeHalfBuffer(sequential16())
    let C = device.makeBuffer(length: 256 * 2, options: .storageModeShared)!
    let debug = device.makeBuffer(length: 32 * 32 * 4, options: .storageModeShared)!

    dispatch1SIMD("probe_identity_matmul", buffers: [A, B, C, debug])

    let result = readHalfBuffer(C, count: 256)
    let expected = sequential16()

    printMatrix("Expected (B)", expected)
    printMatrix("Got (I×B via NAX)", result)

    // Check correctness
    var maxErr: Float = 0
    var errCount = 0
    for i in 0..<256 {
        let err = abs(result[i] - expected[i])
        if err > 0.5 { errCount += 1 }
        maxErr = max(maxErr, err)
    }
    print("  Max error: \(maxErr), positions with >0.5 error: \(errCount)/256")

    // Dump cooperative tensor capacities from debug
    let dbg = readBuffer(debug, count: 32 * 32, as: Float.self)
    let cap_a = Int(dbg[27])
    let cap_b = Int(dbg[28])
    let cap_c = Int(dbg[29])
    print("  Cooperative tensor capacities: ct_a=\(cap_a), ct_b=\(cap_b), ct_c=\(cap_c)")
}

func test2_naive_vs_coord() {
    print("\n== Test 2: Naive Load vs Coord Load ==")
    let A = makeHalfBuffer(identity16())
    let B = makeHalfBuffer(sequential16())

    // Coord-loaded result (from test 1)
    let C_coord = device.makeBuffer(length: 256 * 2, options: .storageModeShared)!
    let debug = device.makeBuffer(length: 32 * 32 * 4, options: .storageModeShared)!
    dispatch1SIMD("probe_identity_matmul", buffers: [A, B, C_coord, debug])

    // Naive-loaded result
    let C_naive = device.makeBuffer(length: 256 * 2, options: .storageModeShared)!
    dispatch1SIMD("probe_naive_load", buffers: [A, B, C_naive])

    let coord_result = readHalfBuffer(C_coord, count: 256)
    let naive_result = readHalfBuffer(C_naive, count: 256)

    printMatrix("Coord-loaded I×B", coord_result)
    printMatrix("Naive-loaded I×B", naive_result)

    // Find the permutation
    var diffs = 0
    for i in 0..<256 {
        if abs(coord_result[i] - naive_result[i]) > 0.5 { diffs += 1 }
    }
    print("  Positions differing: \(diffs)/256")
    if diffs > 0 {
        print("  → Naive load produces DIFFERENT results — hardware expects coord-mapped layout")
    } else {
        print("  → Results identical — hardware may auto-remap (unexpected)")
    }
}

func test3_one_hot_sample() {
    print("\n== Test 3: One-Hot Probe (sample positions) ==")
    let B = makeHalfBuffer(ones16())
    let C = device.makeBuffer(length: 256 * 2, options: .storageModeShared)!

    // Test a few key positions: (0,0), (0,1), (1,0), (8,0), (15,15)
    let positions = [0, 1, 16, 128, 255]
    for pos in positions {
        let row = pos / 16
        let col = pos % 16
        let A = makeHalfBuffer(oneHot16(position: pos))
        dispatch1SIMD("probe_one_hot", buffers: [A, B, C])
        let result = readHalfBuffer(C, count: 256)

        // Find which row of C is nonzero (should be row `row` with all 1s)
        var nonzeroRows = [Int]()
        for r in 0..<16 {
            let rowSum = (0..<16).reduce(Float(0)) { $0 + result[r * 16 + $1] }
            if rowSum > 0.5 { nonzeroRows.append(r) }
        }
        let nonzeroCols: [Int] = {
            guard let r = nonzeroRows.first else { return [] }
            return (0..<16).filter { result[r * 16 + $0] > 0.5 }
        }()
        print("  A[\(row),\(col)]=1 → C nonzero at rows=\(nonzeroRows), cols=\(nonzeroCols)")
    }
}

func test4_fp32_control() {
    print("\n== Test 4: FP16 (NAX) vs FP32 (SIMD ALU) Timing ==")

    // FP16 NAX path
    let A16 = makeHalfBuffer(identity16())
    let B16 = makeHalfBuffer(sequential16())
    let C16 = device.makeBuffer(length: 256 * 2, options: .storageModeShared)!
    let debug = device.makeBuffer(length: 32 * 32 * 4, options: .storageModeShared)!

    let t16 = timeDispatch("probe_identity_matmul", buffers: [A16, B16, C16, debug], iterations: 1000)

    // FP32 SIMD path
    let A32 = makeBuffer(identity16())
    let B32 = makeBuffer(sequential16())
    let C32 = device.makeBuffer(length: 256 * 4, options: .storageModeShared)!

    let t32 = timeDispatch("probe_fp32_control", buffers: [A32, B32, C32], iterations: 1000)

    print(String(format: "  FP16 NAX:     %.3f µs/matmul", t16 * 1e6))
    print(String(format: "  FP32 SIMD:    %.3f µs/matmul", t32 * 1e6))
    print(String(format: "  NAX speedup:  %.2fx", t32 / t16))

    // Verify FP32 produces correct results too
    let result32 = readBuffer(C32, count: 256, as: Float.self)
    let expected = sequential16()
    var maxErr: Float = 0
    for i in 0..<256 { maxErr = max(maxErr, abs(result32[i] - expected[i])) }
    print(String(format: "  FP32 max error: %.6f", maxErr))
}

func test5_throughput() {
    print("\n== Test 5: NAX Throughput (tight loop) ==")
    let A = makeHalfBuffer(ones16())
    let B = makeHalfBuffer(ones16())
    let C = device.makeBuffer(length: 256 * 2, options: .storageModeShared)!

    for iterCount in [1000, 10000, 100000] as [UInt32] {
        var n = iterCount
        let nBuf = device.makeBuffer(bytes: &n, length: 4, options: .storageModeShared)!

        let start = CFAbsoluteTimeGetCurrent()
        dispatch1SIMD("probe_throughput", buffers: [A, B, C, nBuf])
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        let opsPerSec = Double(iterCount) / elapsed
        let flopsPerOp: Double = 2 * 16 * 16 * 16  // 2*M*N*K for matmul
        let gflops = opsPerSec * flopsPerOp / 1e9
        print(String(format: "  %7d iters: %.3f ms (%.0f ops/s, %.1f GFLOPS)",
                     iterCount, elapsed * 1000, opsPerSec, gflops))
    }
}

// MARK: - Main

let args = CommandLine.arguments
let testArg = args.firstIndex(of: "--test").map { args[$0 + 1] } ?? "all"

print("NAX Layout Probe — \(device.name)")
print("=" + String(repeating: "=", count: 50))

switch testArg {
case "0": test0_coord_map()
case "1": test1_identity()
case "2": test2_naive_vs_coord()
case "3": test3_one_hot_sample()
case "4": test4_fp32_control()
case "5": test5_throughput()
case "all":
    test0_coord_map()
    test1_identity()
    test2_naive_vs_coord()
    test3_one_hot_sample()
    test4_fp32_control()
    test5_throughput()
default:
    print("Unknown test: \(testArg). Use 0-5 or all.")
}

print("\nDone.")
