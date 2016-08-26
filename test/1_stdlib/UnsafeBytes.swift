// RUN: %target-run-simple-swiftgyb
// REQUIRES: executable_test

import StdlibUnittest

var UnsafeBytesTestSuite = TestSuite("UnsafeBytes")

UnsafeBytesTestSuite.test("initFromValue") {
  var value1: Int32 = -1
  var value2: Int32 = 0
  // Immutable view of value1's bytes.
  withUnsafeBytes(of: &value1) { bytes1 in
    expectEqual(bytes1.count, 4)
    for b in bytes1 {
      expectEqual(b, 0xFF)
    }
    // Mutable view of value2's bytes.
    withUnsafeMutableBytes(of: &value2) { bytes2 in
      expectEqual(bytes1.count, bytes2.count)
      bytes2[0..<bytes2.count].copyBytes(from: bytes1)
    }
  }
  expectEqual(value2, value1)
}

UnsafeBytesTestSuite.test("initFromArray") {
  var array1: [Int32] = [0, 1, 2, 3]
  var array2 = [Int32](repeating: 0, count: 4)
  // Immutable view of array1's bytes.
  array1.withUnsafeBytes { bytes1 in
    expectEqual(bytes1.count, 16)
    for (i, b) in bytes1.enumerated() {
      if i % 4 == 0 {
        expectEqual(Int(b), i / 4)
      }
      else {
        expectEqual(b, 0)
      }
    }
    // Mutable view of array2's bytes.
    array2.withUnsafeMutableBytes { bytes2 in
      expectEqual(bytes1.count, bytes2.count)
      bytes2[0..<bytes2.count].copyBytes(from: bytes1)
    }
  }
  expectEqual(array2, array1)
}

UnsafeBytesTestSuite.test("Collection") {
  expectCollectionType(UnsafeBytes.self)
  expectMutableCollectionType(UnsafeMutableBytes.self)
  expectSliceType(UnsafeBytes.self)
  expectMutableSliceType(UnsafeMutableBytes.self)

  expectCollectionAssociatedTypes(
    collectionType: UnsafeBytes.self,
    iteratorType: UnsafeBytes.Iterator.self,
    subSequenceType: UnsafeBytes.self,
    indexType: Int.self,
    indexDistanceType: Int.self,
    indicesType: CountableRange<Int>.self)

  expectCollectionAssociatedTypes(
    collectionType: UnsafeMutableBytes.self,
    iteratorType: UnsafeMutableBytes.Iterator.self,
    subSequenceType: UnsafeMutableBytes.self,
    indexType: Int.self,
    indexDistanceType: Int.self,
    indicesType: CountableRange<Int>.self)

  var array1: [Int32] = [0, 1, 2, 3]
  array1.withUnsafeBytes { bytes1 in
    // Initialize an array from a sequence of bytes.
    let byteArray = [UInt8](bytes1)
    for (b1, b2) in zip(byteArray, bytes1) {
      expectEqual(b1, b2)
    }
  }
}

UnsafeBytesTestSuite.test("reinterpret") {
  struct Pair {
    var x: Int32
    var y: Int32
  }
  let numPairs = 2
  let bytes = UnsafeMutableBytes.allocate(
    count: MemoryLayout<Pair>.stride * numPairs)
  defer { bytes.deallocate() }  

  for i in 0..<(numPairs * 2) {
    bytes.storeBytes(of: Int32(i), toByteOffset: i * MemoryLayout<Int32>.stride,
      as: Int32.self)
  }
  let pair1 = bytes.load(as: Pair.self)
  let pair2 = bytes.load(fromByteOffset: MemoryLayout<Pair>.stride,
    as: Pair.self)
  expectEqual(0, pair1.x)
  expectEqual(1, pair1.y)
  expectEqual(2, pair2.x)
  expectEqual(3, pair2.y)

  bytes.storeBytes(of: Pair(x: -1, y: 0), as: Pair.self)
  for i in 0..<MemoryLayout<Int32>.stride {
    expectEqual(0xFF, bytes[i])
  }
  let bytes2 = bytes[MemoryLayout<Int32>.stride..<bytes.count]
  for i in 0..<MemoryLayout<Int32>.stride {
    expectEqual(0, bytes2[i])
  }
}

UnsafeBytesTestSuite.test("inBounds") {
  let numInts = 4
  let bytes = UnsafeMutableBytes.allocate(
    count: MemoryLayout<Int>.stride * numInts)
  defer { bytes.deallocate() }

  for i in 0..<numInts {
    bytes.storeBytes(
      of: i, toByteOffset: i * MemoryLayout<Int>.stride, as: Int.self)
  }
  for i in 0..<numInts {
    let x = bytes.load(
      fromByteOffset: i * MemoryLayout<Int>.stride, as: Int.self)
    expectEqual(x, i)
  }
  let median = (numInts/2 * MemoryLayout<Int>.stride)
  var firstHalf = bytes[0..<median]
  var secondHalf = bytes[median..<bytes.count]
  firstHalf[0..<firstHalf.count] = secondHalf
  expectEqualSequence(firstHalf, secondHalf)
}

UnsafeBytesTestSuite.test("subscript.underflow") {
  if _isDebugAssertConfiguration() {
    expectCrashLater()
  }
  let bytes = UnsafeMutableBytes.allocate(count: 2)
  defer { bytes.deallocate() }

  bytes[-1] = 0
}

UnsafeBytesTestSuite.test("subscript.overflow") {
  if _isDebugAssertConfiguration() {
    expectCrashLater()
  }
  let bytes = UnsafeMutableBytes.allocate(count: 2)
  defer { bytes.deallocate() }

  bytes[2] = 0
}

UnsafeBytesTestSuite.test("load.before") {
  if _isDebugAssertConfiguration() {
    expectCrashLater()
  }
  var x: Int32 = 0
  withUnsafeBytes(of: &x) {
    _ = $0.load(fromByteOffset: -1, as: UInt8.self)
  }
}
UnsafeBytesTestSuite.test("load.after") {
  if _isDebugAssertConfiguration() {
    expectCrashLater()
  }
  var x: Int32 = 0
  withUnsafeBytes(of: &x) {
    _ = $0.load(as: UInt64.self)
  }
}

UnsafeBytesTestSuite.test("store.before") {
  if _isDebugAssertConfiguration() {
    expectCrashLater()
  }
  var x: Int32 = 0
  withUnsafeMutableBytes(of: &x) {
    $0.storeBytes(of: UInt8(0), toByteOffset: -1, as: UInt8.self)
  }
}
UnsafeBytesTestSuite.test("store.after") {
  if _isDebugAssertConfiguration() {
    expectCrashLater()
  }
  var x: Int32 = 0
  withUnsafeMutableBytes(of: &x) {
    $0.storeBytes(of: UInt64(0), as: UInt64.self)
  }
}

UnsafeBytesTestSuite.test("copy.bytes.overflow") {
  expectCrashLater()
  var x: Int64 = 0
  var y: Int32 = 0
  withUnsafeBytes(of: &x) { srcBytes in
    withUnsafeMutableBytes(of: &y) { destBytes in
      destBytes.copyBytes(from: UnsafeMutableBytes(mutating: srcBytes))
    }
  }
}

UnsafeBytesTestSuite.test("copy.sequence.overflow") {
  expectCrashLater()
  var x: Int64 = 0
  var y: Int32 = 0
  withUnsafeBytes(of: &x) { srcBytes in
    withUnsafeMutableBytes(of: &y) { destBytes in
      destBytes.copyBytes(from: srcBytes)
    }
  }
}

UnsafeBytesTestSuite.test("copy.overlap") {
  var bytes = UnsafeMutableBytes.allocate(count: 4)
  // Right Overlap
  bytes[0] = 1
  bytes[1] = 2
  bytes[1..<3] = bytes[0..<2]
  expectEqual(1, bytes[1])
  expectEqual(2, bytes[2])
  // Left Overlap
  bytes[1] = 2
  bytes[2] = 3
  bytes[0..<2] = bytes[1..<3]
  expectEqual(2, bytes[0])
  expectEqual(3, bytes[1])
  // Disjoint
  bytes[2] = 2
  bytes[3] = 3
  bytes[0..<2] = bytes[2..<4]
  expectEqual(2, bytes[0])
  expectEqual(3, bytes[1])
  bytes[0] = 0
  bytes[1] = 1
  bytes[2..<4] = bytes[0..<2]
  expectEqual(0, bytes[2])
  expectEqual(1, bytes[3])
}

runAllTests()
