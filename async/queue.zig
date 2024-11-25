const std = @import("std");
const assert = std.debug.assert;

/// An intrusive queue implementation. The type T must have a field
/// "next" of type `?*T`.
///
/// For those unaware, an intrusive variant of a data structure is one in which
/// the data type in the list has the pointer to the next element, rather
/// than a higher level "node" or "container" type. The primary benefit
/// of this (and the reason we implement this) is that it defers all memory
/// management to the caller: the data structure implementation doesn't need
/// to allocate "nodes" to contain each element. Instead, the caller provides
/// the element and how its allocated is up to them.
pub fn Intrusive(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Head is the front of the queue and tail is the back of the queue.
        head: ?*T = null,
        tail: ?*T = null,

        /// Enqueue a new element to the back of the queue.
        pub fn push(self: *Self, v: *T) void {
            assert(v.next == null);

            if (self.tail) |tail| {
                // If we have elements in the queue, then we add a new tail.
                tail.next = v;
                self.tail = v;
            } else {
                // No elements in the queue we setup the initial state.
                self.head = v;
                self.tail = v;
            }
        }

        /// Dequeue the next element from the queue.
        pub fn pop(self: *Self) ?*T {
            // The next element is in "head".
            const next = self.head orelse return null;

            // If the head and tail are equal this is the last element
            // so we also set tail to null so we can now be empty.
            if (self.head == self.tail) self.tail = null;

            // Head is whatever is next (if we're the last element,
            // this will be null);
            self.head = next.next;

            // We set the "next" field to null so that this element
            // can be inserted again.
            next.next = null;
            return next;
        }

        /// Returns true if the queue is empty.
        pub fn empty(self: *const Self) bool {
            return self.head == null;
        }
    };
}

test Intrusive {
    const testing = std.testing;

    // Types
    const Elem = struct {
        const Self = @This();
        next: ?*Self = null,
    };
    const Queue = Intrusive(Elem);
    var q: Queue = .{};
    try testing.expect(q.empty());

    // Elems
    var elems: [10]Elem = .{.{}} ** 10;

    // One
    try testing.expect(q.pop() == null);
    q.push(&elems[0]);
    try testing.expect(!q.empty());
    try testing.expect(q.pop().? == &elems[0]);
    try testing.expect(q.pop() == null);
    try testing.expect(q.empty());

    // Two
    try testing.expect(q.pop() == null);
    q.push(&elems[0]);
    q.push(&elems[1]);
    try testing.expect(q.pop().? == &elems[0]);
    try testing.expect(q.pop().? == &elems[1]);
    try testing.expect(q.pop() == null);

    // Interleaved
    try testing.expect(q.pop() == null);
    q.push(&elems[0]);
    try testing.expect(q.pop().? == &elems[0]);
    q.push(&elems[1]);
    try testing.expect(q.pop().? == &elems[1]);
    try testing.expect(q.pop() == null);
}
