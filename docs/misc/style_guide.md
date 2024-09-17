
# ZML Style Guide

We prefer to keep it simple and adhere to the [Zig Style Guide](https://ziglang.org/documentation/0.13.0/#Style-Guide).

We use ZLS to auto-format code.

In addition, we try to adhere to the following house-rules:

### We favor:

```zig
const x: Foo = .{ .bar = 1 } 
// over: const x = Foo{ .bar = 1}

pub fn method(self: Foo) void
// over: pub fn method(self: Self) void

const foo = import("foo.zig"); foo.bar()
// over: const bar = import("foo.zig").bar;
//       bar();

const Foo = import("foo.zig").Foo 
// over: const Foo = import("Foo.zig")
//
// Importing types directly instead of using 
// a namespace should be reserved for very 
// frequent types.


/// Foo does X and returns Y
pub fn foo() usize {
// Descriptive doc comments over imperative ones
 ```

As with the Zig Style Guide: use common sense 😊.
