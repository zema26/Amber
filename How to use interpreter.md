tiny compiler in Rust for the "Amber" language syntax

This compiler will be implemented as an interpreter, which is a common first step for building a language. It will perform three main stages:

1.  **Lexical Analysis (Lexing):** It scans the source code and converts it into a sequence of "tokens" (like keywords, identifiers, and operators).
2.  **Parsing:** It takes the tokens and builds an Abstract Syntax Tree (AST), which is a hierarchical representation of the code's structure.
3.  **Interpretation:** It walks the AST and executes the program's logic directly.

Here are the files for the compiler and the example program.

http://googleusercontent.com/immersive_entry_chip/0

"compiler.rs"

http://googleusercontent.com/immersive_entry_chip/1

"euclidean.amb"

### How to Run This Compiler

1.  **Save the Files:** Save the two files above as `compiler.rs` and `euclidean.amb` in the same directory.
2.  **Setup Rust:** Make sure you have the Rust programming language installed. If not, you can get it from [rust-lang.org](https://www.rust-lang.org/).
3.  **Compile and Run:** Open a terminal in the directory where you saved the files and run the following commands:
      * `rustc compiler.rs`
      * `./compiler` (on Linux/macOS) or `compiler.exe` (on Windows)

When you run the compiled program, it will first print the list of tokens and the structure of the AST. Then, it will prompt you to enter two numbers. For example:

```text
Enter 2 integer values separated by space:
54 24
Result: 6
```
