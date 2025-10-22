use std::collections::HashMap;
use std::io::{self, Write};
use std::iter::Peekable;
use std::str::Chars;

// --- 1. Lexer (Tokenizer) ---
// This section turns the raw source code string into a stream of tokens.

#[derive(Debug, Clone, PartialEq)]
enum Token {
    // Keywords (from {tag})
    Module, Fun, While, If, Else, Return,
    EndModule, EndFun, EndWhile, EndIf, EndElse,

    // Identifiers and Types
    Identifier(String),
    Type(String),

    // Literals
    Integer(i64),

    // Operators
    Arrow,      // ->
    Minus,      // -
    NotEqual,   // !=
    GreaterThan,// >

    // Punctuation
    Comma,      // ,
    LeftBracket,// [
    RightBracket,// ]

    // Special I/O
    In,         // in
    Out,        // out
    
    Eof,        // End of File
}

fn lex(source: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut chars = source.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            ' ' | '\t' | '\r' | '\n' => {}, // Skip whitespace
            '!' => { // Skip comments
                while let Some(c) = chars.peek() {
                    if *c == '\n' { break; }
                    chars.next();
                }
            }
            '{' => { // Keyword tags
                let mut tag = String::new();
                let is_end_tag = chars.peek() == Some(&'/');
                if is_end_tag { chars.next(); } // consume '/'

                while let Some(c) = chars.peek() {
                    if *c == '}' { break; }
                    tag.push(chars.next().unwrap());
                }
                chars.next(); // consume '}'

                let token = match tag.as_str() {
                    "module" if !is_end_tag => Token::Module,
                    "fun" if !is_end_tag => Token::Fun,
                    "while" if !is_end_tag => Token::While,
                    "if" if !is_end_tag => Token::If,
                    "else" if !is_end_tag => Token::Else,
                    "return" if !is_end_tag => Token::Return,
                    "module" if is_end_tag => Token::EndModule,
                    "fun" if is_end_tag => Token::EndFun,
                    "while" if is_end_tag => Token::EndWhile,
                    "if" if is_end_tag => Token::EndIf,
                    "else" if is_end_tag => Token::EndElse,
                    _ => panic!("Unknown tag: {}", tag),
                };
                tokens.push(token);
            }
            'a'..='z' | 'A'..='Z' | '_' => { // Identifiers, types, in/out
                let mut ident = String::new();
                ident.push(ch);
                while let Some(&c) = chars.peek() {
                    if !c.is_alphanumeric() && c != '_' { break; }
                    ident.push(chars.next().unwrap());
                }

                let token = match ident.as_str() {
                    "int" | "string" => Token::Type(ident),
                    "in" => Token::In,
                    "out" => Token::Out,
                    _ => Token::Identifier(ident),
                };
                tokens.push(token);
            }
            '0'..='9' => { // Integers
                let mut num_str = String::new();
                num_str.push(ch);
                while let Some(&c) = chars.peek() {
                    if !c.is_digit(10) { break; }
                    num_str.push(chars.next().unwrap());
                }
                tokens.push(Token::Integer(num_str.parse().unwrap()));
            }
            '-' => {
                if chars.peek() == Some(&'>') {
                    chars.next(); // consume '>'
                    tokens.push(Token::Arrow);
                } else {
                    tokens.push(Token::Minus);
                }
            }
            '!' => {
                if chars.peek() == Some(&'=') {
                    chars.next(); // consume '='
                    tokens.push(Token::NotEqual);
                }
            }
            '>' => tokens.push(Token::GreaterThan),
            ',' => tokens.push(Token::Comma),
            '[' => tokens.push(Token::LeftBracket),
            ']' => tokens.push(Token::RightBracket),
            _ => panic!("Unexpected character: {}", ch),
        }
    }
    tokens.push(Token::Eof);
    tokens
}

// --- 2. AST (Abstract Syntax Tree) ---
// These structs define the structure of the program.

#[derive(Debug, Clone)]
enum Expr {
    Literal(i64),
    Variable(String),
    BinaryOp(Box<Expr>, Op, Box<Expr>),
    FunctionCall(String, Vec<String>),
}

#[derive(Debug, Clone)]
enum Op { Minus, NotEqual, GreaterThan }

#[derive(Debug, Clone)]
enum Stmt {
    VarDecl(String, String), // name, type
    Assignment(String, Expr),
    While(Expr, Vec<Stmt>),
    If(Expr, Vec<Stmt>, Option<Vec<Stmt>>),
    Return(Expr),
    FunctionDecl(String, Vec<(String, String)>, Vec<Stmt>), // name, params, body
    IoChain(Vec<String>, String), // input vars, function name
}

#[derive(Debug)]
struct Module {
    name: String,
    functions: Vec<Stmt>,
}


// --- 3. Parser ---
// This section builds the AST from the token stream.

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Parser { tokens, pos: 0 }
    }

    fn current(&self) -> &Token { &self.tokens[self.pos] }
    fn advance(&mut self) { self.pos += 1; }
    fn consume(&mut self, expected: Token) {
        if self.current() == &expected {
            self.advance();
        } else {
            panic!("Expected {:?}, found {:?}", expected, self.current());
        }
    }
    fn consume_ident(&mut self) -> String {
        match self.current() {
            Token::Identifier(s) => {
                let name = s.clone();
                self.advance();
                name
            }
            _ => panic!("Expected Identifier, found {:?}", self.current()),
        }
    }
    fn consume_type(&mut self) -> String {
        match self.current() {
            Token::Type(s) => {
                let name = s.clone();
                self.advance();
                name
            }
            _ => panic!("Expected Type, found {:?}", self.current()),
        }
    }
    
    fn parse(&mut self) -> Module {
        self.consume(Token::Module);
        let name = self.consume_ident();
        let mut functions = Vec::new();
        while self.current() != &Token::EndModule {
            functions.push(self.parse_function_decl());
        }
        self.consume(Token::EndModule);
        Module { name, functions }
    }

    fn parse_function_decl(&mut self) -> Stmt {
        self.consume(Token::Fun);
        let mut params = Vec::new();
        // Parse parameters until arrow
        while self.current() != &Token::Arrow {
            let param_type = self.consume_type();
            let param_name = self.consume_ident();
            params.push((param_name, param_type));
            if self.current() == &Token::Comma { self.advance(); }
            // Handle array syntax for main args, simplified
            if self.current() == &Token::LeftBracket {
                self.advance();
                self.advance(); // skip ']'
            }
        }
        self.consume(Token::Arrow);
        let name = self.consume_ident();
        self.consume_type(); // Consume return type, ignored for now
        
        let mut body = Vec::new();
        while self.current() != &Token::EndFun {
            body.push(self.parse_statement());
        }
        self.consume(Token::EndFun);
        Stmt::FunctionDecl(name, params, body)
    }

    fn parse_statement(&mut self) -> Stmt {
        match self.current() {
            Token::Type(_) => self.parse_var_decl(),
            Token::While => self.parse_while(),
            Token::If => self.parse_if(),
            Token::Return => self.parse_return(),
            Token::In => self.parse_io_chain(),
            _ => { // Assume assignment or expression statement
                let expr = self.parse_expr();
                if let Expr::Variable(target) = expr {
                    if self.current() == &Token::Arrow {
                        self.advance();
                        let target_name = self.consume_ident();
                        // This is a simplified parse for a-b -> a
                        return Stmt::Assignment(target_name, Expr::Variable(target));
                    }
                }
                panic!("Unexpected statement starting with {:?}", self.current());
            }
        }
    }

    fn parse_io_chain(&mut self) -> Stmt {
        self.consume(Token::In);
        self.consume(Token::Arrow);
        let mut vars = Vec::new();
        while self.current() != &Token::Arrow {
            vars.push(self.consume_ident());
            if self.current() == &Token::Comma { self.advance(); }
        }
        self.consume(Token::Arrow);
        let func_name = self.consume_ident();
        self.consume(Token::Arrow);
        self.consume(Token::Out);
        Stmt::IoChain(vars, func_name)
    }
    
    fn parse_var_decl(&mut self) -> Stmt {
        let var_type = self.consume_type();
        let name = self.consume_ident();
        // Simplified: assumes one declaration per line for now
        Stmt::VarDecl(name, var_type)
    }

    fn parse_while(&mut self) -> Stmt {
        self.consume(Token::While);
        let cond = self.parse_expr();
        let mut body = Vec::new();
        while self.current() != &Token::EndWhile {
            body.push(self.parse_statement());
        }
        self.consume(Token::EndWhile);
        Stmt::While(cond, body)
    }

    fn parse_if(&mut self) -> Stmt {
        self.consume(Token::If);
        let cond = self.parse_expr();
        let mut then_branch = Vec::new();
        while self.current() != &Token::EndIf && self.current() != &Token::Else {
            then_branch.push(self.parse_statement());
        }
        
        let mut else_branch = None;
        if self.current() == &Token::Else {
            self.advance();
            let mut else_stmts = Vec::new();
            while self.current() != &Token::EndElse {
                else_stmts.push(self.parse_statement());
            }
            self.consume(Token::EndElse);
            else_branch = Some(else_stmts);
        }

        self.consume(Token::EndIf);
        Stmt::If(cond, then_branch, else_branch)
    }
    
    fn parse_return(&mut self) -> Stmt {
        self.consume(Token::Return);
        let val = self.parse_expr();
        // The parser has to deal with the closing return tag. We can just consume it.
        // A more robust parser would ensure it's a {/return}, but this is tiny.
        while self.current() != &Token::EndFun { 
             // This is a simple way to find the end of the return value
            if let Token::Identifier(_) = self.current() { break; }
             self.advance(); 
        }
        Stmt::Return(val)
    }

    fn parse_expr(&mut self) -> Expr {
        let mut left = self.parse_term();
        
        // This handles simple binary ops like `b != 0` or `a > b`
        loop {
            let op = match self.current() {
                Token::NotEqual => Op::NotEqual,
                Token::GreaterThan => Op::GreaterThan,
                Token::Minus => Op::Minus,
                _ => break,
            };
            self.advance();
            let right = self.parse_term();

            // This part is tricky for `a - b -> a`. The `-` is part of an expression
            // that gets assigned. The parser needs to be smarter to handle this properly.
            // For this tiny compiler, we'll parse `a - b` as one expression.
             if let Expr::Variable(lhs) = &left {
                 if let Expr::Variable(rhs) = &right {
                    if self.current() == &Token::Arrow {
                         self.advance(); // consume '->'
                         let target = self.consume_ident();
                         return Expr::FunctionCall(target, vec![lhs.clone(), rhs.clone()]);
                    }
                 }
             }

            left = Expr::BinaryOp(Box::new(left), op, Box::new(right));
        }
        left
    }

    fn parse_term(&mut self) -> Expr {
        match self.current() {
            Token::Integer(val) => {
                let v = *val;
                self.advance();
                Expr::Literal(v)
            }
            Token::Identifier(name) => {
                let n = name.clone();
                self.advance();
                Expr::Variable(n)
            }
            _ => panic!("Unexpected token in expression: {:?}", self.current()),
        }
    }
}


// --- 4. Interpreter ---
// This section executes the logic from the AST.

type Value = i64;
struct Interpreter {
    functions: HashMap<String, Stmt>,
}

impl Interpreter {
    fn new(module: Module) -> Self {
        let mut functions = HashMap::new();
        for func in module.functions {
            if let Stmt::FunctionDecl(name, _, _) = &func {
                functions.insert(name.clone(), func);
            }
        }
        Interpreter { functions }
    }

    fn run(&self) {
        let main_fn = self.functions.get("main").expect("No main function found");
        if let Stmt::FunctionDecl(_, _, body) = main_fn {
            let mut context = HashMap::new();
            for stmt in body {
                self.execute_stmt(stmt, &mut context);
            }
        }
    }

    fn eval_expr(&self, expr: &Expr, context: &HashMap<String, Value>) -> Value {
        match expr {
            Expr::Literal(val) => *val,
            Expr::Variable(name) => *context.get(name).expect(&format!("Variable not found: {}", name)),
            Expr::BinaryOp(left, op, right) => {
                let lval = self.eval_expr(left, context);
                let rval = self.eval_expr(right, context);
                match op {
                    Op::Minus => lval - rval,
                    Op::NotEqual => (lval != rval) as Value,
                    Op::GreaterThan => (lval > rval) as Value,
                }
            }
            Expr::FunctionCall(name, args) => {
                let result = self.call_function(name, args, context);
                result.unwrap_or(0)
            }
        }
    }

    fn execute_stmt(&self, stmt: &Stmt, context: &mut HashMap<String, Value>) -> Option<Value> {
        match stmt {
            Stmt::VarDecl(name, _) => {
                context.insert(name.clone(), 0); // Initialize with 0
            }
            Stmt::Assignment(name, expr) => {
                let value = self.eval_expr(expr, context);
                context.insert(name.clone(), value);
            }
            Stmt::While(cond, body) => {
                while self.eval_expr(cond, context) != 0 {
                    for s in body {
                        if let Some(ret_val) = self.execute_stmt(s, context) {
                            return Some(ret_val);
                        }
                    }
                }
            }
            Stmt::If(cond, then_branch, else_branch) => {
                if self.eval_expr(cond, context) != 0 {
                    for s in then_branch {
                        if let Some(ret_val) = self.execute_stmt(s, context) {
                            return Some(ret_val);
                        }
                    }
                } else if let Some(else_stmts) = else_branch {
                    for s in else_stmts {
                        if let Some(ret_val) = self.execute_stmt(s, context) {
                            return Some(ret_val);
                        }
                    }
                }
            }
            Stmt::Return(expr) => return Some(self.eval_expr(expr, context)),
            Stmt::FunctionDecl(_, _, _) => { /* Handled in new() */ },
            Stmt::IoChain(vars, func_name) => {
                println!("Enter {} integer values separated by space:", vars.len());
                let mut input = String::new();
                io::stdin().read_line(&mut input).unwrap();
                let numbers: Vec<Value> = input.trim().split_whitespace()
                    .map(|s| s.parse().unwrap()).collect();

                let mut args = Vec::new();
                for (i, var_name) in vars.iter().enumerate() {
                    context.insert(var_name.clone(), numbers[i]);
                    args.push(var_name.clone());
                }

                let result = self.call_function(func_name, &args, context).unwrap();
                println!("Result: {}", result);
            }
        }
        None
    }

    fn call_function(&self, name: &str, args: &[String], context: &HashMap<String, Value>) -> Option<Value> {
        let func = self.functions.get(name).expect("Function not found");
        if let Stmt::FunctionDecl(_, params, body) = func {
            let mut func_context = HashMap::new();
            for (i, (param_name, _)) in params.iter().enumerate() {
                let arg_val = context.get(&args[i]).unwrap();
                func_context.insert(param_name.clone(), *arg_val);
            }

            for stmt in body {
                // This is a special case for the Euclidean algorithm's subtractions
                if let Stmt::Assignment(target, expr) = stmt {
                    if let Expr::BinaryOp(l, op, r) = expr {
                        let val = self.eval_expr(&Expr::BinaryOp(l.clone(), op.clone(), r.clone()), &func_context);
                         func_context.insert(target.clone(), val);
                         continue;
                    }
                }
                if let Some(ret_val) = self.execute_stmt(stmt, &mut func_context) {
                    return Some(ret_val);
                }
            }
        }
        None
    }
}


fn main() {
    let source_code = std::fs::read_to_string("euclidean.amb")
        .expect("Could not read euclidean.amb file.");

    // 1. Lexing
    let tokens = lex(&source_code);
    println!("--- Tokens ---");
    for token in &tokens {
        println!("{:?}", token);
    }

    // 2. Parsing
    let mut parser = Parser::new(&tokens);
    let ast = parser.parse();
    println!("\n--- AST ---");
    println!("{:#?}", ast);

    // 3. Interpretation
    println!("\n--- Running Program ---");
    let interpreter = Interpreter::new(ast);
    interpreter.run();
}
