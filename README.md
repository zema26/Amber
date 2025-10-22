# Amber
Reversed Syntax Programming Language


Changing syntax of programming languages:

1. Reversed assignment operator  a + b -> a

Allows assignment to flow from left to right like we write

in more natural direction

2. Tagging keywords  {while} ... {/while}

This allows to easy introduce new keywords into language

and to easy find mistakes in control flow of programs

It offer flexibility of markup languages and rigor of programming ones

3. Reversed function declaration:  {fun} int a, b -> f int  and  call:  a, b -> f
 
These changes allow to seamlessly combine input and output in one line


Program Euclidean

Comments start with symbol "!"         

{module} Euclidean 		! module itself

     {fun} int a, int b -> Euclid int  ! function with type int
							
	{while} b!= 0
		{if} a > b
			a - b -> a
		{else}
			b - a -> b
		{/else}
		{/if}
	{/while}

	{return} a {/return}

    {/fun}

  
  {fun}	string args[] -> main int	! main function

	int a, b

         in -> a, b -> Euclid -> out

        ! combined input and output and function call

	! here is introduced streamlined logic and
	
	! seamless input to output syntax

  {/fun}

{/module} 

