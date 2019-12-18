---
title: Asterisk in Python
date: 2019-12-17 7：49
categories: [Technical Tools, Python programming]
tags: [Python]
---

The use of `*` in python is related to the way of passing parameters to a function. First, in python, we have three types of parameters, *Positional-or-Keyword Arguments*, *Positional-Only Parameters* and *Keyword-Only Arguments*. The way most frequently used is *Positional-or-Keyword Arguments*, so we can either specify the name of argument or not when passing arguments in order. `*` and `**` are especially useful when the function has a varying/arbitrary number of parameters. The difference between `*` and `**` is:
- `*args` = list of argument, positional arguments
- `**kwargs` = a dictionary of arguments, whose keys become separate keyword arguments and the values. become values of these arguments

## use `*` and `**` when defining function parameter

**a) When defining a function, `*args` can pack an arbitrary number of positional arguments.**
```python
def tuple_args(*args):
    ''' all arguments form a tuple '''
    for element in args:
        print(element)
    return '-'*10 + '{0}'.format(args[0]*5) + '-'*10
```
```python
tuple_args(1)
```
    1
    '----------5----------'

```python
tuple_args(1, 2, 3)
```

    1
    2
    3
    '----------5----------'

```python
tuple_args([1, 2, 3])
```

    [1, 2, 3]
    '----------[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]----------'
    

**b) `**kwargs` can pack an arbitrary number of keyword arguments.**

```python
def dict_args(**kwargs):
    ''' all arguemnts in (keyword, value) pairs form a dictionary '''
    for (key, value) in kwargs.items():
        print(key, ':', value)
        
dict_args(one = 1, two = 2)
```
    
    one : 1
    two : 2

**c) mix the way of passing arguments**

Both idioms can be mixed with normal arguments to allow a set of fixed and some variable arguments:

```python
def mix_args(*args, arg1, **kwargs):
    print('arg1 is', arg1)
    for element in args:
        print (element)
    for key, value in kwargs.items():
        print (key, ':', value)

mix_args(1, 2, arg1 = 3, five = 5, six = 6)
```

    arg1 is 3
    1
    2
    five : 5
    six : 6
    
**Note:** `arg1` must be explicitly specified so that we tell the function to pack the arguments after `arg1` into `kwargs` and packge the arguments before `arg1` into `args`. Otherwise, like the following call, it will cause an error.

```python
mix_args(1, 2, 3, five = 5, six = 6)
```

    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
    <ipython-input-149-89543e4f8596> in <module>
    ----> 1 mix_args(1, 2, 3, five = 5, six = 6)

    TypeError: mix_args() missing 1 required keyword-only argument: 'arg1'


But the more natural way to mix them is `def mix_args(arg1, *args, **kwargs):` so that the first argument can be passed by position.

## use `*` and `**` calling a function

When calling a function, the functionality of `*` and `**` is opposite to that in defining a function; that is, `*` **unpacks** a list/tuple (any iterable data structure) and `**` **unpacks** a dictionary  into several arguments.

```python
def many_argments(arg1, arg2, arg3, arg4, arg5, arg6):
    print('arg1:', arg1, ', arg2:', arg2, ', arg3:', arg3, ', arg4:', arg4,  ', arg5:', arg5, ', arg6:', arg6)
```
**a) `*` unpacks a list/tuple**
```python
many_argments(1, *[2, 3, 4], arg5 = 5, arg6 = 6) # or *(2, 3, 4)
```
    arg1: 1 , arg2: 2 , arg3: 3 , arg4: 4 , arg5: 5 , arg6: 6
    
**b) `**` unpacks a dictionary**

```python 
many_argments(1, *(2, 3, 4), **{'arg5': 5, 'arg6': 6})
### or 
values = list(range(2, 7))
keys = ['arg'+str(i) for i in values]
dict_arg = dict(zip(keys, values))
many_argments(arg1 = 1, **dict_arg)
```
    arg1: 1 , arg2: 2 , arg3: 3 , arg4: 4 , arg5: 5 , arg6: 6

**c) position arguments can NOT after keyword arguments**

```python
many_argments(arg1 = 1, *(2, 3, 4), **{'arg5': 5, 'arg6': 6})
```
    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
    <ipython-input-202-c1452a56ac4f> in <module>
    ----> 1 many_argments(arg1 = 1, *(2, 3, 4), **{'arg5': 5, 'arg6': 6})

    TypeError: many_argments() got multiple values for argument 'arg1'


## unpack a list

```python
first, *middle, last = list(range(1, 6))
print('first is', first, '-'*3,  'middle is', middle, '-'*3, 'last is', last)
```
    first is 1 --- middle is [2, 3, 4] --- last is 5

## References

1. [`*` and `**` as parameters when defining a function](https://stackoverflow.com/questions/36901/what-does-double-star-asterisk-and-star-asterisk-do-for-parameters)
2. [`*` in function call](https://stackoverflow.com/questions/5239856/asterisk-in-function-call)
3. [Python document of `*` and `**`](https://docs.python.org/dev/tutorial/controlflow.html#id2)