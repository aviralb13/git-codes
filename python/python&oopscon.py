"""
#print
print("hello world")

# comments
'''
this is 
multi line 
comment
'''
# this is single line comment


#escape chracters 
'''
\n 	Inserts a new line in the text at the point
\\	Inserts a backslash character in the text at the point
\"	Inserts a double quote character in the text at that point
\'	Inserts a single quote character in the text at that point
\t	Inserts a tab in the text at that point
\f	Inserts a form feed ln the text at that point
\r	Inserts a carriage return in the text at that point
\b	Inserts a backspace in the text at that point
'''

# leaving line (\n)
print("hi \nmy name is aviral")

#datatypes 
'''
string
variable
float
boolean
'''

a = 10
b = 10.0
c = "aviral"


print(type(a)) #tell the datatype 
print(type(b)) #tell the datatype 
print(type(c)) #tell the datatype 

print (a + b) #we can add numbers

d = " bhardwaj"
print(c + d)  #same goes for strings

#typecasting

aa = "5"
ab = 5
ac = int(aa)
print(ac+ab)

#input

print ("enter your age")
age = input()           #takes input from user
print('my age is' ,age)

print("Enter first number")
n1 = input()
print("Enter second number")
n2 = input()
print("Sum of these two numbers is", int(n1) + int(n2))


# strings slicing and operations 

'''
string.endswith()   : This function allows the user to check whether a given string ends with passed argument or not. It returns True or False.
string.count()      : This function counts the total no. of occurrence of any character in the string. It takes the character whose occurrence you want to find as an argument.
string.upper()      : It returns the copy of the string converted to the uppercase.
string.lower()      : It returns the copy of the string converted to lower case.
string.find()       : This function finds any given character or word in the entire string. It returns the index of first character from that word.
string.capitalize() : This function capitalizes the first character of any string
'''

mystr = "my name is aviral"
print (len(mystr))#gives length
print(mystr[0:3]) #slice the string
print(mystr[0:4:2])

#list ,dictionary and tuple
list1 = ['mac','iphone','ipad']
print(list1)

tupl1 = ('watch', 'mobile' ,'laptop')
print(tupl1)

dict1 = {'key':'value','apple':'iphone'}
print(dict1)
'''
l1.sort()           : sort
l1.reverse()        : reverse all elements
list1.append(7)     :add value is last                       # This will add 7 in the last of list 
list1.insert(3,8)   :insert value at particular position     # This will add 8 at 3 index in list
list1.remove(1)     :removes value                            #This will remove 1 from the list 
list1.pop(2)                                                  #This will delete and return index 2 value.

'''

'''
d.copy(): copy dictionary
d.update({"key":"value"})    : updates dictionary
print(d.items())             : prints all the items in dictionary
'''


#swapping of two numbers
a = 10
b = 5
print (a,b)
a,b =b,a
print (a,b)

#sets

s = set()
l = [1, 2, 3, 4]


list_into_set = set(l)
print(list_into_set)
'''
s.add()     : add value to set
s.remove()  : remove value to set
'''



#if else and elif

a = 20
b = 10

if (a>b):
    print("a is > b")
elif (a<b):
    print("a is < b")    
else :
    print("a=b")

#for loops 
lst = ['1','2','3','4']
for i in lst:
    print('the numbers are', i)

dct ={'mac':'macbook','phone':'iphone','tab':'ipad' }
for key,value in dct.items():
    print (key,"is" ,value)



#while loops
while(True):
    inp = int(input("Enter a Number\n"))
    if inp>100:
        print("Congrats you have entered a number greater than 100\n")
        break                        #breaks the loop when condition get satisfied
    else:
        print("Try again!\n")
        continue                     #executed till the condition is not satisfied


#operators

'''
Arithmetic Operators:
Basic mathematical operations such as addition, multiplication, subtraction, division, etc. are performed with the help of arithmetic Operations. It contains nearly all operations that we can perform with the help of a calculator. Symbols for such operators include *, /, %, -, //, etc.    

Assignment Operators:
The assignment operator is used to assign values to a variable. In some cases, we have to assign a variable’s value to another variable, in such cases the value of the right operand is assigned to the left operand. One of the basic signs from which we can recognize an assignment operator is that it must have an equal-to(=) sign. Some commonly used assignment operators include +=, -=, /=, etc.

Comparison Operators:
They are also known as relational operators. They compare the values on either side of the operator and decide the relation among them. Commonly used comparison operators include ==, >, < , >=,etc.

Logical Operators:
Logical operators perform logical AND, OR and NOT, operations. They are usually used in conditional statements to join multiple conditions. AND, OR and NOT keywords are used to perform logical operations.

Identity Operations:
Identity operator checks if two operands share the same identity or not, which means that they share the same location in memory or different. “is” and “is not” are the keywords used for identity operands.

Membership Operands:
 Membership operand checks if the value or variable is a part of a sequence or not. The sequence could be string, list, tuple, etc. “in” and “not in” are keywords used for membership operands.

Bitwise Operand:
Bitwise operands are used to perform bit by bit operation on binary numbers. First, we have to change the format of the number from decimal to binary and then compare them using AND, OR, XOR, NOT, etc.
'''

print("Arithmetic Operators")
print("5 + 6 is ", 5+6)

print("Assignment Operators")
x = 5
print(x)
x +=7  
print(x)

print("Comparison Operators")
a = 10 
b = 10
print(a==b)

print("Membership Operators")
list = [3, 3,2, 2,39, 33, 35,32]
print(324 not in list)

print("Identity Operators")
print(5 is not 5)


#short hand if else

a = 10
b = 11

print("a and b are equal") if a==b else print("a and b are not equal")


if a==b:
    print("a and b are equal")
else:
    print("a and b are not equal") # we can write in any format we want

"""
'''
# functions
def function1 (a,b):
    """this is a docstring
    and this function is used for addition of two numbers"""   
    print (a+b)

function1(1,2)
print(function1.__doc__) #calling docstring

'''
"""

#try and expect
print("Enter num 1")
num1 = input()
print("Enter num 2")
num2 = input()
try:
    print("The sum of these two numbers is",
          int(num1)+int(num2))
except Exception as e:
    print(e)



print("This line is very important")


# file opening
'''
r : r mode opens a file for read-only. We do not have permission to update or change any data in this mode.
w : w mode does not concern itself with what is present in the file. It just opens a file for writing and if there is already some data present in the file, it overwrites it.
x : x is used to create a new file. It does not work for an already existing file, as in such cases the operation fails.
a : a stands for append, which means to add something to the end of the file. It does exactly the same. It just adds the data we like in write(w) mode but instead of overwriting it just adds it to the end of the file. It also does not have the permission of reading the file.
t : t mode is used to open our file in text mode and only proper text files can be opened by it. It deals with the file data as a string.
b : b stands for binary and this mode can only open the binary files, that are read in bytes. The binary files include images, documents, or all other files that require specific software to be read.
+ : In plus mode, we can read and write a file simultaneously. The mode is mostly used in cases where we want to update our file.
'''
f=open("Test01.py", "w") # way of opening fie
f.close()

#seek and tell 

'''
Syntax:  file_pointer .seek(offset, whence).
Offset:   In seek() function, offset is required. Offset is the position of the read/write pointer within the file.
Whence: This is optional. It defines the point of reference. The default is 0, which means absolute file positioning.
 

Value  Meaning

0     Absolute file positioning. The position is relative to the start of the file. The first argument cannot be negative.

1     Seek relative to the current position. The first argument can be negative to move backward or positive to move forward

2     Seek relative to the file’s end. The first argument must be negative.
'''

#global variables

a = 10 #global variable
def tfunc():
    a=8 #local variable
    print("tfunca",a)
    def tfunc2():
        global a #changes the value of global variable
        a= 12
        print("before" , a)
    tfunc2()
    print("after" , a)   #a= 8 because a have been found in tfunc


tfunc()
print("global", a)


#recursions and iterations
print("iterations") #Iteration runs a block of code again and again, depending on a user-defined condition
def iter(n):
    fact = 1
    for i in range(n):
        fact = fact * (i+1)

    return fact

print(iter(5))



print("recursion") #A recursive function terminates to get closer to its base case or base condition
def rec(n):
    if n ==1:
        return 1
    else :
        return n * rec(n-1)
    
print(rec(5))




#lambda or anonymus function


def sum1(a, b): 
    return a + b
print(sum1(15, 10))

add = lambda x,y: x + y
print(add(5,10))


#external modules
import random
random_number = random.randint(0, 1)
print(random_number)
rand = random.random() *100
print(rand)
lst = ["Star Plus", "DD1", "Aaj Tak", "CodeWithHarry"]
choice = random.choice(lst)
print(choice)


#f strings

a = "my name " 
b = "is"
print(f"{a} {b} aviral")


#args and kwargs
def funcarg(normal, *args, **kwargs):
    print(normal)
    for item in args:  # It is used to unpack an argumen
        print(item)
    for key, value in kwargs.items():   #It passes the data to the argument in the form of a dictionary
        print(f"name is {key} and {value}")
    



a =  "company"
b =  ["aviral","bunny","teddy"]
c =  {"aviral":"ceo","bunny":"president","teddy":"director"}


funcarg(a, *b, **c)
print(funcarg)

#join function
list1=['1','2','3','4','5','6','7','8',]


a = ",".join(list1)
print(a, "and 9")


#map filter and reduce

list2 = [1,2,3,4,5,6,7,8]
a = list(map(lambda x: x*2, list2))#A map function executes certain instructions or functionality provided to it on every item of an iterable.
print(a)  #syntax map(function, iterable) 

b = list(filter(lambda x: x>2, list2))# A filter function in Python tests a specific user-defined condition for a function and returns an iterable for the elements and values that satisfy the condition or, in other words, return true.
print(b) #syntax filter(function, iterable)
from functools import reduce # asit is not present by default so we need to import it

c = reduce(lambda x,y: x*y , list2)#Reduce functions apply a function to every item of an iterable and gives back a single value as a resultant
print(c) #reduce(function, iterable)

#decorators 

def dec(fun):
    print("dec is running")
    print(f"now we will run {fun.__name__}")
    fun()
    print (f"{fun.__name__} executed successfully")


@dec
def func():
    print("you are in func")
    return None
"""
#--------------------------------------------------OOPS-------------------------------------------------------------------------------------
"""
class student:
    leaves = 10
    pass

sam = student()
tam = student()
dan = student()

sam.name = "sam"
tam.name = "tam"
dan.name = "dan"

sam.std = 10
tam.std = 8
dan.std = 9

tam.leaves = 100 # we can change leaves


print(sam.name)
print(student.__dict__) 
student.leaves = 101# we can chnage class leaves
print(student.__dict__) # the dictionary will be updated to 101


#self and __init__(constructors)

class student1:
    leaves = 10
    def __init__(self, myname, myage, mystd):
        self.name = myname
        self.age = myage
        self.std = mystd
    def print_details(self):
        return f"name is {self.name} ,age is {self.age} and std is {self.std} "
    pass

sam = student1("sam", 18, 10)
tam = student1("tam", 19, 10)
dan = student1("dan", 20, 10)

print(sam.name,sam.age,sam.std)
print(sam.print_details())

# class method

class student2:
    leaves = 10
    def __init__(self, myname, myage, mystd):
        self.name = myname
        self.age = myage
        self.std = mystd

    def print_details(self):
        return f"name is {self.name} ,age is {self.age} and std is {self.std} "

    @classmethod
    def newleaves(cls,newleaves):
        cls.leaves = newleaves
    pass

sam = student2("sam", 18, 10)
tam = student2("tam", 19, 10)
dan = student2("dan", 20, 10)

sam.newleaves(100)
print (sam.leaves)

#staticmethod
class student2:
    leaves = 10
    def __init__(self, myname, myage, mystd):
        self.name = myname
        self.age = myage
        self.std = mystd
    
    def print_details(self):
        return f"name is {self.name} ,age is {self.age} and std is {self.std} "
    @classmethod
    def dashsplit(cls, string):
        params = string.split("-")
        return cls(params[0], params[1], params[2])# or no need of 2 lines just this return cls(*string.split("-"))
    @staticmethod
    def my_name(string):
        print(f"my {string} is ")
        return


sam = student2("sam", 18, 10)
tam = student2("tam", 19, 10)
dan = student2.dashsplit("dan-20-10")

student2.my_name("tam")

#abstraction:Abstraction refers to hiding the unnecessary details so that the focus could be on the whole product instead of parts of the project separately.
#encapusulation:Encapsulation means hiding under layers. When working with classes and handling sensitive data, global access to all the variables used in the program is not secure.


#inheritence
class parentclass:
    def parent(self):
        print("you are in parent class")


class childclass(parentclass):#single level inheritence
    def child(self):
        print("you are in child class")

print("a")
a = childclass()
a.parent()
a.child()

class grandchild(childclass):#multilevel inheritence
    def grandchild(self):
        print("you are in grand class")

print("b")

b = grandchild()
b.parent()
b.child()
b.grandchild()

class children(grandchild, childclass, parentclass):# order matters # multiple inheritence
    def children(self):
        print("you are in children class")
print("c")
c = children()
c.parent()
c.child()
c.grandchild()
c.children()



# public private and protected access
class student3:
    
    _leaves = 10 #protected
    __fees = 10000 #private
    

    def __init__(self, myname, myage, mystd):
        self.__name = myname
        self._age = myage
        self.std = mystd
    
    def print_details(self):
        return f"name is {self.__name} ,age is {self._age} and std is {self.std} "
    
    

sam = student3("sam", 18, 10)
tam = student3("tam", 19, 10)
dan = student3("dan", 20, 10)


print(sam._student3__fees)

print(sam._student3__name)# how to view private
print(sam._age) # how to view protected


class info (student3):
    pass

print(sam._student3__name) # how to view private in diffrent classes
print(sam._age)# how to view protected in diffrent classes


#polymorphism : Polymorphism means to exist in different states. The same object or thing changing its state from one form to another is known as polymorphic

# dunder methods and operator overloading


class student4:
    leaves = 10
    def __init__(self, myname, myage, mystd, salary):
        self.name = myname
        self.age = myage
        self.std = mystd
        self.salary = salary
    def print_details(self):
        return f"name is {self.name} ,age is {self.age} and std is {self.std} "
    

    def __add__(self, other):
        return self.salary+ other.salary

    def __repr__(self): 
        return f"student4('{self.name}', {self.salary})"

    def __str__(self): # str priority is high
        return f"The Name is {self.name}. Salary is {self.salary} "
    

    
    

sam = student4("sam", 18, 10,1000)
tam = student4("tam", 19, 10,2000)
dan = student4("dan", 20, 10,4000)

print (sam + tam)

print(str(dan))

#abstract method 
from abc import ABC, abstractmethod # we have to import it to use



class shape(ABC):
    @abstractmethod
    def area(self):
        return 0



    pass


class rectangle(shape):

    def __init__(self, length, width):
        self.length = length
        self.width = width
    def area(self, length, width):
        return self.length * self.width


rect = rectangle(4,5)
print(rect.area(4,5))
"""