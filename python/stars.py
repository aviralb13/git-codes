 

print('enter no of rows')
one = int(input())

print('enter a number 1 or 0')
two = int(input())
new = bool(two)

if new == True:
    for i in range(0 ,one+1):
        print("*"*int(i))



elif new == False:
    for i in range(one,0,-1):
        print("*"*int(i))

 
 


'''
class csstudent:

    stream = 'cse'

    def __init__(self, roll) :
     self.roll = roll

     a = csstudent(101)
     b = csstudent(102)
     
     
     print(a.stream)
     print(b.stream)
     print(a.roll)

     print(csstudent.stream)
     '''




# def starfunct(i):
#     for x in range (i):
    
#       print('*'*x)

# starfunct(500)

