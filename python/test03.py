

'''
words = ['aviral' , '18' , 'bhardwaj']
for w in words :
        print(w, len(w), type(w), )


for i in range(1, 10):
    print(i)

print(sum(range(4)))
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