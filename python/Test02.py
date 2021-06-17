


class student:
    school = "abc international"

    def __init__(self, m1, m2, m3):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

    def avg(self):
        return (self.m1 + self.m2 + self.m3)/3

    @staticmethod   
    def info():
        print("this is a class")


s1 = student(1,2,3)
s2 = student(4,5,6)
s3 = student(7,8,9)


print(s1.avg()) 
print(s2.avg())
print(s1.m1)

student.info()



