import random


print('welcome to stone paper scissor game')
lst =['r','p','s']
chancesleft = 10
chances = 0
while  chances < chancesleft :
    input1 = input()
    random1 = random.choice(lst)
    