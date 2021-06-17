#include <stdio.h>
#include <math.h>
int main()
{
    float p, r, t;
    float ci,si;
    printf("entre the pricipal amount :  ");
    scanf("%f", &p);
    printf("entre the intrest rate:  ");
    scanf("%f", &r);
    printf("entre the time period:  ");
    scanf("%f", &t);

    si = (p * r * t) / 100;
    ci = p * (pow((1 + (r/100)),t));
    printf("the value of simple intrest is : %.2f \n",si);
    printf("the value of compound intrest is : %.2f \n",ci);

    return 0;
}

