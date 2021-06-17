#include<stdio.h>
int main() {
      double first, second, c,temp;
      printf("Enter first number: ");
      scanf("%lf", &first);
      printf("Enter second number: ");
      scanf("%lf", &second);

        c= first; //using third variable
      first = second;
      second = temp;
  printf("\nAfter swapping, firstNumber = %.2lf\n",   first);

 printf("After swapping, secondNumber = %.2lf", second);
      return 0;
}
