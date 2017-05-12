#ifndef MAIN_H
#define MAIN_H

#include "math.h"

double CONST_E = 2.718281828459045;

double phi = 12;

double _gamma = 3.7;

double Sigmoid(double in)
{
  return  1 / (1 + pow(CONST_E, -in));
}

double SigClamp(double in)
{
  return  (2*(1 / (1 + pow(CONST_E, -in)))) - 1;
}

double FuncNeural(double sum)
{
//  return tanhf(sum);
//  return ((sum/powf(1+(sum*sum), 0.5))+1)/2;
  return Sigmoid(sum);//
  return ((sum/(1+fabs(sum)))+1)/2;

//  return logf(1+powf(CONST_E, sum));  // Softplus Function
//  return erff(sum);
  return Sigmoid(sum);
  if(sum<0) return 0.00001*sum; // Liner-Rectifier
  return log(sum);
}

double FuncDerivative(double a)
{/*
  return 1/powf(fabs(a)+1, 2); //--> use in combination of sigmoid ->

  return 1/powf(2, fabs(a)+1); //--> use in combination of sigmoid -> 88%

  double b = FuncNeural(a);
  return b*(1 - b);
*/
  double b = tanh(a);
  return (1 - b*b);

  return Sigmoid(a)*(1-Sigmoid(a));

  if(a<0) return 0.00001; // Liner-Rectifier
  return 1/a;
}

double CostFunc(double a, double y)
{
  return y*log(a + 0.00001) + (1-y)*log(1.00001-a);
}

double CostFuncDerivative(double a, double y)
{
  return a-y;//(y/(a+1.00001)) - ((1-y)/(1.00001-a));
}


double Func(double x)
{
//  return Sigmoid(x);
  return tanh(x);//Sigmoid(x);
}

#endif
