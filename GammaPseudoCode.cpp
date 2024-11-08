#include <stdexcept>
#include <math.h>
#include <cassert>
#include <iostream>


// 2 electron integral of two primitive Gaussians
double I2e_pG(arma::vec &Ra, arma::vec &Rb, double sigmaA, double sigmaB){
  double U =  ...  // calculate U using equation 3.8 & 3.11
  double V2 =  ... // calculate V2 using equation 3.9
  double Rd= arma::norm(Ra -Rb, 2);
  if(Rd == 0.0)  // if Ra == Rb
    return U * 2.0 * sqrt(V2/M_PI);  // equation 3.15
  double srT = ...  // equation 3.7 sqrt
  double result = ...  // equation 3.14
  return result; 
}

double Eval_2eI_sAO(AO& ao1, AO& ao2){
  
  int len = ao1.get_len();
  assert(ao2.get_len() == len);

  // also get Ra, Rb, alpha_a, alpha_b, da, db...
  
  double gamma = 0.;
  // loop over k, k', l, l' in equation 3.3
  for (size_t k1 = 0; k1 < len; k1++)
  for (size_t k2 = 0; k2 < len; k2++)
  {
    double sigmaA = ...  // equation 3.10
    for (size_t j1 = 0; j1 < len; j1++)
    for (size_t j2 = 0; j2 < len; j2++)
    {
      double sigmaB = ...  // // equation 3.10
      double I2e = I2e_pG(Ra, Rb, sigmaA, sigmaB);
      gamma += ... // equation 3.3
    }
  }
  return gamma;    
}