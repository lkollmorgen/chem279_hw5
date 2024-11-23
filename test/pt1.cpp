#include <iostream>
#include <fstream>

#include "gaus_overlap.h"

int main(void) {
//test to make sure input file is read correctly
    //input file
    std::string file_path = "../sample_input/H2.txt";
    //std::string file_path = "../sample_input/C2H2.txt";
    conGaussian h2(file_path);

    double x = h2.eval_gradient();
    std::cout << "energy: " << x << std::endl;
    //f_mat.print();

    return 0;
}