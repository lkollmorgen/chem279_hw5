#include <iostream>
#include <fstream>

#include "gaus_overlap.h"

int main(void) {
//test to make sure input file is read correctly
    //input file
    std::string file_path = "../sample_input/H2.txt";
    //std::string file_path = "../sample_input/C2H2.txt";
    conGaussian h2(file_path);

    std::vector<atom> atoms = h2.get_atoms();
    int basis = h2.get_num_basis();
    int elec = h2.get_val_orbitals();
    
    for(int i = 0; i < atoms.size(); i++) {
        std::cout << "id: " << atoms[i].id << std::endl;
        std::cout << "coords: " << std::endl;
        atoms[i].coords.print();
        std::cout << "valence electrons: " << atoms[i].val_electrons << std::endl;
    }

    std::cout << "number of basis functions and number of electrons: " << std::endl;
    std::cout << basis << ", " << elec << std::endl;

    
//test to make sure that angular momentum and the overlap matrix is properly evaluated
    int i = 0;
    mat ang_momentum = h2.get_ang_momentum(i);
    std::cout << "angular momentum values for "<< i << ": " << std::endl;
    ang_momentum.print();

    // mat o_mat = h2.eval_overlap();
    // std::cout << "overlap mat: " << std::endl;
    // o_mat.print();

    double e = h2.eval_scf();
    std::cout << "Electron energy is: " << e << std::endl;
    //f_mat.print();

    return 0;
}