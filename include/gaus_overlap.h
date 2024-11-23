// Homework 3
// Laura Jones
// Chem 279 - Dr. Mayank Agrawal
// Last revisited: 10/17/2024 
// gaus_overlap.h contains all the necessary information about using contracted
//  gaussians to evaluate the energy of the input hydrocarbon

# pragma once

#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <armadillo>
#include <Eigen/Dense>

using namespace arma;

typedef struct {
    std::vector<rowvec> exp; //alpha values
    std::vector<rowvec> coef;    //coefficients(3)
    std::vector<double> e_affin;    //electron affinity
} basis;

typedef struct {
    std::vector<double> norm;    //normalization value
    int id; //num protons
    int val_electrons;  //...valence electrons (z)
    int bonding_param;  //bonding parameters as outlined in table 1.1
    std::vector<basis> func; //basis functions(N)
    //int n_electrons;  //(n)
    rowvec coords;  //coordinates
    mat ang_moment;//angular momentum values(3): l,m,n
} atom;


class conGaussian
{
    private:
        std::vector<atom> _atoms; //atomic_numbers
        int _a; //number of atoms of type 1
        int _b; //number of atoms of tpye 2
        int _num_basis; //N: number of basis functions
        int _val_orbitals; //n: number of occupied valence orbitals
        
    public:
    //getter functions
        std::vector<atom> get_atoms() const {return _atoms;};
        int get_num_basis() const {return _num_basis;};
        int get_val_orbitals() const {return _val_orbitals;};
        mat get_ang_momentum(int i) const {return _atoms[i].ang_moment;};
    
    //initializing functions
        void calc_members();
        void set_exps_coeffs();
        void read_input(std::string);

    //evaluating functions
        mat eval_overlap();
        mat eval_prob(mat&, mat&);
        mat eval_fock(mat&, mat&, mat&);
        mat eval_core(mat&);
        double eval_scf();
        double calc_total_energy(mat&,mat&,mat&,mat&,mat&);
        double i2e_pg(rowvec&, rowvec&, double, double);
        double eval_2ei_sAO(atom, atom);
        int factorial(int);
        int double_factorial(int);
        int combinations(int, int);
        double center_product(double, double, double, double);
        double deriv_center_product(double, double);
        double calc_1d_overlap(double, double, double, double, double, double);
        double eval_gradient();
        double calc_x(int, int);
        double calc_y(int, int);
        double calc_deriv_overlap(int, int, int);
        double calc_nuc_repulsion(int, int);
        double calc_deriv_gamma(int, int, int);
        double deriv_i2e_pg(double, double, double, double);

    //constructor/destructor
        conGaussian(std::string);
        ~conGaussian();

};
