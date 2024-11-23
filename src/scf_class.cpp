#include <iostream>
#include <sstream>
#include <cmath>
#include <armadillo>
#include <Eigen/Dense>
#include <map>
#include <unordered_map>
#include <stdexcept>    //gamma calculations
#include <math.h>
#include <cassert>

#include "gaus_overlap.h"

using namespace arma;


conGaussian::conGaussian(std::string file) {
    conGaussian::read_input(file);
    calc_members();
    set_exps_coeffs();
}

conGaussian::~conGaussian(){}

void conGaussian::calc_members() {
    std::unordered_map<int, int> id_count = {{1,0},{6,0}};
    for(const auto& atom: _atoms) {
        id_count[atom.id]++;
    }
    _a = id_count[6];
    _b = id_count[1];

    _num_basis = (4 * _a) + _b;
    double valence_e = (2 * _a) + (_b / 2);

    if (valence_e != static_cast<int>(valence_e)) {
        std::cout << "Number of valence electrons is not an integer!!!" << std::endl;
        exit(0);
    }
    else _val_orbitals = valence_e;
}

void conGaussian::set_exps_coeffs() {
    rowvec exp_1s = {3.42525091, 0.62391373, 0.16885540};
    rowvec exp_2s = {2.94124940, 0.68348310, 0.22228990};
    rowvec exp_2p = {2.94124940, 0.68348310, 0.22228990};

    rowvec con_1s = {0.15432897, 0.53532814, 0.44463454};
    rowvec con_2s = {-0.09996723, 0.39951283, 0.70011547}; 
    rowvec con_2p = {0.15591627, 0.60768372, 0.39195739};

    //table 1.1 parameters defining ionization energy + electron affinity
    std::map<int, double> electron_affin_s = {{1,7.176},{6,14.051},{7,19.316},
                                              {8,25.390}, {9,32.272}};
    std::map<int, double> electron_affin_p = {{6,5.572},{7,7.275},
                                              {8,9.111}, {9,11.080}};

    //angular momentum values
    mat L0 = { 0,0,0};
    mat L1 = {{1,0,0},
              {0,1,0},
              {0,0,1}};


    for(int i = 0; i < _atoms.size(); i++) {
        basis b;
        int id = _atoms[i].id;
        if(id == 1) {
            b.exp.push_back(exp_1s);
            b.coef.push_back(con_1s);
            _atoms[i].ang_moment = L0;
            b.e_affin.push_back(electron_affin_s[1]);
        }
        else {
            b.exp.push_back(exp_2s);
            b.exp.push_back(exp_2p);
            b.coef.push_back(con_2s);
            b.coef.push_back(con_2p);
            _atoms[i].ang_moment = L1;
            b.e_affin.push_back(electron_affin_s[id]);  //s orbital e affinity
            for(int j = 0; j < 3; j++) {    //3 p subshells
                b.e_affin.push_back(electron_affin_p[id]);
            }
        }
        _atoms[i].func.push_back(b);
    }
}

void conGaussian::read_input(std::string file) {
    std::fstream data_file;
    std::string line;
    std::string s;
    int num_atoms;

    //valence electrons for each atomic number
    std::map<int, int> val_map = {{1,1}, {6,4}, {7,5}, {8,6}, {9,7}};
    std::map<int, int> bonding_params = {{1,9}, {6,21}, {7,25}, {8,31}, {9,39}};

    data_file.open(file);
    getline(data_file,line,'\n');
    std::stringstream ss(line);
    getline(ss,s,' '); //select first value from .txt
    num_atoms = std::stoi(s);

    for(int i = 0; i < num_atoms; i++) {
        getline(data_file,line, '\n');
        ss.str(line);
        ss.clear();

        atom a;
        getline(ss,s, ' ');
        a.id = std::stoi(s);
        //conversion factor from eV to a.u. is 27.211 ev/a.u.
        a.val_electrons = (val_map[a.id]);// / 27.211); //assign valence electrons
        a.bonding_param = (bonding_params[a.id]);// / 27.211);
        getline(ss,s, ' '); //skip atom type
        
        rowvec c(3);
        for(int j = 0; j < 3; j++) {
            c(j) = std::stod(s);
            getline(ss, s, ' ');
        }
        a.coords = c;
        _atoms.push_back(a);
    }
}

/* =======================Evaluating Functions============================ */

mat conGaussian::eval_overlap() {
    mat overlap_mat = zeros(_num_basis,_num_basis);

    for(int i = 0; i < _num_basis; i++) {
        for(int j = 0; j < _num_basis; j++) {
            double Sab = 0.0;

            double s_p;
            int exponent_j;
            for(int exponent_i = 0; exponent_i < 3; exponent_i++) {
                for(int exponent_j = 0; exponent_j < 3; exponent_j++) { 

                    double alpha = _atoms[i].func[0].exp[0](exponent_i);
                    double beta = _atoms[j].func[0].exp[0](exponent_j);

                    s_p = 1;
                    //double s_p = 1;
                    for (int dim = 0; dim < 3; dim++) {
                        double xa = _atoms[i].coords[dim];
                        double xb = _atoms[j].coords[dim];

                        double ang_a = _atoms[i].ang_moment(0, dim);
                        double ang_b = _atoms[j].ang_moment(0, dim);

                        s_p *=  calc_1d_overlap(xa, xb, ang_a, ang_b, alpha, beta);
                        // std::cout << "xa, xb: " << xa << ", " << xb << " ang_a, ang_b: " << ang_a << ", " << ang_b;
                        // std::cout << " alpha, beta: " << alpha << ", " << beta <<  " s_p: " << s_p << std::endl;
                    }
                    if(exponent_i == exponent_j) Sab += s_p;
                }
                if(i == j)_atoms[i].norm.push_back(1.0 / sqrt(Sab));
            }
            overlap_mat(i,j) = Sab;
        }
    }

    // for(int i = 0; i < _num_basis; i++) {
    //     std::cout << " Atom " << i << ": ";
    //     for(int n = 0; n < _atoms[i].norm.size(); n++) {
    //         std::cout <<  _atoms[i].norm[n] << " ";
    //     }
    //     std::cout << "|||";
    // }
    // std::cout << std::endl;

    for(int i = 0; i < _num_basis; i++) {
        for(int j = 0; j < _num_basis; j++) {
            double c_gaus = 0.0;
            for(int k = 0; k < 3; k++) {
                for(int l = 0; l < 3; l++) {
                    double coef_a = _atoms[i].func[0].coef[0](k);
                    double coef_b = _atoms[j].func[0].coef[0](l);

                    double norm_a = _atoms[i].norm[k];
                    double norm_b = _atoms[j].norm[l];

                    double S = overlap_mat(i,j);
                    // std::cout << "norm a, b: " << norm_a << ", " << norm_b << std::endl;
                    // std::cout << "coef a, b: " << coef_a << ", " << coef_b << std::endl << std::endl;
                    c_gaus += coef_a * coef_b * norm_a * norm_b * S;
                }
            }
            overlap_mat(i,j) = c_gaus;
        }
    }
    return overlap_mat;
}

mat conGaussian::eval_prob(mat& p, mat& c) {
    mat p_update(_num_basis, _num_basis);
    int occ_a = _atoms[0].val_electrons;
    int occ_b = _atoms[1].val_electrons;

    for(int i = 0; i < occ_a + 1; i++) {
        for(int j = 0; j < occ_b + 1; j++) {
            p_update(i,j) = c(i,j) * c(i,j); 
        }
    }
    return p_update;
}

mat conGaussian::eval_fock(mat& prob_a, mat& prob_b, mat& overlap) {
    mat gamma_mat(_num_basis, _num_basis);
    mat f_mat(_num_basis, _num_basis);

    //eval gamma matrix
    for(int i = 0; i < _num_basis; i++) {
        for(int j = 0; j < _num_basis; j++) {
            double gamma = eval_2ei_sAO(_atoms[i],_atoms[j]);
            //in this implementation, I am only using hydrogen's s-orbital,
            //  so i don't pass in atomic orbitals into eval_2ei_sAO
            gamma_mat(i,j) = gamma;
        }
    }
    double gamma_aa = gamma_mat(0,0);
    double gamma_ab = gamma_mat(0,1);
    //eval fock matrix
    for(int i = 0; i < _num_basis; i++) {
        for(int j = 0; j < _num_basis; j++) {
            
            double val = 0.0;
            if(i == j) {
                val += - _atoms[i].func[0].e_affin[0];

                val += (((prob_a(i,j) + prob_b(i,j)) - _atoms[i].val_electrons) - \
                            (prob_a(i,j) - .5)) * gamma_aa;

                val += ((prob_a(0,1) + prob_b(0,1)) - _atoms[j].val_electrons) * gamma_ab;
            }
            else {
                val = (.5 * (-_atoms[i].bonding_param + - _atoms[j].bonding_param) * overlap(i,j)) \
                       - (prob_a(i,j) * gamma_ab); 
            }
            f_mat(i,j) = val;
        }
    }
    std::cout << "gamma mat: " << std::endl;
    gamma_mat.print();
    std::cout << std::endl;
    return f_mat;
}

mat conGaussian::eval_core(mat& overlap) {
    mat gamma_mat(_num_basis, _num_basis);
    mat hc_mat(_num_basis, _num_basis);

    //eval gamma matrix
    for(int i = 0; i < _num_basis; i++) {
        for(int j = 0; j < _num_basis; j++) {
            double gamma = eval_2ei_sAO(_atoms[i],_atoms[j]);
            gamma_mat(i,j) = gamma;
        }
    }
    double gamma_aa = gamma_mat(0,0);
    double gamma_ab = gamma_mat(0,1);
    //eval fock matrix
    for(int i = 0; i < _num_basis; i++) {
        for(int j = 0; j < _num_basis; j++) {
            
            double val = 0.0;
            if(i == j) {
                val += - _atoms[i].func[0].e_affin[0];

                val -= ((_atoms[i].val_electrons - .5) * gamma_aa);

                val -= _atoms[j].val_electrons * gamma_ab;
            }
            else {
                val = (.5 * (-_atoms[i].bonding_param + - _atoms[j].bonding_param) * overlap(i,j)); 
            }
            hc_mat(i,j) = val;
        }
    }
    return hc_mat;
}

double conGaussian::calc_total_energy(mat& hc, mat& fa, mat& fb, mat& prob_a, mat& prob_b) {

    mat emat(_num_basis, _num_basis, fill::zeros);
    for(int i = 0; i < _num_basis; i++) {
        for(int j = 0; j < _num_basis; j++) {
            double val = 0.0;
            val += .5 * (prob_a(i,j) * (hc(i,j) + fa(i,j)));
            val += .5 * (prob_b(i,j) * (hc(i,j) + fb(i,j)));
            //i'm going to ignore repulsion tbh
            emat(i,j) = val;
        }
    }
    double total_e = accu(emat);
    return total_e;
}

double conGaussian::eval_scf() {
    // double tol = 1.0;
    // while(tol < 1e-5)
    
    mat p_init;
    mat prob_a = p_init.zeros(_num_basis,_num_basis);
    mat prob_b = p_init.zeros(_num_basis,_num_basis);

    mat s = arma::mat("1.0000 0.6599; 0.6599 1.0000");
    std::cout << "Overlap matrix" << std::endl;
    s.print();

    mat ca; //eigenvectors for fa
    mat cb; //eigenvectors for fb
    double g;  //final energy
    for(int i = 0; i < 2; i++) {

        std::cout << "H core" << std::endl;
        mat hc = eval_core(s);
        hc.print();

        mat fa = eval_fock(prob_a, prob_b, s);
        mat fb = fa;
        std::cout << "Iteration " << i << std::endl;

        std::cout << "Fa" << std::endl;
        fa.print();
        std::cout << "Fb" << std::endl;
        fb.print();

        //create an eigen matrix because armadillo isn't working??
        Eigen::MatrixXd F(fa.n_rows, fa.n_cols);
        for (size_t i = 0; i < fa.n_rows; ++i) {
            for (size_t j = 0; j < fa.n_cols; ++j) {
                F(i, j) = fa(i, j);  // Copy data element by element
            }
        }
        
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(F);
        //get the values
        Eigen::VectorXd eeigvals = solver.eigenvalues().real();
        eeigvals = eeigvals.transpose();
        //save eigen to arma
        arma::rowvec e(eeigvals.data(), eeigvals.size(), false, true);

        //get the vectors
        Eigen::MatrixXd eigen_eigvecs = solver.eigenvectors().real();
        arma::mat c(eigen_eigvecs.data(), eigen_eigvecs.rows(), eigen_eigvecs.cols(), false, true);

        ca, cb = c;  //our eigenvectors
        //e.print();  //our energy, yeet

        //update p
        std::cout << "Ca, Cb" << std::endl;
        c.print();
        prob_a = eval_prob(prob_a, c);
        prob_b = eval_prob(prob_b, c);
        std::cout << "Pa_new" << std::endl;
        prob_a.print();
        std::cout << "Pb_new" << std::endl;
        prob_b.print();

        g = calc_total_energy(hc, fa, fb, prob_a, prob_b);
        //if max amount of change in p < 10-6, you converged
    }
    return g;
}

// 2 electron integral of two primitive Gaussians
double conGaussian::i2e_pg(rowvec &Ra, rowvec &Rb, double sigmaA, double sigmaB){
  double U =  pow(M_PI*sigmaA,3/2) * pow(M_PI*sigmaB,3/2);// calculate U using equation 3.8 & 3.11
  double V2 =  1 / (sigmaA * sigmaB); // calculate V2 using equation 3.9
  double Rd = arma::norm(Ra - Rb, 2);
  if(Rd == 0.0) {  // if Ra == Rb
    return U * 2.0 * sqrt(V2 / M_PI);  // equation 3.15
  }
  double srT = sqrt(V2 * pow(Rd, 2));  // equation 3.7 sqrt
  double result = U * sqrt(1 / pow(Rd,2)) * erf(srT);  // equation 3.14
  return result; 
}

double conGaussian::eval_2ei_sAO(atom ao1, atom ao2){
  
//   int len = ao1.get_len();
//   assert(ao2.get_len() == len);
  int len = 3;

  // also get Ra, Rb, alpha_a, alpha_b, da, db...

  rowvec Ra = ao1.coords;
  rowvec Rb = ao2.coords;
  rowvec alpha_a = ao1.func[0].exp[0];
  rowvec alpha_b = ao2.func[0].exp[0];
  rowvec da = ao1.func[0].coef[0];
  rowvec db = ao2.func[0].coef[0];

  double gamma = 0.;
  // loop over k, k', l, l' in equation 3.3
  for (size_t k1 = 0; k1 < len; k1++)
  for (size_t k2 = 0; k2 < len; k2++)
  {
    double sigmaA = 1 / (alpha_a(k1) + alpha_b(k2));  // equation 3.10
    for (size_t j1 = 0; j1 < len; j1++)
    for (size_t j2 = 0; j2 < len; j2++)
    {
      double sigmaB = 1 / (alpha_b(k1) + alpha_a(k2));  // equation 3.10
      double I2e = i2e_pg(Ra, Rb, sigmaA, sigmaB);
      gamma += da(j1) * db(j2) * da(k1) * db(k2) * I2e; // equation 3.3
    }
  }
  return gamma;    
}

double conGaussian::eval_gradient() {
    mat xmat = zeros(_num_basis,_num_basis);
    mat ymat = zeros(_num_basis, _num_basis); 
    mat smat = zeros(3, _num_basis + _num_basis); //x,y,z for # rows
    mat gmat = zeros(3, _num_basis + _num_basis); //gamma
    mat nmat = zeros(_num_basis, _num_basis);   //nuclear repulsion
    //mat emat = zeros(3, _num_basis);    //gradient
    double energy = 0.0;
    int col_count = 0;
    for(int i = 0; i < _num_basis; i++) {
        for(int j = 0; j < _num_basis; j++) {
            xmat(i, j) = calc_x(i, j);
            for(int dim = 0; dim < 3; dim++) {
                smat(dim,col_count) = calc_deriv_overlap(i, j, dim);
                if(i != j) energy += xmat(i,j) * smat(dim, col_count);
            }
        col_count++;
        }
    }
    col_count = 0;
    for(int a = 0; a < _atoms.size(); a++) {
        for(int b = 0; b < _atoms.size(); b++) {
            ymat(a, b) = calc_y(a, b);
            nmat(a, b) = calc_nuc_repulsion(a, b);
            for(int dim = 0; dim < 3; dim++) {
                gmat(dim, col_count) = calc_deriv_gamma(a, b, dim);
                if(a != b) energy += ymat(a,b) * gmat(dim, col_count) + nmat(a,b);
            }
        col_count++;
        }
    }
    xmat.print("x matrix:");
    ymat.print("y matrix:");
    smat.print("deriv overlap matrix:");
    gmat.print("deriv gamma matrix:");
    nmat.print("nuclear repulsion matrix:");
    return energy;
}

double conGaussian::deriv_i2e_pg(double xa, double xb, double sigmaA, double sigmaB){
  double U =  pow(M_PI*sigmaA,3/2) * pow(M_PI*sigmaB,3/2);// calculate U using equation 3.8 & 3.11
  double V2 =  1 / (sigmaA * sigmaB); // calculate V2 using equation 3.9
  double Rd = xa - xb;
  if(Rd == 0.0) {  // if Ra == Rb
    return U * 2.0 * sqrt(V2 / M_PI);  // equation 3.15
  }
  double srT = sqrt(V2 * pow(Rd, 2));  // equation 3.7 sqrt
  double result1 = (pow(U,2) * (Rd)) / (pow(Rd,2));  // equation 3.14
  return result1 * - (erf(srT)/ abs(Rd)) * (2 * V2 / sqrt(M_PI)) * exp(sqrt(srT)); 
}

double conGaussian::calc_deriv_gamma(int a, int b, int dim) {
    if(a == b) return 0.0;
    double xa = _atoms[a].coords[dim];
    double xb = _atoms[b].coords[dim];
    rowvec alpha_a = _atoms[a].func[0].exp[0];
    rowvec alpha_b = _atoms[b].func[0].exp[0];
    rowvec da = _atoms[a].func[0].coef[0];
    rowvec db = _atoms[b].func[0].coef[0];

    int len = 3;

    double gamma = 0.0;
    for (size_t k1 = 0; k1 < len; k1++)
    for (size_t k2 = 0; k2 < len; k2++) {
        double sigmaA = 1 / (alpha_a(k1) + alpha_b(k2));
        for (size_t j1 = 0; j1 < len; j1++)
        for (size_t j2 = 0; j2 < len; j2++) {
            double sigmaB = 1 / (alpha_b(k1) + alpha_a(k2));
            double I2e_d = deriv_i2e_pg(xa, xb, sigmaA, sigmaB);
            gamma += da(j1) * db(j2) * da(k1) * db(k2) * I2e_d;
        }
  }
  return gamma;    
}

double conGaussian::calc_nuc_repulsion(int a, int b){
    if(a == b) return 0.0;
    double xa = _atoms[a].coords[0];
    double xb = _atoms[b].coords[0];

    int za = _atoms[a].val_electrons;
    int zb = _atoms[b].val_electrons;

    return - za * zb * (xa - xb) / pow(xa - xb, 3);
}

double conGaussian::calc_deriv_overlap(int i, int j, int dim) {
    if(i == j) return 0.0;
    double xa = _atoms[i].coords[dim];
    double xb = _atoms[j].coords[dim];

    double ang_a = _atoms[i].ang_moment(0, dim);
    double ang_b = _atoms[j].ang_moment(0, dim);

    double alpha_a = _atoms[i].func[0].exp[0][0];

    double dist = xa - xb;

    double phi_a = exp(-alpha_a * pow(dist,2));
    double phi_b = exp(-alpha_a * pow(dist,2));

    return -2 * alpha_a * dist * phi_a * phi_b;
}

double conGaussian::calc_x(int i, int j) {

    int dim = 0;    //x dimension only
    double x_u = _atoms[i].coords[dim];
    double x_v = _atoms[j].coords[dim];

    auto alpha_a = _atoms[i].func[0].exp[0][i];
    auto alpha_b = _atoms[j].func[0].exp[0][j];

    double phi_u = exp(-alpha_a * pow(x_u - x_v,2));
    double phi_u_pr = - alpha_a * pow(x_u - x_v, 2) * phi_u;
    double phi_v = exp(-alpha_b * pow(x_u - x_v,2));
    double phi_v_pr = - alpha_b * pow(x_u - x_v, 2) * phi_v;
    
    return (2*phi_u * phi_u_pr) + (2*phi_v * phi_v_pr);
}

double conGaussian::calc_y(int a, int b) {

    if(a == b) return 0.0;

    double x_u = _atoms[a].coords[0];
    double x_v = _atoms[b].coords[0];
    double dist = arma::norm(_atoms[a].coords - _atoms[b].coords);

    double s = 0.0;
    for(int k = 0; k < 3; k++) {
        double dk_a = _atoms[a].func[0].coef[0](k);
        double dk_b = _atoms[b].func[0].coef[0](k);

        double alpha_a = _atoms[a].func[0].exp[0](k);
        double alpha_b = _atoms[b].func[0].exp[0](k);

        double omega = center_product(x_u, x_v, alpha_a, alpha_b);
        double omega_pr = deriv_center_product(alpha_a, alpha_b);

        s += (dk_a * omega * dk_a * omega_pr) + \
             (dk_b * omega * dk_b * omega_pr); 
    }
    return (2 * s)/ dist;
    //return a_repel + b_repel;
}

int conGaussian::factorial(int n) {
    int result = n;
    if(n==0 || n==1) return 1;
    for(int i = n-1; i > 0; i--) {
        result *= i;
    }
    return result;
}

int conGaussian::double_factorial(int n) {
    if (n <= 1) return 1;
    int result = n;
    while(n > 2) {
        n-=2;
        result*= n;
    }
    return result;
}

int conGaussian::combinations(int m, int n) {
    if (n>m) return 0;
    return factorial(m) / (factorial(n) * factorial(m - n));
}

double conGaussian::center_product(double xa, double xb, double alpha, double beta) {
    return ((alpha * xa) + (beta * xb)) / (alpha + beta); 
}

double conGaussian::deriv_center_product(double alpha_a, double alpha_b) {
    return alpha_a / (alpha_a + alpha_b);
}

double conGaussian::calc_1d_overlap(double xa, double xb, double ang_a, double ang_b,
                                    double alpha, double beta) {
//constants
    double pi = M_PI;
    double rp = center_product(xa, xb, alpha, beta);
//begin calculation in parts
    double exponent = exp( -((alpha * beta / (alpha + beta)) * pow((xa - xb), 2.0)));
    double prefactor = sqrt(pi / (alpha + beta));

    double gaussian = 0.0;
    for(int i = 0; i <= ang_a; i++) {
        for(int j = 0; j <= ang_b; j++) {
            if((i+j) % 2 == 0 ) {
                double c1 = combinations(ang_a, i);
                double c2 = combinations(ang_b, j);

                double double_fact = double_factorial(i + j - 1);
                
                gaussian += c1 * c2 * double_fact * \
                            (pow((rp - xa), (ang_a - i)) * pow((rp - xb), (ang_b - j))) \
                            / pow((2.0 * (alpha + beta)), ((i + j) / 2.0));    
            }
        }
    }
    return exponent * prefactor * gaussian;
}