#include "numerics.hpp"

// Run root solver
void RootSolver::solve(const function<double(double)> func,
		       const vector<double> guess){
  // Set up function
  GslFunctionWrap<decltype(func)> Fp(func);
  F = static_cast<gsl_function*>(&Fp);
  // Set up solver
  gsl_root_fsolver_set(rs, F, guess.at(0), guess.at(1));
  // Call solver
  do{
    status = gsl_root_fsolver_iterate(rs);
    sol = gsl_root_fsolver_root(rs);
    double solLo = gsl_root_fsolver_x_lower(rs);
    double solHi = gsl_root_fsolver_x_upper(rs);
    status = gsl_root_test_interval(solLo, solHi, 0, relErr);
    iter++;
  } while (status == GSL_CONTINUE && iter < maxIter);
}

// Compute 1D integrals 
void Integrator1D::compute(const function<double(double)> func,
			   const double xMin, double xMax){
  // Set up function
  GslFunctionWrap<decltype(func)> Fp(func);
  F = static_cast<gsl_function*>(&Fp);
  // Integrate
  gsl_integration_cquad(F, xMin, xMax, 
			0.0, relErr, 
			wsp, &sol,
			&err, &nEvals);
}

// Compute 1D integrals of Fourier type 
void Integrator1DFourier::compute(const function<double(double)> func){
  // Set up function
  GslFunctionWrap<decltype(func)> Fp(func);
  F = static_cast<gsl_function*>(&Fp);
  // Set wave-vector
  gsl_integration_qawo_table_set(qtab, r, 1.0, GSL_INTEG_SINE);
  // Integrate
  gsl_integration_qawf(F, 0.0, relErr, 
		       limit, wsp, wspc,
		       qtab, &sol, &err);
}

