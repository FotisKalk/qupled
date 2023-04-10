#ifndef QSTLS_HPP
#define QSTLS_HPP

#include <map>
#include "stls.hpp"

// -----------------------------------------------------------------
// Solver for the qSTLS-based schemes
// -----------------------------------------------------------------

class Qstls : public Stls {
  
private: 

  // Input data
  const QstlsInput qin;
  // Auxiliary density response
  vecUtil::Vector2D adr;
  vecUtil::Vector2D adrOld;
  vecUtil::Vector3D adrFixed;
  map<int,pair<string,bool>> adrFixedIetFileInfo;
  // Static structure factor (for iterations)
  vector<double> ssfOld;
  // Compute auxiliary density response
  void computeAdr();
  void computeAdrFixed();
  void loadAdrFixed();
  int  checkAdrFixedFromFile(const vector<double> &wvg_,
			     const double Theta_,
			     const int nl_) const;
  void computeAdrIet();
  void computeAdrFixedIet();
  void getAdrFixedIetFileInfo();
  void writeAdrFixedIetFile(const vecUtil::Vector3D &res,
			    const int i) const;
  void readAdrFixedIetFile(vecUtil::Vector3D &res,
			   const int i) const;
  // Compute static structure factor at finite temperature
  void computeSsf();
  void computeSsfFinite();
  // Iterations to solve the stls scheme
  void doIterations();
  void initialGuess();
  void initialGuessSsf(const vector<double> &wvg_,
		       const vector<double> &adr_);
  void initialGuessAdr(const vector<double> &wvg_,
		       const vecUtil::Vector2D &adr_);
  double computeError();
  void updateSolution();
   // Write output files
  void writeOutput() const;
  void writeAdr() const;
  // Restart files
  void writeRestart() const;
  void readRestart(const string &fileName,
		   vector<double> &wvg_,
		   vecUtil::Vector3D &adrFixed_,
		   double &Theta,
		   int &nl) const;
  void readRestart(const string &fileName,
		   vector<double> &wvg_,
		   vector<double> &ssf_,
		   vecUtil::Vector2D &adr_,
		   vecUtil::Vector3D &adrFixed_,
		   double &Theta,
		   int &nl) const;
  // Check if iet schemes should be used
  void checkIet() { useIet = in.getTheory() == "QSTLS-HNC" ||
      in.getTheory() == "QSTLS-IOI" ||
      in.getTheory() == "QSTLS-LCT";}

public:

  // Constructor
  Qstls(const StlsInput &in_,
	const QstlsInput &qin_)
    : Stls(in_), qin(qin_) { checkIet(); };
  // Compute qstls scheme
  void compute();
  // Getters
  vecUtil::Vector2D getAdr() const { return adr; }
  vecUtil::Vector3D getAdrFixed() const { return adrFixed; }

};

// -----------------------------------------------------------------
// Class for the static structure factor
// -----------------------------------------------------------------

class Qssf : public Ssf {

private:

  // Auxiliary density response
  const double *adr;
  // Bridge function
  const double bf;
  
public:

  // Constructor for quantum schemes
  Qssf(const double x_,
       const double Theta_,
       const double rs_,
       const double ssfHF_,
       const int nl_,
       const double *idr_,
       const double *adr_,
       const double bf_)
    : Ssf(x_, Theta_, rs_, ssfHF_, 0, nl_, idr_),
      adr(adr_), bf(bf_) {;};
  // Get static structure factor
  double get() const;
 
  
};
// -----------------------------------------------------------------
// Classes for the auxiliary density response
// -----------------------------------------------------------------

class AdrBase {

protected:
  
  // Degeneracy parameter
  const double Theta;
  // Integration limits
  const double yMin;
  const double yMax;
  // Wave-vector
  const double x;
  // Interpolator for the static structure factor
  const Interpolator1D &ssfi;
  // Integrand scaling constants
  const double isc;
  const double isc0;
  // Compute static structure factor
  double ssf(const double y) const;
  
public:

  // Constructor
  AdrBase(const double Theta_,
	  const double yMin_,
	  const double yMax_,
	  const double x_,
	  const Interpolator1D &ssfi_)
    : Theta(Theta_), yMin(yMin_), yMax(yMax_), x(x_),
      ssfi(ssfi_), isc(-3.0/8.0), isc0(isc*2.0/Theta) {;};
  
};

class AdrFixedBase {

protected:

  // Degeneracy parameter
  const double Theta;
  // Integration limits
  const double qMin;
  const double qMax;
  // Wave-vector
  const double x;
  // Chemical potential
  const double mu;
  
public:

  // Constructor for finite temperature calculations
  AdrFixedBase(const double Theta_,
	       const double qMin_,
	       const double qMax_,
	       const double x_,
	       const double mu_)
    : Theta(Theta_), qMin(qMin_), qMax(qMax_), x(x_), mu(mu_) {;};
  
};

class Adr : public AdrBase {

private:

  // Compute fixed component
  double fix(const double y) const;
  // integrand 
  double integrand(const double y) const;
  // Interpolator for the fixed component
  Interpolator1D fixi;
  // Integrator object
  Integrator1D &itg;
  
public:

  // Constructor for finite temperature calculations
  Adr(const double Theta_,
      const double yMin_,
      const double yMax_,
      const double x_,
      const Interpolator1D &ssfi_,
      Integrator1D &itg_)
    : AdrBase(Theta_, yMin_, yMax_, x_, ssfi_), itg(itg_) {;};
  
  // Get result of integration
  void get(const vector<double> &wvg,
	   const vecUtil::Vector3D &fixed,
	   vecUtil::Vector2D &res);
  
};

class AdrFixed : public AdrFixedBase {
  
private:
  
  // Integrands 
  double integrand1(const double q,
		    const double l) const;
  double integrand2(const double t,
		    const double y,
		    const double l) const;
  // Integrator object
  Integrator2D &itg;
  // Grid for 2D integration
  const vector<double> &itgGrid;
  
public:

  // Constructor for finite temperature calculations
  AdrFixed(const double Theta_,
	   const double qMin_,
	   const double qMax_,
	   const double x_,
	   const double mu_,
	   const vector<double> &itgGrid_,
	   Integrator2D &itg_)
    : AdrFixedBase(Theta_, qMin_, qMax_, x_, mu_),
      itg(itg_), itgGrid(itgGrid_) {;};
  
  // Get integration result
  void get(vector<double> &wvg,
	   vecUtil::Vector3D &res) const;
  
};

// Class for the auxiliary density response calculation in the IET scheme
class AdrIet : public AdrBase {

private:

   // Integration limits
  const double &qMin = yMin;
  const double &qMax = yMax;
  // Integrands 
  double integrand1(const double q,
		    const int l) const;
  double integrand2(const double y) const;
  // Integrator object
  Integrator2D &itg;
  // Grid for 2D integration
  const vector<double> &itgGrid;
  // Interpolator for the dynamic local field correction
  const vector<Interpolator1D> &dlfci;
  // Interpolator for the bridge function contribution
  const Interpolator1D &bfi;
  // Interpolator for the fixed component 
  Interpolator2D fixi;
  // Compute dynamic local field correction
  double dlfc(const double y,
	      const int l) const;
  // Compute bridge function contribution
  double bf(const double y) const;
  // Compute fixed component
  double fix(const double x,
	     const double y) const;
  
public:

  // Constructor for finite temperature calculations
  AdrIet(const double Theta_,
	 const double qMin_,
	 const double qMax_,
	 const double x_,
	 const Interpolator1D &ssfi_,
	 const vector<Interpolator1D> &dlfci_,
	 const Interpolator1D &bfi_,
	 const vector<double> &itgGrid_,
	 Integrator2D &itg_)
    : AdrBase(Theta_, qMin_, qMax_, x_, ssfi_),
      itg(itg_), itgGrid(itgGrid_), dlfci(dlfci_), bfi(bfi_) {;};
  
  // Get integration result
  void get(const vector<double> &wvg,
	   const vecUtil::Vector3D &fixed,
	   vecUtil::Vector2D &res);
  
};

class AdrFixedIet : public AdrFixedBase {

private:

  // Integration limits
  const double &tMin = qMin;
  const double &tMax = qMax;
  // Integrands 
  double integrand(const double t,
		   const double y,
		   const double q,
		   const double l) const;
  // Integrator object
  Integrator1D &itg;
  
public:

  // Constructor for finite temperature calculations
  AdrFixedIet(const double Theta_,
	      const double qMin_,
	      const double qMax_,
	      const double x_,
	      const double mu_,
	      Integrator1D &itg_)
    : AdrFixedBase(Theta_, qMin_, qMax_, x_, mu_),
      itg(itg_) {;};
  
  //Get integration result
  void get(vector<double> &wvg,
	   vecUtil::Vector3D &res) const;
  
};


#endif
