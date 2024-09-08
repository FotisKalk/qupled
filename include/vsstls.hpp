#ifndef VSSTLS_HPP
#define VSSTLS_HPP

#include "input.hpp"
#include "stls.hpp"
#include "vsbase.hpp"
#include <limits>
#include <map>

class ThermoProp;
class StructProp;
class StlsCSR;

// -----------------------------------------------------------------
// VSStls class
// -----------------------------------------------------------------

class VSStls : public VSBase, public Stls {

public:

  // Constructor from initial data
  explicit VSStls(const VSStlsInput &in_);
  // Constructor for recursive calculations
  VSStls(const VSStlsInput &in_, const ThermoProp &thermoProp_);
  // Solve the scheme
  using VSBase::compute;

private:

  // Input
  VSStlsInput in;
  // Verbosity
  using VSBase::verbose;
  // Thermodynamic properties
  std::shared_ptr<ThermoProp> thermoProp;
  // Initialize
  void initScheme();
  void initFreeEnergyIntegrand();
  // Compute free parameter
  double computeAlpha();
  // Iterations to solve the vs-stls scheme
  void updateSolution();
};

// -----------------------------------------------------------------
// ThermoProp class
// -----------------------------------------------------------------

class ThermoProp : public ThermoPropBase {

public:

  // Constructor
  explicit ThermoProp(const VSStlsInput &in_);

private:

  // Structural properties
  std::shared_ptr<StructProp> structProp;
};

// -----------------------------------------------------------------
// StructProp class
// -----------------------------------------------------------------

class StructProp : public StructPropBase {

public:

  explicit StructProp(const VSStlsInput &in_);

private:

  // Vector containing NPOINTS state points to be solved simultaneously
  std::vector<std::shared_ptr<StlsCSR>> csr;
  // setup the csr vector
  std::vector<VSStlsInput> setupCSRInput(const VSStlsInput &in);
  void setupCSR(const VSStlsInput &in_);
  //
  void doIterations();
};

// -----------------------------------------------------------------
// StlsCSR class
// -----------------------------------------------------------------

class StlsCSR : public CSR, public Stls {

public:

  // Constructor
  explicit StlsCSR(const VSStlsInput &in_)
      : CSR(in_),
        Stls(in_.toStlsInput(), false, false),
        in(in_) {}

  // Compute static local field correction
  void computeSlfcStls();
  void computeSlfc();

  // Publicly esposed private stls methods
  void init() { Stls::init(); }
  void initialGuess() { Stls::initialGuess(); }
  void computeSsf() { Stls::computeSsf(); }
  double computeError() { return Stls::computeError(); }
  void updateSolution() { Stls::updateSolution(); }

  // Getters
  std::vector<double> getSsf() const { return Stls::getSsf(); }
  std::vector<double> getSlfc() const { return Stls::getSlfc(); }
  std::vector<double> getWvg() const { return Stls::getWvg(); }

private:

  // Input parameters
  VSStlsInput in;
};

#endif
