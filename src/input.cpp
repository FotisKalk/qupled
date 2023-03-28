#include "input.hpp"

// --- Input ---

double Input::initCoupling(const double &rs_){
  if (rs_ <= 0) {
    throw runtime_error("The quantum coupling parameter must be larger than zero");
  }
  return rs_;
}

double Input::initDegeneracy(const double &Theta_){
  if (Theta_ < 0.0) {
    throw runtime_error("The quantum degeneracy parameter can't be negative");
  }
  return Theta_;
}

string Input::initTheory(const string &theory_){
  const vector<string> cTheories = {"STLS", "STLS-HNC",
				    "STLS-IOI", "STLS-LCT"};
  const vector<string> qTheories = {"QSTLS", "QSTLS-HNC",
				    "QSTLS-IOI", "QSTLS-LCT"};
  isClassicTheory = count(cTheories.begin(), cTheories.end(), theory_) != 0;
  isQuantumTheory = count(qTheories.begin(), qTheories.end(), theory_) != 0;
  if (!isClassicTheory && !isQuantumTheory) {
    throw runtime_error("Unknown theory: " + theory_);
  }
  // A theory can't both be classical and quantum at the same time
  assert(!isClassicTheory || !isQuantumTheory);
  return theory_;
}

void Input::setCoupling(const double &rs){
  this->rs = initCoupling(rs);
}

void Input::setDegeneracy(const double &Theta){
  this->Theta = initDegeneracy(Theta);
}


void Input::setInt2DScheme(const string  &int2DScheme){
  const vector<string> schemes = {"full", "segregated"};
  if (count(schemes.begin(), schemes.end(), int2DScheme) == 0) {
    throw runtime_error("Unknown scheme for 2D integrals: " + int2DScheme);
  }
  this->int2DScheme = int2DScheme;
}

void Input::setNThreads(const int &nThreads){
  if (nThreads <= 0) {
    throw runtime_error("The number of threads must be larger than zero");
  }
  this->nThreads = nThreads;
}

void Input::setTheory(const string &theory){
  this->theory = initTheory(theory);
}

void Input::print() const {
  cout << "Coupling parameter = " << rs << endl;
  cout << "Degeneracy parameter = " << Theta << endl;
  cout << "Number of OMP threads = " << nThreads << endl;
  cout << "Scheme for 2D integrals = " << int2DScheme << endl;
  cout << "Theory to be solved = " << theory << endl;
}

bool Input::isEqual(const Input &in) const {
  return ( int2DScheme == in.int2DScheme &&
	   nThreads == in.nThreads &&
	   rs == in.rs &&
	   theory == in.theory &&
	   Theta == in.Theta ); 
}

// --- StlsInput ---

void StlsInput::setChemicalPotentialGuess(const double &muMin,
					  const double &muMax){
  if (muMin >= muMax) {
    throw runtime_error("Invalid guess for chemical potential calculation");
  }
  muGuess.resize(2);
  muGuess[0] = muMin;
  muGuess[1] = muMax;
}

void StlsInput::setErrMin(const double &errMin){
  if (errMin <= 0.0) {
    throw runtime_error("The minimum error for convergence must be larger than zero");
  }
  this->errMin = errMin;
}

void StlsInput::setMixingParameter(const double &aMix){
  if (aMix < 0.0 || aMix > 1.0) {
    throw runtime_error("The mixing parameter must be a number between zero and one");
  }
  this->aMix = aMix;
}

void StlsInput::setNMatsubara(const int &nl){
  if (nl < 0) {
    throw runtime_error("The number of matsubara frequencies can't be negative");
  }
  this->nl = nl;
}

void StlsInput::setNIter(const int &nIter){
  if (nIter < 0) {
    throw runtime_error("The maximum number of iterations can't be negative");
  }
  this->nIter = nIter; 
}

void StlsInput::setOutIter(const int &outIter){
  if (outIter < 0) {
    throw runtime_error("The output frequency can't be negative");
  }
  this->outIter = outIter; 
}

void StlsInput::setIETMapping(const string &IETMapping){
  const vector<string> mappings = {"standard", "sqrt", "linear"};
  if (count(mappings.begin(), mappings.end(), IETMapping) == 0) {
    throw runtime_error("Unknown IET mapping: " + IETMapping);
  }
  this->IETMapping = IETMapping;
}

void StlsInput::setRestartFileName(const string &restartFileName){
  this->restartFileName = restartFileName;
}

void StlsInput::setWaveVectorGridRes(const double &dx){
  if (dx <= 0.0) {
    throw runtime_error("The wave-vector grid resolution must be larger than zero");
  }
  this->dx = dx;
}

void StlsInput::setWaveVectorGridCutoff(const double &xmax){
  if (xmax <= 0.0) {
    throw runtime_error("The wave-vector grid cutoff must be larger than zero");
  }
  if (xmax < dx) {
    throw runtime_error("The wave-vector grid cutoff must be larger than the resolution");
  }
  this->xmax = xmax;
}

void StlsInput::print() const {
  Input::print();
  cout << "##### STLS-related input #####" << endl;
  cout << "Guess for chemical potential = " << muGuess.at(0) << "," << muGuess.at(1) << endl;
  cout << "Iet mapping scheme" << IETMapping << endl;
  cout << "Maximum number of iterations = " << nIter << endl;
  cout << "Minimum error for convergence = " << errMin << endl;
  cout << "Mixing parameter = " << aMix << endl;
  cout << "Number of Matsubara frequencies = " << nl << endl;
  cout << "Output frequency = " << outIter << endl;
  cout << "File with restart data = " << restartFileName << endl;
  cout << "Wave-vector resolution = " << dx << endl;
  cout << "Wave-vector cutoff = " << xmax << endl;
}

bool StlsInput::isEqual(const StlsInput &in) const {
  return ( Input::isEqual(in) &&
	   aMix == in.aMix && 
	   dx == in.dx && 
	   errMin == in.errMin &&
	   IETMapping == in.IETMapping &&
	   muGuess == in.muGuess &&
	   nl == in.nl &&
	   nIter == in.nIter &&
	   outIter == in.outIter &&
	   restartFileName == in.restartFileName &&
	   xmax == in.xmax);
}

// --- QstlsInput ---

void QstlsInput::setFixedFileName(const string &fixedFileName){
  this->fixedFileName = fixedFileName;
} 

void QstlsInput::print() const {
  cout << "##### qSTLS-related input #####" << endl;
  cout << "File with fixed adr component = " << fixedFileName  << endl;
}

bool QstlsInput::isEqual(const QstlsInput &in) const {
  return ( fixedFileName == in.fixedFileName );
}
