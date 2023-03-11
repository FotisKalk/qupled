#include <omp.h>
#include "util.hpp"
#include "qstls.hpp"

using namespace stringUtil;
using namespace binUtil;

// -----------------------------------------------------------------
// QSTLS class
// -----------------------------------------------------------------

void Qstls::compute(){
  if (in.getDegeneracy() == 0.0) {
    throw runtime_error("Ground state calculations are not available "
			"for the quantum schemes");
  }
  Stls::init();
  if (verbose) cout << "Structural properties calculation ..." << endl;
  doIterations();
  if (verbose) cout << "Done" << endl;
  if (verbose) cout << "Writing output files: ";
  writeOutput();
  if (verbose) cout << "Done" << endl;
}


// qstls iterations
void Qstls::doIterations() {
  const auto &statIn = in.getStaticInput();
  const int maxIter = statIn.getNIter();
  const int outIter = statIn.getOutIter();
  const double minErr = statIn.getErrMin();
  double err = 1.0;
  int counter = 0;
  // Define initial guess
  initialGuess();
  while (counter < maxIter+1 && err > minErr ) {
    // Start timing
    double tic = omp_get_wtime();
    // Update auxiliary density response
    computeAdr();
    err = 0;
    // Update static structure factor
    computeSsf();
    // Write output
    if (counter % outIter == 0) { writeOutput();};
    // Update diagnostic
    counter++;
    err = computeError();
    updateSolution();
    // End timing
    double toc = omp_get_wtime();
    // Print diagnostic
    if (verbose) {
       printf("--- iteration %d ---\n", counter);
       printf("Elapsed time: %f seconds\n", toc - tic);
       printf("Residual error: %.5e\n", err);
       fflush(stdout);
    }
  }
}

// Compute auxiliary density response
void Qstls::computeAdr() {
  assert(ssfOld.size() > 0);
  const auto &statIn = in.getStaticInput();
  const int nx = wvg.size();
  const int nl = statIn.getNMatsubara();
  if (adr.size() == 0) adr.resize(nx, nl);
  if (slfc.size() == 0) slfc.resize(nx);
  assert(wvg.size() == ssfOld.size());
  if (adrFixed.size() == 0) computeAdrFixed();
  const Interpolator ssfi = Interpolator(wvg[0], ssfOld[0], nx);
  for (int i=0; i<nx; ++i) {
    Adr adrTmp(nl, in.getDegeneracy(), wvg.front(),
	       wvg.back(), ssfi, itg);
    //adrTmp.get(wvg, adrFixed[i], adr[i]);
  }
  if (useIet) computeAdrIet();
  for (int i=0; i<nx; ++i) {slfc[i] = adr(i,0); };
}

void Qstls::computeAdrFixed() {
  cout << "#######" << endl;
  cout << "Computing fixed component of the auxiliary density response: ";
  fflush(stdout);
  loadAdrFixed();
  if (adrFixed.size() > 0) { return; }
  const auto &statIn = in.getStaticInput();
  const int nx = wvg.size();
  const int nl = statIn.getNMatsubara();
  //adrFixed.resize(nx);
#pragma omp parallel for
  for (int i=0; i<nx; ++i) {
    Integrator2D itg2;
    AdrFixed adrTmp(nl, wvg[i], in.getDegeneracy(), mu,
		    wvg.front(), wvg.back(), itg2);
    //adrTmp.get(wvg, adrFixed[i]);
  }
  writeRestart();
  cout << "Done" << endl;
  cout << "#######" << endl;
}

void Qstls::loadAdrFixed() {
  const auto &qstlsIn = in.getQstlsInput();
  const string fileName = qstlsIn.getFixedFileName();
  if (fileName == "") return;
  decltype(wvg) wvg_;
  decltype(adrFixed) adrFixed_;
  double Theta_;
  readRestart(fileName, wvg_, adrFixed_, Theta_);
  //checkAdrFixedFromFile(wvg_, Theta_, adrFixed_[0].size());
  adrFixed = adrFixed_;
  cout << endl << "Loaded from file " << fileName << endl;
}

void Qstls::checkAdrFixedFromFile(const decltype(wvg) &wvg_,
				  const double Theta_,
				  const int nl_) const {
  const double tol = 1e-15;
  const string errMsg = "Data loaded from file for the fixed component "
                        "of the auxiliary density response is not "
                        "compatible with input";
  if (nl_ != in.getStaticInput().getNMatsubara()) {
    throw runtime_error(errMsg + ", wrong number of Matsubara frequencies.");
  }
  if (abs(Theta_ - in.getDegeneracy()) > tol) {
    throw runtime_error(errMsg + ", wrong degeneracy parameter.");
  }
  if (wvg_.size() != wvg.size()) {
    throw runtime_error(errMsg + ", wrong wave-vector grid.");
  }
  const vector<double> wvgDiff = diff(wvg_, wvg);
  if (abs(*max_element(wvgDiff.begin(), wvgDiff.end())) > tol) {
    throw runtime_error(errMsg + ", wrong wave-vector grid.");
  }
}

void Qstls::computeAdrIet() {
  if (bf.size() == 0) computeBf();
  const int nx = wvg.size();
  computeAdrFixedIet();
  for (int i=0; i<nx; ++i) {
    assert(adrFixedIetFileInfo.at(i).second);
    //Vector3D<double> fix;
    //readAdrFixedIetFile(fix, i);
  }
}

void Qstls::getAdrFixedIetFileInfo() {
  const auto &statIn = in.getStaticInput();
  const int nx = wvg.size();
  const int nl = statIn.getNMatsubara();
  const double Theta = in.getDegeneracy();
  adrFixedIetFileInfo.clear();
  for (int i=0; i<nx; ++i) {
    const string name = format<double,double>("adr_iet_fixed_rs%.3f_theta%.3f_"
					      + in.getTheory() + "_wv%.5f.bin",
					      in.getCoupling(), Theta, wvg[i]);
    const bool found = __fs::filesystem::exists(name);
    const pair<string,bool> filePair = pair<string,bool>(name,found);
    adrFixedIetFileInfo.insert(pair<int,decltype(filePair)>(i,filePair));
  }
}

void Qstls::computeAdrFixedIet() {
  const auto &statIn = in.getStaticInput();
  const int nx = wvg.size();
  const int nl = statIn.getNMatsubara();
  vector<int> idx;
  // Check which files have to be created
  getAdrFixedIetFileInfo();
  for (const auto &member : adrFixedIetFileInfo) {
    if(!member.second.second) { idx.push_back(member.first); };
  }
  if (idx.size() == 0) return;
  // Write necessary files
  cout << "#######" << endl;
  cout << "Writing files for the fixed component "
          "of the iet auxiliary density response" << endl;
  cout << "#######" << endl;
#pragma omp parallel for
  for (int i=0; i<idx.size(); ++i) {
    Integrator1D itgT;
    decltype(adrFixed) res;
    AdrFixedIet adrTmp(nl, wvg[idx[i]], in.getDegeneracy(), mu,
		       wvg.front(), wvg.back(), itgT);
    //adrTmp.get(wvg, res);
    writeAdrFixedIetFile(res, idx[i]);
  }
  // Update content of adrFixedIetFileInfo
  getAdrFixedIetFileInfo();
}

void Qstls::writeAdrFixedIetFile(const decltype(adrFixed) &res,
				 const int i) const {
  const auto &statIn = in.getStaticInput();
  const int nx = wvg.size();
  const int nl = statIn.getNMatsubara();
  const double Theta = in.getDegeneracy();
  const string fileName = adrFixedIetFileInfo.at(i).first;
  ofstream file;
  file.open(fileName, ios::binary);
  if (!file.is_open()) {
    throw runtime_error("Output file " + fileName + " could not be created.");
  }
  writeDataToBinary<int>(file, nx);
  writeDataToBinary<int>(file, nl);
  writeDataToBinary<double>(file, Theta);
  writeDataToBinary<decltype(wvg)>(file, wvg);
  writeDataToBinary<decltype(res)>(file, res);
  file.close();
  if (!file) {
    throw runtime_error("Error in writing the file " + fileName);
  }
}

void Qstls::readAdrFixedIetFile(decltype(adrFixed) &res,
				const int i) const {
  const auto &statIn = in.getStaticInput();
  const int nx = wvg.size();
  const int nl = statIn.getNMatsubara();
  const double Theta = in.getDegeneracy();
  const string fileName = adrFixedIetFileInfo.at(i).first;
  ifstream file;
  file.open(fileName, ios::binary);
  if (!file.is_open()) {
    throw runtime_error("Input file " + fileName + " could not be opened.");
  }
  int nx_;
  int nl_;
  decltype(wvg) wvg_;
  double Theta_;
  readDataFromBinary<int>(file, nx_);
  readDataFromBinary<int>(file, nl_);
  readDataFromBinary<double>(file, Theta_);
  wvg_.resize(nx);
  //res.resize(nx, nl, nx);
  readDataFromBinary<decltype(wvg)>(file, wvg_);
  readDataFromBinary<decltype(res)>(file, res);
  file.close();
  if (!file) {
    throw runtime_error("Error in reading the file " + fileName);
  }
  checkAdrFixedFromFile(wvg_, Theta_, nl_); 
}

// Compute static structure factor
void Qstls::computeSsf(){
  const int nx = wvg.size();
  if (ssf.size() == 0) ssf.resize(nx);
  computeSsfFinite();
}

// Compute static structure factor at finite temperature
void Qstls::computeSsfFinite(){
  assert(computedChemicalPotential);
  assert(adr.size() > 0);
  assert(idr.size() > 0);
  const double Theta = in.getDegeneracy();
  const double rs = in.getCoupling();
  const int nx = wvg.size();
  const int nl = idr.size(1);
  if (ssf.size() == 0) ssf.resize(nx);
  for (int i=0; i<nx; ++i){
    Qssf ssfTmp(wvg[i], Theta, rs, ssfHF[i], nl, idr(i), adr(i));
    ssf[i] = ssfTmp.get();
  }
}

// Initial guess for qstls iterations
void Qstls::initialGuess() {
  const auto &statIn = in.getStaticInput();
  const int nx = wvg.size();
  const int nl = statIn.getNMatsubara();
  ssfOld.resize(nx);
  ssf.resize(nx);
  adrOld.resize(nx, nl);
  // From file
  const auto &stlsIn = in.getStlsInput();
  const string fileName = stlsIn.getRestartFileName();
  if (fileName != "") {
    vector<double> wvg_;
    vector<double> ssf_;
    Vector2D adr_;
    readRestart(fileName, wvg_, ssf_, adr_);
    assert(wvg_.size() == ssf_.size());
    assert(wvg_.size() == adr_.size());
    initialGuessSsf(wvg_, ssf_);
    if (useIet) initialGuessAdr(wvg_, adr_);
    return;
  }
  // Default
  Stls stls(in, false, false);
  stls.compute();
  stls.getSsf(ssfOld);
  if (useIet) {
    adrOld.resize(nx, nl);
    adrOld.fill(0.0);
    //for (auto &el : adrOld) { fill(el.begin(), el.end(), 0.0); }
  }
}

void Qstls::initialGuessSsf(const decltype(wvg) &wvg_,
			    const decltype(ssf) &ssf_) {
  const auto &statIn = in.getStaticInput();
  const int nx = wvg.size();
  const int nl = statIn.getNMatsubara();
  const Interpolator ssfi(wvg_[0], ssf_[0], wvg_.size());
  const double xMax = wvg_.back();
  for (int i=0; i<nx; ++i) {
    const double x = wvg[i];
    if (x > xMax) {
      ssfOld[i] = 1.0;
      continue;
    }
    ssfOld[i] = ssfi.eval(x);
  }
}

void Qstls::initialGuessAdr(const decltype(wvg) &wvg_,
			    const decltype(adr) &adr_) {
  const auto &statIn = in.getStaticInput();
  const int nx = wvg.size();
  const int nl = statIn.getNMatsubara();
  const int nlMax = adr_.size(1);
  const int xMax = wvg_.back();
  vector<Interpolator> itp;
  for (int l=0; l<nlMax; ++l) {
    vector<double> tmp;
    for (int i=0; i<adr_.size(); ++i){
      tmp.push_back(adr_(i,l));
    }
    itp.push_back(Interpolator(wvg_[0],tmp[0], wvg_.size()));
  }
  for (int i=0; i<nx; ++i) {
    const double x = wvg[i];
    if (x > xMax) {
      adrOld.fill(i, 0.0);
      continue;
    }
    for (int l=0; l<nl; ++l) {
      if (l <= nlMax) { adrOld(i,l) = itp[l].eval(x); }
      else { adrOld(i,l) = 0.0; }
    }
  }
}

// Compute residual error for the qstls iterations
double Qstls::computeError(){
  return rms(ssf, ssfOld, false);
}

// Update solution during qstls iterations
void Qstls::updateSolution(){
  const auto &statIn = in.getStaticInput();
  const double aMix = statIn.getMixingParameter();
  ssfOld = sum(mult(ssf, aMix), mult(ssfOld, 1 - aMix));
}

// Write output files
void Qstls::writeOutput() const{
  Stls::writeOutput();
  writeAdr();
  writeRestart();
}

void Qstls::writeAdr() const {
  if (in.getDegeneracy() == 0.0) return;
  const string fileName = format<double,double>("adr_rs%.3f_theta%.3f_"
						+ in.getTheory() + ".dat",
						in.getCoupling(),
						in.getDegeneracy());
  ofstream file;
  file.open(fileName);
  if (!file.is_open()) {
    throw runtime_error("Output file " + fileName + " could not be created.");
  }
  const int nx = adr.size(0);
  const int nl = adr.size(1);
  for (int i=0; i<nx; ++i){
    const string el1 = format<double>("%.8e", wvg[i]);
    file << el1;
    for (int l=0; l<nl; ++l) {
      const string el2 = format<double>("%.8e ", adr(i,l));
      file << el2;
    }
    file << endl;
  }
  file.close();
}

// Restart files
void Qstls::writeRestart() const {
  const string fileName = format<double,double>("restart_rs%.3f_theta%.3f_"
						+ in.getTheory() + ".bin",
						in.getCoupling(),
						in.getDegeneracy());
  ofstream file;
  file.open(fileName, ios::binary);
  if (!file.is_open()) {
    throw runtime_error("Output file " + fileName + " could not be created.");
  }
  const auto &statIn = in.getStaticInput();
  int nx = wvg.size();
  int nl = statIn.getNMatsubara();
  writeDataToBinary<int>(file, nx);
  writeDataToBinary<int>(file, nl);
  writeDataToBinary<double>(file, in.getDegeneracy());
  writeDataToBinary<decltype(wvg)>(file, wvg);
  writeDataToBinary<decltype(ssf)>(file, ssf);
  writeDataToBinary<decltype(adr)>(file, adr);
  writeDataToBinary<decltype(adrFixed)>(file, adrFixed);
  file.close();
  if (!file) {
    throw runtime_error("Error in writing the file " + fileName);
  }
}

void Qstls::readRestart(const string &fileName,
			decltype(wvg) &wvg_,
			decltype(ssf) &ssf_,
			decltype(adr) &adr_) const {
  decltype(adrFixed) tmp1;
  double tmp2;
  readRestart(fileName, wvg_, ssf_, adr_, tmp1, tmp2);
}

void Qstls::readRestart(const string &fileName,
			decltype(wvg) &wvg_,
			decltype(adrFixed) &adrFixed_,
			double &Theta) const {
  decltype(wvg) tmp1;
  decltype(adr) tmp2;
  readRestart(fileName, wvg_, tmp1, tmp2, adrFixed_, Theta);
}

void Qstls::readRestart(const string &fileName,
			decltype(wvg) &wvg_,
			decltype(ssf) &ssf_,
			decltype(adr) &adr_,
			decltype(adrFixed) &adrFixed_,
			double &Theta) const {
  ifstream file;
  file.open(fileName, ios::binary);
  if (!file.is_open()) {
    throw runtime_error("Input file " + fileName + " could not be opened.");
  }
  int nx;
  int nl;
  readDataFromBinary<int>(file, nx);
  readDataFromBinary<int>(file, nl);
  readDataFromBinary<double>(file, Theta);
  wvg_.resize(nx);
  ssf_.resize(nx);
  adr_.resize(nx, nl);
  //adrFixed_.resize(nx, nl, nx);
  readDataFromBinary<decltype(wvg_)>(file, wvg_);
  readDataFromBinary<decltype(ssf_)>(file, ssf_);
  readDataFromBinary<decltype(adr_)>(file, adr_);
  readDataFromBinary<decltype(adrFixed_)>(file, adrFixed_);
  file.close();
  if (!file) {
    throw runtime_error("Error in reading the file " + fileName);
  }
}

// -----------------------------------------------------------------
// Qssf class
// -----------------------------------------------------------------

// Get static structure factor
double Qssf::get() const {
  if (x == 0.0) return 0.0;
  const double fact1 = 4.0*lambda*rs/M_PI;
  const double x2 = x*x;
  double fact2 = 0.0;
  for (int l=0; l<nl; ++l) {
    const double fact3 = 1.0 + fact1/x2*(1.0 - adr[l]/idr[l])*idr[l];
    double fact4 = idr[l]*idr[l]*(1.0 - adr[l]/idr[l])/fact3;
    if (l>0) fact4 *= 2;
    fact2 += fact4;
  }
  return ssfHF - 1.5 * fact1/x2 * Theta * fact2;
}


// -----------------------------------------------------------------
// AdrBase class
// -----------------------------------------------------------------

// Compute static structure factor
double AdrBase::ssf(const double y) const {
  return ssfi.eval(y);
}

// -----------------------------------------------------------------
// Adr class
// -----------------------------------------------------------------

// Compute fixed component
double Adr::fix(const double y) const {
  return fixi.eval(y);
}

// Integrand
double Adr::integrand(const double y) const{
  return y * fix(y) * (ssf(y) - 1.0);
}

// Get result of integration
void Adr::get(const vector<double> &wvg,
	      const Vector2D &fixed,
	      vector<double> &res) {
  const int nx = wvg.size();
  res.resize(nl);
  assert(fixed.size(1) > 0);
  assert(fixed.size(1) == nx);
  for (int l = 0; l < nl; ++l){
    fixi = Interpolator(&wvg[0], fixed(l), nx);
    auto func = [&](double y)->double{return integrand(y);};
    itg.compute(func, yMin, yMax);
    res[l] = itg.getSolution();
    res[l] *= (l==0) ? -3.0/(4.0*Theta) : -3.0/8.0;
  }
}

// -----------------------------------------------------------------
// AdrFixed class
// -----------------------------------------------------------------

// Get fixed component
void AdrFixed::get(vector<double> &wvg,
		   Vector2D &res) const {
  const int nx = wvg.size();
  res.resize(nl, nx);
  if ( x == 0.0 ) { res.fill(0.0); };
  const double x2 = x*x;
  for (int l = 0; l < nl; ++l){
    for (int i = 0; i < nx; ++i) {
      const double xq = x*wvg[i];
      auto tMin = [&](double q)->double{return x2 - xq;};
      auto tMax = [&](double q)->double{return x2 + xq;};
      auto func1 = [&](double q)->double{return integrand1(q, l);};
      auto func2 = [&](double t)->double{return integrand2(t, wvg[i], l);};
      itg.compute(func1, func2, qMin, qMax, tMin, tMax);
      res(l,i) = itg.getSolution();
    }
  }
}

// Integrands for the fixed component
double AdrFixed::integrand1(const double q, const double l) const {
  if (l == 0) return q/(exp(q*q/Theta - mu) + exp(-q*q/Theta + mu) + 2.0);
  return q/(exp(q*q/Theta - mu) + 1.0);
}

double AdrFixed::integrand2(const double t, const double y, const double l) const {
  const double q = itg.getX();
  if (q == 0) return 0;
  const double x2 = x*x;
  const double y2 = y*y;
  const double q2 = q*q;
  const double txq = 2.0 * x * q;
  if (l == 0) {
    if (t == txq) return 2.0*q2/(y2 + 2.0*txq - x2);
    const double t2 = t*t;
    double logarg = (t + txq)/(t - txq);
    logarg = (logarg < 0.0) ? -logarg : logarg;
    return 1.0/(2.0*t + y2 - x2)*((q2 - t2/(4.0*x2))*log(logarg) + q*t/x);
  }
  const double tplT = 2.0 * M_PI * l * Theta;
  const double tplT2 = tplT*tplT;
  const double txqpt = txq + t;
  const double txqmt = txq - t;
  const double txqpt2 = txqpt*txqpt;
  const double txqmt2 = txqmt*txqmt;
  const double logarg = (txqpt2 + tplT2)/(txqmt2 + tplT2);
  return 1.0/(2.0*t + y*y - x*x)*log(logarg);
}

// -----------------------------------------------------------------
// AdrIet class
// -----------------------------------------------------------------

// Compute ideal density response
double AdrIet::idr(const double y) const {
  return idri.eval(y);
}

// Compute auxiliary density response
double AdrIet::adr(const double y) const {
  return adri.eval(y);
}

// Compute auxiliary density response
double AdrIet::bf(const double y) const {
  return bfi.eval(y);
}

// Compute fixed component
double AdrIet::fix(const double x, const double y) const {
  return fixi.eval(x,y);
}

// Integrands
double AdrIet::integrand1(const double q) const {
  if (q == 0.0) return 0.0;
  return (1 - bf(q)) * ssf(q) + (adr(q)/idr(q) * (ssf(q) - 1.0) - 1.0) / q;
}

double AdrIet::integrand2(const double y) const {
  const double q = itg.getX();
  return y * fix(q,y) * (ssf(y) - 1.0);
}

// Get result of integration
// for now we work with a single matsubara frequency. If we want results for all
// frequencies then fixed has to become Vector3D and res a vector<double>
void AdrIet::get(const vector<double> &wvg,
		 const Vector2D &fixed,
		 double &res) {
  const int nx = wvg.size();
  if (x == 0.0) {
    res = 0.0;
    return;
  }
  // NOT SURE ABOUT THIS INTERPOLATOR CALL
  fixi = Interpolator2D(&wvg[0], &wvg[0], fixed(l), nx, nx);
  auto yMin = [&](double q)->double{return q + x;};
  auto yMax = [&](double q)->double{return (q > x) ? q - x : x - q;};
  auto func1 = [&](double q)->double{return integrand1(q);};
  auto func2 = [&](double y)->double{return integrand2(y);};
  itg.compute(func1, func2, qMin, qMax, yMin, yMax);
  res = itg.getSolution();
  //res[l][i] = itg.getSolution();
  // for (int l = 0; l < nl; ++l){
  //   res[l].resize(nx);
  //   if (x == 0.0) {
  //     fill(res[l].begin(), res[l].end(), 0.0);
  //     continue;
  //   }
  //   for (int i = 0; i < nx; ++i) {
  //     const double xq = x*wvg[i];
  //     auto tMin = [&](double q)->double{return x2 - xq;};
  //     auto tMax = [&](double q)->double{return x2 + xq;};
  //     auto func1 = [&](double q)->double{return integrand1(q, l);};
  //     auto func2 = [&](double t)->double{return integrand2(t, wvg[i], l);};
  //     itg.compute(func1, func2, qMin, qMax, tMin, tMax);
  //     res[l][i] = itg.getSolution();
  //   }
  // }
}

// -----------------------------------------------------------------
// AdrFixedIet class
// -----------------------------------------------------------------

// get fixed component
// void AdrFixedIet::get(vector<double> &wvg,
// 		      Vector3D<double> &res) const {
//   res.resize(nl);
//   const int nx = wvg.size();
//   for (int l = 0; l < nl; ++l){
//     res[l].resize(nx);
//     for (int i = 0; i < nx; ++i) {
//       res[l][i].resize(nx);
//       if (x == 0 || wvg[i] == 0.0) {
// 	fill(res[l][i].begin(), res[l][i].end(), 0.0);
// 	continue;
//       }
//       for (int j = 0; j < nx; ++j) {
// 	auto func = [&](double t)->double{return integrand(t, wvg[j], wvg[i], l);};
// 	itg.compute(func, tMin, tMax);
// 	res[l][i][j] = itg.getSolution();
//       }  
//     }
//   }
// }

// Integrand for the fixed component
double AdrFixedIet::integrand(const double t, const double y,
			      const double q, const double l) const {
  // l ---> l
  // x ---> x
  // u ---> q
  // w ---> y
  // y ---> t
  const double x2 = x*x;
  const double q2 = q*q;
  const double y2 = y*y;
  const double t2 = t*t;
  const double fxt = 4.0*x*t;
  const double qmypx = q2 - y2 + x2;
  if (l == 0) {
    double logarg = (qmypx + fxt)/(qmypx - fxt);
    if (logarg < 0.0) logarg = -logarg;
    return t / (exp(t2/Theta - mu) + exp(-t2/Theta + mu) + 2.0)*
      ((t2 - qmypx*qmypx/(16.0*x2))*log(logarg) + (t/x)*qmypx/2.0);
  }
  const double fplT = 4.0*M_PI*l*Theta;
  const double fplT2 = fplT*fplT;
  const double logarg = ((qmypx + fxt)*(qmypx + fxt) + fplT2)/
    ((qmypx - fxt)*(qmypx - fxt) + fplT2);
  return t / (exp(y2/Theta - mu) + 1.0)*log(logarg);
}
