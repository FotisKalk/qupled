#include <string>
#include <fstream>
#include <sstream>
#include "input.hpp"

// --- Input ---
 
Input::Input(){
  theory = "stls";
  Theta = 1.0;
  rs = 1.0;
  nThreads = 1;
  stat = make_shared<StaticInput>();
  stls = make_shared<StlsInput>();
}

string Input::getTheory(){
  return theory;
}

double Input::getDegeneracy(){
  return Theta;
}

double Input::getCoupling(){
  return rs;
}

void Input::setTheory(cString &theory){
  if (theory != "stls") {
    throw runtime_error("Unknown theory: " + theory);
  }
  this->theory = theory;
}

void Input::setDegeneracy(cString &Theta){
  if (isNegative<double>(Theta)) {
    throw runtime_error("The quantum degeneracy parameter can't be negative");
  }
  this->Theta = stod(Theta);
}

void Input::setCoupling(cString &rs){
  if (isNotPositive<double>(rs)) {
    throw runtime_error("The quantum coupling parameter must be larger than zero");
  }
  this->rs = stod(rs);
}

void Input::setThreads(cString  &nThreads){
  if (isNotPositive<int>(nThreads)) {
    throw runtime_error("The number of threads must be positive");
  }
  this->nThreads = stoi(nThreads);
}

void Input::readInput(cString &fileName){
  ifstream file(fileName);
  if (file.is_open()) {
    string line;
    while (getline(file, line)) {
      parseInputLine(line);
    }
    file.close();
  }
  else {
    throw runtime_error("Input file " + fileName + " could not be opened.");    
  }   
}

void Input::parseInputLine(cString &line){
  bool isComment = line[0] == '#';
  bool isEmpty = line[0] == '\n' || line.length()==0;
  if (!isComment && !isEmpty) {
    vector<string> tokens = tokenize(line, ' ');
    if (tokens.size() < 2) {
      throw runtime_error("wrong line format: " + line);
    }
    assignInputToData(tokens);    
  }
}

void Input::assignInputToData(cVector<string> &input){
  cVector<string> keyword = tokenize(input[0], '.');
  map<string, function<void(cString &, cString &)>> funcArr;
  funcArr["base"] = [this](cString &s1, cString &s2) {this->assignInputToBaseData(s1, s2);};
  funcArr["static"] = [this](cString &s1, cString &s2) {this->assignInputToStaticData(s1, s2);};
  funcArr["stls"] = [this](cString &s1, cString &s2) {this->assignInputToStlsData(s1, s2);};
  try{
    matchKeyAndData(keyword, input[1], funcArr);
  }
  catch (const runtime_error& err) {
    cerr << err.what() << endl;
  }
}

void Input::assignInputToBaseData(cString &keyword, cString &value){
  map<string, function<void(cString &)>> funcArr;
  funcArr["theory"] = [this](cString &s1) {this->setTheory(s1);};
  funcArr["degeneracy"] = [this](cString &s1) {this->setDegeneracy(s1);};
  funcArr["coupling"] = [this](cString &s1) {this->setCoupling(s1);};
  funcArr["threads"] = [this](cString &s1) {this->setThreads(s1);};
  try{
    matchKeyAndData(keyword, value, funcArr);
  }
  catch (const runtime_error& err) {
    cerr << err.what() << endl;
  }
}

void Input::assignInputToStaticData(cString &keyword, cString &value){
  stat->assignInputToData(keyword, value);
}

void Input::assignInputToStlsData(cString &keyword, cString &value){
  stls->assignInputToData(keyword, value);
}

void Input::print(){
  cout << "base.theory = " << theory << endl;
  cout << "base.degeneracy = " << Theta << endl;
  cout << "base.coupling = " << rs << endl;
  cout << "base.threads = " << nThreads << endl;
  stat->print();
  stls->print();
}

// --- StaticInput ---

StaticInput::StaticInput(){
  aMix = 1.0;
  errMin = 1e-5;
  dx = 0.1;
  xmax = 10.0;
  vector<double> muGuessDefault = {-10, 10};
  muGuess.assign(muGuessDefault.begin(), muGuessDefault.end());
  nl = 128;
  nIter = 1000;
}

double StaticInput::getMixingParameter(){
  return aMix;
}

double StaticInput::getErrMin(){
  return errMin;
}

double StaticInput::getWaveVectorGridRes(){
  return dx;
}

double StaticInput::getWaveVectorGridCutoff(){
  return xmax;
}
 
vector<double> StaticInput::getChemicalPotentialGuess(){
  return muGuess;
}

size_t StaticInput::getNMatsubara(){
  return nl;
}

size_t StaticInput::getNIter(){
  return nIter;
}

void StaticInput::setMixingParameter(cString &aMix){
  if (isNegative<double>(aMix) || isLarger<double>(aMix, 1.0)) {
    throw runtime_error("The mixing parameter must be a number between zero and one");
  }
  this->aMix = stod(aMix);
}

void StaticInput::setErrMin(cString &errMin){
  if (isNotPositive<double>(errMin)) {
    throw runtime_error("The minimum error for convergence must be larger than zero");
  }
  this->errMin = stod(errMin);
}

void StaticInput::setChemicalPotentialGuess(cString &muGuess){
  vector<string> muGuessVec = tokenize(muGuess, ',');
  if (muGuessVec.size() != 2) {
    throw runtime_error("Wrong format for the chemical potential input.");
  }
  vector<double> muGuessNum;
  for (string mu : muGuessVec) muGuessNum.push_back(stod(mu));
  this->muGuess.assign(muGuessNum.begin(), muGuessNum.end());
}
 
void StaticInput::setWaveVectorGridRes(cString &dx){
  if (isNotPositive<double>(dx)) {
    throw runtime_error("The wave-vector grid resolution must be larger than zero");
  }
  this->dx = stod(dx);
}

void StaticInput::setWaveVectorGridCutoff(cString &xmax){
  if (isNotPositive<double>(xmax)) {
    throw runtime_error("The wave-vector grid cutoff must be larger than zero");
  }
  if (!isLarger<double>(xmax, dx)) {
    throw runtime_error("The wave-vector grid cutoff must be larger than the resolution");
  }
  this->xmax = stod(xmax);
}

void StaticInput::setNMatsubara(cString &nl){
  if (isNegative<int>(nl)) {
    throw runtime_error("The number of matsubara frequencies can't be negative");
  }
  this->nl = stoi(nl);
}

void StaticInput::setNIter(cString &nIter){
  if (isNegative<int>(nIter)) {
    throw runtime_error("The maximum number of iterations can't be negative");
  }
  this->nIter = stoi(nIter); 
}

void StaticInput::assignInputToData(const string &keyword, const string &value){
  map<string, function<void(cString &)>> funcArr;
  funcArr["mixing"] = [this](cString &s1) {this->setMixingParameter(s1);};
  funcArr["error"] = [this](cString &s1) {this->setErrMin(s1);};
  funcArr["waveVectorResolution"] = [this](cString &s1) {this->setWaveVectorGridRes(s1);};
  funcArr["waveVectorCutoff"] = [this](cString &s1) {this->setWaveVectorGridCutoff(s1);};
  funcArr["chemicalPotential"] = [this](cString &s1) {this->setChemicalPotentialGuess(s1);};
  funcArr["matsubara"] = [this](cString &s1) {this->setNMatsubara(s1);};
  funcArr["iterations"] = [this](cString &s1) {this->setNIter(s1);};
  try{
    matchKeyAndData(keyword, value, funcArr);
  }
  catch (const runtime_error& err) {
    cerr << err.what() << endl;
  }
}

void StaticInput::print(){
  cout << "static.mixing = " << aMix << endl;
  cout << "static.error = " << errMin << endl;
  cout << "static.waveVectorResolution = " << dx << endl;
  cout << "static.waveVectorCutoff = " << xmax << endl;
  cout << "static.chemicalPotential = " << muGuess.at(0) << "," << muGuess.at(1) << endl;
  cout << "static.matsubara = " << nl << endl;
  cout << "static.iterations = " << nIter << endl;
}

// --- StlsInput ---

StlsInput::StlsInput(){
  IETMapping = "standard";
  restartFileName = NO_FILE_NAME;
}

string StlsInput::getIETMapping(){
  return IETMapping;
}

string StlsInput::getRestartFileName(){
  return restartFileName;
}

void StlsInput::setIETMapping(cString &IETMapping){
  if (IETMapping != "standard" &&
      IETMapping != "sqrt" &&
      IETMapping != "linear") {
    throw runtime_error("Unknown IET mapping" + IETMapping);
  }
  this->IETMapping = IETMapping;
}

void StlsInput::setRestartFileName(cString &restartFileName){
  if (restartFileName == NO_FILE_NAME) {
    throw runtime_error("Restart file name " + restartFileName + "is not allowed");
  }
  this->restartFileName = restartFileName;
} 
  
void StlsInput::assignInputToData(const string &keyword, const string &value){
  map<string, function<void(cString &)>> funcArr;
  funcArr["iet"] = [this](cString &s1) {this->setIETMapping(s1);};
  funcArr["restart"] = [this](cString &s1) {this->setRestartFileName(s1);};
  try{
    matchKeyAndData(keyword, value, funcArr);
  }
  catch (const runtime_error& err) {
    cerr << err.what() << endl;
  }
} 

void StlsInput::print(){
  string outputName = (restartFileName == NO_FILE_NAME) ? "" : restartFileName;
  cout << "stls.iet = " << IETMapping << endl;
  cout << "stls.restart = " << outputName  << endl;
}