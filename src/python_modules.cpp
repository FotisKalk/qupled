#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "input.hpp"
#include "stls.hpp"
#include "qstls.hpp"

namespace bp = boost::python;
namespace bn = boost::python::numpy;

// Methods that need wrapping to pass arrays between native and python

namespace arrayWrapper {

  void CheckRowMajor(const bn::ndarray &nda) {
    const bn::ndarray::bitflag flags = nda.get_flags();
    const bool isRowMajor = flags & bn::ndarray::C_CONTIGUOUS;
    if (!isRowMajor) {
      throw runtime_error("The numpy array is not stored in row major order (c-contiguous)");
    }
  }
  
  vector<double> toVector(const bn::ndarray &nda){
    if (nda.get_nd() != 1) {
      throw runtime_error("Incorrect numpy array dimensions");
    }
    const Py_intptr_t* shape = nda.get_shape();
    const int dim = nda.get_nd();
    // the numpy array is flattened to a one dimensional std::vector
    Py_intptr_t n = 1;
    for (int i = 0; i < dim; ++i){ n *= shape[i]; }
    double* ptr = reinterpret_cast<double*>(nda.get_data());
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) { v[i] = *(ptr + i); }
    return v;
  }

  vector<double> toVector(const bp::list &list){
    int n = len(list);
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i){ v[i] = bp::extract<double>(list[i]); }
    return v;
  }

  vecUtil::Vector2D toVector2D(const bn::ndarray &nda){
    if (nda.get_nd() != 2) {
      throw runtime_error("Incorrect numpy array dimensions");
    }
    CheckRowMajor(nda);
    const Py_intptr_t* shape = nda.get_shape();
    const int sz1 = shape[0];
    const int sz2 = shape[1];
    vecUtil::Vector2D v(sz1, sz2);
    double* ptr = reinterpret_cast<double*>(nda.get_data());
    for (int i = 0; i < sz1; ++i){
      for (int j = 0; j < sz2; ++j) {
	v(i,j) = *(ptr + j + i*sz2);
      }
    }
    return v;
  }
  
  template<typename T>
  bn::ndarray toNdArray(const T &v){
    Py_intptr_t shape[1];
    shape[0] = v.size();
    bn::ndarray result = bn::zeros(1, shape, bn::dtype::get_builtin<double>());
    std::copy(v.begin(), v.end(), reinterpret_cast<double*>(result.get_data()));
    return result;
  }
  
}

namespace StlsInputWrapper {
  
  bn::ndarray getChemicalPotentialGuess(StlsInput &in){
    return arrayWrapper::toNdArray(in.getChemicalPotentialGuess());
  }
  
  void setChemicalPotentialGuess(StlsInput &in,
				 const bp::list &muGuess){
    in.setChemicalPotentialGuess(arrayWrapper::toVector(muGuess));
  }

  struct SlfcGuess {
    bn::ndarray wvg = arrayWrapper::toNdArray(vector<double>(0));
    bn::ndarray slfc = arrayWrapper::toNdArray(vector<double>(0));
  };
    
  StlsInputWrapper::SlfcGuess getGuess(StlsInput &in){
    StlsInput::SlfcGuess guess_ = in.getGuess();
    StlsInputWrapper::SlfcGuess guess;
    guess.wvg = arrayWrapper::toNdArray(guess_.wvg);
    guess.slfc = arrayWrapper::toNdArray(guess_.slfc);
    return guess;
  }
  
  void setGuess(StlsInput &in,
		const StlsInputWrapper::SlfcGuess &guess){
    StlsInput::SlfcGuess guess_;
    guess_.wvg = arrayWrapper::toVector(guess.wvg);
    guess_.slfc = arrayWrapper::toVector(guess.slfc);
    in.setGuess(guess_);
  }

}

namespace StlsWrapper {

  bn::ndarray getBf(const Stls &stls){
    return arrayWrapper::toNdArray(stls.getBf());
  }

  bn::ndarray getIdr(const Stls &stls){
    const vecUtil::Vector2D &idrNative = stls.getIdr();
    const size_t nx = idrNative.size(0);
    const size_t nl = idrNative.size(1);
    bn::ndarray idr = arrayWrapper::toNdArray(idrNative);
    bp::tuple shape = bp::make_tuple(nx, nl);
    idr = idr.reshape(shape);
    return idr;
  }

  bn::ndarray getRdf(const Stls &stls,
		     const bn::ndarray &r){
    return arrayWrapper::toNdArray(stls.getRdf(arrayWrapper::toVector(r)));
  }
  
  bn::ndarray getSdr(const Stls &stls){
    return arrayWrapper::toNdArray(stls.getSdr());
  }
  
  bn::ndarray getSlfc(const Stls &stls){
    return arrayWrapper::toNdArray(stls.getSlfc());
  }
  
  bn::ndarray getSsf(const Stls &stls){
    return arrayWrapper::toNdArray(stls.getSsf());
  }

  bn::ndarray getSsfHF(const Stls &stls){
    return arrayWrapper::toNdArray(stls.getSsfHF());
  }
  
  bn::ndarray getWvg(const Stls &stls){
    return arrayWrapper::toNdArray(stls.getWvg());
  }

}


namespace QstlsInputWrapper {
  
  struct QstlsGuess {
    bn::ndarray wvg = arrayWrapper::toNdArray(vector<double>(0));
    bn::ndarray ssf = arrayWrapper::toNdArray(vector<double>(0));
    bn::ndarray adr = arrayWrapper::toNdArray(vector<double>(0));
    int matsubara = 0;
  };

  QstlsInputWrapper::QstlsGuess getGuess(QstlsInput &in){
    QstlsInput::QstlsGuess guess_ = in.getGuess();
    QstlsInputWrapper::QstlsGuess guess;
    const int sz1 = guess_.adr.size(0);
    const int sz2 = guess_.adr.size(1);
    guess.wvg = arrayWrapper::toNdArray(guess_.wvg);
    guess.ssf = arrayWrapper::toNdArray(guess_.ssf);
    bn::ndarray adrTmp = arrayWrapper::toNdArray(guess_.adr);
    guess.adr = adrTmp.reshape(bp::make_tuple(sz1, sz2));
    guess.matsubara = guess_.matsubara;
    return guess;
  }
  
  void setGuess(QstlsInput &in,
		const QstlsInputWrapper::QstlsGuess &guess){
    QstlsInput::QstlsGuess guess_;
    guess_.wvg = arrayWrapper::toVector(guess.wvg);
    guess_.ssf = arrayWrapper::toVector(guess.ssf);
    if (guess.adr.shape(0) > 0) {
      guess_.adr = arrayWrapper::toVector2D(guess.adr);
    }
    guess_.matsubara = guess.matsubara;
    in.setGuess(guess_);
  }
  
}

namespace QstlsWrapper {
    
  bn::ndarray getAdr(const Qstls &qstls){
    const vecUtil::Vector2D &adrNative = qstls.getAdr();
    const size_t nx = adrNative.size(0);
    const size_t nl = adrNative.size(1);
    bn::ndarray adr = arrayWrapper::toNdArray(adrNative);
    bp::tuple shape = bp::make_tuple(nx, nl);
    adr = adr.reshape(shape);
    return adr;
  }
  
  bn::ndarray getAdrFixed(const Qstls &qstls){
    const vecUtil::Vector3D &adrNative = qstls.getAdrFixed();
    const size_t nx = adrNative.size(0);
    const size_t nl = adrNative.size(1);
    bn::ndarray adr = arrayWrapper::toNdArray(adrNative);
    bp::tuple shape = bp::make_tuple(nx, nl, nx);
    adr = adr.reshape(shape);
    return adr;
  }

}

namespace thermoWrapper {

  bn::ndarray computeRdf(const bn::ndarray &rIn,
			 const bn::ndarray &wvgIn,
			 const bn::ndarray &ssfIn) {
    const vector<double> &r = arrayWrapper::toVector(rIn);
    const vector<double> &wvg = arrayWrapper::toVector(wvgIn);
    const vector<double> &ssf = arrayWrapper::toVector(ssfIn);
    return arrayWrapper::toNdArray(thermoUtil::computeRdf(r, wvg, ssf));
  }

  double computeInternalEnergy(const bn::ndarray &wvgIn,
			       const bn::ndarray &ssfIn,
			       const double &coupling) {
    const vector<double> &wvg = arrayWrapper::toVector(wvgIn);
    const vector<double> &ssf = arrayWrapper::toVector(ssfIn);
    return thermoUtil::computeInternalEnergy(wvg, ssf, coupling);
  }
  
}

// Classes exposed to Python
BOOST_PYTHON_MODULE(qupled)
{

  // Docstring formatting
  bp::docstring_options docopt;
  docopt.enable_all();
  docopt.disable_cpp_signatures();
	
  // Numpy library initialization
  bn::initialize();
    
  // Wrapper for vector<double>
  bp::class_<std::vector<double>>("vector<double>")
    .def(bp::vector_indexing_suite<std::vector<double>>());
  
  // Classes to manage the input
  bp::class_<Input>("Input", 
		    "Base class to handle the inputs ",
		    bp::init<const double, const double, const string>())
    .add_property("coupling",
		  &Input::getCoupling,
		  &Input::setCoupling,
		  "float: Coupling parameter")
    .add_property("degeneracy",
		  &Input::getDegeneracy,
		  &Input::setDegeneracy,
		  "float: Degeneracy parameter")
    .add_property("int2DScheme",
		  &Input::getInt2DScheme,
		  &Input::setInt2DScheme,
		  "str: Scheme used to solve two-dimensional integrals \n"
		  "allowed options include: \n\n"
		  "- full (the inner integral is evaluated at arbitrary points "
		  "selected automatically by the quadrature rule \n\n"
		  "- segregated (the inner integral is evaluated on a fixed "
		  "grid that depends on the integrand that is being processed\n\n"
		  "Segregated is usually faster than full but it could become "
		  "less accurate if the fixed points are not chosen correctly"
		  )
    .add_property("threads",
		  &Input::getNThreads,
		  &Input::setNThreads,
		  "int: Number of OMP threads for parallel calculations")
    .add_property("theory",
		  &Input::getTheory,
		  &Input::setTheory,
		  "str: Theory to be solved")
    .def("print", &Input::print, "Prints the content of the input structure")
    .def("isEqual", &Input::isEqual, "Compares two input structures and returns "
	 "true if they are identical");

  bp::class_<StlsInputWrapper::SlfcGuess>("SlfcGuess", "Class used to define an initial guess"
					  "for STLS and STLS-IET schemes")
    .def_readwrite("wvg", &StlsInputWrapper::SlfcGuess::wvg, "Wave-vector grid")
    .def_readwrite("slfc", &StlsInputWrapper::SlfcGuess::slfc, "Static local field correction");
    
  bp::class_<StlsInput, bp::bases<Input>>("StlsInput",
					  bp::init<const double, const double, const string>())
    .add_property("chemicalPotential",
		  StlsInputWrapper::getChemicalPotentialGuess,
		  StlsInputWrapper::setChemicalPotentialGuess)
    .add_property("error",
		  &StlsInput::getErrMin,
		  &StlsInput::setErrMin)
    .add_property("mixing",
		  &StlsInput::getMixingParameter,
		  &StlsInput::setMixingParameter)
    .add_property("iet",
		  &StlsInput::getIETMapping,
		  &StlsInput::setIETMapping)
    .add_property("matsubara",
		  &StlsInput::getNMatsubara,
		  &StlsInput::setNMatsubara)
    .add_property("iterations",
		  &StlsInput::getNIter,
		  &StlsInput::setNIter)
    .add_property("outputFrequency",
		  &StlsInput::getOutIter,
		  &StlsInput::setOutIter)
    .add_property("recoveryFile",
		  &StlsInput::getRecoveryFileName,
		  &StlsInput::setRecoveryFileName)
    .add_property("guess",
		  StlsInputWrapper::getGuess,
		  StlsInputWrapper::setGuess)
    .add_property("resolution",
		  &StlsInput::getWaveVectorGridRes,
		  &StlsInput::setWaveVectorGridRes)
    .add_property("cutoff",
		  &StlsInput::getWaveVectorGridCutoff,
		  &StlsInput::setWaveVectorGridCutoff)
    .def("print", &StlsInput::print)
    .def("isEqual", &StlsInput::isEqual);

  bp::class_<QstlsInputWrapper::QstlsGuess>("QstlsGuess")
    .def_readwrite("wvg", &QstlsInputWrapper::QstlsGuess::wvg)
    .def_readwrite("ssf", &QstlsInputWrapper::QstlsGuess::ssf)
    .def_readwrite("adr", &QstlsInputWrapper::QstlsGuess::adr)
    .def_readwrite("matsubara", &QstlsInputWrapper::QstlsGuess::matsubara);

  bp::class_<QstlsInput>("QstlsInput")
    .add_property("guess",
		  QstlsInputWrapper::getGuess,
		  QstlsInputWrapper::setGuess)
    .add_property("fixed",
		  &QstlsInput::getFixed,
		  &QstlsInput::setFixed)
    .add_property("fixediet",
		  &QstlsInput::getFixedIet,
		  &QstlsInput::setFixedIet)
    .def("print", &QstlsInput::print)
    .def("isEqual", &QstlsInput::isEqual);
    
  // Class to solve classical schemes
  bp::class_<Stls>("Stls",
		   bp::init<const StlsInput>())
    .def("compute", &Stls::compute)
    .def("getRdf", StlsWrapper::getRdf)
    .add_property("bf", StlsWrapper::getBf)
    .add_property("idr", StlsWrapper::getIdr)
    .add_property("recovery", &Stls::getRecoveryFileName)
    .add_property("sdr", StlsWrapper::getSdr)
    .add_property("slfc", StlsWrapper::getSlfc)
    .add_property("ssf", StlsWrapper::getSsf)
    .add_property("ssfHF", StlsWrapper::getSsfHF)
    .add_property("uInt", &Stls::getUInt)
    .add_property("wvg", StlsWrapper::getWvg);
  
  // Class to solve quantum schemes
  bp::class_<Qstls, bp::bases<Stls>>("Qstls",
				     bp::init<const StlsInput, const QstlsInput>())
    .def("compute", &Qstls::compute)
    .add_property("adr", QstlsWrapper::getAdr);

  // Post-process methods
  bp::def("computeRdf", thermoWrapper::computeRdf);
  bp::def("computeInternalEnergy", thermoWrapper::computeInternalEnergy);
  
}
