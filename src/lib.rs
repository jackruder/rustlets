use numpy::{PyReadonlyArray1, PyReadwriteArray2, PyArray1, PyArray2};
use pyo3::{prelude::*};
use num_complex::{Complex64};

pub mod waveletanalysis;

use crate::waveletanalysis::*;




//
//fn cwt_morlet(timeseries: Vec<f64>, hz: f64, steps: u32) -> (Vec<Vec<Complex64>>, Vec<f64>) {
 //   cwt(&timeseries, morlet_fourier, hz, steps as f64, false) 
//} 

//fn cwt_morlet_ext(timeseries: Vec<f64>, hz: f64, steps: u32, normalize: bool) -> (Vec<Vec<Complex64>>, Vec<f64>) {
//    cwt(&timeseries, morlet_fourier, hz, steps as f64, normalize)
//} 

#[pyfunction]
#[pyo3(name = "cwt_morlet")]
fn cwt_morlet_py<'py>(py: Python<'py>, timeseries: PyReadonlyArray1<f64>, hz: f64, steps: u32) -> (PyArray2<Complex64>, PyArray1<f64>) {
    let timeseries = timeseries.as_array();
    let (cwtm, scales) = cwt(&timeseries, morlet_fourier, hz, steps as f64, false);
    (cwtm.into_py_array(), scales.into_py_array())
}

#[pyfunction]
#[pyo3(name = "cwt_morlet_ext")]
fn cwt_morlet_ext_py<'py>(py: Python<'py>, timeseries: PyReadonlyArray1<f64>, hz: f64, steps: u32, normalize: bool) -> (PyArray2<Complex64>, PyArray1<f64>) {
    let timeseries = timeseries.as_array();
    let (cwtm, scales) = cwt(&timeseries, morlet_fourier, hz, steps as f64, normalize);
    (cwtm.into_py_array(), scales.into_py_array())
}

#[pyfunction]
#[pyo3(name = "icwt_morlet")]
fn icwt_morlet_py(cwtm: Vec<Vec<Complex64>>, scales: Vec<f64>, times: Vec<f64>) -> Vec<f64> {
    icwt_morlet(&cwtm, &scales, &times)
} 

#[pyfunction]
#[pyo3(name = "morlet_wavelength")]
fn morlet_wavelength_py(omega_0: f64) -> f64 {
    morlet_wavelength(omega_0)
}



#[pymodule]
fn rustlets(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cwt_morlet_py, m)?)?;
    m.add_function(wrap_pyfunction!(cwt_morlet_ext_py, m)?)?;
    m.add_function(wrap_pyfunction!(icwt_morlet_py, m)?)?;
    m.add_function(wrap_pyfunction!(morlet_wavelength_py, m)?)?;
    Ok(())
}





