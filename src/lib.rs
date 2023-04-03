use pyo3::{prelude::*};
use num_complex::{Complex64};

pub mod waveletanalysis;

use crate::waveletanalysis::*;





#[pyfunction]
#[pyo3(name = "cwt_morlet")]
fn cwt_morlet_py(timeseries: Vec<f64>, hz: f64, steps: u32) -> (Vec<Vec<Complex64>>, Vec<f64>) {
    cwt(&timeseries, morlet_fourier, hz, steps as f64, false) 
} 

#[pyfunction]
#[pyo3(name = "cwt_morlet_ext")]
fn cwt_morlet_ext_py(timeseries: Vec<f64>, hz: f64, steps: u32, normalize: bool) -> (Vec<Vec<Complex64>>, Vec<f64>) {
    cwt(&timeseries, morlet_fourier, hz, steps as f64, normalize)
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





