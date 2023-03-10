use pyo3::{prelude::*, impl_::pyfunction};
use num_complex::{Complex64};

pub mod waveletanalysis;

use crate::waveletanalysis::*;





#[pyfunction]
#[pyo3(name = "cwt_morlet")]
fn cwt_morlet_py(timeseries: Vec<f64>, hz: f64, steps: u32, normalize: bool) -> (Vec<Complex64>, Vec<f64>) {
    cwt(&timeseries, morlet_fourier, hz, steps as f64, normalize)
} 

#[pyfunction]
#[pyo3(name = "icwt_morlet")]
fn icwt_morlet_py(cwtm: Vec<Complex64>, scales: Vec<f64>, times: Vec<f64>) -> Vec<f64> {
    icwt_morlet(&cwtm, &scales, &times)
} 


#[pyfunction]
#[pyo3(name = "morlet_wavelength")]
fn morlet_wavelength_py(omega_0: f64) -> f64 {
    morlet_wavelength(omega_0)
}

#[pyfunction]
#[pyo3(name = "diff")]
fn diff_py(f: Vec<Complex64>, x: Vec<f64>) -> Vec<Complex64>{ //Complex64ODO handle mismatch lengths
    let mut target = vec![Complex64{re: 0.0, im: 0.0}; f.len()];
    calctools::diff(&f[..], &x, &mut target[..]);
    return target;
}

#[pyfunction]
#[pyo3(name = "trapz")]
fn trapz_py(f: Vec<Complex64>, x: Vec<Complex64>) -> Complex64 { //Complex64ODO handle mismatch lengths
    calctools::trapz(&f, &x) 
}

#[pyfunction]
#[pyo3(name = "trapz_step")]
fn trapz_step_py(f: Vec<Complex64>, x: Vec<f64>, col: usize) -> Complex64 { //Complex64ODO handle mismatch lengths
    calctools::trapz_step(&f, &x, col)
}


#[pymodule]
fn rustlets(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cwt_morlet_py, m)?)?;
    m.add_function(wrap_pyfunction!(icwt_morlet_py, m)?)?;
    m.add_function(wrap_pyfunction!(morlet_wavelength_py, m)?)?;
    m.add_function(wrap_pyfunction!(diff_py, m)?)?;
    m.add_function(wrap_pyfunction!(trapz_py, m)?)?;
    m.add_function(wrap_pyfunction!(trapz_step_py, m)?)?;
    Ok(())
}





