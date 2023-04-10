use numpy::{PyArray1, PyArray2, PyArray, PyReadonlyArray1};
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
fn cwt_morlet_py<'py>(py: Python<'py>, timeseries: &PyArray1<f64>, hz: f64, steps: u32) -> (&'py PyArray2<Complex64>, &'py PyArray1<f64>) {
    let timeseries_ro = timeseries.readonly();
    let ts = timeseries_ro.as_array();
    let (cwtm, scales) = cwt(&ts, morlet_fourier, hz, steps as f64, false);
    (PyArray::from_array(py, &cwtm), PyArray::from_array(py, &scales))
}

#[pyfunction]
#[pyo3(name = "cwt_morlet_ext")]
fn cwt_morlet_ext_py<'py>(py: Python<'py>, timeseries: &PyArray1<f64>, hz: f64, steps: u32, normalize: bool) -> (&'py PyArray2<Complex64>, &'py PyArray1<f64>) {
    let timeseries_ro = timeseries.readonly();
    let ts = timeseries_ro.as_array();
    let (cwtm, scales) = cwt(&ts, morlet_fourier, hz, steps as f64, normalize);
    (PyArray::from_array(py, &cwtm), PyArray::from_array(py, &scales))
}

#[pyfunction]
#[pyo3(name = "icwt_morlet")]
fn icwt_morlet_py<'py>(py: Python<'py>, cwtm: &PyArray2<Complex64>, scales: &PyArray1<f64>, times: &PyArray1<f64>) -> &'py PyArray1<f64> {
    let cwtm_ro = cwtm.readonly();
    let scales_ro = scales.readonly();
    let times_ro = times.readonly();

    icwt_morlet(&cwtm_ro.as_array(),
                &scales_ro.as_array(),
                &times.as_array())
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





