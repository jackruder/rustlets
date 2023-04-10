pub mod waveletanalysis;
use crate::waveletanalysis::*;

fn main() {
    let duration = 1.0;
    let hz = 44180.0;
    let n = (duration * hz) as usize;
    let times: Vec<f64> = (0..n).map(|t| t as f64 / hz).collect();
    let f: Vec<f64> = times.iter()
                            .map(|t| chirp(*t,1.0,10000.0,7000.0))
                            .collect();
    
}

