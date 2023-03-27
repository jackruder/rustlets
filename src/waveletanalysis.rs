use std::f64::consts::PI;
use num_complex::{Complex64, ComplexFloat};
use rustfft::{self, FftPlanner};
use collect_slice::CollectSlice;

use self::calctools::phase_unwrap;

pub fn chirp(t_0: f64, t_1: f64, f_0: f64, f_1: f64) -> f64 {
    f64::sin(f_0 + (f_1 - f_0) * t_0 / t_1) 
}


pub fn morlet_fourier(s: f64,omega: f64,omega_0: f64) -> Complex64{
    if omega > 0.0 { // implement heaviside
        let exp = -(s * omega - omega_0).powi(2)/2.0;
        return Complex64{re: PI.powf(-0.25) * exp.exp(), im: 0.0};
    }
    else {
        return Complex64{re: 0.0, im: 0.0};
    }
}

pub fn morlet_wavelength(omega_0: f64) -> f64 {
    4.0 * PI / (omega_0 + f64::sqrt(2.0 + omega_0.powi(2)))
}

fn gen_scales(t: f64, hz: f64, steps: f64) -> Vec<f64> {
    let min_scale: f64 = 2.0 / hz; // smallest scale is twice the timestep
    let num_scales: usize = (f64::log2(t / min_scale) * steps) as usize; // log2 spacing of scales
    
    let mut scales: Vec<f64> = Vec::with_capacity(num_scales+1);
    for j in 0..(num_scales+1){
        scales.push(min_scale * f64::powf(2.0, j as f64 / steps)) // compute the scale
    }
    return scales;
}

pub fn cwt(timeseries: &Vec<f64>, wavelet: fn(f64, f64, f64) -> Complex64, hz: f64, steps: f64, normalize: bool) -> (Vec<Complex64>, Vec<f64>){
    let steps = steps as f64;

    let mut nrm = 0;
    if normalize { // should we normalize wavelets for equal energy?
        nrm = 1; // set normalization exponent to 1, nfac^1 = nfac
    }

    // cast real timeseries to complex
    let mut data: Vec<Complex64> = timeseries.iter().map(|x| Complex64{re: *x, im: 0.0}).collect();

    let n: usize = data.len().next_power_of_two(); // pad for fourier transform
    data.resize(n,Complex64{re: 0.0, im: 0.0});

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    
    fft.process(&mut data); // do the fft
    for i in 0..data.len() {
        data[i] = data[i] / (n as f64); // normalize by 1/len, we will transform back to time later
                                        // rustfft documentation states output is not normalized
    }

    let t = data.len() as f64 / hz;
    let scales = gen_scales(t, hz, steps);

    //let newscales: Vec<f64> = scales.iter().map(|s| s * rescale).collect();

    // generate angular frequencies for wavelet
    let mut angs: Vec<f64> = Vec::with_capacity(n); 
    let c: f64 = 2.0 * PI / (n as f64) * hz;
    for i in 0..n {
        if i <= n/2 {
            angs.push(c * i as f64) // pos to n/2
        } else {
            angs.push(c * (i as f64 - n as f64)) // wrap to [-n/2, -1]
        }
    }

    let ifft = planner.plan_fft_inverse(n);

    let mut cwtm = vec![Complex64{ re: 0.0f64, im: 0.0f64 }; scales.len() * n]; //output
    for i in 0..scales.len() {
        let target = &mut cwtm[(i * n)..((i+1)*n)]; // slice of output we wish to write
        let norm = f64::sqrt(2.0 * PI * scales[i] * hz).powi(nrm); // 1 if normalize = false
                                                                      // TODO maybe just scales and
                                                                      // not newscales in norm?
        angs.iter()
            .zip(&data)
            .map(|(&w, f)| f *norm* wavelet(scales[i],w,2.0*PI))
            .collect_slice(target);

        ifft.process(target)

    }

    return (cwtm, scales)
}


pub fn icwt_morlet(cwtm: &Vec<Complex64>, scales: &Vec<f64>,times: &Vec<f64>) -> Vec<f64> {
    let mut diffs: Vec<Complex64> = vec![Complex64{re:0.0, im:0.0}; cwtm.len()];
    let n = times.len();
    for i in 0..scales.len() { 
        let target = &mut diffs[(i * n)..((i+1)*n)]; //target slice for output
        let cwtvals = &cwtm[(i * n)..((i+1)*n)]; //target slice for input
        calctools::diff(cwtvals, times, target); //differentiate each row
    }
    
    //recovered signal
    let mut rec: Vec<f64> = Vec::with_capacity(times.len());

    for j in 0..times.len() { //integrate over each column and push the scaled result
        let integral: Complex64 = calctools::trapz_step(&mut diffs, scales, j);
        rec.push(PI * integral.im() / (2.0 * PI).sqrt());
    }
    return rec;


}

pub fn vocode_cwt(timeseries: &Vec<f64>, wavelet: fn(f64, f64, f64) -> Complex64, hz: f64, steps: f64, times: &Vec<f64>, rescale: f64) -> Vec<f64>{
    // slow, I wouldn't use this
    let (mut cwtm, mut scales) = cwt(timeseries, wavelet, hz, steps, false);
    let mut phases: Vec<f64> = cwtm.iter()
                    .map(|c| f64::atan2(c.im, c.re))
                    .collect();
    let n = timeseries.len();
    for j in 0..scales.len() {
        let i = j * n;
        phase_unwrap(&mut phases[i..(i+n)]);

    }

    for i in 0..cwtm.len() {
        let exp = Complex64{re:0.0f64, im:1.0f64} * phases[i] * rescale;
        cwtm[i] = cwtm[i].abs() * exp.exp();
    }

    for s in scales.iter_mut() {
        *s /= rescale;
    }

    icwt_morlet(&cwtm, &scales, &times)

}
mod calctools {
    use num_complex::{Complex64};
    use std::f64::consts::PI;

    fn centraldiff3(f_prev: &Complex64, f_next: &Complex64, x_prev: &f64, x_next: &f64) -> Complex64 {
        (*f_next - *f_prev)/(*x_next - *x_prev)
    }

    fn forwarddiff(f: &Complex64, f_next: &Complex64, x: &f64, x_next: &f64) -> Complex64 {
        (*f_next - *f)/(*x_next - *x)
    }

    fn backwarddiff(f_prev: &Complex64, f: &Complex64, x_prev: &f64, x: &f64) -> Complex64 {
        (*f - *f_prev)/(*x - *x_prev)
    }

    pub fn diff(f: &[Complex64], x: &Vec<f64>, target: &mut[Complex64]){ //Complex64ODO handle mismatch lengths
        for i in 0..f.len() {
            if i == 0 {
                target[i] = forwarddiff(&f[0], &f[1], &x[0], &x[1]);
            } 
            else if i == f.len() - 1 {
                target[i] = backwarddiff(&f[i-1], &f[i], &x[i-1], &x[i]);
            }
            else {
                target[i] = centraldiff3(&f[i-1], &f[i+1], &x[i-1], &x[i+1]);
            }
        }
    }


    pub fn trapz(f: &Vec<Complex64>, x: &Vec<Complex64>) -> Complex64 { //Complex64ODO handle mismatch lengths
        let mut sum = Complex64{re: 0.0f64, im: 0.0f64}; 
        for i in 0..(f.len()-1){
            let h = x[i+1] - x[i];
            let v = h * (f[i] + f[i+1]) / 2.0;
            sum += v;
        }
        return sum;
        
    }
    
    // compute the integral over the column of a flattened 2d array, with width equal to the length
    // of x
    pub fn trapz_step(f: &Vec<Complex64>, x: &Vec<f64>, col: usize) -> Complex64 { //Complex64ODO handle mismatch lengths
        let mut sum = Complex64{re: 0.0f64, im: 0.0f64}; 
        let m = f.len() / x.len();
        for row in 0..(x.len()-1) {
            let i = row * m + col;
            let i_next = (row + 1) * m + col;
            let h = x[row + 1] - x[row];
            sum += h * (f[i] + f[i_next]) / 2.0;
        }
        return sum;
        
    }

    pub fn phase_unwrap(phases: &mut[f64]) { // T(N^2/2 - N/2), bad
        let op = phases.to_vec();// old phases
        let n = phases.len();
        for i in 1..n {
            let diff = op[i] - op[i-1];
            if diff > PI {
                for p in phases[i..n].iter_mut() {
                    *p -= 2.0 * PI;
                }
            } else if diff < -PI {
                for p in phases[i..n].iter_mut() {
                    *p += 2.0 * PI;
                }
            }
        }
    }

}

