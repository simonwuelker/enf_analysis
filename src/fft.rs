use num_complex::Complex32;
use num_traits::Zero;

/// Use the Cooley-Tukey algorithm to compute the fourier transform in O(n log n)
/// the invert argument is used to specify whether the IFFT is to be calculated
fn fft_recursive(values: Vec<Complex32>, invert: bool) -> Vec<Complex32> {
    let n = values.len();

    // Cooley-Tukey only works on powers of two
    assert!(n.is_power_of_two());

    // base case
    if n == 1 {
        return values;
    }

    // Split into odd and even indices
    let mut even = Vec::with_capacity(n / 2);
    let mut odd = Vec::with_capacity(n / 2);

    values.iter().enumerate().for_each(|(ix, val)| {
        if ix % 2 == 0 {
            even.push(val.clone());
        } else {
            odd.push(val.clone());
        }
    });

    let f_even = fft_recursive(even, invert);
    let f_odd = fft_recursive(odd, invert);

    let mut result = vec![Complex32::zero(); n];

    let exponent = -2. * std::f32::consts::PI * Complex32::i() * (1. / n as f32);
    let twiddle = exponent.exp();

    for k in 0..n / 2 {
        result[k] = f_even[k] + f_odd[k] * twiddle.powi(k as i32);
        result[k + n / 2] = f_even[k] - f_odd[k] * twiddle.powi(k as i32);
    }
    result
}

/// Handle scaling of IDFT vs DFT
pub fn fft(vals_real: Vec<f32>, invert: bool) -> Vec<Complex32> {
    let n = vals_real.len();

    // pad the values to be a power of two and convert them to complex numbers
    let vals: Vec<Complex32> = if n.is_power_of_two() {
        vals_real.iter().map(|x| Complex32::new(*x, 0.)).collect()
    } else {
        let pad_to = n.next_power_of_two();
        let mut pad = vec![Complex32::zero(); pad_to];
        for i in 0..n {
            pad[i] = Complex32::new(vals_real[i], 0.);
        }
        pad

    };

    let y = fft_recursive(vals, invert);

    if invert {
        return y.iter().map(|x| x / n as f32).collect()
    }
    y 
}
