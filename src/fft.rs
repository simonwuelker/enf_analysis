use num_complex::Complex32;


/// Use the Cooley-Tukey algorithm to compute the fourier transform in O(n log n)
pub fn fft(n: usize, values: Vec<Complex32>) -> Vec<Complex32> {
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

    let f_even = fft(n / 2, even);
    let f_odd = fft(n / 2, odd);

    let mut result = Vec::with_capacity(n);

    // twiddle can be reused
    let exponent = -2. std::f32::consts::PI * Complex32::i() * (1. / n as f32);
    let twiddle = exponent.exp();

    for k in 0..n / 2 {
        result[k] = f_even[k] + f_odd[k] * twiddle;
        result[k + n / 2] = f_even[k] - f_odd[k] * twiddle;
    }
    result
}
