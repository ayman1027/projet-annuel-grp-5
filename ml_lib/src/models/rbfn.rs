pub struct RBFN {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    centers: Vec<f32>,
    weights: Vec<f32>,
    gamma: f32,
    lr: f32,
}

impl RBFN {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, gamma: f32, lr: f32) -> Self {
        let mut seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos() as u64;

        let mut rand_f32 = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        let centers = (0..hidden_size * input_size).map(|_| rand_f32()).collect();
        let weights = (0..hidden_size * output_size).map(|_| rand_f32() * 0.1).collect();

        RBFN { input_size, hidden_size, output_size, centers, weights, gamma, lr }
    }

    fn rbf(&self, x: &[f32], center_idx: usize) -> f32 {
        let mut dist = 0.0f32;
        for i in 0..self.input_size {
            let diff = x[i] - self.centers[center_idx * self.input_size + i];
            dist += diff * diff;
        }
        (-self.gamma * dist).exp()
    }

    fn predict(&self, x: &[f32]) -> Vec<f32> {
        let hidden: Vec<f32> = (0..self.hidden_size).map(|j| self.rbf(x, j)).collect();
        let mut output = vec![0.0f32; self.output_size];
        for k in 0..self.output_size {
            for j in 0..self.hidden_size {
                output[k] += self.weights[j * self.output_size + k] * hidden[j];
            }
        }
        output
    }

    fn train(&mut self, x: &[f32], y: &[f32]) {
        let hidden: Vec<f32> = (0..self.hidden_size).map(|j| self.rbf(x, j)).collect();
        let output = self.predict(x);
        for k in 0..self.output_size {
            let erreur = y[k] - output[k];
            for j in 0..self.hidden_size {
                self.weights[j * self.output_size + k] += self.lr * erreur * hidden[j];
            }
        }
    }
}

#[unsafe(no_mangle)]
extern "C" fn create_rbfn(input_size: usize, hidden_size: usize, output_size: usize, gamma: f32, lr: f32) -> *mut RBFN {
    let model = Box::new(RBFN::new(input_size, hidden_size, output_size, gamma, lr));
    Box::into_raw(model)
}

#[unsafe(no_mangle)]
extern "C" fn rbfn_set_centers(rbfn: *mut RBFN, centers: *const f32, n: usize) {
    let m = unsafe { &mut *rbfn };
    let centers_slice = unsafe { std::slice::from_raw_parts(centers, n) };
    m.centers = centers_slice.to_vec();
}

#[unsafe(no_mangle)]
extern "C" fn rbfn_train_step(rbfn: *mut RBFN, x: *const f32, n: usize, y: *const f32, ny: usize) {
    let m = unsafe { &mut *rbfn };
    let x_slice = unsafe { std::slice::from_raw_parts(x, n) };
    let y_slice = unsafe { std::slice::from_raw_parts(y, ny) };
    m.train(x_slice, y_slice);
}

#[unsafe(no_mangle)]
extern "C" fn rbfn_predict(rbfn: *mut RBFN, x: *const f32, n: usize, output: *mut f32) {
    let m = unsafe { &*rbfn };
    let x_slice = unsafe { std::slice::from_raw_parts(x, n) };
    let result = m.predict(x_slice);
    for i in 0..result.len() {
        unsafe { *output.add(i) = result[i]; }
    }
}

#[unsafe(no_mangle)]
extern "C" fn free_rbfn(rbfn: *mut RBFN) {
    if !rbfn.is_null() {
        unsafe { drop(Box::from_raw(rbfn)) };
    }
}

#[unsafe(no_mangle)]
extern "C" fn save_rbfn(rbfn: *mut RBFN, path: *const i8) {
    let m = unsafe { &*rbfn };
    let path_str = unsafe { std::ffi::CStr::from_ptr(path).to_str().unwrap() };
    let mut data = vec![m.input_size as f32, m.hidden_size as f32, m.output_size as f32, m.gamma, m.lr];
    data.extend_from_slice(&m.centers);
    data.extend_from_slice(&m.weights);
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(path_str, bytes).unwrap();
}

#[unsafe(no_mangle)]
extern "C" fn load_rbfn(path: *const i8) -> *mut RBFN {
    let path_str = unsafe { std::ffi::CStr::from_ptr(path).to_str().unwrap() };
    let bytes = std::fs::read(path_str).unwrap();
    let data: Vec<f32> = bytes.chunks(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let input_size  = data[0] as usize;
    let hidden_size = data[1] as usize;
    let output_size = data[2] as usize;
    let gamma       = data[3];
    let lr          = data[4];
    let centers_size = hidden_size * input_size;
    let weights_size = hidden_size * output_size;
    let centers = data[5..5+centers_size].to_vec();
    let weights = data[5+centers_size..5+centers_size+weights_size].to_vec();
    let model = Box::new(RBFN { input_size, hidden_size, output_size, centers, weights, gamma, lr });
    Box::into_raw(model)
}