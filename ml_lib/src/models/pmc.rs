pub struct PMC {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    w1: Vec<f32>,
    w2: Vec<f32>,
    b1: Vec<f32>,
    b2: Vec<f32>,
    lr: f32,
}

impl PMC {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, lr: f32) -> Self {
        let w1_size = input_size * hidden_size;
        let w2_size = hidden_size * output_size;

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

        let w1 = (0..w1_size).map(|_| rand_f32() * 0.5).collect();
        let w2 = (0..w2_size).map(|_| rand_f32() * 0.5).collect();
        let b1 = vec![0.0f32; hidden_size];
        let b2 = vec![0.0f32; output_size];

        PMC { input_size, hidden_size, output_size, w1, w2, b1, b2, lr }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut hidden = vec![0.0f32; self.hidden_size];
        for j in 0..self.hidden_size {
            let mut sum = self.b1[j];
            for i in 0..self.input_size {
                sum += self.w1[i * self.hidden_size + j] * x[i];
            }
            hidden[j] = Self::sigmoid(sum);
        }

        let mut output = vec![0.0f32; self.output_size];
        for k in 0..self.output_size {
            let mut sum = self.b2[k];
            for j in 0..self.hidden_size {
                sum += self.w2[j * self.output_size + k] * hidden[j];
            }
            output[k] = sum;
        }

        (hidden, output)
    }

    fn train(&mut self, x: &[f32], y: &[f32]) {
        let (hidden, output) = self.forward(x);

        let mut output_delta = vec![0.0f32; self.output_size];
        for k in 0..self.output_size {
            output_delta[k] = (y[k] - output[k]) * 1.0;
        }

        let mut hidden_delta = vec![0.0f32; self.hidden_size];
        for j in 0..self.hidden_size {
            let mut err = 0.0f32;
            for k in 0..self.output_size {
                err += self.w2[j * self.output_size + k] * output_delta[k];
            }
            hidden_delta[j] = err * hidden[j] * (1.0 - hidden[j]);
        }

        for j in 0..self.hidden_size {
            for k in 0..self.output_size {
                self.w2[j * self.output_size + k] += self.lr * output_delta[k] * hidden[j];
            }
        }
        for k in 0..self.output_size {
            self.b2[k] += self.lr * output_delta[k];
        }

        for j in 0..self.hidden_size {
            for i in 0..self.input_size {
                self.w1[i * self.hidden_size + j] += self.lr * hidden_delta[j] * x[i];
            }
            self.b1[j] += self.lr * hidden_delta[j];
        }
    }

    fn predict(&self, x: &[f32]) -> Vec<f32> {
        let (_, output) = self.forward(x);
        output
    }
}

#[unsafe(no_mangle)]
extern "C" fn create_pmc(input_size: usize, hidden_size: usize, output_size: usize, lr: f32) -> *mut PMC {
    let model = Box::new(PMC::new(input_size, hidden_size, output_size, lr));
    Box::into_raw(model)
}

#[unsafe(no_mangle)]
extern "C" fn pmc_train_step(pmc: *mut PMC, x: *const f32, n: usize, y: *const f32, ny: usize) {
    let m = unsafe { &mut *pmc };
    let x_slice = unsafe { std::slice::from_raw_parts(x, n) };
    let y_slice = unsafe { std::slice::from_raw_parts(y, ny) };
    m.train(x_slice, y_slice);
}

#[unsafe(no_mangle)]
extern "C" fn pmc_predict(pmc: *mut PMC, x: *const f32, n: usize, output: *mut f32) {
    let m = unsafe { &*pmc };
    let x_slice = unsafe { std::slice::from_raw_parts(x, n) };
    let result = m.predict(x_slice);
    for i in 0..result.len() {
        unsafe { *output.add(i) = result[i]; }
    }
}

#[unsafe(no_mangle)]
extern "C" fn free_pmc(pmc: *mut PMC) {
    if !pmc.is_null() {
        unsafe { drop(Box::from_raw(pmc)) };
    }
}

#[unsafe(no_mangle)]
extern "C" fn save_pmc(pmc: *mut PMC, path: *const i8) {
    let m = unsafe { &*pmc };
    let path_str = unsafe { std::ffi::CStr::from_ptr(path).to_str().unwrap() };
    let mut data = vec![m.input_size as f32, m.hidden_size as f32, m.output_size as f32, m.lr];
    data.extend_from_slice(&m.w1);
    data.extend_from_slice(&m.w2);
    data.extend_from_slice(&m.b1);
    data.extend_from_slice(&m.b2);
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(path_str, bytes).unwrap();
}

#[unsafe(no_mangle)]
extern "C" fn load_pmc(path: *const i8) -> *mut PMC {
    let path_str = unsafe { std::ffi::CStr::from_ptr(path).to_str().unwrap() };
    let bytes = std::fs::read(path_str).unwrap();
    let data: Vec<f32> = bytes.chunks(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let input_size  = data[0] as usize;
    let hidden_size = data[1] as usize;
    let output_size = data[2] as usize;
    let lr          = data[3];
    let w1_size     = input_size * hidden_size;
    let w2_size     = hidden_size * output_size;
    let w1 = data[4..4+w1_size].to_vec();
    let w2 = data[4+w1_size..4+w1_size+w2_size].to_vec();
    let b1 = data[4+w1_size+w2_size..4+w1_size+w2_size+hidden_size].to_vec();
    let b2 = data[4+w1_size+w2_size+hidden_size..4+w1_size+w2_size+hidden_size+output_size].to_vec();
    let model = Box::new(PMC { input_size, hidden_size, output_size, w1, w2, b1, b2, lr });
    Box::into_raw(model)
}