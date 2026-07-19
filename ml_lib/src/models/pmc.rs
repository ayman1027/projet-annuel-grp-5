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

    fn forward(&self, entree: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut hidden = vec![0.0f32; self.hidden_size];
        for j in 0..self.hidden_size {
            let mut somme = self.b1[j];
            for i in 0..self.input_size {
                somme += self.w1[i * self.hidden_size + j] * entree[i];
            }
            hidden[j] = Self::sigmoid(somme);
        }

        let mut sortie = vec![0.0f32; self.output_size];
        for k in 0..self.output_size {
            let mut somme = self.b2[k];
            for j in 0..self.hidden_size {
                somme += self.w2[j * self.output_size + k] * hidden[j];
            }
            sortie[k] = somme;
        }

        (hidden, sortie)
    }

    fn train(&mut self, entree: &[f32], cible: &[f32]) {
        let (hidden, sortie) = self.forward(entree);

        let mut delta_sortie = vec![0.0f32; self.output_size];
        for k in 0..self.output_size {
            delta_sortie[k] = cible[k] - sortie[k];
        }

        let mut delta_hidden = vec![0.0f32; self.hidden_size];
        for j in 0..self.hidden_size {
            let mut erreur = 0.0f32;
            for k in 0..self.output_size {
                erreur += self.w2[j * self.output_size + k] * delta_sortie[k];
            }
            delta_hidden[j] = erreur * hidden[j] * (1.0 - hidden[j]);
        }

        for j in 0..self.hidden_size {
            for k in 0..self.output_size {
                self.w2[j * self.output_size + k] += self.lr * delta_sortie[k] * hidden[j];
            }
        }
        for k in 0..self.output_size {
            self.b2[k] += self.lr * delta_sortie[k];
        }

        for j in 0..self.hidden_size {
            for i in 0..self.input_size {
                self.w1[i * self.hidden_size + j] += self.lr * delta_hidden[j] * entree[i];
            }
            self.b1[j] += self.lr * delta_hidden[j];
        }
    }

    fn predict(&self, entree: &[f32]) -> Vec<f32> {
        let (_, sortie) = self.forward(entree);
        sortie
    }
}

#[unsafe(no_mangle)]
extern "C" fn create_pmc(input_size: usize, hidden_size: usize, output_size: usize, lr: f32) -> *mut PMC {
    let modele = Box::new(PMC::new(input_size, hidden_size, output_size, lr));
    Box::into_raw(modele)
}

#[unsafe(no_mangle)]
extern "C" fn pmc_train_step(pmc: *mut PMC, entree: *const f32, n: usize, cible: *const f32, nc: usize) {
    let m = unsafe { &mut *pmc };
    let entree_slice = unsafe { std::slice::from_raw_parts(entree, n) };
    let cible_slice = unsafe { std::slice::from_raw_parts(cible, nc) };
    m.train(entree_slice, cible_slice);
}

#[unsafe(no_mangle)]
extern "C" fn pmc_predict(pmc: *mut PMC, entree: *const f32, n: usize, sortie: *mut f32) {
    let m = unsafe { &*pmc };
    let entree_slice = unsafe { std::slice::from_raw_parts(entree, n) };
    let resultat = m.predict(entree_slice);
    for i in 0..resultat.len() {
        unsafe { *sortie.add(i) = resultat[i]; }
    }
}

#[unsafe(no_mangle)]
extern "C" fn free_pmc(pmc: *mut PMC) {
    if !pmc.is_null() {
        unsafe { drop(Box::from_raw(pmc)) };
    }
}

#[unsafe(no_mangle)]
extern "C" fn save_pmc(pmc: *mut PMC, chemin: *const i8) {
    let m = unsafe { &*pmc };
    let chemin_str = unsafe { std::ffi::CStr::from_ptr(chemin).to_str().unwrap() };
    let mut data = vec![m.input_size as f32, m.hidden_size as f32, m.output_size as f32, m.lr];
    data.extend_from_slice(&m.w1);
    data.extend_from_slice(&m.w2);
    data.extend_from_slice(&m.b1);
    data.extend_from_slice(&m.b2);
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(chemin_str, bytes).unwrap();
}

#[unsafe(no_mangle)]
extern "C" fn load_pmc(chemin: *const i8) -> *mut PMC {
    let chemin_str = unsafe { std::ffi::CStr::from_ptr(chemin).to_str().unwrap() };
    let bytes = std::fs::read(chemin_str).unwrap();
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
    let modele = Box::new(PMC { input_size, hidden_size, output_size, w1, w2, b1, b2, lr });
    Box::into_raw(modele)
}