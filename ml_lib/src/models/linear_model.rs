use std::os::raw::c_float;
pub struct LinearModel {
    weights: Vec<f32>,
    bias: f32,
    lr: f32,
}
impl LinearModel {

    // Je cree le modele avec des petits poids de depart
    fn new(n: usize, lr: f32) -> Self {
        let mut weights = vec![0.0f32; n];
        for i in 0..n {
            weights[i] = (i as f32 * 0.001) % 0.1 - 0.05;
        }
        LinearModel { weights, bias: 0.0, lr }
    }

    // Je calcule la prediction : somme des poids * features + bias
    // Si le resultat est positif c'est une classe, negatif c'est l'autre
    fn predict(&self, x: &[f32]) -> f32 {
        let mut sum = self.bias;
        for i in 0..self.weights.len() {
            sum += self.weights[i] * x[i];
        }
        sum
    }

    // J'entraine le modele sur un exemple
    // Je calcule l'erreur et je corrige les poids proportionnellement
    // C'est la descente de gradient
    fn train(&mut self, x: &[f32], y: f32) {
        let prediction = self.predict(x);
        let erreur = y - prediction;
        for i in 0..self.weights.len() {
            self.weights[i] += self.lr * erreur * x[i];
        }
        self.bias += self.lr * erreur;
    }

    // Je sauvegarde les poids dans un fichier binaire
    // Comme ca je dois pas reentrainer a chaque fois
    fn save(&self, path: &str) {
        let mut data = vec![self.weights.len() as f32, self.bias, self.lr];
        data.extend_from_slice(&self.weights);
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(path, bytes).unwrap();
    }

    // Je charge les poids depuis un fichier
    fn load(path: &str) -> Self {
        let bytes = std::fs::read(path).unwrap();
        let data: Vec<f32> = bytes.chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let n = data[0] as usize;
        let bias = data[1];
        let lr = data[2];
        let weights = data[3..3+n].to_vec();
        LinearModel { weights, bias, lr }
    }
}

// Fonctions appelables depuis Python via ctypes
// Box::new alloue la memoire, Box::into_raw donne un pointeur a Python

#[unsafe(no_mangle)]
extern "C" fn create_linear_model(n: usize, lr: c_float) -> *mut LinearModel {
    let model = Box::new(LinearModel::new(n, lr));
    Box::into_raw(model)
}

#[unsafe(no_mangle)]
extern "C" fn free_linear_model(model: *mut LinearModel) {
    if !model.is_null() {
        unsafe { drop(Box::from_raw(model)) };
    }
}

#[unsafe(no_mangle)]
extern "C" fn predict(model: *mut LinearModel, x: *const f32, n: usize) -> c_float {
    let m = unsafe { &*model };
    let x_slice = unsafe { std::slice::from_raw_parts(x, n) };
    m.predict(x_slice)
}

#[unsafe(no_mangle)]
extern "C" fn train_step(model: *mut LinearModel, x: *const f32, n: usize, y: c_float) {
    let m = unsafe { &mut *model };
    let x_slice = unsafe { std::slice::from_raw_parts(x, n) };
    m.train(x_slice, y);
}

#[unsafe(no_mangle)]
extern "C" fn save_linear_model(model: *mut LinearModel, path: *const i8) {
    let m = unsafe { &*model };
    let path_str = unsafe { std::ffi::CStr::from_ptr(path).to_str().unwrap() };
    m.save(path_str);
}

#[unsafe(no_mangle)]
extern "C" fn load_linear_model(path: *const i8) -> *mut LinearModel {
    let path_str = unsafe { std::ffi::CStr::from_ptr(path).to_str().unwrap() };
    let model = Box::new(LinearModel::load(path_str));
    Box::into_raw(model)
}
