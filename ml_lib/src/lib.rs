// Point d entree de la lib Rust
// Je declare les modules et je les rends accessibles depuis Python
mod models;

pub use models::linear_model::*;
pub use models::pmc::*;
pub use models::rbfn::*;

// Fonctions de test pour valider que la tuyauterie Rust/Python fonctionne
#[unsafe(no_mangle)]
extern "C" fn addition(a: i32, b: i32) -> i32 {
    a + b
}

#[unsafe(no_mangle)]
extern "C" fn soustraction(a: i32, b: i32) -> i32 {
    a - b
}