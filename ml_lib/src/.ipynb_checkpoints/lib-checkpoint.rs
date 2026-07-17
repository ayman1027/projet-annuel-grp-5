// Point d'entrée de la bibliothèque Rust on déclare les modules et on les rend accessibles depuis Python
mod models;

pub use models::linear_model::*;
pub use models::pmc::*;

#[no_mangle]
pub extern "C" fn addition(a: i32, b: i32) -> i32 {
    a + b
}

#[no_mangle]
pub extern "C" fn soustraction(a: i32, b: i32) -> i32 {
    a - b
}