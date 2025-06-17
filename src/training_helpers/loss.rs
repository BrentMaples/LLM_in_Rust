// Standard library
use std::cmp::min;
// External crates
use tiktoken_rs::{CoreBPE, tokenizer};
use tch::{
    no_grad, Device, Kind, Tensor,
    nn::{self, AdamW, EmbeddingConfig, Module, ModuleT},
    data, COptimizer, IndexOp
};
// Internal modules
use crate::architecture::{generate_text_simple, GPTModel, TransformerBlock, CONFIG_124M};
use crate::training_helpers::{fine_tuned::*, text_sampling::*};
use crate::dataloader::DataLoader;


//calculate the loss for each batch size
pub fn calc_loss_batch(input_batch: &Tensor, target_batch: &Tensor, model: &GPTModel, train: bool) -> Tensor {
   let mut logits = model.forward_t(input_batch, train);
   logits = logits.flatten(0, 1);
   let loss = Tensor::cross_entropy_loss::<Tensor>(&logits, &target_batch.flatten(0,-1), None, tch::Reduction::Mean, -100, 0.0);
   return loss;
}

//loading the loss and doing it here - default batch 2 for now
pub fn calc_loss_loader(data_loader: &mut DataLoader, model: &GPTModel, max_batches: Option<i64>, train: bool) -> f64 {
    let mut total_loss = 0.0;
    let mut batch_count = 0;

    for (input_batch, target_batch) in data_loader.by_ref() {
        let loss = calc_loss_batch(&input_batch, &target_batch, model, train);
        total_loss += loss.double_value(&[]);
        batch_count += 1;

        if let Some(max) = max_batches {
            if batch_count >= max {
                break;
            }
        }
    }
    if batch_count == 0 {
        return f64::NAN;
    }
    total_loss / batch_count as f64
}


//this evaluates the module
pub fn evaluate_model(model: &GPTModel, train_loader: &DataLoader, val_loader: &DataLoader, eval_iter: i64) -> (f64, f64) {
    let mut train_loader_copy = DataLoader::init(train_loader.dataset.clone(), train_loader.batch_size, false, false);
    let mut val_loader_copy = DataLoader::init(val_loader.dataset.clone(), val_loader.batch_size, false, false);

    let train_loss = no_grad(|| calc_loss_loader(&mut train_loader_copy, model, Some(eval_iter), false));
    let val_loss = no_grad(|| calc_loss_loader(&mut val_loader_copy, model, Some(eval_iter), false));

    return (train_loss, val_loss);
}
