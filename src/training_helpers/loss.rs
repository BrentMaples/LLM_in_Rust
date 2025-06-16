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


//calculate the loss for each batch size
pub fn calc_loss_batch(input_batch: &Tensor, target_batch: &Tensor, model: &GPTModel, train: bool) -> Tensor {
   let mut logits = model.forward_t(input_batch, train);
   logits = logits.flatten(0, 1);
   let loss = Tensor::cross_entropy_loss::<Tensor>(&logits, &target_batch.flatten(0,-1), None, tch::Reduction::Mean, -100, 0.0);
   return loss;
}

//loading the loss and doing it here - default batch 2 for now
pub fn calc_loss_loader(data_loader: &(Tensor, Tensor), model: &GPTModel, mut num_batches: Option<i64>, train: bool) -> f64{
   let mut total_loss = 0.0;
   let (input_tensor, target_tensor) = data_loader;
   if input_tensor.size()[0] == 0 {
        return f64::NAN;
   }
   else if num_batches == None {
      num_batches = Some(input_tensor.size()[0]);
   }
   else {
      num_batches = min(num_batches, Some(input_tensor.size()[0]));
   }

   for i in 0..num_batches.unwrap() {
      let input_batch = input_tensor.narrow(0, i, 2);
      let target_batch = target_tensor.narrow(0, i, 2);
      let loss = calc_loss_batch(&input_batch, &target_batch, model, train);
      //can grab it from loss this way I think - will panic if it's not a single value
      total_loss += loss.double_value(&[]);
   }

   return total_loss / num_batches.unwrap() as f64;
}

//this evaluates the module
pub fn evaluate_model (model: &GPTModel, train_loader: &(Tensor,Tensor), val_loader: &(Tensor,Tensor), eval_iter: i64) -> (f64, f64) { 
   let mut train = false;
   // || equates FnOnce closures
   let train_loss = no_grad(|| calc_loss_loader(&train_loader, model, Some(eval_iter), train));
   let val_loss = no_grad(|| calc_loss_loader(&val_loader, model, Some(eval_iter), train));

   return (train_loss, val_loss);
}