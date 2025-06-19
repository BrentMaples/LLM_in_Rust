// ─── Standard Library ──────────────────────────────────────────────────────────
use std::cmp::min;

// ─── External Crates ───────────────────────────────────────────────────────────
use tiktoken_rs::{CoreBPE, tokenizer};
use tch::{
    data, nn::{self, embedding, AdamW, EmbeddingConfig, Module, ModuleT, Optimizer}, no_grad, Device, IndexOp, Kind, Tensor
};

// ─── Internal Project Modules ──────────────────────────────────────────────────
use crate::{dataloader::DataLoader, training_helpers::{fine_tuned::*, loss::*, text_sampling::*}};
use crate::architecture::GPTModel;



//printing for project
pub fn generate_and_print_sample(model: &GPTModel, tokenizer: CoreBPE, start_context: &str){
   let mut train = false;
   //grabbing first [0] so we get an i64, not a Vec<i64> -- ws represents weights
   let context_size = model.pos_emb.ws.size()[0];
   let encoded = text_to_token_ids(start_context, tokenizer.clone()).to_device(Device::Cpu);
   //temp of 1 for just in-case
   let token_ids = no_grad(|| generate(model, encoded, 50, context_size, 1.0, Some(3), None, train));
   let decoded_text = token_ids_to_text(token_ids, tokenizer);
   println!("{}", decoded_text.replace("\n", " "));

}


//Below is the actual training function for the model
pub fn train_model_simple(model: &mut GPTModel, train_loader: DataLoader, val_loader: DataLoader,
                           mut optimizer: Optimizer, device: Device, num_epochs: i64, 
                          eval_freq: i64, eval_iter: i64, start_context: String, 
                          tokenizer: CoreBPE, train: bool, batch_size: usize) -> (Vec<f64>, Vec<f64>, Vec<i64>) { //train should be true here
      let mut train_losses: Vec<f64> = Vec::new();
      let mut val_losses: Vec<f64> = Vec::new();
      let mut track_tokens_seen: Vec<i64> = Vec::new();
      let mut tokens_seen = 0;
      let mut global_step = -1;

      for epoch in 0..num_epochs {
         let mut train_iter = DataLoader::init(train_loader.dataset.clone(), batch_size, train_loader.shuffle, train_loader.drop_last);

         for (input_batch, target_batch) in train_iter.by_ref() {
            let input_batch = input_batch.to(device);
            let target_batch = target_batch.to(device);
            // Train step
            optimizer.zero_grad();
            let loss = calc_loss_batch(&input_batch, &target_batch, &model, train);
            loss.backward();
            optimizer.step();

            tokens_seen += input_batch.numel();
            global_step += 1;


            //optional evaluation step
            if global_step % eval_freq == 0 {
                  let (train_loss, val_loss) = evaluate_model(&model, &train_loader, &val_loader, eval_iter);
                  train_losses.push(train_loss);
                  val_losses.push(val_loss);
                  track_tokens_seen.push(tokens_seen as i64);
                  println!(
                     "Ep {} (Step {:06}): Train loss {:.3}, Val loss {:.3}",
                     epoch + 1,
                     global_step,
                     train_loss,
                     val_loss
                  );
            }
         }

         generate_and_print_sample(&model, tokenizer.clone(), &start_context);
      }

      return (train_losses, val_losses, track_tokens_seen)
}
