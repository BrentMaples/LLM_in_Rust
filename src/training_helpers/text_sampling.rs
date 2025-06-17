//These are functions to convert the token ids to text and likewise 

use std::cmp::min;
use tiktoken_rs::{CoreBPE, tokenizer};
use tch::{
    data, no_grad, Device, IndexOp, Kind, Tensor,
    nn::{self, AdamW, EmbeddingConfig, Module, ModuleT}
};
// Internal modules
use crate::architecture::GPTModel;
use crate::training_helpers::{fine_tuned::*, loss::*};

//convert given string to token id 
pub fn text_to_token_ids(txt: &'static str, tokenizer: CoreBPE ) -> Tensor {
   let tokens_ids: Vec<i64> = tokenizer
            .encode_with_special_tokens(&txt)
            .into_iter()
            .map(|x| x as i64)
            .collect();
   let mut encoded_tensor = Tensor::from_slice(&tokens_ids)
            .to_kind(Kind::Int64)
            .to_device(Device::Cpu);
   encoded_tensor = encoded_tensor.unsqueeze(0);

   return encoded_tensor;
}

//reconvert token id to text
pub fn token_ids_to_text(token_ids: Tensor, tokenizer: CoreBPE) -> String {
   let flat = token_ids.squeeze();
   //so I need to iterate through the vec and then unwrap and map it to usize before collecting it
   let converted_tensor: Vec<u32> = flat.iter::<i64>().unwrap().map(|x| x as u32).collect();
   //unwrap to raise panic 
   let decoded_tensor = tokenizer.decode(converted_tensor).unwrap();
   return decoded_tensor;
}

//Generating text for multi-token prediction using top-k and temperature. May add top-p in the future!
pub fn generate(model: &GPTModel, mut idx: Tensor, max_new_tokens: i64,
               context_size: i64, temperature: f64, top_k: Option<i64>,
               eos_id: Option<i64>, train: bool) -> Tensor {
      for _ in 0..max_new_tokens{
         //rust tensor slicing - must clamp in Rust because Python automatically clamps
         let seq_len = idx.size()[1];
         let start_idx = if seq_len > context_size {
            seq_len - context_size
         } else {
            0
         };
         let idx_cond = idx.i((.., start_idx..));

         let mut logits = no_grad(|| model.forward_t(&idx_cond, train));

         logits = logits.i((.., -1, ..));

         if top_k != None {
            //-1 for last dim, largest for true,  and sorted true since it's default options for top_k
            let (top_logits, _) = Tensor::topk(&logits, top_k.unwrap(), -1, true, true);
            let min_val = top_logits.i((..,-1));
            
            //logits < min_val
            let mask= logits.lt_tensor(&min_val);
            let neg_inf= Tensor::from(f64::NEG_INFINITY)
                                 .to_device(logits.device());  // broadcastable scalar
            //if condition mask is true, then it is neg_inf, otherwise logits
            logits = neg_inf.where_self(&mask, &logits);
         }
         let mut idx_next = Tensor::new();
         if temperature > 0.0 {
            logits = logits / temperature;
            let probs = Tensor::softmax(&logits, -1, Kind::Float);
            idx_next = Tensor::multinomial(&probs, 1, false);
         }
         else {
            idx_next = Tensor::argmax(&logits, -1, true);
         }
         //idx_next == eos_id
         if eos_id.is_some() {
            let eos_tensor = Tensor::from(eos_id.unwrap()).to_device(idx_next.device());
            if idx_next.eq_tensor(&eos_tensor).any().int64_value(&[]) != 0 {
               break;
            }
         }
         idx = Tensor::cat(&[idx, idx_next], 1);

      }

      return idx;
}


