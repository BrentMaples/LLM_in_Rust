/* Implementation for supervised instruction fine-tuning */
use serde::Deserialize;
use tiktoken_rs::{tokenizer, CoreBPE};
use tch::{Tensor, Kind};
use crate::dataset::*;

/* Custom entry struct for the JSON data */
#[derive(Debug, Deserialize, Clone)]
pub struct Entry {
   pub instruction: String,
   pub input: String,
   pub output: String
}
#[derive(Clone)]
pub struct InstructionDataset{
   pub data: Vec<Entry>,
   pub encoded_texts: Vec<Vec<u32>>
}

impl InstructionDataset {
   pub fn init(data: Vec<Entry>, tokenizer:CoreBPE) -> Self {
      //only need one new for this
      let mut encoded_texts: Vec<Vec<u32>> = Vec::new();
      let vec_data = data.clone();

      //padding the data
      for entry in data{
         let instruction_plus_input = format_input(&entry);
         let response_text = format!("\n\n### Response:\n{}", &entry.output);
         let full_text = instruction_plus_input + &response_text;
         encoded_texts.push(tokenizer.encode_with_special_tokens(&full_text));
      }  
      Self {data: vec_data, encoded_texts}
   }
}
impl Dataset for InstructionDataset {
    fn get_sample(&self, index: usize) -> (Tensor, Tensor) {
        let token_ids = &self.encoded_texts[index];
        let input_ids: Vec<i64> = token_ids.iter().map(|&x| x as i64).collect();

        let input = Tensor::from_slice(&input_ids[..input_ids.len() - 1]).to_kind(Kind::Int64);
        let target = Tensor::from_slice(&input_ids[1..]).to_kind(Kind::Int64);
        (input, target)
    }

    fn indices(&self) -> Vec<usize> {
        (0..self.encoded_texts.len()).collect()
    }

    fn len(&self) -> usize {
        self.encoded_texts.len()
    }
}

pub fn format_input(entry: &Entry) -> String {
    let instruction_text = format!(
        "Below is an instruction that describes a task. \
Write a response that appropriately completes the request.\n\n### Instruction:\n{}",
        entry.instruction
    );

    let input_text = if entry.input.trim().is_empty() {
        String::new()
    } else {
        format!("\n\n### Input:\n{}", entry.input)
    };

    format!("{}{}", instruction_text, input_text)
}
