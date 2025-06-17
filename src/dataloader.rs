/* I'm going to implement my own Dataloader in Rust - Inspiration from: https://github.com/hughperkins/pytorch-pytorch/blob/master/torch/utils/data/dataloader.py
This will not be as dynamic as Python's version, but I have my own use cases, so I will use it as I see fit.
It will still build based on the C++ bindings for Rust - tch.
*/
use crate::architecture::{generate_text_simple, GPTModel, TransformerBlock, CONFIG_124M};
use crate::dataset::GPTDataset;
use crate::mha::MultiHeadAttention;
use crate::train::train_model_simple;
use crate::training_helpers::{fine_tuned::*, loss::*, text_sampling::*};
use tch::{Tensor};
use rand::{seq::SliceRandom, thread_rng};
/* For my first implementation, the items listed below are going to be ignored: 
      1. Sampling and batch sampling will be ignored, as I have no use for them at the moment.
      2. Pin memory - I am going to run this on the CPU for the moment, so I don't need to worry about CUDA pinned memory.
      3. Number of Workers - I am running 0 workers for the time being, so no implementation is need.
      4. Timeout - My number of workers will be 0 until I understand the dataloader more.
 */
/* For now, I will personally embed the collate function so that it is the standard call */
pub struct DataLoader {
    pub dataset: GPTDataset,
    pub batch_size: usize,
    pub shuffle: bool,
    pub drop_last: bool,
    pub indices: Vec<usize>,
    pub current_index: usize,
}

//I have vectors in gptdataset that I can use to stack. I will just pass those here for the collate??
impl DataLoader {
   pub fn init(dataset: GPTDataset, batch_size: usize, shuffle:bool, drop_last: bool) -> Self {
      let mut indices = dataset.indices();
      if shuffle {
         let mut rng = thread_rng();
         indices.shuffle(&mut rng);
      }
      Self { dataset, batch_size, shuffle, drop_last, indices, current_index: 0}
   }
   //works like a default collate that just stacks everything to makde them a 2D set of tensors
   pub fn collate_fn(batch: &[(Tensor, Tensor)]) -> (Tensor, Tensor) {
        let input_tensors: Vec<Tensor> = batch.iter().map(|(x, _)| x.unsqueeze(0)).collect();
        let target_tensors: Vec<Tensor> = batch.iter().map(|(_, y)| y.unsqueeze(0)).collect();

        let input_batch = Tensor::cat(&input_tensors, 0);
        let target_batch = Tensor::cat(&target_tensors, 0);
        return (input_batch, target_batch);
   }
}
//Iterator implementation for Dataloader
impl Iterator for DataLoader {
   //this is what it will yield on each next call
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.indices.len() {
            return None;
        }
        //Gets end of current batch
        let end_index = (self.current_index + self.batch_size).min(self.indices.len());
        //when drop_last is true, we skip the last one
        if self.drop_last && (end_index - self.current_index < self.batch_size) {
            return None;
        }
        //extracting indice slice
        let batch_indices = &self.indices[self.current_index..end_index];
        //getting information from each batch
        let batch: Vec<(Tensor, Tensor)> = batch_indices
            .iter()
            .map(|&idx| self.dataset.get_sample(idx))
            .collect();
        //then we advance the index
        self.current_index = end_index;
        //then collate
        Some(Self::collate_fn(&batch))
    }
}
