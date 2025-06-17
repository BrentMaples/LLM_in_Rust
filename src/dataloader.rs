/* I'm going to implement my own Dataloader in Rust - Inspiration from: https://github.com/hughperkins/pytorch-pytorch/blob/master/torch/utils/data/dataloader.py
This will not be as dynamic as Python's version, but I have my own use cases, so I will use it as I see fit.
It will still build based on the C++ bindings for Rust - tch.
*/
use crate::architecture::{generate_text_simple, GPTModel, TransformerBlock, CONFIG_124M};
use crate::dataset::*;
use crate::mha::MultiHeadAttention;
use crate::train::train_model_simple;
use crate::training_helpers::{fine_tuned::*, loss::*, text_sampling::*};
use tch::{Device, Tensor, Kind, IndexOp};
use rand::{seq::SliceRandom, thread_rng};
/* For my first implementation, the items listed below are going to be ignored: 
      1. Sampling and batch sampling will be ignored, as I have no use for them at the moment.
      2. Pin memory - I am going to run this on the CPU for the moment, so I don't need to worry about CUDA pinned memory.
      3. Number of Workers - I am running 0 workers for the time being, so no implementation is need.
      4. Timeout - My number of workers will be 0 until I understand the dataloader more.
 */
/* For now, I will personally embed the collate function so that it is the standard call */


pub struct DataLoader {
    pub dataset: Box<dyn Dataset>,
    pub batch_size: usize,
    pub shuffle: bool,
    pub drop_last: bool,
    pub indices: Vec<usize>,
    pub current_index: usize
}

//I have vectors in gptdataset that I can use to stack. I will just pass those here for the collate??
impl DataLoader {
    pub fn init(dataset: Box<dyn Dataset>, batch_size: usize, shuffle: bool, drop_last: bool) -> Self {
        let mut indices = dataset.indices();
        if shuffle {
            let mut rng = thread_rng();
            indices.shuffle(&mut rng);
        }
        Self {
            dataset,
            batch_size,
            shuffle,
            drop_last,
            indices,
            current_index: 0,
        }
    }

   //works like a default collate that just stacks everything to makde them a 2D set of tensors
   pub fn default_collate_fn(batch: &[(Tensor, Tensor)]) -> (Tensor, Tensor) {
        let input_tensors: Vec<Tensor> = batch.iter().map(|(x, _)| x.unsqueeze(0)).collect();
        let target_tensors: Vec<Tensor> = batch.iter().map(|(_, y)| y.unsqueeze(0)).collect();

        let input_batch = Tensor::cat(&input_tensors, 0);
        let target_batch = Tensor::cat(&target_tensors, 0);
        return (input_batch, target_batch);
   }
}

impl DataLoader{
    //defaults for pad_token_id: 50256, ignore_index: -100 (we set ignore_index to -100 since CrossEntropy ignores values at -100)
    pub fn custom_collate_fn(batch: &[(Tensor, Tensor)], pad_token_id: i64, ignore_index: i64, allowed_max_length: Option<usize>, device: Device) -> (Tensor, Tensor){
        let batch_max_length = batch.iter().map(|(input, _)| input.size()[0] + 1).max().unwrap_or(0);
        let inputs_list: Vec<Tensor> = Vec::new();
        let targets_list: Vec<Tensor> = Vec::new();
        let mut inputs_list = Vec::with_capacity(batch.len());
        let mut targets_list = Vec::with_capacity(batch.len());
        for (input_tensor, _) in batch.iter() {
                // Append pad_token_id to input
                let pad_tensor = Tensor::from_slice(&[pad_token_id]).to_device(input_tensor.device());
                let new_item = Tensor::cat(&[input_tensor, &pad_tensor], 0);

                // Pad to batch_max_length
                let pad_len = batch_max_length as i64 - new_item.size()[0];
                let padded = if pad_len > 0 {
                    let pad = Tensor::full(&[pad_len], pad_token_id, (Kind::Int64, input_tensor.device()));
                    Tensor::cat(&[&new_item, &pad], 0)
                } else {
                    new_item
                };

                let inputs = padded.slice(0, 0, padded.size()[0] - 1, 1);
                let mut targets = padded.slice(0, 1, padded.size()[0], 1);

                // Mask pad_token_id in targets except first occurrence
                let mask = targets.eq(pad_token_id);
                let indices = mask.nonzero();
                if indices.size()[0] > 1 {
                    let tail_indices = indices.i(1..).squeeze();

                    // Ensure the index tensor is 1D
                    let tail_indices = if tail_indices.dim() == 0 {
                        tail_indices.unsqueeze(0)
                    } else {
                        tail_indices
                    };

                    let ignore_tensor = Tensor::full(&tail_indices.size(), ignore_index, (Kind::Int64, device));

                    targets = targets.scatter(0, &tail_indices, &ignore_tensor);
                }

                // Truncate
                let inputs = if let Some(max_len) = allowed_max_length {
                    inputs.slice(0, 0, max_len as i64, 1)
                } else {
                    inputs
                };

                let targets = if let Some(max_len) = allowed_max_length {
                    targets.slice(0, 0, max_len as i64, 1)
                } else {
                    targets
                };

                inputs_list.push(inputs);
                targets_list.push(targets);
            }

        let inputs_tensor = Tensor::stack(&inputs_list, 0).to_device(device);
        let targets_tensor = Tensor::stack(&targets_list, 0).to_device(device);

        return (inputs_tensor, targets_tensor)
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
        //then collate - for now we will have two collates, one I define as default, the other as not 
        //return Some(DataLoader::default_collate_fn(&batch));
        return Some(DataLoader::custom_collate_fn(&batch,50256, -100, Some(1024), Device::Cpu));
    }
}
