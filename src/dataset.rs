/* This creates my tokenized text for me */
//bpe implementation
use tch::{data, Kind, Tensor};
//GPT2
use tiktoken_rs::{r50k_base, tokenizer, CoreBPE};

use crate::dataloader::DataLoader;


/* to make my code dynamic for both datasets */
pub trait Dataset: DatasetClone {
    fn get_sample(&self, index: usize) -> (Tensor, Tensor);
    fn indices(&self) -> Vec<usize>;
    fn len(&self) -> usize;
}

pub trait DatasetClone {
    fn clone_box(&self) -> Box<dyn Dataset>;
}
impl<T> DatasetClone for T
where
    T: Dataset + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn Dataset> {
        Box::new(self.clone())
    }
}
impl Clone for Box<dyn Dataset> {
    fn clone(&self) -> Box<dyn Dataset> {
        self.clone_box()
    }
}

//at the moment, implementing GPTDataset from my notes -- equiv to pytorch dataset in rust

pub struct GPTDataset{
    pub input_ids: Vec<Tensor>,
    pub target_ids: Vec<Tensor>,
    pub txt: String,
    pub max_len: usize,
    pub stride: usize,
}

//no init in rust
//so we just build a function that does what we want inside of the implementations

//So we have built the GPT dataset equivalent in Rust
impl GPTDataset {
    //defined self for no return
    // takes the string data from the file and we can stuff it into the BPE, may not need to pass tokenizer itself here
    pub fn init (txt: String, tokenizer: CoreBPE, max_len: usize, 
                stride: usize) -> Self{
        //pytorch wrapper tensors
        let mut input_ids: Vec<Tensor> = Vec::new();
        let mut target_ids: Vec<Tensor> = Vec::new();
        let token_ids: Vec<u32> = tokenizer.encode_with_special_tokens(&txt);
        //println!("Token count: {}", token_ids.len());
    
        //this is where I would do the tokenizer stuff here
        let loop_range = token_ids.len() - max_len;

        //then do the loop here
        for i in (0..loop_range).step_by(stride) {
            // now that we have tokenizer, we can do stuff with it
            let input_chunk = &token_ids[i..i + max_len];
            let target_chunk = &token_ids[i+1..i + max_len + 1];
            let input_tensor = Tensor::from_slice(
                &input_chunk.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            ).to_kind(Kind::Int64);
            
            let target_tensor = Tensor::from_slice(
                &target_chunk.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            ).to_kind(Kind::Int64);
            
            input_ids.push(input_tensor);
            target_ids.push(target_tensor);

        }
        //required to return the changes made in this init
        Self {
            input_ids,
            target_ids,
            txt,
            max_len,
            stride
        }

    }

}
impl Dataset for GPTDataset {
    fn get_sample(&self, index: usize) -> (Tensor, Tensor) {
        self.get_sample(index)
    }

    fn indices(&self) -> Vec<usize> {
        self.indices()
    }

    fn len(&self) -> usize {
        self.len()
    }
}
//had to implement the copy trait
impl Clone for GPTDataset {
    fn clone(&self) -> Self {
        GPTDataset {
            input_ids: self.input_ids.iter().map(|t| t.shallow_clone()).collect(),
            target_ids: self.target_ids.iter().map(|t| t.shallow_clone()).collect(),
            txt: self.txt.clone(),
            max_len: self.max_len,
            stride: self.stride,
        }
    }
}

//so here I need to implement the create_dataloader_v1 fn
pub fn create_dataloader_v1(txt: String, batch_size: usize, max_len: usize, stride: usize, shuffle: bool, drop_last: bool, tokenizer: CoreBPE,
                            collate_fn: Box<dyn Fn(&[(Tensor, Tensor)]) -> (Tensor, Tensor)>,) -> DataLoader{
    let dataset = GPTDataset::init(txt, tokenizer, max_len, stride);
    let loader = DataLoader::init(Box::new(dataset), batch_size, true, true);
    return loader
}