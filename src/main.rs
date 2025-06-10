#![allow(warnings)]
use std::fs::File;
use std::io::prelude::*;
use tiktoken_rs::r50k_base;
use tch::{Tensor, nn::Module, nn, Device};
use tch::nn::{EmbeddingConfig, embedding};
//includes dataset.rs fcns
mod dataset;
//includes class implementations 
use crate::dataset::GPTDataset;



fn main() {
    //Reading the file in and making it a txt
    let file_path = "./data/the_verdict.txt";
    let mut file = File::open(file_path).expect("File Path Invalid");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Contents Failed");
    
    // Create dataset
    let max_len = 4;
    let stride = 4;
    let batch_size = 8;
    let shuffle = false;
    let drop_last = true;
    //my own dataset
    let tokenizer = r50k_base().unwrap();
    let dataset = GPTDataset::init(contents, tokenizer, max_len, stride);
    // For embedding reqs:
    let vs = nn::VarStore::new(Device::Cpu);
    let root = &vs.root();
    let vocab_size = 50257;
    let output_dim = 256;
    let default_config = EmbeddingConfig::default();
    let token_embedding_layer = nn::embedding(root, vocab_size, output_dim, default_config);

    //now we simulate printing - this only does one iteration for the time being before breaking
    let (input_batch, target_batch) = dataset::batch_printing(batch_size, dataset);      
    //println!("Total elements: {}", input_batch.numel());
    input_batch.print();
    println!("Inputs shape: {:?}", input_batch.size());
    let embedded = token_embedding_layer.forward(&input_batch);
    println!("Embedded shape: {:?}", embedded.size());
    /*
    At this point in the code I have done the relative position approach for token embeddings. This is what I would consider the first step of the LLM journey.
    The next step now is to implement the attention mechanism.
    */
    return;

}
