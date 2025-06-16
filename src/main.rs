#![allow(warnings)]
// ─── Standard Library ──────────────────────────────────────────────────────────
use std::fs::File;
use std::io::prelude::*;

// ─── External Crates ───────────────────────────────────────────────────────────
use tiktoken_rs::r50k_base;
use tch::{
    CModule, Device, Kind, Tensor,
    nn::{self, EmbeddingConfig, Module, ModuleT, embedding}
};

// ─── Internal Crate Modules ────────────────────────────────────────────────────
use crate::architecture::{generate_text_simple, GPTModel, TransformerBlock, CONFIG_124M};
use crate::dataset::GPTDataset;
use crate::mha::MultiHeadAttention;
use crate::training_helpers::{fine_tuned::*, loss::*, text_sampling::*};

// ─── Project Module Declarations ───────────────────────────────────────────────
mod architecture;
mod dataset;
mod ffn_layer;
mod mha;
mod train;
mod training_helpers;




fn main() {
    //training fcn
    let train = true;
    //seed
    tch::manual_seed(123);

    /* Here begins the tokenization methods to be used in our LLM */
    //Reading the file in and making it a txt
    let file_path = "./data/the_verdict.txt";
    let mut file = File::open(file_path).expect("File Path Invalid");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Contents Failed");
    
    //my own dataset - for preprocessing and making 
    let tokenizer = r50k_base().unwrap();
    let dataset = GPTDataset::init(contents, tokenizer.clone(), 4, 4);
    // For embedding reqs:
    let vs = nn::VarStore::new(Device::Cpu);
    let root = &vs.root();
    // Let us create our model configuration
    let model_config = CONFIG_124M{
        vocab_size: 50257,
        context_length: 256,
        emb_dim: 768,
        n_heads: 12,
        n_layers: 12,
        drop_rate: 0.1,
        qkv_bias: false
    };
    let mut model = GPTModel::init(&model_config, root);
    


    

    return;

}
