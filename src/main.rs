#![allow(warnings)]
// ─── Standard Library ──────────────────────────────────────────────────────────
use std::{default, fs::File};
use std::io::prelude::*;

// ─── External Crates ───────────────────────────────────────────────────────────
use tiktoken_rs::r50k_base;
use tch::{
    CModule, Device, Kind, Tensor,
    nn::{self, EmbeddingConfig, Module, ModuleT, embedding, AdamW, OptimizerConfig}
};

// ─── Internal Crate Modules ────────────────────────────────────────────────────
use crate::architecture::{generate_text_simple, GPTModel, TransformerBlock, CONFIG_124M};
use crate::dataset::{create_dataloader_v1, GPTDataset};
use crate::mha::MultiHeadAttention;
use crate::train::train_model_simple;
use crate::training_helpers::{fine_tuned::*, loss::*, text_sampling::*};

// ─── Project Module Declarations ───────────────────────────────────────────────
mod architecture;
mod dataset;
mod ffn_layer;
mod mha;
mod train;
mod training_helpers;
mod dataloader;




fn main() {
    //training fcn
    let train = true;
    //seed
    tch::manual_seed(123);

    //Reading the file in and making it a txt
    let file_path = "./data/the_verdict.txt";
    let mut file = File::open(file_path).expect("File Path Invalid");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Contents Failed");

    // Perform 90/10 split on raw string
    let total_len = contents.len();
    let split_index = (total_len as f32 * 0.8).floor() as usize;

    let train_txt = contents[..split_index].to_string();
    let valid_txt = contents[split_index..].to_string();
    
    //my own dataset - for preprocessing and making 
    let tokenizer = r50k_base().unwrap();
    // let dataset = GPTDataset::init(contents, tokenizer.clone(), 4, 4);
    // For device and model requirements
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
    //model uses cpu because it utilizes the varstore
    let mut model = GPTModel::init(&model_config, root);
    //all are default parameters except for weight decay (wd) being 0.1
    let adamw_config = nn::AdamW { wd: 0.1, beta1: 0.9, beta2: 0.999, eps: 1e-8, amsgrad: false};
    //must declare AdamW this way, as .build requires self. vs is equivalent to model.parameters() in python
    let mut optimizer = adamw_config.build(&vs, 0.0004).unwrap();
    
    let num_epochs = 10;
    let batch_size = 2;


    //that was annoying but at least the data loader is implemented now. 
    let train_loader = create_dataloader_v1(train_txt, batch_size, model_config.context_length as usize, model_config.context_length as usize, 
                                                        true, true, tokenizer.clone());
    let val_loader = create_dataloader_v1(valid_txt, batch_size, model_config.context_length as usize, model_config.context_length as usize, 
                                                        false, false, tokenizer.clone());
    let (train_losses, val_losses, tokens_seen) = train_model_simple(model, train_loader, val_loader, optimizer, Device::Cpu, 
        num_epochs, 5, 5, "Every effort moves you", tokenizer.clone(), train, batch_size);



    return;

}
