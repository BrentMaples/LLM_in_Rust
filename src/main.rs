#![allow(warnings)]
// ─── Standard Library ──────────────────────────────────────────────────────────
use std::{default, fs, fs::File};
use std::io::prelude::*;

// ─── External Crates ───────────────────────────────────────────────────────────
use tiktoken_rs::r50k_base;
use tch::{
    CModule, Device, Kind, Tensor,
    nn::{self, EmbeddingConfig, Module, ModuleT, embedding, AdamW, OptimizerConfig}
};
use serde_json;

// ─── Internal Crate Modules ────────────────────────────────────────────────────
use crate::architecture::{generate_text_simple, GPTModel, TransformerBlock, CONFIG_124M};
use crate::dataset::{create_dataloader_v1, GPTDataset};
use crate::mha::MultiHeadAttention;
use crate::train::train_model_simple;
use crate::training_helpers::{fine_tuned::*, loss::*, text_sampling::*};
use crate::dataloader::*;

// ─── Project Module Declarations ───────────────────────────────────────────────
mod architecture;
mod dataset;
mod ffn_layer;
mod mha;
mod train;
mod training_helpers;
mod dataloader;




fn main() {
    //training legality
    let train = true;
    //seed
    tch::manual_seed(123);

    
    //my own dataset - for preprocessing and making 
    let tokenizer = r50k_base().unwrap();
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
    let mut optimizer = adamw_config.build(&vs, 0.0005).unwrap();


    /* This is the Instruction Based Tuning Step - It uses a special format for instructions, such as the ALPACA version we are using. */
    let json_data = fs::read_to_string("data/instruction-data.json").expect("Failed to read file");
    let entries: Vec<Entry> = serde_json::from_str(&json_data).expect("Failed to parse JSON");

    let entries_len = entries.len();
    let train_split = (entries_len as f64 * 0.85).floor() as usize;
    let test_split = (entries_len as f64 * 0.1).floor() as usize;

    //first 85%
    let train_data = &entries[0..train_split];
    //next 10%
    let test_data = &entries[train_split..train_split+test_split];
    //last 5%
    let validation_data = &entries[train_split + test_split..];

    let train_dataset = InstructionDataset::init(train_data.to_vec(), tokenizer.clone());
    let test_dataset = InstructionDataset::init(test_data.to_vec(), tokenizer.clone());
    let validation_dataset = InstructionDataset::init(validation_data.to_vec(), tokenizer.clone());

    let batch_size = 8;

    // DataLoader creation
    let train_loader = DataLoader::init(Box::new(train_dataset), batch_size, true, true);  // shuffle, drop_last
    let val_loader = DataLoader::init(Box::new(validation_dataset.clone()), batch_size, false, false);   // no shuffle, no drop
    let test_loader = DataLoader::init(Box::new(test_dataset), batch_size, false, false); // no shuffle, no drop

    let example_prompt = format_input(&validation_dataset.data[0]);


    let (train_losses, val_losses, tokens_seen) = train_model_simple(&mut model, train_loader, val_loader, optimizer, Device::Cpu, 
        2, 5, 5, example_prompt, tokenizer.clone(), train, batch_size);

    // This is where I text my model's output.
    for entry in test_data.iter().take(3) {
        let input_text = format_input(entry);

        let input_ids = text_to_token_ids(&input_text, tokenizer.clone()).to_device(Device::Cpu);

        let token_ids = generate(
            &model,
            input_ids,
            256,                         // max_new_tokens
            model_config.context_length,
            1.0, Some(3), Some(50256), true                        // eos_id
        );

        let generated_text = token_ids_to_text(token_ids, tokenizer.clone());

        // Remove the prompt portion and clean the model output
        let response_text = generated_text[input_text.len()..]
            .replace("### Response:", "")
            .trim()
            .to_string();

        println!("Input:\n{}", input_text);
        println!("\nCorrect response:\n>> {}", entry.output);
        println!("\nModel response:\n>> {}", response_text);
        println!("------------------------------------");
    }



    return;

}
