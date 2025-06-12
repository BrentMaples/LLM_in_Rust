#![allow(warnings)]
use std::fs::File;
use std::io::prelude::*;
use tiktoken_rs::r50k_base;
use tch::{Tensor, nn::Module, nn, Device, Kind};
use tch::nn::{EmbeddingConfig, embedding};
//includes dataset.rs fcns
mod dataset;
mod mha;
mod architecture;
mod ffn_layer;
use crate::architecture::{CONFIG_124M, TransformerBlock};
//includes class implementations 
use crate::dataset::GPTDataset;
use crate::mha::MultiHeadAttention;




fn main() {
    tch::manual_seed(123);

    /* Here begins the tokenization methods to be used in our LLM */
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
    //input_batch.print();
    //println!("Inputs shape: {:?}", input_batch.size());
    let embedded = token_embedding_layer.forward(&input_batch);
    //println!("Embedded shape: {:?}", embedded.size());
    
    /*
    At this point in the code I have done the relative position approach for token embeddings. This is what I would consider the first step of the LLM journey.
    The next step now is to implement the attention mechanism.
    */
    //Begin Efficient Multi-Head Attention Implementation
    
    //Initializing tensor
    let inputs = Tensor::from_slice(&[
        0.43, 0.15, 0.89,  // x^1
        0.55, 0.87, 0.66,  // x^2
        0.57, 0.85, 0.64,  // x^3
        0.22, 0.58, 0.33,  // x^4
        0.77, 0.25, 0.10,  // x^5
        0.05, 0.80, 0.55   // x^6
    ])
    .reshape(&[6, 3])
    .to_kind(Kind::Float);//need reshape

    
    
    
    let batch = Tensor::stack(&[&inputs, &inputs], 0);
    //extracting shape from vec
    let [batch_size, context_length, d_in]: [i64; 3] = batch.size().try_into().unwrap();
    let d_out = 2;
    let num_heads = 2;
    let mha = MultiHeadAttention::init(root,d_in, d_out, context_length, 0.0, num_heads, false);
    //mha is an instance of MHA, so do . notation 
    let context_vecs = mha.forward(&batch);
    // println!("Tensors after MHA:");
    // context_vecs.print();
    

    /* Now that Multi-Head Attention has been implemented efficiently, we will now carry on to building the LLM architecture.
        This includes the transformer block, model, and text conversion. */

    // Let us create the struct for our configuration
    let model_config = CONFIG_124M{
        vocab_size: 50257,
        context_length: 256,
        emb_dim: 768,
        n_heads: 12,
        n_layers: 12,
        drop_rate: 0.1,
        qkv_bias: false
    };
    
    let transform_input = Tensor::randn([2,4,model_config.emb_dim], (Kind::Float, Device::Cpu));
    let block = TransformerBlock::init(&model_config, root);
    let output = block.forward(&transform_input);

    //shape is maintained, which is good
    println!("{:?}", transform_input.size());
    println!("{:?}", output.size());
    return;

}
