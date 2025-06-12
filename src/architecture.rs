//This is the architecture file for my LLM
use crate::mha::MultiHeadAttention;
use tch::nn::Sequential;
use crate::ffn_layer::{LayerNorm, FeedForward};
use tch::nn::{Init, LinearConfig, Linear, linear, Module};
use tch::nn::init::{NormalOrUniform, FanInOut, NonLinearity, DEFAULT_KAIMING_UNIFORM};

pub struct CONFIG_124M{
   pub vocab_size: i64,
   pub context_length: i64,
   pub emb_dim: i64, 
   pub n_heads: i64,
   pub n_layers: i64,
   pub drop_rate: f64,
   pub qkv_bias: bool
}

//now let us implement the transformer block - stop for now, need to implement the other stuff

pub struct TransformerBlock {
   pub att: MultiHeadAttention,
   pub ff: Sequential,

}

impl TransformerBlock{
   pub fn init(cfg: CONFIG_124M){

   }
}