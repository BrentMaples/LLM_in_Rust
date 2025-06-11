//This is the Multi-Head Attention implementation in Rust
use tch::{Tensor, Device, Kind, nn};
use tch::nn::{LinearConfig, Linear, linear, Module};

pub struct MultiHeadAttention{
    pub d_out: i64,
    pub num_heads: i64,
    pub W_query: Linear,
    pub W_key: Linear,
    pub W_value: Linear,
    pub out_proj: Linear,
    pub dropout_val: f64,
   // pub dropout: Tensor, //use it on tensor we are modifying, not like python. Call it on attn_weights tensor as a dropout
    pub head_dim: i64,
    //register buff will be a fcn call that I create myself 
    pub mask: Tensor// I guess this is how I will do the register_buffer
}

impl MultiHeadAttention{
    //vs is root call for var store
    pub fn init(vs: &nn::Path,d_in:i64,d_out:i64, context_length: i64, dropout_val: f64, num_heads: i64, qkv_bias: bool) -> Self{
        
        let d_out = d_out;
        let num_heads = num_heads;
        //auto floor div in rust so no need to floor
        let head_dim = d_out / num_heads;
        //defaults for ws_init and bs_init but setting bias to qkv_bias val
        let config = LinearConfig {bias: qkv_bias,..Default::default()};
        //need to look at variable stores more later on
        
        let W_query = linear(vs, d_in, d_out,config);
        let W_key = linear(vs, d_in, d_out,config);
        let W_value = linear(vs, d_in, d_out,config);
        
        let out_proj = linear(vs, d_out, d_out, config);
        
        
        //Now we need to implement dropout and the register buffer
        let dropout_val = dropout_val; // dropout is registered differently
        //now for the register buffer
        let mask = Tensor::ones([context_length,context_length], (Kind::Float, Device::Cpu)).triu(1);
        Self {
            d_out,
            num_heads,
            head_dim,
            W_query,
            W_key,
            W_value,
            out_proj,
            dropout_val,
            mask
        }
    }
    
    pub fn forward(&self,x: Tensor) -> Tensor{
        let [b, num_tokens, d_in]: [i64; 3] = x.size().try_into().unwrap();
        //println!("{:?}", x);
        //tensor shape for arrays
        let mut keys = self.W_key.forward(&x);
        let mut queries = self.W_query.forward(&x);
        let mut values = self.W_value.forward(&x);
        
        //How do I make this work?
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim));
        
        
        return x;
    }
}