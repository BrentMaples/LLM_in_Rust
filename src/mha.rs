//This is the Multi-Head Attention implementation in Rust
use tch::{Tensor, nn::Module, nn, Device};
use tch::nn::{LinearConfig, Linear, linear};


pub struct MultiHeadAttention{
    pub d_out: i64,
    pub num_heads: i64,
    pub W_query: Linear,
    pub W_key: Linear,
    pub W_value: Linear,
    pub out_proj: Linear,
    pub dropout: f32,
    pub head_dim: i64
    //register buff will be a fcn call that I create myself 
}

impl MultiHeadAttention{
    //vs is root call for var store
    pub fn init(vs: &nn::Path,d_in:i64,d_out:i64, context_length: i32, dropout: f32, num_heads: i64, qkv_bias: bool) -> Self{
        
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
        
        //Now we need to implement dropout and the register_buffer
        Self {
            d_out,
            num_heads,
            head_dim,
            W_query,
            W_key,
            W_value,
            out_proj,
            dropout
        }
    }
}