use core::{f64, num};

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
    pub mask: Tensor,// I guess this is how I will do the register_buffer,
    pub train: bool
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
            mask,
            train: true //default val for train
        }
    }
    
    pub fn forward(&self,x: Tensor) -> Tensor{
        let [b, num_tokens, d_in]: [i64; 3] = x.size().try_into().unwrap();
        //println!("{:?}", x);
        //tensor shape for arrays
        let mut keys = self.W_key.forward(&x);
        let mut queries = self.W_query.forward(&x);
        let mut values = self.W_value.forward(&x);
        
        let new_shape = [b, num_tokens, self.num_heads, self.head_dim];
        //view and reshape are similar here. We want my tensor to look at the same data so we use view here.
        //view will fail and never copy while reshape CAN copy if the tensors are not contiguous.
        //No errors were raised so it is irrelevant here.
        //keys = keys.reshape(&*new_shape);
        //These are reshaped dimensions
        keys = keys.view(new_shape); 
        queries = queries.view(new_shape);
        values = values.view(new_shape);
        
        // transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2);
        queries = queries.transpose(1, 2);
        values = values.transpose(1, 2);

        //now dot product for each head
        let mut attn_scores = queries.matmul(&(keys.transpose(2, 3)));
        
        //first convert tensor to a bool mask - Kind is a typecast.
        let bool_tens = self.mask.to_kind(Kind::Bool);
        //then I build the slicing by dim, equiv to [:num, num:] in python
        let mask_bool = bool_tens
                        .narrow(0,0,num_tokens)
                        .narrow(1,0,num_tokens);
        //using the mask to fill attention scores
        attn_scores.masked_fill_(&mask_bool, f64::NEG_INFINITY);
        
        //to get last element -> *keys.size() returns Vec<i64> -> .last for vec -> .unwrap for element
        let last_element = *keys.size().last().unwrap() as f64;
        let mut attn_weights = Tensor::softmax(&(attn_scores/ last_element.powf(0.5)) , -1, (Kind::Float));
        attn_weights = Tensor::dropout(&attn_weights, self.dropout_val, self.train);

        let mut context_vec = (attn_weights.matmul(&values)).transpose(1, 2);
        context_vec = context_vec.contiguous().view([b, num_tokens, self.d_out]);
        
        //optional linear projection - applies linear transformation of out_proj on context_vec
        context_vec = self.out_proj.forward(&context_vec);
        return context_vec;
    }

    //for the sake of train and eval, we will do this
    pub fn train(&mut self) {
        self.train = true;
    }

    pub fn eval(&mut self) {
        self.train = false;
    }
}