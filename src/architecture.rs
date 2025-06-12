//This is the architecture file for my LLM
use crate::mha::MultiHeadAttention;
use tch::{nn::{self, Sequential}, Tensor};
use crate::ffn_layer::{LayerNorm, FeedForward};


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
   pub ff: FeedForward,
   pub norm_1: LayerNorm,
   pub norm_2: LayerNorm,
   pub drop_shortcut: f64,
   pub train: bool
}
/*
    def forward(self,x):
        # shortcut connection for attention block
        shortcut=x
        x=self.norm1(x)
        x=self.att(x)
        x=self.drop_shortcut(x)
        # then we add the original input back
        x = x+shortcut
        
        # this is the shortcut connection for the feedforward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        
        return x
 */

impl TransformerBlock{
    pub fn init(cfg: &CONFIG_124M, root: &nn::Path) -> Self {
        let att = MultiHeadAttention::init(root, cfg.emb_dim, cfg.emb_dim, 
            cfg.context_length, cfg.drop_rate, cfg.n_heads, cfg.qkv_bias);
        let ff = FeedForward::init(cfg, root);
        let norm_1 = LayerNorm::init(cfg.emb_dim, root);
        let norm_2 = LayerNorm::init(cfg.emb_dim, root);
        let drop_shortcut = cfg.drop_rate;

        Self { att, ff, norm_1, norm_2, drop_shortcut, train: true}
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shortcut1 = x;
        let mut out = self.norm_1.forward(x);
        out = self.att.forward(&out);
        out = Tensor::dropout(&out, self.drop_shortcut, self.train);
        out = out + shortcut1;

        //use shallow_clone to solve borrowing issue: Returns a new tensor that share storage with the input tensor.
        let shortcut2 = out.shallow_clone();
        out = self.norm_2.forward(&out);
        out = self.ff.forward(&out);
        out = Tensor::dropout(&out, self.drop_shortcut, self.train);
        out = out + shortcut2;


        return out;
    }

    pub fn train(&mut self) {
        self.train = true;
    }

    pub fn eval(&mut self) {
        self.train = false;
    }
}