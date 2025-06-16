//This is the architecture file for my LLM
// External crates - tch
use tch::{
    no_grad, Device, Kind, Tensor,
    nn::{
        self, embedding, init::{DEFAULT_KAIMING_UNIFORM, FanInOut, NonLinearity, NormalOrUniform},
        Embedding, EmbeddingConfig, Init, Linear, LinearConfig, Module, ModuleT, SequentialT, linear
    }
};
// Internal modules
use crate::ffn_layer::{FeedForward, LayerNorm};
use crate::mha::MultiHeadAttention;




pub struct CONFIG_124M{
   pub vocab_size: i64,
   pub context_length: i64,
   pub emb_dim: i64, 
   pub n_heads: i64,
   pub n_layers: i64,
   pub drop_rate: f64,
   pub qkv_bias: bool
}

//now let us implement the transformer block

#[derive(Debug)]
pub struct TransformerBlock {
   pub att: MultiHeadAttention,
   pub ff: FeedForward,
   pub norm_1: LayerNorm,
   pub norm_2: LayerNorm,
   pub drop_shortcut: f64,
   
}

impl TransformerBlock{
    pub fn init(cfg: &CONFIG_124M, root: &nn::Path) -> Self {
        let att = MultiHeadAttention::init(root, cfg.emb_dim, cfg.emb_dim, 
            cfg.context_length, cfg.drop_rate, cfg.n_heads, cfg.qkv_bias);
        let ff = FeedForward::init(cfg, root);
        let norm_1 = LayerNorm::init(cfg.emb_dim, root);
        let norm_2 = LayerNorm::init(cfg.emb_dim, root);
        let drop_shortcut = cfg.drop_rate;

        Self { att, ff, norm_1, norm_2, drop_shortcut}
    }
}
// train is passed here
impl ModuleT for TransformerBlock {
    fn forward_t(&self, x: &Tensor, train: bool) -> Tensor {
        let shortcut1 = x;
        let mut out = self.norm_1.forward(x);
        out = self.att.forward_t(&out, train);
        out = out.dropout(self.drop_shortcut, train);
        out = out + shortcut1;

        let shortcut2 = out.shallow_clone();
        out = self.norm_2.forward(&out);
        out = self.ff.forward(&out);
        out = out.dropout(self.drop_shortcut, train);
        out + shortcut2
    }
}

#[derive(Debug)]
pub struct GPTModel{
    pub tok_emb: Embedding,
    pub pos_emb: Embedding,
    pub drop_val: f64,
    pub trf_blocks: SequentialT,
    pub final_norm: LayerNorm,
    pub out_head: Linear
}

impl GPTModel{
    pub fn init(cfg: &CONFIG_124M, root: &nn::Path) -> Self{
        //this is this is the default, I just prefer to write it out for clarity
        let embed_config = EmbeddingConfig{
            sparse: false,
            scale_grad_by_freq: false,
            ws_init: Init::Randn { mean: 0., stdev: 1. },
            padding_idx: -1 //equates to None in Rust 
        };
        
        let tok_emb = embedding(root, cfg.vocab_size, cfg.emb_dim, embed_config);
        let pos_emb = embedding(root, cfg.context_length, cfg.emb_dim, embed_config);
        //do not init dropout here
        let mut trf_blocks = nn::seq_t();
        for _ in 0..cfg.n_layers {
            //prevents moving of values
            let block = TransformerBlock::init(cfg, &root);
            trf_blocks = trf_blocks.add(block); //adding it to sequential
        }


        let final_norm = LayerNorm::init(cfg.emb_dim, root);
        let lin_config = LinearConfig {
            ws_init: Init::Kaiming {
                dist: NormalOrUniform::Uniform,
                fan: FanInOut::FanIn,
                non_linearity: NonLinearity::ReLU,
            },
            bs_init: None,
            bias: cfg.qkv_bias
        };
        /* IMPORTANT. If we wanted to fix this to a classification task such as a binary 0 and 1 (spam and not spam),
            we would change the output layer shown by cfg.vocab_size to be 2. This is relevant to classification fine-tuning
        */
        let out_head = linear(root, cfg.emb_dim, cfg.vocab_size, lin_config);
        Self { tok_emb, pos_emb , drop_val: cfg.drop_rate, trf_blocks, final_norm, out_head}
    }
}
//makes this 
impl ModuleT for GPTModel{
        fn forward_t(&self, in_idx: &Tensor, train:bool) -> Tensor {
            let [batch_size, seq_len]: [i64; 2] = in_idx.size().try_into().unwrap();
            let tok_embeds = self.tok_emb.forward(&in_idx);
            //allows us to train data with a CPU or GPU, depending on which device the input data sits on
            let pos_embed_inp = Tensor::arange(seq_len, (Kind::Int64, Device::Cpu));
            let pos_embeds = self.pos_emb.forward(&pos_embed_inp);

            let mut x = tok_embeds + pos_embeds;
            x = x.dropout(self.drop_val,train);
            x = self.trf_blocks.forward_t(&x, train);
            x = self.final_norm.forward(&x);

            let logits = self.out_head.forward(&x);

            return logits;
    }
}

pub fn generate_text_simple(model: GPTModel, idx: &Tensor, max_new_tokens: i64, context_size: i64, train: bool) -> Tensor {
   
    let mut idx = idx.unsqueeze(0);

    for _ in 0..max_new_tokens {
        //grab the last of each (since we do this for next-word prediction)
        let idx_cond = idx.slice(1, idx.size()[1] - context_size, idx.size()[1], 1);
        //FnOnce implementation for no_grad, so expects || (no args) and returns T (ambiguous type)
        let mut logits = no_grad(|| {model.forward_t(&idx_cond, train)});
        //Focuses only on the last time step so that (batch, n_token, vocab_size) becomes (batch, vocab_size)
        //choose dim 1 since n_token = dim 1. We get the last token and then remove the n_token dimension
        logits = logits.slice(1, logits.size()[1] - 1, logits.size()[1], 1).squeeze_dim(1);

        let probas = Tensor::softmax(&logits, -1, Kind::Float);

        let idx_next = Tensor::argmax(&probas, -1, true);

        idx = Tensor::cat(&[idx, idx_next], 1);
    }
    return idx;
}