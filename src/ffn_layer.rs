//This implements Layer Normalization, Feed Forward networks, and GELU
use tch::{nn::{self, Module, Path, Sequential}, Device,Kind, Tensor};
use tch::nn::{Init, LinearConfig, Linear, linear};
use tch::nn::init::{NormalOrUniform, FanInOut, NonLinearity, DEFAULT_KAIMING_UNIFORM};
use crate::architecture::CONFIG_124M;

#[derive(Debug)]
pub struct LayerNorm{
   pub eps: f64,
   pub scale: Tensor,
   pub shift: Tensor
}

impl LayerNorm{
   pub fn init(emb_dim: i64, root: &nn::Path) -> Self{
     //parameter type storing -> docs: https://docs.rs/tch/latest/tch/nn/struct.Path.html#method.ones
     let scale = root.ones("scale", &[emb_dim]);
     let shift = root.zeros("shift", &[emb_dim]);

     Self { eps: 1e-5, scale, shift }
   }

   pub fn forward(&self, x: &Tensor) -> Tensor {
      let mean = x.mean_dim(-1,true, Kind::Float);
      let var = x.var_dim(-1,false,true);
      let norm_x = (x-&mean) / Tensor::sqrt(&(var + self.eps));

      //to borrow, & the self references here, otherwise shifting
      let return_val = &self.scale * norm_x + &self.shift;

      return return_val;

   }
}
#[derive(Debug)]
 pub struct FeedForward{
   pub layers: Sequential
}
impl FeedForward{
   pub fn init(cfg: &CONFIG_124M, root: &nn::Path) -> Self {
      let lin_config = LinearConfig {
            ws_init: Init::Kaiming {
                dist: NormalOrUniform::Uniform,
                fan: FanInOut::FanIn,
                non_linearity: NonLinearity::ReLU,
            },
            bs_init: None,
            bias: cfg.qkv_bias
      };
      let linear_lyr_1 = linear(root, cfg.emb_dim, 4 * cfg.emb_dim,lin_config);
      let linear_lyr_2 = linear(root, 4*cfg.emb_dim, cfg.emb_dim, lin_config);
      //initialized sequential layer
      let mut layers = nn::seq();
      layers = layers
         .add(linear_lyr_1)
         .add_fn(|x| x.gelu("tanh"))
         .add(linear_lyr_2);
      Self {
         layers
      }
   }
   pub fn forward(&self, x:&Tensor) -> Tensor{
      return self.layers.forward(&x);
   }
}