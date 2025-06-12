//This implements Layer Normalization, Feed Forward networks, and GELU
use tch::{nn::{self, Module, Path, Sequential}, Device,Kind, Tensor};
use tch::nn::{Init, LinearConfig, Linear, linear};
use tch::nn::init::{NormalOrUniform, FanInOut, NonLinearity, DEFAULT_KAIMING_UNIFORM};
use crate::architecture::CONFIG_124M;

 /*
 class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x-mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
 */
//batch_example = torch.randn(2,5)
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
 /* class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0/torch.pi)) *
            (x + 0.044715 * torch.pow(x,3))
        ))
 */
//just pass .gelu("tanh") to do the same thing above
/* class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # First linear layer increase the embedding dimension by a factor of 4
        # While the last one decreases the embedding dimension by a factor of 4
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # input, expand to 4 * input
            GELU(), # do GELU
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]) # then input is the expanded, and reduce it back to original
        )
    def forward(self, x):
        return self.layers(x)
 */
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