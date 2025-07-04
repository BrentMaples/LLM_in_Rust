# LLM in Rust
I have been a longtime programmer of Python and find myself implementing many of my machine learning tasks in this language. Recently, I have found that Rust is a great companion to Python and may help improve my implementation and understanding of PyTorch and machine learning techiniques. The objective of this repository is to record what I have implemented in my LLM journey.

# What Have I Learned?
It's much harder than I originally thought to make an LLM in Rust. Python overly simplifies the techniques used to create a powerful model and provides far too much abstraction when implementing its methods. After doing this project in Rust, I have learned a lot more about the inner workings of the Pytorch library. I created my own dataloader, learned about collating, tokenization, and all the other messy processes that go into making LLMs. From an LLM perspective, this model was a failure. From a learning perspective, this project was a complete success.

# Libraries
Although there are many Rust libraries that can be used, I chose to use the rust bindings for the C++ API of PyTorch: https://github.com/LaurentMazare/tch-rs.
Additionally, I used BPE encoding taken from the GPT-2 model: https://github.com/zurawiki/tiktoken-rs.

# File Breakdown
1. ## main.rs
   1. File containing each of the model calls and my tests.
1. ## dataset.rs
   1. This represents the token embedding step of the LLM with a relative positional embeddings implementation. We receive our token IDs from this code.
1. ## mha.rs
   1. This file implements an efficient Multi Head Attention mechanism with parallelism and multiple Causal Attention heads.
1. ## ffn_layer.rs
   1. Implementations for this codes Feed Forward and Layer Normalization to be used in architecture.rs
1. ## architecture.rs
   1. The LLM backbone with model configuration, the transformer, and the model itself.
1. ## train.rs
   1. The training implementation for the model
   1. ### training_helpers
      1. These are all the files I used in the train.rs code to create and train my model.
1. ## dataloader.rs
   1. Since tch-rs does not have an official Dataloader implementation in Rust, I created my own. This doesn't match the full functionality of its Python counterpart, but rather does what I need it to do for the given. There is no sampler or batch sampling, and CUDA was ignored as I am just using CPU for this project.



