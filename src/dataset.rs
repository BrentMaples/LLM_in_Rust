//bpe implementation
use tch::{Tensor, Kind};
//GPT2
use tiktoken_rs::r50k_base;
use tiktoken_rs::CoreBPE;

//at the moment, implementing GPTDataset from my notes
pub struct GPTDataset{
    pub input_ids: Vec<Tensor>,
    pub target_ids: Vec<Tensor>,
    pub txt: String,
    pub max_len: usize,
    pub stride: usize,
    // implement token_ids using BPE encoding here or pass tokenizer
    // and call tokenizer here using its method
    pub tokenizer: CoreBPE
}

//no init in rust
//so we just build a function that does what we want inside of the implementations

//So we have built the GPT dataset equivalent in Rust
impl GPTDataset {

    //defined self for no return
    // takes the string data from the file and we can stuff it into the BPE, may not need to pass tokenizer itself here
    pub fn init (txt: String, tokenizer: CoreBPE, max_len: usize, 
                stride: usize) -> Self{
        //pytorch wrapper tensors
        let mut input_ids: Vec<Tensor> = Vec::new();
        let mut target_ids: Vec<Tensor> = Vec::new();
        // Equivalent to: tokenizer = tiktoken.get_encoding("gpt2")        
        let tokenizer = r50k_base().unwrap();
        let token_ids: Vec<u32> = tokenizer.encode_with_special_tokens(&txt);
                
        //this is where I would do the tokenizer stuff here
        let loop_range = token_ids.len() - max_len;

        //then do the loop here
        for i in (0..loop_range).step_by(stride) {
            // now that we have tokenizer, we can do stuff with it
            let input_chunk = &token_ids[i..i + max_len];
            let target_chunk = &token_ids[i+1..i + max_len + 1];
            let input_tensor = Tensor::from_slice(
                &input_chunk.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            ).to_kind(Kind::Int64);
            
            let target_tensor = Tensor::from_slice(
                &target_chunk.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            ).to_kind(Kind::Int64);

        
        }
        //required to return the changes made in this init
        Self {
            input_ids,
            target_ids,
            txt,
            max_len,
            stride,
            tokenizer,
        }

    }
    //must ref self at all times
    // should still be right bc of it being a tensor
    pub fn inp_len(&self) -> usize{
        return self.input_ids.len();
    }

    
    pub fn get_item(&self, index:usize) -> (&Tensor, &Tensor){
        //must be a tuple for returning
        return (&self.input_ids[index], &self.target_ids[index]);
    }
}


//Now let us implement the dataloader