//bpe implementation
use bpe_tokenizer::BytePairEncoder;
use tch::Tensor;

//at the moment, implementing GPTDataset from my notes
pub struct MyDataset{
    pub input_ids: Vec<Tensor>,
    pub target_ids: Vec<Tensor>,
    pub text: String,
    pub max_len: i32,
    pub stride: i32,
    // implement token_ids using BPE encoding here or pass tokenizer
    // and call tokenizer here using its method
    pub tokenizer: BytePairEncoder
}

//no init in rust
//so we just build a function that does what we want inside of the implementations

//So we have built the GPT dataset equivalent in Rust
impl MyDataset {

    //defined self for no return
    // takes the string data from the file and we can stuff it into the BPE, may not need to pass tokenizer itself here
    pub fn init (txt: String, tokenizer: BytePairEncoder, max_len: i32, 
                stride: i32) -> Self{
        //pytorch wrapper tensors
        let mut input_ids: Vec<Tensor> = Vec::new();
        let mut target_ids: Vec<Tensor> = Vec::new();
        // Equivalent to: tokenizer = tiktoken.get_encoding("gpt2")
        let encoder = BytePairEncoder::new_default_small().unwrap(); // Or medium/large, depending on your needs

        // Equivalent to: token_ids = tokenizer.encode(txt)
        let tokenizer: Vec<String> = encoder.tokenize(txt);
        //this is where I would do the tokenizer stuff here
        let loop_range = tokenizer.len() - max_len;

        //then do the loop here
        for i in 0..loop_range.step_by(stride) {
            // now that we have tokenizer, we can do stuff with it
            let input_chunk = &tokenizer[i..i + max_len];
            let target_chunk = &tokenizer[i+1..i + max_len + 1];
            input_ids.push(Tensor::from_slice(input_chunk));
            target_ids.push(Tensor::from_slice(target_chunk));
        
        }
        //required to return the changes made in this init
        Self {
            input_ids,
            target_ids,
            text,
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

    
    pub fn get_item(&self, index:usize){
        //must be a tuple for returning
        return (self.input_ids[index], self.target_ids[idx]);
    }


}