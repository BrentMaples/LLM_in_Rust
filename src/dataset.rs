//bpe implementation
use bpe_tokenizer::BytePairEncoder;

//at the moment, implementing GPTDataset from my notes
pub struct MyDataset{
    pub input_ids: Vec<i32>,
    pub target_ids: Vec<i32>,
    pub text: String,
    pub max_len: i32,
    pub stride: i32,
    // implement token_ids using BPE encoding here or pass tokenizer
    // and call tokenizer here using its method
    pub tokenizer: BytePairEncoder
}

//no init in rust
//so we just build a function that does what we want inside of the implementations

impl MyDataset {

    //defined self for no return
    pub fn init (txt: String, tokenizer: BytePairEncoder, max_len: i32, 
                stride: i32) -> Self{
        let mut input_ids = Vec::new();
        let mut target_ids = Vec::new();
        let token_ids = tokenizer::new_from_str(txt).unwrap();
        //this is where I would do the tokenizer stuff here
        let loop_range = tokenizer.len() - max_len;

        //then do the loop here
        for i in 0..loop_range.step_by(stride) {
            //now we need to call a BPE for this to work

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
    pub fn inp_len(&self) -> usize{
        return self.input_ids.len();
    }

    
    pub fn get_item(&self, index:usize){
        //must be a tuple for returning
        return (self.input_ids[index], self.target_ids[idx]);
    }


}