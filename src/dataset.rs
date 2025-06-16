/* This creates my tokenized text for me */
//bpe implementation
use tch::{Tensor, Kind};
//GPT2
use tiktoken_rs::{r50k_base, CoreBPE};



//at the moment, implementing GPTDataset from my notes -- equiv to pytorch dataset in rust
pub struct GPTDataset{
    pub input_ids: Vec<Tensor>,
    pub target_ids: Vec<Tensor>,
    pub txt: String,
    pub max_len: usize,
    pub stride: usize,
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
        let token_ids: Vec<u32> = tokenizer.encode_with_special_tokens(&txt);
        //println!("Token count: {}", token_ids.len());
    
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
            
            input_ids.push(input_tensor);
            target_ids.push(target_tensor);

        }
        //required to return the changes made in this init
        Self {
            input_ids,
            target_ids,
            txt,
            max_len,
            stride
        }

    }

    pub fn len(&self) -> usize {
            self.input_ids.len()
        }
    
    pub fn get_item(&self, index:usize) -> (&Tensor, &Tensor){
        //must be a tuple for returning
        return (&self.input_ids[index], &self.target_ids[index]);
    }
}
//this is equivalent to dataloader for one batch (iter)
pub fn batch_printing(batch_size: usize, dataset: GPTDataset) -> (Tensor, Tensor){
    let mut input_batch: Tensor = Tensor::new();
    let mut target_batch: Tensor = Tensor::new();
    for batch_start in (0..dataset.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(dataset.len());
        let inputs: Vec<_> = (batch_start..batch_end)
            .map(|i| dataset.input_ids[i].shallow_clone())
            .collect();
        let targets: Vec<_> = (batch_start..batch_end)
            .map(|i| dataset.target_ids[i].shallow_clone())
            .collect();

        input_batch = Tensor::stack(&inputs, 0);
        target_batch = Tensor::stack(&targets, 0);
        break; // only one batch for demonstration
    }   
    return (input_batch, target_batch);
}