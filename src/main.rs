use std::fs::File;
use std::io::prelude::*;

fn main() {
  
   let file_path = "./data/the_verdict.txt";
   let mut file = File::open(file_path).expect("File Path Invalid");
   let mut contents = String::new();
   file.read_to_string(&mut contents).expect("Contents Failed");

   return;

}
