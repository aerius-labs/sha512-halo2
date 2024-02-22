use std::ops::Add;

use halo2_proofs::{
    circuit::{ Layouter, SimpleFloorPlanner, Value },
    dev::MockProver,
    halo2curves::bn256,
    plonk::{ Circuit, ConstraintSystem, Error },
};
use sha2::{ Digest, Sha512 };
use sha512_halo2::sha512::{
    BlockWord,
    Sha512 as OtherSha512,
    Table16Chip,
    Table16Config,
    BLOCK_SIZE,
    IV,
};


pub fn get_hex(data:&[u64]) -> String{
	let mut output:String = String::with_capacity(data.len()*8*2+1);
	for i in 0..data.len() {
		let next_hex = format!("{:016x}", data[i]);
		output.push_str(&next_hex);
	}
	return output;
}


fn preprocess_message(message: &str) -> Vec<Vec<u8>> {
    let mut bits = translate(message);
    let msg_len = bits.len();
    println!("msg lenght={}",msg_len);
    let message_len = format!("{:0128b}", msg_len)
        .chars()
        .map(|c| c.to_digit(10).unwrap() as u8)
        .collect::<Vec<_>>();
    let k = (896 - ((msg_len + 1) % 1024)) % 1024;
    bits.push(1);
    for _ in 0..k {
        bits.push(0);
    }
    println!("bits length after padding ={:?}", bits.len());
    bits.extend_from_slice(&message_len);
    println!("bits length after adding message length ={:?}", bits.len());
    assert_eq!(bits.len() % 1024, 0);

    chunker(&bits, 1024)
}

fn translate(message: &str) -> Vec<u8> {
    // string characters to unicode values
    let charcodes = message.chars().map(|c| c as u32);
    // unicode values to 8-bit strings (removed binary indicator)
    let bytes = charcodes.map(|char| format!("{:08b}", char)).collect::<Vec<_>>();
    // 8-bit strings to list of bits as integers
    let mut bits = Vec::new();
    for byte in bytes {
        for bit in byte.chars() {
            bits.push(bit.to_digit(10).unwrap() as u8);
        }
    }
    bits
}

fn chunker(bits: &[u8], chunk_length: usize) -> Vec<Vec<u8>> {
    bits.chunks(chunk_length)
        .map(|chunk| chunk.to_vec())
        .collect()
}

#[test]
fn sha512_test() {
    #[derive(Default)]
    struct MyCircuit {}

    impl Circuit<bn256::Fr> for MyCircuit {
        type Config = Table16Config;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self::default()
        }

        fn configure(meta: &mut ConstraintSystem<bn256::Fr>) -> Self::Config {
            Table16Chip::configure(meta)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<bn256::Fr>
        ) -> Result<(), Error> {
            Table16Chip::load(config.clone(), &mut layouter)?;
            let table16_chip = Table16Chip::construct(config);

            // Test vector: "12"
            let str =
                "0123456789ABCDEF0123456789ABCCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF";
            
      //      let str = "abc";
            let result = preprocess_message(&str);
      
            let mut res_new: Vec<Vec<Vec<u8>>> = Vec::new();
            for y in result {
                let chnk = y
                    .chunks(64)
                    .map(|chunk| chunk.to_vec())
                    .collect::<Vec<_>>();
                res_new.push(chnk);
            }
    
            let mut str_vec: Vec<BlockWord> = Vec::new();
            for y in res_new {
                for x in y {
                    let bin_str = x
                        .iter()
                        .map(|&b| b.to_string())
                        .collect::<String>();
                    let bin_num = u64::from_str_radix(&bin_str, 2).unwrap();
                   
                   // println!("num ={:?}", bin_num);
                    str_vec.push(BlockWord(Value::known(bin_num)));
                }
            }
            
            
            let expected_digest = Sha512::digest(&str);       
            let digest = OtherSha512::digest(
                table16_chip,
                layouter.namespace(|| "'abc' * 2"),
                &str_vec
            )?;
    
            let mut s: Vec<u64> = Vec::new();
            for i in 0..8 {
                let temp = &expected_digest[8 * i..8 * i + 8];
                let mut string = String::from("0b");
                for num in temp {
                    string.push_str(&format!("{:08b}", num));
                }
                s.push(u64::from_str_radix(&string[2..], 2).unwrap());
            }
    
            for (idx, digest_word) in digest.0.iter().enumerate() {
                let x = digest_word.0;
                let y = x.add(Value::known(IV[idx]));
                let s = Value::known(s[idx]);
            }
          
            Ok(())
        }
    }
    let circuit: MyCircuit = MyCircuit {};
    let prover = match MockProver::<bn256::Fr>::run(19, &circuit, vec![]) {
        Ok(prover) => prover,
        Err(e) => panic!("{:?}", e),
    };
    prover.assert_satisfied();
}
#[test]
fn test_preproces_msg() {
    let _ = preprocess_message(
        &"0123456789ABCDEF0123456789ABCCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF".to_string()
    );
}
