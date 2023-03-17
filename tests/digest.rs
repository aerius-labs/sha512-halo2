use hex_literal::hex;
use sha2::{Sha512, Digest, digest::{generic_array::GenericArray, typenum::{U64, U8, U2}}};
use sha512_halo2::sha512::{BlockWord, Sha512 as OtherSha512, Table16Chip, Table16Config,IV,BLOCK_SIZE};
use halo2_proofs::{
    circuit::{AssignedCell, Chip, Layouter, Region, Value, SimpleFloorPlanner},
    halo2curves::bn256,
    plonk::{Advice, Any, Assigned, Column, ConstraintSystem, Error, Circuit}, dev::MockProver,
};
use halo2_proofs::arithmetic::FieldExt;

fn preprocess_message(message: &str) -> Vec<Vec<u8>> {
    // translate message into bits
    let bits = translate(message);
    // message length
    let length = bits.len();
    // get length in bits of message (64 bit block)
    let message_len = format!("{:0128b}", length)
        .chars()
        .map(|c| c.to_digit(10).unwrap() as u8)
        .collect::<Vec<_>>();
    // if length smaller than 448 handle block individually otherwise
    // if exactly 448 then add single 1 and add up to 1024 and if longer than 448
    // create multiple of 512 - 64 bits for the length at the end of the message (big endian)
    if length < 896 {
        // append single 1
        let mut bits = bits;
        bits.push(1);
        // fill zeros little endian wise
        fill_zeros(&mut bits, 896, "LE");
        // add the 64 bits representing the length of the message
        bits.extend_from_slice(&message_len);
        // return as list
        vec![bits]
    } else if length == 896 {
        let mut bits = bits;
        bits.push(1);
        // moves to next message block - total length = 1024
        fill_zeros(&mut bits, 1024, "LE");
        // replace the last 64 bits of the multiple of 512 with the original message length
        let ln = bits.len();
        bits[ln - 128..].copy_from_slice(&message_len);
        // returns it in 512 bit chunks
        chunker(&bits, 1024)
    } else {
        let mut bits = bits;
        bits.push(1);
        // loop until multiple of 512 if message length exceeds 448 bits
        while bits.len() % 1024 != 0 {
            bits.push(0);
        }
        // replace the last 64 bits of the multiple of 512 with the original message length
        let ln = bits.len();
        bits[ln - 128..].copy_from_slice(&message_len);
        // returns it in 512 bit chunks
        chunker(&bits, 1024)
    }
}

fn translate(message: &str) -> Vec<u8> {
    // string characters to unicode values
    let charcodes = message.chars().map(|c| c as u32);
    // unicode values to 8-bit strings (removed binary indicator)
    let bytes = charcodes
        .map(|char| format!("{:08b}", char))
        .collect::<Vec<_>>();
    // 8-bit strings to list of bits as integers
    let mut bits = Vec::new();
    for byte in bytes {
        for bit in byte.chars() {
            bits.push(bit.to_digit(10).unwrap() as u8);
        }
    }
    bits
}

fn fill_zeros(bits: &mut Vec<u8>, length: usize, endian: &str) {
    let l = bits.len();
    if endian == "LE" {
        for _ in l..length {
            bits.push(0);
        }
    } else {
        while bits.len() < length {
            bits.insert(0, 0);
        }
    }
}

fn chunker(bits: &[u8], chunk_length: usize) -> Vec<Vec<u8>> {
    bits.chunks(chunk_length).map(|chunk| chunk.to_vec()).collect()
}



#[test]
fn sha512_test(){
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
            mut layouter: impl Layouter<bn256::Fr>,
        ) -> Result<(), Error> {
            Table16Chip::load(config.clone(), &mut layouter)?;
            let table16_chip = Table16Chip::construct(config);

            // Test vector: "12"
            let str = "12";
            let result = preprocess_message(&str.repeat(55));
            let mut res_new : Vec<Vec<Vec<u8>>> = Vec::new();
            for y in result{
                let chnk = y.chunks(64).map(|chunk| chunk.to_vec()).collect::<Vec<_>>();
                res_new.push(chnk);
            }
            let mut str_vec : Vec<BlockWord> = Vec::new();
            for y in res_new {
                for x in y {
                    let bin_str = x.iter().map(|&b| b.to_string()).collect::<String>();
                    let bin_num = u64::from_str_radix(&bin_str, 2).unwrap();
                    // let num = format!("0b{:064b}", bin_num);
                    str_vec.push(BlockWord(Value::known(bin_num)));
                }
            }   
            let expected_digest = Sha512::digest(&str.repeat(55));
            let digest = OtherSha512::digest(table16_chip, layouter.namespace(|| "'abc' * 2"), &str_vec)?;
            let mut s: Vec<u64> = Vec::new();
            for i in 0..8 {
                let temp = &expected_digest[8*i..8*i+8];
                let mut string = String::from("0b");
                for num in temp {
                    string.push_str(&format!("{:08b}", num));
                }
                s.push(u64::from_str_radix(&string[2..], 2).unwrap());
            }
            for (idx, digest_word) in digest.0.iter().enumerate() {
                digest_word.0.assert_if_known(|digest_word| {
                    (*digest_word as u128 + IV[idx] as u128) as u64
                        == s[idx] as u64
                    });
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



