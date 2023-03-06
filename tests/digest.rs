use hex_literal::hex;
use sha2::{Sha512, Digest, digest::{generic_array::GenericArray, typenum::{U64, U8, U2}}};
use sha512_halo2::sha512::{BlockWord, Sha512 as OtherSha512, Table16Chip, Table16Config,IV,BLOCK_SIZE};
use halo2_proofs::{
    circuit::{AssignedCell, Chip, Layouter, Region, Value, SimpleFloorPlanner},
    halo2curves::bn256,
    plonk::{Advice, Any, Assigned, Column, ConstraintSystem, Error, Circuit}, dev::MockProver,
};
use halo2_proofs::arithmetic::FieldExt;

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

            // Test vector: "abc"
            let test_input = [
                BlockWord(Value::known(0b0110000101100010011000111000000000000000000000000000000000000000)),
                BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
                BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
                BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
                BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
                BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
                BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
                BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
                BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
                BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
                BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
                BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
                BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
                BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
                BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
                BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000011000)),
            ];

            let mut input = test_input;
            // let mut input = Vec::with_capacity(63);
            // for _ in 0..63 {
            //     input.extend_from_slice(&test_input);
            // }
    
            let expected_digest = Sha512::digest(b"abc");
            let digest = OtherSha512::digest(table16_chip, layouter.namespace(|| "'abc' * 2"), &input)?;
            println!("{:?}",expected_digest);
            println!("{:?}",digest.0);

            let mut s: Vec<u64> = Vec::new();
            for i in 0..8 {
                let temp = &expected_digest[8*i..8*i+8];
                let mut string = String::from("0b");
                for num in temp {
                    string.push_str(&format!("{:08b}", num));
                }
                s.push(u64::from_str_radix(&string[2..], 2).unwrap());
            }
            println!("{:?}", s);
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




//for num in expected_digest[8*i: 8*i + 8]:
// string = ''
// string.concat(num_to_bin)



