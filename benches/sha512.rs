use ark_std::{end_timer, start_timer};
use halo2_proofs::circuit::{SimpleFloorPlanner, Layouter, Value};
use halo2_proofs::plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, ConstraintSystem, Error};
use halo2_proofs::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG, ParamsVerifierKZG};
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::{
    halo2curves::bn256::{Bn256, Fr, G1Affine},
    poly::commitment::ParamsProver,
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_xorshift::XorShiftRng;

use criterion::{criterion_group, criterion_main, Criterion};
use std::{
    fs::{create_dir_all, File},
    io::{prelude::*, BufReader},
    path::Path,
};

use sha512_halo2::sha512::{BlockWord, Sha512, Table16Chip, Table16Config, BLOCK_SIZE};

pub const SETUP_PREFIX: &str = "[Setup generation]";
pub const PROOFGEN_PREFIX: &str = "[Proof generation]";
pub const PROOFVER_PREFIX: &str = "[Proof verification]";

#[test]
fn bench() {
    use std::env::var;

    use halo2_proofs::arithmetic::Field;

    #[derive(Default)]
    struct MyCircuit {}

    impl Circuit<Fr> for MyCircuit {
        type Config = Table16Config;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self::default()
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            Table16Chip::configure(meta)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            Table16Chip::load(config.clone(), &mut layouter)?;
            let table16_chip = Table16Chip::construct(config);

            // Test vector: "abc"
            let test_input = 
                [
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

            // Create a message of length 31 blocks
            let mut input = Vec::with_capacity(31 * BLOCK_SIZE);
            for _ in 0..31 {
                input.extend_from_slice(&test_input);
            }

            Sha512::digest(table16_chip, layouter.namespace(|| "'abc' * 2"), &input)?;

            Ok(())
        }
    }
    let setup_prfx = SETUP_PREFIX;
    let proof_gen_prfx = PROOFGEN_PREFIX;
    let proof_ver_prfx = PROOFVER_PREFIX;
    let degree: u32 = var("DEGREE")
    .unwrap_or_else(|_| "19".to_string())
    .parse()
    .expect("Cannot parse DEGREE env var as u32");

    //Unique string used by bench results module for parsing the result
    const BENCHMARK_ID: &str = "SHA512 Circuit";

    // Initialize the polynomial commitment parameters
    let mut rng = XorShiftRng::from_seed([
        0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
        0xbc, 0xe5,
    ]);
    // Create the circuit
    let circuit: MyCircuit = MyCircuit {};


    // Bench setup generation
    let setup_message = format!("{} {} with degree = {}", BENCHMARK_ID, setup_prfx, degree);
    let start1 = start_timer!(|| setup_message);
    let general_params = ParamsKZG::<Bn256>::setup(degree as u32, &mut rng);
    let verifier_params: ParamsVerifierKZG<Bn256> = general_params.verifier_params().clone();
    end_timer!(start1);

    // Initialize the proving key
    let vk = keygen_vk(&general_params, &circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&general_params, vk, &circuit).expect("keygen_pk should not fail");
    // Create a proof
    let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);

    // Bench proof generation time
    let proof_message = format!(
        "{} {} with degree = {}",
        BENCHMARK_ID, proof_gen_prfx, degree
    );
    let start2 = start_timer!(|| proof_message);
    create_proof::<
        KZGCommitmentScheme<Bn256>,
        ProverSHPLONK<'_, Bn256>,
        Challenge255<G1Affine>,
        XorShiftRng,
        Blake2bWrite<Vec<u8>, G1Affine, Challenge255<G1Affine>>,
        MyCircuit,
    >(&general_params, &pk, &[circuit], &[], rng, &mut transcript)
    .expect("proof generation should not fail");
    let proof = transcript.finalize();
    end_timer!(start2);

    // Bench verification time
    let start3 = start_timer!(|| format!("{} {}", BENCHMARK_ID, proof_ver_prfx));
    let mut verifier_transcript = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof[..]);
    let strategy = SingleStrategy::new(&general_params);

    verify_proof::<
        KZGCommitmentScheme<Bn256>,
        VerifierSHPLONK<'_, Bn256>,
        Challenge255<G1Affine>,
        Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
        SingleStrategy<'_, Bn256>,
    >(
        &verifier_params,
        pk.get_vk(),
        strategy,
        &[],
        &mut verifier_transcript,
    )
    .expect("failed to verify bench circuit");
    end_timer!(start3);
}

 
