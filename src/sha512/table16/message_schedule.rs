use std::convert::TryInto;

use super::{super::BLOCK_SIZE, AssignedBits, BlockWord, SpreadInputs, Table16Assignment, ROUNDS};
use halo2_proofs::{
    circuit::Layouter,
    pasta::pallas,
    plonk::{Advice, Column, ConstraintSystem, Error, Selector},
    poly::Rotation,
};

mod schedule_gates;
mod schedule_util;
mod subregion1;
mod subregion2;
mod subregion3;

use schedule_gates::ScheduleGate;
use schedule_util::*;

#[cfg(test)]
pub use schedule_util::msg_schedule_test_input;

#[derive(Clone, Debug)]
pub(super) struct MessageWord(AssignedBits<64>);

impl std::ops::Deref for MessageWord {
    type Target = AssignedBits<64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Debug)]
pub(super) struct MessageScheduleConfig {
    lookup: SpreadInputs,
    message_schedule: Column<Advice>,
    extras: [Column<Advice>; 6],

    /// Construct a word using reduce_4.
    s_word: Selector,
    /// Decomposition gate for W_0, W_78, W_79.
    s_decompose_0: Selector,
    /// Decomposition gate for W_[1..14]
    s_decompose_1: Selector,
    /// Decomposition gate for W_[14..65]
    s_decompose_2: Selector,
    /// Decomposition gate for W_[65..77]
    s_decompose_3: Selector,
    /// sigma_0 gate for W_[1..14]
    s_lower_sigma_0: Selector,
    /// sigma_1 gate for W_[65..77]
    s_lower_sigma_1: Selector,
    /// sigma_0_v2 gate for W_[14..65]
    s_lower_sigma_0_v2: Selector,
    /// sigma_1_v2 gate for W_[14..65]
    s_lower_sigma_1_v2: Selector,
}

impl Table16Assignment for MessageScheduleConfig {}

impl MessageScheduleConfig {
    /// Configures the message schedule.
    ///
    /// `message_schedule` is the column into which the message schedule will be placed.
    /// The caller must create appropriate permutations in order to load schedule words
    /// into the compression rounds.
    ///
    /// `extras` contains columns that the message schedule will only use for internal
    /// gates, and will not place any constraints on (such as lookup constraints) outside
    /// itself.
    #[allow(clippy::many_single_char_names)]
    pub(super) fn configure(
        meta: &mut ConstraintSystem<pallas::Base>,
        lookup: SpreadInputs,
        message_schedule: Column<Advice>,
        extras: [Column<Advice>; 6],
    ) -> Self {
        // Create fixed columns for the selectors we will require.
        let s_word = meta.selector();
        let s_decompose_0 = meta.selector();
        let s_decompose_1 = meta.selector();
        let s_decompose_2 = meta.selector();
        let s_decompose_3 = meta.selector();
        let s_lower_sigma_0 = meta.selector();
        let s_lower_sigma_1 = meta.selector();
        let s_lower_sigma_0_v2 = meta.selector();
        let s_lower_sigma_1_v2 = meta.selector();

        // Rename these here for ease of matching the gates to the specification.
        let a_0 = lookup.tag;
        let a_1 = lookup.dense;
        let a_2 = lookup.spread;
        let a_3 = extras[0];
        let a_4 = extras[1];
        let a_5 = message_schedule;
        let a_6 = extras[2];
        let a_7 = extras[3];
        let a_8 = extras[4];
        let a_9 = extras[5];

        // s_word for W_[16..80]
        meta.create_gate("s_word for W_[16..80]", |meta| {
            let s_word = meta.query_selector(s_word);

            let sigma_0_lo = meta.query_advice(a_6, Rotation::prev());
            let sigma_0_hi = meta.query_advice(a_6, Rotation::cur());

            let sigma_1_lo = meta.query_advice(a_7, Rotation::prev());
            let sigma_1_hi = meta.query_advice(a_7, Rotation::cur());

            let w_minus_9_lo = meta.query_advice(a_8, Rotation::prev());
            let w_minus_9_hi = meta.query_advice(a_8, Rotation::cur());

            let w_minus_16_lo = meta.query_advice(a_3, Rotation::prev());
            let w_minus_16_hi = meta.query_advice(a_4, Rotation::prev());

            let word = meta.query_advice(a_5, Rotation::cur());
            let carry = meta.query_advice(a_9, Rotation::cur());

            ScheduleGate::s_word(
                s_word,
                sigma_0_lo,
                sigma_0_hi,
                sigma_1_lo,
                sigma_1_hi,
                w_minus_9_lo,
                w_minus_9_hi,
                w_minus_16_lo,
                w_minus_16_hi,
                word,
                carry,
            )
        });

        // s_decompose_0 for all words
        meta.create_gate("s_decompose_0", |meta| {
            let s_decompose_0 = meta.query_selector(s_decompose_0);
            let lo = meta.query_advice(a_3, Rotation::cur());
            let hi = meta.query_advice(a_4, Rotation::cur());
            let word = meta.query_advice(a_5, Rotation::cur());

            ScheduleGate::s_decompose_0(s_decompose_0, lo, hi, word)
        });

        // s_decompose_1 for W_[1..14]
        // (1, 6, 1, 56)-bit chunks
        meta.create_gate("s_decompose_1", |meta| {
            let s_decompose_1 = meta.query_selector(s_decompose_1);
            let a = meta.query_advice(a_3, Rotation::next()); // 1-bit chunk
            let b = meta.query_advice(a_4, Rotation::next()); // 6-bit chunk
            let c = meta.query_advice(a_3, Rotation(2)); // 1-bit chunk
            // let tag_c = meta.query_advice(a_0, Rotation::next());
            let d_lo_lo = meta.query_advice(a_1, Rotation::cur()); // 14-bit chunk
            let d_lo_hi = meta.query_advice(a_1, Rotation::next()); // 14-bit chunk
            let d_hi_lo = meta.query_advice(a_1, Rotation(2)); // 14-bit chunk
            let d_hi_hi = meta.query_advice(a_1, Rotation(3)); // 14-bit chunk
            let tag_d_lo_lo = meta.query_advice(a_0, Rotation::cur());
            let tag_d_lo_hi = meta.query_advice(a_0, Rotation::next());
            let tag_d_hi_lo = meta.query_advice(a_0, Rotation(2));
            let tag_d_hi_hi = meta.query_advice(a_0, Rotation(3));
            let word = meta.query_advice(a_5, Rotation::cur());

            ScheduleGate::s_decompose_1(s_decompose_1, a, b, c, d_lo_lo, d_lo_hi, d_hi_lo, d_hi_hi, tag_d_lo_lo, tag_d_lo_hi, tag_d_hi_lo, tag_d_hi_hi, word)
        });

        // s_decompose_2 for W_[14..65]
        // (1, 5, 1, 1, 11, 42, 3)-bit chunks
        meta.create_gate("s_decompose_2", |meta| {
            let s_decompose_2 = meta.query_selector(s_decompose_2);
            let a = meta.query_advice(a_3, Rotation::prev()); // 1-bit chunk
            let b = meta.query_advice(a_1, Rotation(4)); // 5-bit chunk
            let c = meta.query_advice(a_4, Rotation::prev()); // 1-bit chunk
            let d = meta.query_advice(a_4, Rotation::next()); // 1-bit chunk
            // let tag_d = meta.query_advice(a_0, Rotation::cur());
            let e = meta.query_advice(a_1, Rotation::prev()); // 11-bit chunk
            let tag_e = meta.query_advice(a_0, Rotation::prev());
            let f_lo_lo = meta.query_advice(a_1, Rotation::cur()); // 11-bit chunk
            let f_lo_hi = meta.query_advice(a_1, Rotation::next()); // 10-bit chunk
            let f_hi_lo = meta.query_advice(a_1, Rotation(2)); // 11-bit chunk
            let f_hi_hi = meta.query_advice(a_1, Rotation(3)); // 10-bit chunk
            let tag_f_lo_lo = meta.query_advice(a_0, Rotation::cur());
            let tag_f_lo_hi = meta.query_advice(a_0, Rotation::next());
            let tag_f_hi_lo = meta.query_advice(a_0, Rotation(2));
            let tag_f_hi_hi = meta.query_advice(a_0, Rotation(3));
            let g = meta.query_advice(a_3, Rotation::next()); // 3-bit chunk
            // let tag_g = meta.query_advice(a_0, Rotation::prev());
            let word = meta.query_advice(a_5, Rotation::cur());

            ScheduleGate::s_decompose_2(s_decompose_2, a, b, c, d, e, tag_e, f_lo_lo, f_lo_hi, f_hi_lo, f_hi_hi, tag_f_lo_lo, tag_f_lo_hi, tag_f_hi_lo, tag_f_hi_hi, g, word)
        });

        // s_decompose_3 for W_65 to W_77
        // (6, 13, 42, 3)-bit chunks
        meta.create_gate("s_decompose_3", |meta| {
            let s_decompose_3 = meta.query_selector(s_decompose_3);
            let a = meta.query_advice(a_4, Rotation::next()); // 6-bit chunk
            let b = meta.query_advice(a_1, Rotation::cur()); // 13-bit chunk
            let tag_b = meta.query_advice(a_0, Rotation::cur());
            let c_lo_lo = meta.query_advice(a_1, Rotation::next()); // 11-bit chunk
            let c_lo_hi = meta.query_advice(a_1, Rotation(2)); // 10-bit chunk
            let c_hi_lo = meta.query_advice(a_1, Rotation(3)); // 11-bit chunk
            let c_hi_hi = meta.query_advice(a_1, Rotation(4)); // 10-bit chunk
            let tag_c_lo_lo = meta.query_advice(a_0, Rotation::next());
            let tag_c_lo_hi = meta.query_advice(a_0, Rotation(2));
            let tag_c_hi_lo = meta.query_advice(a_0, Rotation(3));
            let tag_c_hi_hi = meta.query_advice(a_0, Rotation(4));
            let d = meta.query_advice(a_3, Rotation::next()); // 3-bit chunk
            let word = meta.query_advice(a_5, Rotation::cur());

            ScheduleGate::s_decompose_3(s_decompose_3, a, b, tag_b, c_lo_lo, c_lo_hi, c_hi_lo, c_hi_hi, tag_c_lo_lo, tag_c_lo_hi, tag_c_hi_lo, tag_c_hi_hi, d, word)
        });

        // sigma_0 v1 on W_[1..14]
        // (1, 6, 1, 56)-bit chunks
        meta.create_gate("sigma_0 v1", |meta| {
            ScheduleGate::s_lower_sigma_0(
                meta.query_selector(s_lower_sigma_0),
                meta.query_advice(a_2, Rotation::prev()), // spread_r0_even_lo
                meta.query_advice(a_2, Rotation::cur()), // spread_r0_even_hi
                meta.query_advice(a_2, Rotation::next()),  // spread_r0_odd_lo
                meta.query_advice(a_2, Rotation(2)),  // spread_r0_odd_hi
                meta.query_advice(a_2, Rotation(3)), // spread_r1_even_lo
                meta.query_advice(a_2, Rotation(4)), // spread_r1_even_hi
                meta.query_advice(a_2, Rotation(5)),  // spread_r1_odd_lo
                meta.query_advice(a_2, Rotation(6)),  // spread_r1_odd_hi
                // meta.query_advice(a_5, Rotation::next()), // a
                meta.query_advice(a_6, Rotation::next()), // spread_a
                meta.query_advice(a_6, Rotation::cur()),  // b
                meta.query_advice(a_3, Rotation::prev()), // b_lo
                meta.query_advice(a_4, Rotation::prev()), // spread_b_lo
                meta.query_advice(a_5, Rotation::prev()), // b_hi
                meta.query_advice(a_6, Rotation::prev()), // spread_b_hi
                meta.query_advice(a_4, Rotation::cur()),  // spread_c
                meta.query_advice(a_5, Rotation::next()),  // spread_d_lo_lo
                meta.query_advice(a_5, Rotation::cur()),  // spread_d_lo_hi
                meta.query_advice(a_4, Rotation::next()),  // spread_d_hi_lo
                meta.query_advice(a_3, Rotation::next()),  // spread_d_hi_hi
            )
        });

        // sigma_0 v2 on W_[14..65]
        // (1, 5, 1, 1, 11, 42, 3)-bit chunks
        meta.create_gate("sigma_0 v2", |meta| {
            ScheduleGate::s_lower_sigma_0_v2(
                meta.query_selector(s_lower_sigma_0_v2),
                meta.query_advice(a_2, Rotation::prev()), // spread_r0_even_lo
                meta.query_advice(a_2, Rotation::cur()), // spread_r0_even_hi
                meta.query_advice(a_2, Rotation::next()),  // spread_r0_odd_lo
                meta.query_advice(a_2, Rotation(2)),  // spread_r0_odd_hi
                meta.query_advice(a_2, Rotation(3)), // spread_r1_even_lo
                meta.query_advice(a_2, Rotation(4)), // spread_r1_even_hi
                meta.query_advice(a_2, Rotation(5)),  // spread_r1_odd_lo
                meta.query_advice(a_2, Rotation(6)),  // spread_r1_odd_hi
                // meta.query_advice(a_3, Rotation::next()), // a
                meta.query_advice(a_4, Rotation::next()), // spread_a
                meta.query_advice(a_6, Rotation::cur()),  // b
                meta.query_advice(a_3, Rotation::prev()), // b_lo
                meta.query_advice(a_4, Rotation::prev()), // spread_b_lo
                meta.query_advice(a_5, Rotation::prev()), // b_hi
                meta.query_advice(a_6, Rotation::prev()), // spread_b_hi
                // meta.query_advice(a_5, Rotation::next()), // c
                meta.query_advice(a_6, Rotation::next()), // spread_c
                meta.query_advice(a_4, Rotation::cur()),  // spread_d
                meta.query_advice(a_7, Rotation::cur()),  // spread_e
                meta.query_advice(a_7, Rotation::next()), // spread_f_lo_lo
                meta.query_advice(a_7, Rotation(2)), // spread_f_lo_hi
                meta.query_advice(a_4, Rotation(2)), // spread_f_hi_lo
                meta.query_advice(a_4, Rotation(3)), // spread_f_hi_hi
                meta.query_advice(a_5, Rotation::next()), // g
                meta.query_advice(a_5, Rotation::cur()),  // spread_g
            )
        });

        // sigma_1 v2 on W_14 to W_64
        // (1, 5, 1, 1, 11, 42, 3)-bit chunks
        meta.create_gate("sigma_1 v2", |meta| {
            ScheduleGate::s_lower_sigma_1_v2(
                meta.query_selector(s_lower_sigma_1_v2),
                meta.query_advice(a_2, Rotation::prev()), // spread_r0_even_lo
                meta.query_advice(a_2, Rotation::cur()), // spread_r0_even_hi
                meta.query_advice(a_2, Rotation::next()),  // spread_r0_odd_lo
                meta.query_advice(a_2, Rotation(2)),  // spread_r0_odd_hi
                meta.query_advice(a_2, Rotation(3)), // spread_r1_even_lo
                meta.query_advice(a_2, Rotation(4)), // spread_r1_even_hi
                meta.query_advice(a_2, Rotation(5)),  // spread_r1_odd_lo
                meta.query_advice(a_2, Rotation(6)),  // spread_r1_odd_hi
                // meta.query_advice(a_3, Rotation::next()), // a
                meta.query_advice(a_4, Rotation::next()), // spread_a
                meta.query_advice(a_6, Rotation::cur()),  // b
                meta.query_advice(a_3, Rotation::prev()), // b_lo
                meta.query_advice(a_4, Rotation::prev()), // spread_b_lo
                meta.query_advice(a_5, Rotation::prev()), // b_hi
                meta.query_advice(a_6, Rotation::prev()), // spread_b_hi
                // meta.query_advice(a_5, Rotation::next()), // c
                meta.query_advice(a_6, Rotation::next()), // spread_c
                meta.query_advice(a_4, Rotation::cur()),  // spread_d
                meta.query_advice(a_7, Rotation::cur()),  // spread_e
                meta.query_advice(a_7, Rotation::next()), // spread_f_lo_lo
                meta.query_advice(a_7, Rotation(2)), // spread_f_lo_hi
                meta.query_advice(a_4, Rotation(2)), // spread_f_hi_lo
                meta.query_advice(a_4, Rotation(3)), // spread_f_hi_hi
                meta.query_advice(a_5, Rotation::next()), // g
                meta.query_advice(a_5, Rotation::cur()),  // spread_g
            )
        });

        // sigma_1 v1 on W_49 to W_77
        // (6, 13, 42, 3)-bit chunks
        meta.create_gate("sigma_1 v1", |meta| {
            ScheduleGate::s_lower_sigma_1(
                meta.query_selector(s_lower_sigma_1),
                meta.query_advice(a_2, Rotation::prev()), // spread_r0_even_lo
                meta.query_advice(a_2, Rotation::cur()), // spread_r0_even_hi
                meta.query_advice(a_2, Rotation::next()),  // spread_r0_odd_lo
                meta.query_advice(a_2, Rotation(2)),  // spread_r0_odd_hi
                meta.query_advice(a_2, Rotation(3)), // spread_r1_even_lo
                meta.query_advice(a_2, Rotation(4)), // spread_r1_even_hi
                meta.query_advice(a_2, Rotation(5)),  // spread_r1_odd_lo
                meta.query_advice(a_2, Rotation(6)),  // spread_r1_odd_hi
                meta.query_advice(a_6, Rotation::cur()),  // a
                meta.query_advice(a_3, Rotation::prev()), // a_lo
                meta.query_advice(a_6, Rotation::prev()), // a_hi
                meta.query_advice(a_4, Rotation::prev()), // spread_a_lo
                meta.query_advice(a_6, Rotation::next()), // spread_a_hi
                meta.query_advice(a_5, Rotation::prev()), // spread_b
                // meta.query_advice(a_4, Rotation::cur()),  // spread_a
                // meta.query_advice(a_5, Rotation::prev()), // b_mid
                // meta.query_advice(a_6, Rotation::prev()), // spread_b_mid
                // meta.query_advice(a_3, Rotation::next()), // c
                meta.query_advice(a_5, Rotation::cur()), // spread_c_lo_lo
                meta.query_advice(a_4, Rotation::cur()), // spread_c_lo_hi
                meta.query_advice(a_4, Rotation(2)), // spread_c_hi_lo
                meta.query_advice(a_6, Rotation(2)), // spread_c_hi_hi
                meta.query_advice(a_3, Rotation::next()),  // d
                meta.query_advice(a_4, Rotation::next()), // spread_d
            )
        });

        MessageScheduleConfig {
            lookup,
            message_schedule,
            extras,
            s_word,
            s_decompose_0,
            s_decompose_1,
            s_decompose_2,
            s_decompose_3,
            s_lower_sigma_0,
            s_lower_sigma_1,
            s_lower_sigma_0_v2,
            s_lower_sigma_1_v2,
        }
    }

    #[allow(clippy::type_complexity)]
    pub(super) fn process(
        &self,
        layouter: &mut impl Layouter<pallas::Base>,
        input: [BlockWord; BLOCK_SIZE],
    ) -> Result<
        (
            [MessageWord; ROUNDS],
            [(AssignedBits<32>, AssignedBits<32>); ROUNDS],
        ),
        Error,
    > {
        let mut w = Vec::<MessageWord>::with_capacity(ROUNDS);
        let mut w_halves = Vec::<(AssignedBits<32>, AssignedBits<32>)>::with_capacity(ROUNDS);

        layouter.assign_region(
            || "process message block",
            |mut region| {
                w = Vec::<MessageWord>::with_capacity(ROUNDS);
                w_halves = Vec::<(AssignedBits<32>, AssignedBits<32>)>::with_capacity(ROUNDS);

                // Assign all fixed columns
                for index in 1..14 {
                    let row = get_word_row(index);
                    self.s_decompose_1.enable(&mut region, row)?;
                    self.s_lower_sigma_0.enable(&mut region, row + 6)?;
                }

                for index in 14..65 {
                    let row = get_word_row(index);
                    self.s_decompose_2.enable(&mut region, row)?;
                    self.s_lower_sigma_0_v2.enable(&mut region, row + 6)?;
                    self.s_lower_sigma_1_v2
                        .enable(&mut region, row + SIGMA_0_V2_ROWS + 6)?;

                    let new_word_idx = index + 2;
                    self.s_word
                        .enable(&mut region, get_word_row(new_word_idx - 16) + 1)?;
                }

                for index in 65..78 {
                    let row = get_word_row(index);
                    self.s_decompose_3.enable(&mut region, row)?;
                    self.s_lower_sigma_1.enable(&mut region, row + 6)?;

                    let new_word_idx = index + 2;
                    self.s_word
                        .enable(&mut region, get_word_row(new_word_idx - 16) + 1)?;
                }

                for index in 0..80 {
                    let row = get_word_row(index);
                    self.s_decompose_0.enable(&mut region, row)?;
                }

                // Assign W[0..16]
                for (i, word) in input.iter().enumerate() {
                    let (word, halves) = self.assign_word_and_halves(&mut region, word.0, i)?;
                    w.push(MessageWord(word));
                    w_halves.push(halves);
                }

                // Returns the output of sigma_0 on W_[1..14]
                let lower_sigma_0_output = self.assign_subregion1(&mut region, &input[1..14])?;


                // sigma_0_v2 and sigma_1_v2 on W_[14..65]
                // Returns the output of sigma_0_v2 on W_[36..49], to be used in subregion3 //this could be wrong, but we are not sure
                let lower_sigma_0_v2_output = self.assign_subregion2(
                    &mut region,
                    lower_sigma_0_output,
                    &mut w,
                    &mut w_halves,
                )?;

                // sigma_1 v1 on W[65..78]
                self.assign_subregion3(
                    &mut region,
                    lower_sigma_0_v2_output,
                    &mut w,
                    &mut w_halves,
                )?;

                Ok(())
            },
        )?;

        Ok((w.try_into().unwrap(), w_halves.try_into().unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        super::BLOCK_SIZE, util::lebs2ip, BlockWord, SpreadTableChip, Table16Chip, Table16Config,
    };
    use super::schedule_util::*;
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner},
        dev::MockProver,
        pasta::pallas,
        plonk::{Circuit, ConstraintSystem, Error},
    };

    #[test]
    fn message_schedule() {
        struct MyCircuit {}

        impl Circuit<pallas::Base> for MyCircuit {
            type Config = Table16Config;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                MyCircuit {}
            }

            fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
                Table16Chip::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<pallas::Base>,
            ) -> Result<(), Error> {
                // Load lookup table
                SpreadTableChip::load(config.lookup.clone(), &mut layouter)?;

                // Provide input
                // Test vector: "abc"
                let inputs: [BlockWord; BLOCK_SIZE] = msg_schedule_test_input();

                // Run message_scheduler to get W_[0..80]
                let (w, _) = config.message_schedule.process(&mut layouter, inputs)?;
                let mut ctr = 0;
                for (word, test_word) in w.iter().zip(MSG_SCHEDULE_TEST_OUTPUT.iter()) {
                    word.value().assert_if_known(|bits| {
                        let word: u64 = lebs2ip(bits) as u64;
                        println!("{},{},{}",word,test_word,ctr);
                        ctr = ctr + 1;
                        word == *test_word
                        
                    });
                }
                Ok(())
            }
        }

        let circuit: MyCircuit = MyCircuit {};
        let prover = match MockProver::<pallas::Base>::run(17, &circuit, vec![]) {
            Ok(prover) => prover,
            Err(e) => panic!("{:?}", e),
        };
        prover.assert_satisfied();
    }
}