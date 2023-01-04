use super::super::{util::*, AssignedBits, Bits, SpreadVar, SpreadWord, Table64Assignment};
use super::{schedule_util::*, MessageScheduleConfig, MessageWord};
use ff::PrimeField;
use halo2_proofs::{
    circuit::{Region, Value},
    pasta::pallas,
    plonk::Error,
};
use std::convert::TryInto;

// A word in subregion 3
// (6, 13, 42, 3)-bit chunks
pub struct Subregion3Word {
    index: usize,
    #[allow(dead_code)]
    a: AssignedBits<6>,
    b: AssignedBits<13>,
    c: AssignedBits<42>,
    #[allow(dead_code)]
    d: AssignedBits<3>,
    spread_b: AssignedBits<26>,
    spread_c: AssignedBits<84>,
}

impl Subregion3Word {
    fn spread_a(&self) -> Value<[bool; 12]> {
        self.a.value().map(|v| v.spread())
    }

    fn spread_b(&self) -> Value<[bool; 26]> {
        self.spread_b.value().map(|v| v.0)
    }

    fn spread_c(&self) -> Value<[bool; 84]> {
        self.spread_c.value().map(|v| v.0)
    }

    fn spread_d(&self) -> Value<[bool; 26]> {
        self.d.value().map(|v| v.spread())
    }

    fn xor_lower_sigma_1(&self) -> Value<[bool; 128]> {
        self.spread_a()
            .zip(self.spread_b())
            .zip(self.spread_c())
            .zip(self.spread_d())
            .map(|(((a, b), c), d)| {
                let xor_0 = b
                    .iter()
                    .chain(c.iter())
                    .chain(d.iter())
                    .chain(std::iter::repeat(&false).take(12))
                    .copied()
                    .collect::<Vec<_>>();

                let xor_1 = c
                    .iter()
                    .chain(d.iter())
                    .chain(a.iter())
                    .chain(b.iter())
                    .copied()
                    .collect::<Vec<_>>();
                let xor_2 = d
                    .iter()
                    .chain(a.iter())
                    .chain(b.iter())
                    .chain(c.iter())
                    .copied()
                    .collect::<Vec<_>>();

                let xor_0 = lebs2ip::<128>(&xor_0.try_into().unwrap());
                let xor_1 = lebs2ip::<128>(&xor_1.try_into().unwrap());
                let xor_2 = lebs2ip::<128>(&xor_2.try_into().unwrap());

                i2lebsp(xor_0 + xor_1 + xor_2)
            })
    }
}

impl MessageScheduleConfig {
    // W_[65..78]
    pub fn assign_subregion3(
        &self,
        region: &mut Region<'_, pallas::Base>,
        lower_sigma_0_v2_output: Vec<(AssignedBits<32>, AssignedBits<32>)>,
        w: &mut Vec<MessageWord>,
        w_halves: &mut Vec<(AssignedBits<32>, AssignedBits<32>)>,
    ) -> Result<(), Error> {
        let a_5 = self.message_schedule;
        let a_6 = self.extras[2];
        let a_7 = self.extras[3];
        let a_8 = self.extras[4];
        let a_9 = self.extras[5];

        // Closure to compose new word
        // W_i = sigma_1(W_{i - 2}) + W_{i - 7} + sigma_0(W_{i - 15}) + W_{i - 16}
        // e.g. W_51 = sigma_1(W_49) + W_44 + sigma_0(W_36) + W_35

        // sigma_0_v2(W_[52..65]) will be used to get the new W_[67..80]
        // sigma_1(W_[65..78]) will also be used to get the W_[67..80]
        // The lowest-index words involved will be W_[51..74]
        let mut new_word = |idx: usize| -> Result<(), Error> {
            // Decompose word into (6, 13, 42, 3)-bit chunks
            let subregion3_word = self.decompose_subregion3_word(region, w[idx].value(), idx)?;

            // sigma_1 on subregion3_word
            let (r_0_even, r_1_even) = self.lower_sigma_1(region, subregion3_word)?;

            let new_word_idx = idx + 2;

            // Copy sigma_0_v2(W_{i - 15}) output from Subregion 2
            lower_sigma_0_v2_output[idx - 65].0.copy_advice(
                || format!("sigma_0(W_{})_lo", new_word_idx - 15),
                region,
                a_6,
                get_word_row(new_word_idx - 16),
            )?;
            lower_sigma_0_v2_output[idx - 65].1.copy_advice(
                || format!("sigma_0(W_{})_hi", new_word_idx - 15),
                region,
                a_6,
                get_word_row(new_word_idx - 16) + 1,
            )?;

            // Copy sigma_1(W_{i - 2})
            r_0_even.copy_advice(
                || format!("sigma_1(W_{})_lo", new_word_idx - 2),
                region,
                a_7,
                get_word_row(new_word_idx - 16),
            )?;
            r_1_even.copy_advice(
                || format!("sigma_1(W_{})_hi", new_word_idx - 2),
                region,
                a_7,
                get_word_row(new_word_idx - 16) + 1,
            )?;

            // Copy W_{i - 7}
            w_halves[new_word_idx - 7].0.copy_advice(
                || format!("W_{}_lo", new_word_idx - 7),
                region,
                a_8,
                get_word_row(new_word_idx - 16),
            )?;
            w_halves[new_word_idx - 7].1.copy_advice(
                || format!("W_{}_hi", new_word_idx - 7),
                region,
                a_8,
                get_word_row(new_word_idx - 16) + 1,
            )?;

            // Calculate W_i, carry_i
            let (word, carry) = sum_with_carry(vec![
                (r_0_even.value_u32(), r_1_even.value_u32()),
                (
                    w_halves[new_word_idx - 7].0.value_u32(),
                    w_halves[new_word_idx - 7].1.value_u32(),
                ),
                (
                    lower_sigma_0_v2_output[idx - 49].0.value_u32(),
                    lower_sigma_0_v2_output[idx - 49].1.value_u32(),
                ),
                (
                    w_halves[new_word_idx - 16].0.value_u32(),
                    w_halves[new_word_idx - 16].1.value_u32(),
                ),
            ]);

            // Assign W_i, carry_i
            region.assign_advice(
                || format!("W_{}", new_word_idx),
                a_5,
                get_word_row(new_word_idx - 16) + 1,
                || word.map(|word| pallas::Base::from_u128(word as u128)),
            )?;
            region.assign_advice(
                || format!("carry_{}", new_word_idx),
                a_9,
                get_word_row(new_word_idx - 16) + 1,
                || carry.map(|carry| pallas::Base::from_u128(carry as u128)),
            )?;
            let (word, halves) = self.assign_word_and_halves(region, word, new_word_idx)?;
            w.push(MessageWord(word));
            w_halves.push(halves);

            Ok(())
        };

        for i in 65..78 {
            new_word(i)?;
        }

        Ok(())
    }

    /// Pieces of length [6, 13, 42, 3]
    fn decompose_subregion3_word(
        &self,
        region: &mut Region<'_, pallas::Base>,
        word: Value<&Bits<64>>,
        index: usize,
    ) -> Result<Subregion3Word, Error> {
        let row = get_word_row(index);

        // Rename these here for ease of matching the gates to the specification.
        let a_3 = self.extras[0];
        let a_4 = self.extras[1];

        let pieces = word.map(|word| {
            vec![
                word[0..6].to_vec(),
                word[6..19].to_vec(),
                word[19..61].to_vec(),
                word[61..64].to_vec(),
            ]
        });
        let pieces = pieces.transpose_vec(4);

        // Assign `a` (6-bit piece)
        let a = AssignedBits::<6>::assign_bits(region, || "a", a_4, row + 1, pieces[0].clone())?;

        // Assign `b` (13-bit piece)
        let spread_b = pieces[1].clone().map(SpreadWord::try_new);
        let spread_b = SpreadVar::with_lookup(region, &self.lookup, row + 1, spread_b)?;

        // Assign `c` (42-bit piece)
        let spread_c = pieces[2].clone().map(SpreadWord::try_new);
        let spread_c = SpreadVar::with_lookup(region, &self.lookup, row, spread_c)?;

        // Assign `d` (3-bit piece) lookup
        let d = AssignedBits::<3>::assign_bits(region, || "d", a_3, row + 1, pieces[3].clone())?;



        Ok(Subregion3Word {
            index,
            a,
            b: spread_b.dense,
            c: spread_c.dense,
            d,
            spread_b: spread_b.spread,
            spread_c: spread_c.spread,
        })
    }

    fn lower_sigma_1(
        &self,
        region: &mut Region<'_, pallas::Base>,
        word: Subregion3Word,
    ) -> Result<(AssignedBits<32>, AssignedBits<32>), Error> {
        let a_3 = self.extras[0];
        let a_4 = self.extras[1];
        let a_5 = self.message_schedule;
        let a_6 = self.extras[2];

        let row = get_word_row(word.index) + 3;

        // Split `a` (6-bit chunk) into (3, 3)-bit `a_lo`, `a_hi`.
        // Assign `a_lo`, `spread_a_lo`, `a_hi`, `spread_a_hi`.

        // a_lo (3-bit chunk)
        {
            let a_lo: Value<[bool; 3]> = word.a.value().map(|v| v[0..3].try_into().unwrap());
            let a_lo = a_lo.map(SpreadWord::<3, 6>::new);
            SpreadVar::without_lookup(region, a_3, row - 1, a_4, row - 1, a_lo)?;
        }

        // a_hi (3-bit chunk)
        {
            let a_hi: Value<[bool; 3]> = word.a.value().map(|v| v[3..6].try_into().unwrap());
            let a_hi = a_hi.map(SpreadWord::<3, 6>::new);
            SpreadVar::without_lookup(region, a_5, row + 1, a_6, row + 1, a_hi)?;
        }

        // Assign `a` and copy constraint
        word.a.copy_advice(|| "a", region, a_4, row)?;

        // Assign `spread_b` and copy constraint
        word.spread_b.copy_advice(|| "spread_b", region, a_6, row)?;

        // Assign `spread_c` and copy constraint
        word.spread_c.copy_advice(|| "c", region, a_5, row)?;

        // Assign `d` and copy constraint
        word.d.copy_advice(|| "d", region, a_3, row + 1)?;

         // Witness `spread_d`
         {
            let spread_d = word.d.value().map(spread_bits);
            AssignedBits::<6>::assign_bits(region, || "spread_d", a_4, row + 1, spread_d)?;
        }

        // (6, 13, 42, 3)
        // Calculate R_0^{even}, R_0^{odd}, R_1^{even}, R_1^{odd}
        let r = word.xor_lower_sigma_1();
        let r_0: Value<[bool; 64]> = r.map(|r| r[..64].try_into().unwrap());
        let r_0_even = r_0.map(even_bits);
        let r_0_odd = r_0.map(odd_bits);

        let r_1: Value<[bool; 64]> = r.map(|r| r[64..].try_into().unwrap());
        let r_1_even = r_1.map(even_bits);
        let r_1_odd = r_1.map(odd_bits);

        self.assign_sigma_outputs(
            region,
            &self.lookup,
            a_3,
            row,
            r_0_even,
            r_0_odd,
            r_1_even,
            r_1_odd,
        )
    }
}
