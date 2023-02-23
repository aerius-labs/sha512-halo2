use super::super::{util::*, AssignedBits, Bits, SpreadVar, SpreadWord, Table16Assignment};
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
    _b: AssignedBits<13>,
    _c_lo_lo: AssignedBits<11>,
    _c_lo_hi: AssignedBits<10>,
    _c_hi_lo: AssignedBits<11>,
    _c_hi_hi: AssignedBits<10>,
    #[allow(dead_code)]
    d: AssignedBits<3>,
    spread_b: AssignedBits<26>,
    spread_c_lo_lo: AssignedBits<22>,
    spread_c_lo_hi: AssignedBits<20>,
    spread_c_hi_lo: AssignedBits<22>,
    spread_c_hi_hi: AssignedBits<20>,
}

impl Subregion3Word {
    fn spread_a(&self) -> Value<[bool; 12]> {
        self.a.value().map(|v| v.spread())
    }

    fn spread_b(&self) -> Value<[bool; 26]> {
        self.spread_b.value().map(|v| v.0)
    }

    fn spread_c_lo_lo(&self) -> Value<[bool; 22]> {
        self.spread_c_lo_lo.value().map(|v| v.0)
    }

    fn spread_c_lo_hi(&self) -> Value<[bool; 20]> {
        self.spread_c_lo_hi.value().map(|v| v.0)
    }

    fn spread_c_hi_lo(&self) -> Value<[bool; 22]> {
        self.spread_c_hi_lo.value().map(|v| v.0)
    }

    fn spread_c_hi_hi(&self) -> Value<[bool; 20]> {
        self.spread_c_hi_hi.value().map(|v| v.0)
    }

    fn spread_d(&self) -> Value<[bool; 6]> {
        self.d.value().map(|v| v.spread())
    }

    fn xor_lower_sigma_1(&self) -> Value<[bool; 128]> {
        self.spread_a()
            .zip(self.spread_b())
            .zip(self.spread_c_lo_lo())
            .zip(self.spread_c_lo_hi())
            .zip(self.spread_c_hi_lo())
            .zip(self.spread_c_hi_hi())
            .zip(self.spread_d())
            .map(|((((((a, b), c_lo_lo), c_lo_hi), c_hi_lo), c_hi_hi), d)| {
                let xor_0 = b
                    .iter()
                    .chain(c_lo_lo.iter())
                    .chain(c_lo_hi.iter())
                    .chain(c_hi_lo.iter())
                    .chain(c_hi_hi.iter())
                    .chain(d.iter())
                    .chain(std::iter::repeat(&false).take(12))
                    .copied()
                    .collect::<Vec<_>>();

                let xor_1 = c_lo_lo
                    .iter()
                    .chain(c_lo_hi.iter())
                    .chain(c_hi_lo.iter())
                    .chain(c_hi_hi.iter())
                    .chain(d.iter())
                    .chain(a.iter())
                    .chain(b.iter())
                    .copied()
                    .collect::<Vec<_>>();
                let xor_2 = d
                    .iter()
                    .chain(a.iter())
                    .chain(b.iter())
                    .chain(c_lo_lo.iter())
                    .chain(c_lo_hi.iter())
                    .chain(c_hi_lo.iter())
                    .chain(c_hi_hi.iter())
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

            // // Copy W_{i - 7}
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
                    lower_sigma_0_v2_output[idx - 65].0.value_u32(),
                    lower_sigma_0_v2_output[idx - 65].1.value_u32(),
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
                word[19..30].to_vec(),
                word[30..40].to_vec(),
                word[40..51].to_vec(),
                word[51..61].to_vec(),
                word[61..64].to_vec(),
            ]
        });
        let pieces = pieces.transpose_vec(7);

        // Assign `a` (6-bit piece)
        let a = AssignedBits::<6>::assign_bits(region, || "a", a_4, row + 1, pieces[0].clone())?;

        // Assign `b` (13-bit piece)
        let spread_b = pieces[1].clone().map(SpreadWord::try_new);
        let spread_b = SpreadVar::with_lookup(region, &self.lookup, row, spread_b)?;

        // Assign `c_lo_lo` (11-bit piece)
        let spread_c_lo_lo = pieces[2].clone().map(SpreadWord::try_new);
        let spread_c_lo_lo = SpreadVar::with_lookup(region, &self.lookup, row + 1, spread_c_lo_lo)?;

        // Assign `c_lo_hi` (10-bit piece)
        let spread_c_lo_hi = pieces[3].clone().map(SpreadWord::try_new);
        let spread_c_lo_hi = SpreadVar::with_lookup(region, &self.lookup, row + 2, spread_c_lo_hi)?;

        // Assign `c_hi_lo` (11-bit piece)
        let spread_c_hi_lo = pieces[4].clone().map(SpreadWord::try_new);
        let spread_c_hi_lo = SpreadVar::with_lookup(region, &self.lookup, row + 3, spread_c_hi_lo)?;

        // Assign `c_hi_hi` (10-bit piece)
        let spread_c_hi_hi = pieces[5].clone().map(SpreadWord::try_new);
        let spread_c_hi_hi = SpreadVar::with_lookup(region, &self.lookup, row + 4, spread_c_hi_hi)?;

        // Assign `d` (3-bit piece) lookup
        let d = AssignedBits::<3>::assign_bits(region, || "d", a_3, row + 1, pieces[6].clone())?;



        Ok(Subregion3Word {
            index,
            a,
            _b: spread_b.dense,
            _c_lo_lo: spread_c_lo_lo.dense,
            _c_lo_hi: spread_c_lo_hi.dense,
            _c_hi_lo: spread_c_hi_lo.dense,
            _c_hi_hi: spread_c_hi_hi.dense,
            d,
            spread_b: spread_b.spread,
            spread_c_lo_lo: spread_c_lo_lo.spread,
            spread_c_lo_hi: spread_c_lo_hi.spread,
            spread_c_hi_lo: spread_c_hi_lo.spread,
            spread_c_hi_hi: spread_c_hi_hi.spread,
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

        let row = get_word_row(word.index) + 6;

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
            SpreadVar::without_lookup(region, a_6, row - 1, a_6, row + 1, a_hi)?;
        }

        // Assign `a` and copy constraint
        word.a.copy_advice(|| "a", region, a_6, row)?;

        // Assign `spread_b` and copy constraint
        word.spread_b.copy_advice(|| "spread_b", region, a_5, row - 1)?;

        // Assign `spread_c_lo_lo` and copy constraint
        word.spread_c_lo_lo.copy_advice(|| "c_lo_lo", region, a_5, row)?;

        // Assign `spread_c_lo_hi` and copy constraint
        word.spread_c_lo_hi.copy_advice(|| "c_lo_hi", region, a_4, row)?;

        // Assign `spread_c_hi_lo` and copy constraint
        word.spread_c_hi_lo.copy_advice(|| "c_hi_lo", region, a_4, row + 2)?;

        // Assign `spread_c_hi_hi` and copy constraint
        word.spread_c_hi_hi.copy_advice(|| "c_hi_hi", region, a_6, row + 2)?;

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
