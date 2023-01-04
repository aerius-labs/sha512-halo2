use super::super::{util::*, AssignedBits, Bits, SpreadVar, SpreadWord, Table64Assignment};
use super::{schedule_util::*, MessageScheduleConfig, MessageWord};
use ff::PrimeField;
use halo2_proofs::{
    circuit::{Region, Value},
    pasta::pallas,
    plonk::Error,
};
use std::convert::TryInto;

/// A word in subregion 2
/// (1, 5, 1, 1, 11, 42, 3)-bit chunks
#[derive(Clone, Debug)]
pub struct Subregion2Word {
    index: usize,
    a: AssignedBits<1>,
    b: AssignedBits<5>,
    c: AssignedBits<1>,
    d: AssignedBits<1>,
    _e: AssignedBits<11>,
    _f: AssignedBits<42>,
    g: AssignedBits<3>,
    spread_e: AssignedBits<22>,
    spread_f: AssignedBits<84>,
}

impl Subregion2Word {
    fn spread_a(&self) -> Value<[bool;2]> {
        self.a.value().map(|v| v.spread())
    }

    fn spread_b(&self) -> Value<[bool; 10]> {
        self.b.value().map(|v| v.spread())
    }

    fn spread_c(&self) -> Value<[bool; 2]> {
        self.c.value().map(|v| v.spread())
    }

    fn spread_d(&self) -> Value<[bool; 2]> {
        self.d.value().map(|v| v.spread())
    }

    fn spread_e(&self) -> Value<[bool; 22]> {
        self.spread_e.value().map(|v| v.0)
    }

    fn spread_f(&self) -> Value<[bool; 84]> {
        self.spread_f.value().map(|v| v.0)
    }

    fn spread_g(&self) -> Value<[bool; 6]> {
        self.g.value().map(|v| v.spread())
    }

    fn xor_sigma_0(&self) -> Value<[bool; 128]> {
        self.spread_a()
            .zip(self.spread_b())
            .zip(self.spread_c())
            .zip(self.spread_d())
            .zip(self.spread_e())
            .zip(self.spread_f())
            .zip(self.spread_g())
            .map(|((((((a, b), c), d), e), f), g)| {
                let xor_0 = d
                    .iter()
                    .chain(e.iter())
                    .chain(f.iter())
                    .chain(g.iter())
                    .chain(std::iter::repeat(&false).take(2))
                    .chain(std::iter::repeat(&false).take(10))
                    .chain(std::iter::repeat(&false).take(2))
                    .copied()
                    .collect::<Vec<_>>();

                let xor_1 = b
                    .iter()
                    .chain(c.iter())
                    .chain(d.iter())
                    .chain(e.iter())
                    .chain(f.iter())
                    .chain(g.iter())
                    .chain(a.iter())
                    .copied()
                    .collect::<Vec<_>>();

                let xor_2 = e
                    .iter()
                    .chain(f.iter())
                    .chain(g.iter())
                    .chain(a.iter())
                    .chain(b.iter())
                    .chain(c.iter())
                    .chain(d.iter())
                    .copied()
                    .collect::<Vec<_>>();

                let xor_0 = lebs2ip::<128>(&xor_0.try_into().unwrap());
                let xor_1 = lebs2ip::<128>(&xor_1.try_into().unwrap());
                let xor_2 = lebs2ip::<128>(&xor_2.try_into().unwrap());

                i2lebsp(xor_0 + xor_1 + xor_2)
            })
    }

    fn xor_sigma_1(&self) -> Value<[bool; 128]> {
        self.spread_a()
            .zip(self.spread_b())
            .zip(self.spread_c())
            .zip(self.spread_d())
            .zip(self.spread_e())
            .zip(self.spread_f())
            .zip(self.spread_g())
            .map(|((((((a, b), c), d), e), f), g)| {
                let xor_0 = c
                    .iter()
                    .chain(d.iter())
                    .chain(e.iter())
                    .chain(f.iter())
                    .chain(g.iter())
                    .chain(std::iter::repeat(&false).take(2))
                    .chain(std::iter::repeat(&false).take(10))
                    .copied()
                    .collect::<Vec<_>>();

                let xor_1 = f
                    .iter()
                    .chain(g.iter())
                    .chain(a.iter())
                    .chain(b.iter())
                    .chain(c.iter())
                    .chain(d.iter())
                    .chain(e.iter())
                    .copied()
                    .collect::<Vec<_>>();

                let xor_2 = g
                    .iter()
                    .chain(a.iter())
                    .chain(b.iter())
                    .chain(c.iter())
                    .chain(d.iter())
                    .chain(e.iter())
                    .chain(f.iter())
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
    // W_[14..65]
    pub fn assign_subregion2(
        &self,
        region: &mut Region<'_, pallas::Base>,
        lower_sigma_0_output: Vec<(AssignedBits<32>, AssignedBits<32>)>,
        w: &mut Vec<MessageWord>,
        w_halves: &mut Vec<(AssignedBits<32>, AssignedBits<32>)>,
    ) -> Result<Vec<(AssignedBits<32>, AssignedBits<32>)>, Error> {
        let a_5 = self.message_schedule;
        let a_6 = self.extras[2];
        let a_7 = self.extras[3];
        let a_8 = self.extras[4];
        let a_9 = self.extras[5];

        let mut lower_sigma_0_v2_results =
            Vec::<(AssignedBits<32>, AssignedBits<32>)>::with_capacity(SUBREGION_2_LEN);
        let mut lower_sigma_1_v2_results =
            Vec::<(AssignedBits<32>, AssignedBits<32>)>::with_capacity(SUBREGION_2_LEN);

        // Closure to compose new word
        // W_i = sigma_1(W_{i - 2}) + W_{i - 7} + sigma_0(W_{i - 15}) + W_{i - 16}
        // e.g. W_16 = sigma_1(W_14) + W_9 + sigma_0(W_1) + W_0

        // sigma_0(W_[1..14]) will be used to get the new W_[16..29]
        // sigma_0_v2(W_[14..52]) will be used to get the new W_[29..67]
        // sigma_1_v2(W_[14..65]) will be used to get the W_[16..67]
        // The lowest-index words involved will be W_[0..13]
        let mut new_word = |idx: usize,
                            sigma_0_output: &(AssignedBits<32>, AssignedBits<32>)|
         -> Result<Vec<(AssignedBits<32>, AssignedBits<32>)>, Error> {
            // Decompose word into (1, 5, 1, 1, 11, 42, 3)-bit chunks
            let word = self.decompose_word(region, w[idx].value(), idx)?;

            // sigma_0 v2 and sigma_1 v2 on word
            lower_sigma_0_v2_results.push(self.lower_sigma_0_v2(region, word.clone())?);
            lower_sigma_1_v2_results.push(self.lower_sigma_1_v2(region, word)?);

            let new_word_idx = idx + 2;

            // Copy sigma_0(W_{i - 15}) output from Subregion 1
            sigma_0_output.0.copy_advice(
                || format!("sigma_0(W_{})_lo", new_word_idx - 15),
                region,
                a_6,
                get_word_row(new_word_idx - 16),
            )?;
            sigma_0_output.1.copy_advice(
                || format!("sigma_0(W_{})_hi", new_word_idx - 15),
                region,
                a_6,
                get_word_row(new_word_idx - 16) + 1,
            )?;

            // Copy sigma_1(W_{i - 2})
            lower_sigma_1_v2_results[new_word_idx - 16].0.copy_advice(
                || format!("sigma_1(W_{})_lo", new_word_idx - 2),
                region,
                a_7,
                get_word_row(new_word_idx - 16),
            )?;
            lower_sigma_1_v2_results[new_word_idx - 16].1.copy_advice(
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
                (
                    lower_sigma_1_v2_results[new_word_idx - 16].0.value_u32(),
                    lower_sigma_1_v2_results[new_word_idx - 16].1.value_u32(),
                ),
                (
                    w_halves[new_word_idx - 7].0.value_u32(),
                    w_halves[new_word_idx - 7].1.value_u32(),
                ),
                (sigma_0_output.0.value_u32(), sigma_0_output.1.value_u32()),
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
                || carry.map(pallas::Base::from_u128),
            )?;
            let (word, halves) = self.assign_word_and_halves(region, word, new_word_idx)?;
            w.push(MessageWord(word));
            w_halves.push(halves);

            Ok(lower_sigma_0_v2_results.clone())
        };

        let mut tmp_lower_sigma_0_v2_results: Vec<(AssignedBits<32>, AssignedBits<32>)> =
            Vec::with_capacity(SUBREGION_2_LEN);

        // Use up all the output from Subregion 1 lower_sigma_0
        for i in 14..27 {
            tmp_lower_sigma_0_v2_results = new_word(i, &lower_sigma_0_output[i - 14])?;
        }

        for i in 27..65 {
            tmp_lower_sigma_0_v2_results =
                new_word(i, &tmp_lower_sigma_0_v2_results[i + 2 - 15 - 14])?;
        }

        // Return lower_sigma_0_v2 output for W_[52..65]
        Ok(lower_sigma_0_v2_results.split_off(52 - 14))
    }

    /// Pieces of length [1, 5, 1, 1, 11, 42, 3]
    fn decompose_word(
        &self,
        region: &mut Region<'_, pallas::Base>,
        word: Value<&Bits<64>>,
        index: usize,
    ) -> Result<Subregion2Word, Error> {
        let row = get_word_row(index);

        let pieces = word.map(|word| {
            vec![
                vec![word[0]],
                word[1..6].to_vec(),
                vec![word[6]],
                vec![word[7]],
                word[8..19].to_vec(),
                word[19..61].to_vec(),
                word[61..64].to_vec(),
            ]
        });
        let pieces = pieces.transpose_vec(7);

        // Rename these here for ease of matching the gates to the specification.
        let a_3 = self.extras[0];
        let a_4 = self.extras[1];

        // Assign `a` (1-bit piece)
        let a = AssignedBits::<1>::assign_bits(region, || "a", a_3, row - 1, pieces[0].clone())?;

        // Assign `b` (5-bit piece) lookup
        let spread_b: Value<SpreadWord<5, 10>> = pieces[1].clone().map(SpreadWord::try_new);
        let spread_b = SpreadVar::with_lookup(region, &self.lookup, row + 1, spread_b)?;

        // Assign `c` (1-bit piece)
        let c = AssignedBits::<1>::assign_bits(region, || "c", a_4, row - 1, pieces[2].clone())?;

        // Assign `d` (1-bit piece)
        let d = AssignedBits::<1>::assign_bits(region, || "d", a_3, row + 1, pieces[3].clone())?;

        // Assign `e` (11-bit piece)
        let spread_e = pieces[4].clone().map(SpreadWord::try_new);
        let spread_e = SpreadVar::with_lookup(region, &self.lookup, row, spread_e)?;

        // Assign `f` (42-bit piece)
        let spread_f = pieces[5].clone().map(SpreadWord::try_new);
        let spread_f = SpreadVar::with_lookup(region, &self.lookup, row - 1, spread_f)?;

        // Assign `g` (3-bit piece) lookup
        let g = AssignedBits::<3>::assign_bits(region, || "g", a_4, row + 1, pieces[6].clone())?;

        Ok(Subregion2Word {
            index,
            a,
            b: spread_b.dense,
            c,
            d,
            _e: spread_e.dense,
            _f: spread_f.dense,
            g,
            spread_e: spread_e.spread,
            spread_f: spread_f.spread,
        })
    }

    /// A word in subregion 2
    /// (1, 5, 1, 1, 11, 42, 3)-bit chunks
    #[allow(clippy::type_complexity)]
    fn assign_lower_sigma_v2_pieces(
        &self,
        region: &mut Region<'_, pallas::Base>,
        row: usize,
        word: &Subregion2Word,
    ) -> Result<(), Error> {
        let a_3 = self.extras[0];
        let a_4 = self.extras[1];
        let a_5 = self.message_schedule;
        let a_6 = self.extras[2];
        let a_7 = self.extras[3];

        // Assign `a` and copy constraint
        word.a.copy_advice(|| "a", region, a_3, row + 1)?;

        // Split `b` (5-bit chunk) into `b_hi` and `b_lo`
        // Assign `b_lo`, `spread_b_lo`

        let b_lo: Value<[bool; 3]> = word.b.value().map(|b| b.0[..3].try_into().unwrap());
        let spread_b_lo = b_lo.map(spread_bits);
        {
            AssignedBits::<3>::assign_bits(region, || "b_lo", a_3, row - 1, b_lo)?;

            AssignedBits::<6>::assign_bits(region, || "spread_b_lo", a_4, row - 1, spread_b_lo)?;
        };

        // Split `b` (2-bit chunk) into `b_hi` and `b_lo`
        // Assign `b_hi`, `spread_b_hi`
        let b_hi: Value<[bool; 2]> = word.b.value().map(|b| b.0[2..].try_into().unwrap());
        let spread_b_hi = b_hi.map(spread_bits);
        {
            AssignedBits::<2>::assign_bits(region, || "b_hi", a_5, row - 1, b_hi)?;

            AssignedBits::<4>::assign_bits(region, || "spread_b_hi", a_6, row - 1, spread_b_hi)?;
        };

        // Assign `b` and copy constraint
        word.b.copy_advice(|| "b", region, a_6, row)?;

        // Assign `c` and copy constraint
        word.c.copy_advice(|| "c", region, a_5, row)?;

        // Assign `d` and copy constraint
        word.d.copy_advice(|| "d", region, a_4, row)?;

        // Assign `e` and copy constraint
        word.spread_e.copy_advice(|| "spread_e", region, a_7, row)?;

        // Assign `f` and copy constraint
        word.spread_f.copy_advice(|| "spread_f", region, a_7, row + 1)?;

        // Assign `g` and copy constraint
        word.g.copy_advice(|| "g", region, a_5, row+1)?;

        // Witness `spread_g`
        AssignedBits::<6>::assign_bits(region, || "spread_g", a_6, row + 1, word.spread_g())?;

        Ok(())
    }

    fn lower_sigma_0_v2(
        &self,
        region: &mut Region<'_, pallas::Base>,
        word: Subregion2Word,
    ) -> Result<(AssignedBits<32>, AssignedBits<32>), Error> {
        let a_3 = self.extras[0];
        let row = get_word_row(word.index) + 3;

        // Assign lower sigma_v2 pieces
        self.assign_lower_sigma_v2_pieces(region, row, &word)?;

        // Calculate R_0^{even}, R_0^{odd}, R_1^{even}, R_1^{odd}
        let r = word.xor_sigma_0();
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

    fn lower_sigma_1_v2(
        &self,
        region: &mut Region<'_, pallas::Base>,
        word: Subregion2Word,
    ) -> Result<(AssignedBits<32>, AssignedBits<32>), Error> {
        let a_3 = self.extras[0];
        let row = get_word_row(word.index) + SIGMA_0_V2_ROWS + 3;

        // Assign lower sigma_v2 pieces
        self.assign_lower_sigma_v2_pieces(region, row, &word)?;

        // (1, 5, 1, 1, 11, 42, 3)
        // Calculate R_0^{even}, R_0^{odd}, R_1^{even}, R_1^{odd}
        // Calculate R_0^{even}, R_0^{odd}, R_1^{even}, R_1^{odd}
        let r = word.xor_sigma_1();
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
