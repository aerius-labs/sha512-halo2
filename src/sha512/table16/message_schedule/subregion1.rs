use super::super::{util::*, AssignedBits, BlockWord, SpreadVar, SpreadWord, Table16Assignment};
use super::{schedule_util::*, MessageScheduleConfig};
use halo2_proofs::{
    circuit::{Region, Value},
    pasta::pallas,
    plonk::Error,
};
use std::convert::TryInto;

// A word in subregion 1
// (1, 6, 1, 56)-bit chunks
#[derive(Debug)]
pub struct Subregion1Word {
    index: usize,
    a: AssignedBits<1>,
    b: AssignedBits<6>,
    c: AssignedBits<1>,
    _d_lo_lo: AssignedBits<14>,
    _d_lo_hi: AssignedBits<14>,
    _d_hi_lo: AssignedBits<14>,
    _d_hi_hi: AssignedBits<14>,
    spread_d_lo_lo: AssignedBits<28>,
    spread_d_lo_hi: AssignedBits<28>,
    spread_d_hi_lo: AssignedBits<28>,
    spread_d_hi_hi: AssignedBits<28>,
}

impl Subregion1Word {
    fn spread_a(&self) -> Value<[bool; 2]> {
        self.a.value().map(|v| v.spread())
    }

    fn spread_b(&self) -> Value<[bool; 12]> {
        self.b.value().map(|v| v.spread())
    }

    fn spread_c(&self) -> Value<[bool; 2]> {
        self.c.value().map(|v| v.spread())
    }

    fn spread_d_lo_lo(&self) -> Value<[bool; 28]> {
        self.spread_d_lo_lo.value().map(|v| v.0)
    }

    fn spread_d_lo_hi(&self) -> Value<[bool; 28]> {
        self.spread_d_lo_hi.value().map(|v| v.0)
    }
    fn spread_d_hi_lo(&self) -> Value<[bool; 28]> {
        self.spread_d_hi_lo.value().map(|v| v.0)
    }

    fn spread_d_hi_hi(&self) -> Value<[bool; 28]> {
        self.spread_d_hi_hi.value().map(|v| v.0)
    }

    fn xor_lower_sigma_0(&self) -> Value<[bool; 128]> {
        self.spread_a()
            .zip(self.spread_b())
            .zip(self.spread_c())
            .zip(self.spread_d_lo_lo())
            .zip(self.spread_d_lo_hi())
            .zip(self.spread_d_hi_lo())
            .zip(self.spread_d_hi_hi())
            .map(|((((((a, b), c), d_lo_lo), d_lo_hi), d_hi_lo), d_hi_hi)| {
                let xor_0 = c
                    .iter()
                    .chain(d_lo_lo.iter())
                    .chain(d_lo_hi.iter())
                    .chain(d_hi_lo.iter())
                    .chain(d_hi_hi.iter())
                    .chain(std::iter::repeat(&false).take(2))
                    .chain(std::iter::repeat(&false).take(12))
                    .copied()
                    .collect::<Vec<_>>();
                let xor_1 = b
                    .iter()
                    .chain(c.iter())
                    .chain(d_lo_lo.iter())
                    .chain(d_lo_hi.iter())
                    .chain(d_hi_lo.iter())
                    .chain(d_hi_hi.iter())
                    .chain(a.iter())
                    .copied()
                    .collect::<Vec<_>>();
                let xor_2 = d_lo_lo
                    .iter()
                    .chain(d_lo_hi.iter())
                    .chain(d_hi_lo.iter())
                    .chain(d_hi_hi.iter())
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
    pub fn assign_subregion1(
        &self,
        region: &mut Region<'_, pallas::Base>,
        input: &[BlockWord],
    ) -> Result<Vec<(AssignedBits<32>, AssignedBits<32>)>, Error> {
        assert_eq!(input.len(), SUBREGION_1_LEN);
        Ok(input
            .iter()
            .enumerate()
            .map(|(idx, word)| {
                // s_decompose_1 on W_[1..14]
                let subregion1_word = self
                    .decompose_subregion1_word(
                        region,
                        word.0.map(|word| i2lebsp(word.into())),
                        idx + 1,
                    )
                    .unwrap();

                // lower_sigma_0 on W_[1..14]
                self.lower_sigma_0(region, subregion1_word).unwrap()
            })
            .collect::<Vec<_>>())
    }

    /// Pieces of length [1, 6, 1, 56]
    fn decompose_subregion1_word(
        &self,
        region: &mut Region<'_, pallas::Base>,
        word: Value<[bool; 64]>,
        index: usize,
    ) -> Result<Subregion1Word, Error> {
        let row = get_word_row(index);

        // Rename these here for ease of matching the gates to the specification.
        let a_3 = self.extras[0];
        let a_4 = self.extras[1];

        let pieces = word.map(|word| {
            vec![
                word[0..1].to_vec(),
                word[1..7].to_vec(),
                word[7..8].to_vec(),
                word[8..22].to_vec(),
                word[22..36].to_vec(),
                word[36..50].to_vec(),
                word[50..64].to_vec(),
            ]
        });
        let pieces = pieces.transpose_vec(7);

        // Assign `a` (1-bit piece)
        let a =
            AssignedBits::<1>::assign_bits(region, || "word_a", a_3, row + 1, pieces[0].clone())?;
        // Assign `b` (6-bit piece)
        let b =
            AssignedBits::<6>::assign_bits(region, || "word_b", a_4, row + 1, pieces[1].clone())?;

        // Assign `c` (1-bit piece)
        let c =
            AssignedBits::<1>::assign_bits(region, || "word_c", a_3, row, pieces[2].clone())?;

        // Assign `d_lo_lo` (14-bit piece) lookup
        let spread_d_lo_lo = pieces[3].clone().map(SpreadWord::try_new);
        let spread_d_lo_lo = SpreadVar::with_lookup(region, &self.lookup, row - 1, spread_d_lo_lo)?;

        // Assign `d_lo_hi` (14-bit piece) lookup
        let spread_d_lo_hi = pieces[4].clone().map(SpreadWord::try_new);
        let spread_d_lo_hi = SpreadVar::with_lookup(region, &self.lookup, row, spread_d_lo_hi)?;

        // Assign `d_hi_lo` (14-bit piece) lookup
        let spread_d_hi_lo = pieces[5].clone().map(SpreadWord::try_new);
        let spread_d_hi_lo = SpreadVar::with_lookup(region, &self.lookup, row + 1, spread_d_hi_lo)?;

        // Assign `d_hi_hi` (14-bit piece) lookup
        let spread_d_hi_hi = pieces[6].clone().map(SpreadWord::try_new);
        let spread_d_hi_hi = SpreadVar::with_lookup(region, &self.lookup, row + 2, spread_d_hi_hi)?;

        Ok(Subregion1Word {
            index,
            a,
            b,
            c,
            _d_lo_lo: spread_d_lo_lo.dense,
            _d_lo_hi: spread_d_lo_hi.dense,
            _d_hi_lo: spread_d_hi_lo.dense,
            _d_hi_hi: spread_d_hi_hi.dense,
            spread_d_lo_lo: spread_d_lo_lo.spread,
            spread_d_lo_hi: spread_d_lo_hi.spread,
            spread_d_hi_lo: spread_d_hi_lo.spread,
            spread_d_hi_hi: spread_d_hi_hi.spread,
        })
    }

    // sigma_0 v1 on a word in W_1 to W_13
    // (1, 6, 1, 56)-bit chunks
    fn lower_sigma_0(
        &self,
        region: &mut Region<'_, pallas::Base>,
        word: Subregion1Word,
    ) -> Result<(AssignedBits<32>, AssignedBits<32>), Error> {
        let a_3 = self.extras[0];
        let a_4 = self.extras[1];
        let a_5 = self.message_schedule;
        let a_6 = self.extras[2];

        let row = get_word_row(word.index) + 3;

        // Assign `a` and copy constraint
        word.a.copy_advice(|| "a", region, a_6, row + 1)?;

        // Split `b` (6-bit chunk) into `b_hi` and `b_lo`
        // Assign `b_lo`, `spread_b_lo`
        let b_lo: Value<[bool; 3]> = word.b.value().map(|b| b.0[..3].try_into().unwrap());
        let spread_b_lo = b_lo.map(spread_bits);
        {
            AssignedBits::<3>::assign_bits(region, || "b_lo", a_3, row - 1, b_lo)?;

            AssignedBits::<6>::assign_bits(region, || "spread_b_lo", a_4, row - 1, spread_b_lo)?;
        };

        // Split `b` (6-bit chunk) into `b_hi` and `b_lo`
        // Assign `b_hi`, `spread_b_hi`
        let b_hi: Value<[bool; 3]> = word.b.value().map(|b| b.0[3..].try_into().unwrap());
        let spread_b_hi = b_hi.map(spread_bits);
        {
            AssignedBits::<3>::assign_bits(region, || "b_hi", a_5, row - 1, b_hi)?;

            AssignedBits::<6>::assign_bits(region, || "spread_b_hi", a_6, row - 1, spread_b_hi)?;
        };

        // Assign `b` and copy constraint
        word.b.copy_advice(|| "b", region, a_6, row)?;

        // Assign `c` and copy constraint
        word.c.copy_advice(|| "c", region, a_4, row)?;

        // Assign `spread_d_lo_lo` and copy constraint
        word.spread_d_lo_lo.copy_advice(|| "spread_d_lo_lo", region, a_5, row)?;

        // Assign `spread_d_lo_hi` and copy constraint
        word.spread_d_lo_hi.copy_advice(|| "spread_d_lo_hi", region, a_5, row + 1)?;

        // Assign `spread_d_hi_lo` and copy constraint
        word.spread_d_hi_lo.copy_advice(|| "spread_d_hi_lo", region, a_4, row + 1)?;

        // Assign `spread_d_hi_hi` and copy constraint
        word.spread_d_hi_hi.copy_advice(|| "spread_d_hi_hi", region, a_3, row + 1)?;

        // Calculate R_0^{even}, R_0^{odd}, R_1^{even}, R_1^{odd}
        let r = word.xor_lower_sigma_0();
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
