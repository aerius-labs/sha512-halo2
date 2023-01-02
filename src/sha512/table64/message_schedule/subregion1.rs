use super::super::{util::*, AssignedBits, BlockWord, SpreadVar, SpreadWord, Table64Assignment};
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
    _d: AssignedBits<56>,
    spread_d: AssignedBits<112>,
}

impl Subregion1Word {
    fn spread_a(&self) -> Value<[bool; 2]> {
        self.a.value().map(|v| v.spread())
    }

    fn spread_b(&self) -> Value<[bool; 12]> {
        self.b.value().map(|v| v.spread())
    }

    fn spread_c(&self) -> Value<[bool; 2]> {
        self.spread_c.value().map(|v| v.spread())
    }

    fn spread_d(&self) -> Value<[bool; 112]> {
        self.spread_d.value().map(|v| v.0)
    }

    fn xor_lower_sigma_0(&self) -> Value<[bool; 128]> {
        self.spread_a()
            .zip(self.spread_b())
            .zip(self.spread_c())
            .zip(self.spread_d())
            .map(|(((a, b), c), d)| {
                let xor_0 = c
                    .iter()
                    .chain(d.iter())
                    .chain(std::iter::repeat(&false).take(2))
                    .chain(std::iter::repeat(&false).take(12))
                    .copied()
                    .collect::<Vec<_>>();
                let xor_1 = b
                    .iter()
                    .chain(c.iter())
                    .chain(d.iter())
                    .chain(a.iter())
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
                word[8..64].to_vec(),
            ]
        });
        let pieces = pieces.transpose_vec(4);

        // Assign `a` (1-bit piece)
        let a =
            AssignedBits::<1>::assign_bits(region, || "word_a", a_3, row + 1, pieces[0].clone())?;
        // Assign `b` (6-bit piece)
        let b =
            AssignedBits::<6>::assign_bits(region, || "word_b", a_4, row + 1, pieces[1].clone())?;

        // Assign `c` (1-bit piece)
        let c =
            AssignedBits::<1>::assign_bits(region, || "word_c", a_3, row, pieces[2].clone())?;

        // Assign `d` (56-bit piece) lookup
        let spread_d = pieces[3].clone().map(SpreadWord::try_new);
        let spread_d = SpreadVar::with_lookup(region, &self.lookup, row, spread_d)?;

        Ok(Subregion1Word {
            index,
            a,
            b,
            c,
            _d: spread_d.dense,
            spread_d: spread_d.spread,
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
        word.a.copy_advice(|| "a", region, a_5, row + 1)?;

        // Witness `spread_a`
        let spread_a = word.a.value().map(|bits| spread_bits(bits.0));
        AssignedBits::<2>::assign_bits(region, || "spread_a", a_6, row + 1, spread_a)?;

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
        word.c.copy_advice(|| "c", region, a_3, row)?;

        // Witness `spread_c`
        let spread_c = word.c.value().map(|bits| spread_bits(bits.0));
        AssignedBits::<2>::assign_bits(region, || "spread_c", a_3, row, spread_c)?;

        // Assign `spread_d` and copy constraint
        word.spread_d.copy_advice(|| "spread_d", region, a_5, row)?;

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
