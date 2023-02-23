use super::{
    AbcdVar, CompressionConfig, EfghVar, RoundWord, RoundWordA, RoundWordDense, RoundWordE,
    RoundWordSpread, State, UpperSigmaVar,
};
use crate::sha512::table16::{
    util::*, AssignedBits, SpreadVar, SpreadWord, StateWord, Table16Assignment,
};
use halo2_proofs::{
    circuit::{Region, Value},
    pasta::pallas,
    plonk::{Advice, Column, Error},
};
use std::convert::TryInto;

// Test vector 'abc'
#[cfg(test)]
#[allow(dead_code)]
pub const COMPRESSION_OUTPUT: [u64; 8] = [
    0b1101110110101111001101011010000110010011011000010111101010111010,
    0b1010111100110101101000011001001101100001011110101011101011001100,
    0b0011010110100001100100110110000101111010101110101100110001000001,
    0b1010000110010011011000010111101010111010110011000100000101110011,
    0b1001001101100001011110101011101011001100010000010111001101001001,
    0b0110000101111010101110101100110001000001011100110100100110101110,
    0b0111101010111010110011000100000101110011010010011010111000100000,
    0b1011101011001100010000010111001101001001101011100010000001000001,
];

// Rows needed for each gate
pub const SIGMA_0_ROWS: usize = 9;
pub const SIGMA_1_ROWS: usize = 9;
pub const CH_ROWS: usize = 18;
pub const MAJ_ROWS: usize = 9;
pub const DECOMPOSE_ABCD: usize = 4;
pub const DECOMPOSE_EFGH: usize = 5;

// Rows needed for main subregion
pub const SUBREGION_MAIN_LEN: usize = 80;
pub const SUBREGION_MAIN_WORD: usize =
    DECOMPOSE_ABCD + SIGMA_0_ROWS + DECOMPOSE_EFGH + SIGMA_1_ROWS + CH_ROWS + MAJ_ROWS;
pub const SUBREGION_MAIN_ROWS: usize = SUBREGION_MAIN_LEN * SUBREGION_MAIN_WORD;

/// The initial round.
pub struct InitialRound;

/// A main round index.
#[derive(Debug, Copy, Clone)]
pub struct MainRoundIdx(usize);

/// Round index.
#[derive(Debug, Copy, Clone)]
pub enum RoundIdx {
    Init,
    Main(MainRoundIdx),
}

impl From<InitialRound> for RoundIdx {
    fn from(_: InitialRound) -> Self {
        RoundIdx::Init
    }
}

impl From<MainRoundIdx> for RoundIdx {
    fn from(idx: MainRoundIdx) -> Self {
        RoundIdx::Main(idx)
    }
}

impl MainRoundIdx {
    pub(crate) fn as_usize(&self) -> usize {
        self.0
    }
}

impl From<usize> for MainRoundIdx {
    fn from(idx: usize) -> Self {
        MainRoundIdx(idx)
    }
}

impl std::ops::Add<usize> for MainRoundIdx {
    type Output = Self;

    fn add(self, rhs: usize) -> Self::Output {
        MainRoundIdx(self.0 + rhs)
    }
}

impl Ord for MainRoundIdx {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for MainRoundIdx {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for MainRoundIdx {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for MainRoundIdx {}

/// Returns starting row number of a compression round
pub fn get_round_row(round_idx: RoundIdx) -> usize {
    match round_idx {
        RoundIdx::Init => 0,
        RoundIdx::Main(MainRoundIdx(idx)) => {
            assert!(idx < 80);
            idx * SUBREGION_MAIN_WORD
        }
    }
}

pub fn get_decompose_e_row(round_idx: RoundIdx) -> usize {
    get_round_row(round_idx)
}

pub fn get_decompose_f_row(round_idx: InitialRound) -> usize {
    get_decompose_e_row(round_idx.into()) + DECOMPOSE_EFGH
}

pub fn get_decompose_g_row(round_idx: InitialRound) -> usize {
    get_decompose_f_row(round_idx) + DECOMPOSE_EFGH
}

pub fn get_upper_sigma_1_row(round_idx: MainRoundIdx) -> usize {
    get_decompose_e_row(round_idx.into()) + DECOMPOSE_EFGH + 1
}

pub fn get_ch_row(round_idx: MainRoundIdx) -> usize {
    get_decompose_e_row(round_idx.into()) + DECOMPOSE_EFGH + SIGMA_1_ROWS + 1
}

pub fn get_ch_neg_row(round_idx: MainRoundIdx) -> usize {
    get_ch_row(round_idx) + CH_ROWS / 2
}

pub fn get_decompose_a_row(round_idx: RoundIdx) -> usize {
    match round_idx {
        RoundIdx::Init => get_h_row(round_idx) + DECOMPOSE_EFGH,
        RoundIdx::Main(mri) => get_ch_neg_row(mri) - 1 + CH_ROWS / 2,
    }
}

pub fn get_upper_sigma_0_row(round_idx: MainRoundIdx) -> usize {
    get_decompose_a_row(round_idx.into()) + DECOMPOSE_ABCD + 1
}

pub fn get_decompose_b_row(round_idx: InitialRound) -> usize {
    get_decompose_a_row(round_idx.into()) + DECOMPOSE_ABCD
}

pub fn get_decompose_c_row(round_idx: InitialRound) -> usize {
    get_decompose_b_row(round_idx) + DECOMPOSE_ABCD
}

pub fn get_maj_row(round_idx: MainRoundIdx) -> usize {
    get_upper_sigma_0_row(round_idx) + SIGMA_0_ROWS
}

// Get state word rows
pub fn get_h_row(round_idx: RoundIdx) -> usize {
    match round_idx {
        RoundIdx::Init => get_decompose_g_row(InitialRound) + DECOMPOSE_EFGH,
        RoundIdx::Main(mri) => get_ch_row(mri) - 1,
    }
}

pub fn get_h_prime_row(round_idx: MainRoundIdx) -> usize {
    get_ch_row(round_idx)
}

pub fn get_d_row(round_idx: RoundIdx) -> usize {
    match round_idx {
        RoundIdx::Init => get_decompose_c_row(InitialRound) + DECOMPOSE_ABCD,
        RoundIdx::Main(mri) => get_ch_row(mri) + 2,
    }
}

pub fn get_e_new_row(round_idx: MainRoundIdx) -> usize {
    get_d_row(round_idx.into())
}

pub fn get_a_new_row(round_idx: MainRoundIdx) -> usize {
    get_maj_row(round_idx)
}

pub fn get_digest_abcd_row() -> usize {
    SUBREGION_MAIN_ROWS
}

pub fn get_digest_efgh_row() -> usize {
    get_digest_abcd_row() + 4
}

impl CompressionConfig {
    pub(super) fn decompose_abcd(
        &self,
        region: &mut Region<'_, pallas::Base>,
        row: usize,
        val: Value<u64>,
    ) -> Result<AbcdVar, Error> {
        self.s_decompose_abcd.enable(region, row)?;

        let a_3 = self.extras[0];
        let a_4 = self.extras[1];
        // let a_5 = self.message_schedule;
        // let a_6 = self.extras[2];

        let spread_pieces = val.map(AbcdVar::pieces);
        let spread_pieces = spread_pieces.transpose_vec(8);

        let a_lo = SpreadVar::with_lookup(
            region,
            &self.lookup,
            row,
            spread_pieces[0].clone().map(SpreadWord::<14,28>::try_new),
        )?;
        let a_hi = SpreadVar::with_lookup(
            region,
            &self.lookup,
            row + 1,
            spread_pieces[1].clone().map(SpreadWord::<14,28>::try_new),
        )?;
        let b_lo = SpreadVar::without_lookup(
            region,
            a_3,
            row,
            a_4,
            row,
            spread_pieces[2].clone().map(SpreadWord::<3, 6>::try_new),
        )?;
        let b_hi = SpreadVar::without_lookup(
            region,
            a_3,
            row+1,
            a_4,
            row+1,
            spread_pieces[3].clone().map(SpreadWord::<3, 6>::try_new),
        )?;
        let c_lo = SpreadVar::without_lookup(
            region,
            a_3,
            row+2,
            a_4,
            row+2,
            spread_pieces[4].clone().map(SpreadWord::<2, 4>::try_new),
        )?;
        let c_hi = SpreadVar::without_lookup(
            region,
            a_3,
            row + 3,
            a_4,
            row + 3,
            spread_pieces[5].clone().map(SpreadWord::<3, 6>::try_new),
        )?;
        let d_lo = SpreadVar::with_lookup(
            region,
            &self.lookup,
            row+2,
            spread_pieces[6].clone().map(SpreadWord::<14, 28>::try_new),
        )?;
        let d_hi = SpreadVar::with_lookup(
            region,
            &self.lookup,
            row+3,
            spread_pieces[7].clone().map(SpreadWord::<11, 22>::try_new),
        )?;

        Ok(AbcdVar {
            a_lo,
            a_hi,
            b_lo,
            b_hi,
            c_lo,
            c_hi,
            d_lo,
            d_hi,
        })
    }

    pub(super) fn decompose_efgh(
        &self,
        region: &mut Region<'_, pallas::Base>,
        row: usize,
        val: Value<u64>,
    ) -> Result<EfghVar, Error> {
        self.s_decompose_efgh.enable(region, row)?;

        let a_3 = self.extras[0];
        let a_4 = self.extras[1];
        // let a_5 = self.message_schedule;
        // let a_6 = self.extras[2];

        let spread_pieces = val.map(EfghVar::pieces);
        let spread_pieces = spread_pieces.transpose_vec(7);
        let a = SpreadVar::with_lookup(
            region,
            &self.lookup,
            row,
            spread_pieces[0].clone().map(SpreadWord::try_new),
        )?;
        let b_lo = SpreadVar::without_lookup(
            region,
            a_3,
            row,
            a_4,
            row,
            spread_pieces[1].clone().map(SpreadWord::try_new),
        )?;
        let b_hi = SpreadVar::without_lookup(
            region,
            a_3,
            row + 1,
            a_4,
            row + 1,
            spread_pieces[2].clone().map(SpreadWord::try_new),
        )?;
        let c_lo = SpreadVar::with_lookup(
            region,
            &self.lookup,
            row + 1,
            spread_pieces[3].clone().map(SpreadWord::try_new),
        )?;
        let c_hi = SpreadVar::with_lookup(
            region,
            &self.lookup,
            row+2,
            spread_pieces[4].clone().map(SpreadWord::try_new),
        )?;
        let d_lo = SpreadVar::with_lookup(
            region,
            &self.lookup,
            row+3,
            spread_pieces[5].clone().map(SpreadWord::try_new),
        )?;
        let d_hi = SpreadVar::with_lookup(
            region,
            &self.lookup,
            row+4,
            spread_pieces[6].clone().map(SpreadWord::try_new),
        )?;

        Ok(EfghVar {
            a,
            b_lo,
            b_hi,
            c_lo,
            c_hi,
            d_lo,
            d_hi,
        })
    }

    pub(super) fn decompose_a(
        &self,
        region: &mut Region<'_, pallas::Base>,
        round_idx: RoundIdx,
        a_val: Value<u64>,
    ) -> Result<RoundWordA, Error> {
        let row = get_decompose_a_row(round_idx);

        let (dense_halves, spread_halves) = self.assign_word_halves(region, row, a_val)?;
        let a_pieces = self.decompose_abcd(region, row, a_val)?;
        Ok(RoundWordA::new(a_pieces, dense_halves, spread_halves))
    }

    pub(super) fn decompose_e(
        &self,
        region: &mut Region<'_, pallas::Base>,
        round_idx: RoundIdx,
        e_val: Value<u64>,
    ) -> Result<RoundWordE, Error> {
        let row = get_decompose_e_row(round_idx);

        let (dense_halves, spread_halves) = self.assign_word_halves(region, row, e_val)?;
        let e_pieces = self.decompose_efgh(region, row, e_val)?;
        Ok(RoundWordE::new(e_pieces, dense_halves, spread_halves))
    }

    pub(super) fn assign_upper_sigma_0(
        &self,
        region: &mut Region<'_, pallas::Base>,
        round_idx: MainRoundIdx,
        word: AbcdVar,
    ) -> Result<(AssignedBits<32>, AssignedBits<32>), Error> {
        // Rename these here for ease of matching the gates to the specification.
        let a_3 = self.extras[0];
        let a_4 = self.extras[1];
        let a_5 = self.message_schedule;
        // let a_6 = self.extras[2];

        let row = get_upper_sigma_0_row(round_idx);

        self.s_upper_sigma_0.enable(region, row)?;

        // Assign `spread_a_lo` and copy constraint
        word.a_lo
            .spread
            .copy_advice(|| "spread_a_lo", region, a_3, row + 1)?;
        // Assign `spread_a_hi` and copy constraint
        word.a_hi
            .spread
            .copy_advice(|| "spread_a_hi", region, a_4, row - 1)?;
        // Assign `spread_b_lo` and copy constraint
        word.b_lo
            .spread
            .copy_advice(|| "spread_b_lo", region, a_4, row)?;
        // Assign `spread_b_hi` and copy constraint
        word.b_hi
            .spread
            .copy_advice(|| "spread_b_hi", region, a_4, row + 1)?;
        // Assign `spread_c_lo` and copy constraint
        word.c_lo
            .spread
            .copy_advice(|| "spread_c_lo", region, a_5, row - 1)?;
        // Assign `spread_c_hi` and copy constraint
        word.c_hi
            .spread
            .copy_advice(|| "spread_c_hi", region, a_5, row)?;
        // Assign `spread_d_lo` and copy constraint
        word.d_lo
            .spread
            .copy_advice(|| "spread_d_lo", region, a_5, row + 1)?;
        // Assign `spread_d_hi` and copy constraint
        word.d_hi
            .spread
            .copy_advice(|| "spread_d_hi", region, a_3, row - 1)?;

        // Calculate R_0^{even}, R_0^{odd}, R_1^{even}, R_1^{odd}
        let r = word.xor_upper_sigma();
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

    pub(super) fn assign_upper_sigma_1(
        &self,
        region: &mut Region<'_, pallas::Base>,
        round_idx: MainRoundIdx,
        word: EfghVar,
    ) -> Result<(AssignedBits<32>, AssignedBits<32>), Error> {
        // Rename these here for ease of matching the gates to the specification.
        let a_3 = self.extras[0];
        let a_4 = self.extras[1];
        let a_5 = self.message_schedule;

        let row = get_upper_sigma_1_row(round_idx);

        self.s_upper_sigma_1.enable(region, row)?;

        // Assign `spread_a` and copy constraint
        word.a
            .spread
            .copy_advice(|| "spread_a", region, a_3, row + 1)?;
        // Assign `spread_b_lo` and copy constraint
        word.b_lo
            .spread
            .copy_advice(|| "spread_b_lo", region, a_4, row - 1)?;
        // Assign `spread_b_hi` and copy constraint
        word.b_hi
            .spread
            .copy_advice(|| "spread_b_hi", region, a_4, row)?;
        // Assign `spread_c_lo` and copy constraint
        word.c_lo.spread.copy_advice(|| "spread_c_lo", region, a_4, row + 1)?;
        // Assign `spread_c_hi` and copy constraint
        word.c_hi.spread.copy_advice(|| "spread_c_hi", region, a_5, row - 1)?;
        // Assign `spread_d_lo` and copy constraint
        word.d_lo.spread.copy_advice(|| "spread_d_lo", region, a_5, row)?;
        // Assign `spread_d_hi` and copy constraint
        word.d_hi.spread.copy_advice(|| "spread_d_hi", region, a_5, row + 1)?;

        // Calculate R_0^{even}, R_0^{odd}, R_1^{even}, R_1^{odd}
        // Calculate R_0^{even}, R_0^{odd}, R_1^{even}, R_1^{odd}
        let r = word.xor_upper_sigma();
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

    fn assign_ch_outputs(
        &self,
        region: &mut Region<'_, pallas::Base>,
        row: usize,
        r_0_even: Value<[bool; 32]>,
        r_0_odd: Value<[bool; 32]>,
        r_1_even: Value<[bool; 32]>,
        r_1_odd: Value<[bool; 32]>,
    ) -> Result<(AssignedBits<32>, AssignedBits<32>), Error> {
        let a_3 = self.extras[0];

        let (_even, odd) = self.assign_spread_outputs(
            region,
            &self.lookup,
            a_3,
            row,
            r_0_even,
            r_0_odd,
            r_1_even,
            r_1_odd,
        )?;

        Ok(odd)
    }

    pub(super) fn assign_ch(
        &self,
        region: &mut Region<'_, pallas::Base>,
        round_idx: MainRoundIdx,
        spread_halves_e: RoundWordSpread,
        spread_halves_f: RoundWordSpread,
    ) -> Result<(AssignedBits<32>, AssignedBits<32>), Error> {
        let a_3 = self.extras[0];
        let a_4 = self.extras[1];

        let row = get_ch_row(round_idx);

        self.s_ch.enable(region, row)?;

        // Assign and copy spread_e_lo, spread_e_hi
        spread_halves_e
            .0
            .copy_advice(|| "spread_e_lo", region, a_3, row - 1)?;
        spread_halves_e
            .1
            .copy_advice(|| "spread_e_hi", region, a_4, row - 1)?;

        // Assign and copy spread_f_lo, spread_f_hi
        spread_halves_f
            .0
            .copy_advice(|| "spread_f_lo", region, a_3, row + 1)?;
        spread_halves_f
            .1
            .copy_advice(|| "spread_f_hi", region, a_4, row + 1)?;

        let p: Value<[bool; 128]> = spread_halves_e
            .value()
            .zip(spread_halves_f.value())
            .map(|(e, f)| i2lebsp(e + f));

        let p_0: Value<[bool; 64]> = p.map(|p| p[..64].try_into().unwrap());
        let p_0_even = p_0.map(even_bits);
        let p_0_odd = p_0.map(odd_bits);

        let p_1: Value<[bool; 64]> = p.map(|p| p[64..].try_into().unwrap());
        let p_1_even = p_1.map(even_bits);
        let p_1_odd = p_1.map(odd_bits);

        self.assign_ch_outputs(region, row, p_0_even, p_0_odd, p_1_even, p_1_odd)
    }

    pub(super) fn assign_ch_neg(
        &self,
        region: &mut Region<'_, pallas::Base>,
        round_idx: MainRoundIdx,
        spread_halves_e: RoundWordSpread,
        spread_halves_g: RoundWordSpread,
    ) -> Result<(AssignedBits<32>, AssignedBits<32>), Error> {
        let row = get_ch_neg_row(round_idx);

        self.s_ch_neg.enable(region, row)?;

        let a_3 = self.extras[0];
        let a_4 = self.extras[1];
        let a_5 = self.message_schedule;

        // Assign and copy spread_e_lo, spread_e_hi
        spread_halves_e
            .0
            .copy_advice(|| "spread_e_lo", region, a_5, row - 1)?;
        spread_halves_e
            .1
            .copy_advice(|| "spread_e_hi", region, a_5, row)?;

        // Assign and copy spread_g_lo, spread_g_hi
        spread_halves_g
            .0
            .copy_advice(|| "spread_g_lo", region, a_3, row + 1)?;
        spread_halves_g
            .1
            .copy_advice(|| "spread_g_hi", region, a_4, row + 1)?;

        // Calculate neg_e_lo
        let spread_neg_e_lo = spread_halves_e
            .0
            .value()
            .map(|spread_e_lo| negate_spread(spread_e_lo.0));
        // Assign spread_neg_e_lo
        AssignedBits::<64>::assign_bits(
            region,
            || "spread_neg_e_lo",
            a_3,
            row - 1,
            spread_neg_e_lo,
        )?;

        // Calculate neg_e_hi
        let spread_neg_e_hi = spread_halves_e
            .1
            .value()
            .map(|spread_e_hi| negate_spread(spread_e_hi.0));
        // Assign spread_neg_e_hi
        AssignedBits::<64>::assign_bits(
            region,
            || "spread_neg_e_hi",
            a_4,
            row - 1,
            spread_neg_e_hi,
        )?;

        let p: Value<[bool; 128]> = {
            let spread_neg_e = spread_neg_e_lo
                .zip(spread_neg_e_hi)
                .map(|(lo, hi)| lebs2ip(&lo) + (1 << 64) * lebs2ip(&hi));
            spread_neg_e
                .zip(spread_halves_g.value())
                .map(|(neg_e, g)| i2lebsp(neg_e + g))
        };

        let p_0: Value<[bool; 64]> = p.map(|p| p[..64].try_into().unwrap());
        let p_0_even = p_0.map(even_bits);
        let p_0_odd = p_0.map(odd_bits);

        let p_1: Value<[bool; 64]> = p.map(|p| p[64..].try_into().unwrap());
        let p_1_even = p_1.map(even_bits);
        let p_1_odd = p_1.map(odd_bits);

        self.assign_ch_outputs(region, row, p_0_even, p_0_odd, p_1_even, p_1_odd)
    }

    fn assign_maj_outputs(
        &self,
        region: &mut Region<'_, pallas::Base>,
        row: usize,
        r_0_even: Value<[bool; 32]>,
        r_0_odd: Value<[bool; 32]>,
        r_1_even: Value<[bool; 32]>,
        r_1_odd: Value<[bool; 32]>,
    ) -> Result<(AssignedBits<32>, AssignedBits<32>), Error> {
        let a_3 = self.extras[0];
        let (_even, odd) = self.assign_spread_outputs(
            region,
            &self.lookup,
            a_3,
            row,
            r_0_even,
            r_0_odd,
            r_1_even,
            r_1_odd,
        )?;

        Ok(odd)
    }

    pub(super) fn assign_maj(
        &self,
        region: &mut Region<'_, pallas::Base>,
        round_idx: MainRoundIdx,
        spread_halves_a: RoundWordSpread,
        spread_halves_b: RoundWordSpread,
        spread_halves_c: RoundWordSpread,
    ) -> Result<(AssignedBits<32>, AssignedBits<32>), Error> {
        let a_4 = self.extras[1];
        let a_5 = self.message_schedule;

        let row = get_maj_row(round_idx);

        self.s_maj.enable(region, row)?;

        // Assign and copy spread_a_lo, spread_a_hi
        spread_halves_a
            .0
            .copy_advice(|| "spread_a_lo", region, a_4, row - 1)?;
        spread_halves_a
            .1
            .copy_advice(|| "spread_a_hi", region, a_5, row - 1)?;

        // Assign and copy spread_b_lo, spread_b_hi
        spread_halves_b
            .0
            .copy_advice(|| "spread_b_lo", region, a_4, row)?;
        spread_halves_b
            .1
            .copy_advice(|| "spread_b_hi", region, a_5, row)?;

        // Assign and copy spread_c_lo, spread_c_hi
        spread_halves_c
            .0
            .copy_advice(|| "spread_c_lo", region, a_4, row + 1)?;
        spread_halves_c
            .1
            .copy_advice(|| "spread_c_hi", region, a_5, row + 1)?;

        let m: Value<[bool; 128]> = spread_halves_a
            .value()
            .zip(spread_halves_b.value())
            .zip(spread_halves_c.value())
            .map(|((a, b), c)| i2lebsp(a + b + c));

        let m_0: Value<[bool; 64]> = m.map(|m| m[..64].try_into().unwrap());
        let m_0_even = m_0.map(even_bits);
        let m_0_odd = m_0.map(odd_bits);

        let m_1: Value<[bool; 64]> = m.map(|m| m[64..].try_into().unwrap());
        let m_1_even = m_1.map(even_bits);
        let m_1_odd = m_1.map(odd_bits);

        self.assign_maj_outputs(region, row, m_0_even, m_0_odd, m_1_even, m_1_odd)
    }

    // s_h_prime to get H' = H + Ch(E, F, G) + s_upper_sigma_1(E) + K + W
    #[allow(clippy::too_many_arguments)]
    pub(super) fn assign_h_prime(
        &self,
        region: &mut Region<'_, pallas::Base>,
        round_idx: MainRoundIdx,
        h: RoundWordDense,
        ch: (AssignedBits<32>, AssignedBits<32>),
        ch_neg: (AssignedBits<32>, AssignedBits<32>),
        sigma_1: (AssignedBits<32>, AssignedBits<32>),
        k: u64,
        w: &(AssignedBits<32>, AssignedBits<32>),
    ) -> Result<RoundWordDense, Error> {
        let row = get_h_prime_row(round_idx);
        self.s_h_prime.enable(region, row)?;

        let a_4 = self.extras[1];
        let a_5 = self.message_schedule;
        let a_6 = self.extras[2];
        let a_7 = self.extras[3];
        let a_8 = self.extras[4];
        let a_9 = self.extras[5];

        // Assign and copy h
        h.0.copy_advice(|| "h_lo", region, a_7, row - 1)?;
        h.1.copy_advice(|| "h_hi", region, a_7, row)?;

        // Assign and copy sigma_1
        sigma_1.0.copy_advice(|| "sigma_1_lo", region, a_4, row)?;
        sigma_1.1.copy_advice(|| "sigma_1_hi", region, a_5, row)?;

        // Assign k
        let k: [bool; 64] = i2lebsp(k.into());
        let k_lo: [bool; 32] = k[..32].try_into().unwrap();
        let k_hi: [bool; 32] = k[32..].try_into().unwrap();
        {
            AssignedBits::<32>::assign_bits(region, || "k_lo", a_6, row - 1, Value::known(k_lo))?;
            AssignedBits::<32>::assign_bits(region, || "k_hi", a_6, row, Value::known(k_hi))?;
        }

        // Assign and copy w
        w.0.copy_advice(|| "w_lo", region, a_8, row - 1)?;
        w.1.copy_advice(|| "w_hi", region, a_8, row)?;

        // Assign and copy ch
        // ch.0.copy_advice(|| "ch_lo", region, a_9, row)?;
        ch.1.copy_advice(|| "ch_hi", region, a_6, row + 1)?;

        // Assign and copy ch_neg
        ch_neg.0.copy_advice(|| "ch_neg_lo", region, a_5, row - 1)?;
        ch_neg.1.copy_advice(|| "ch_neg_hi", region, a_5, row + 1)?;

        // Assign h_prime_lo, h_prime_hi, h_prime_carry
        {
            let (h_prime, h_prime_carry) = sum_with_carry(vec![
                (h.0.value_u32(), h.1.value_u32()),
                (ch.0.value_u32(), ch.1.value_u32()),
                (ch_neg.0.value_u32(), ch_neg.1.value_u32()),
                (sigma_1.0.value_u32(), sigma_1.1.value_u32()),
                (
                    Value::known(lebs2ip(&k_lo) as u32),
                    Value::known(lebs2ip(&k_hi) as u32),
                ),
                (w.0.value_u32(), w.1.value_u32()),
            ]);

            region.assign_advice(
                || "h_prime_carry",
                a_9,
                row + 1,
                || h_prime_carry.map(pallas::Base::from),
            )?;

            let h_prime: Value<[bool; 64]> = h_prime.map(|w| i2lebsp(w.into()));
            let h_prime_lo: Value<[bool; 32]> = h_prime.map(|w| w[..32].try_into().unwrap());
            let h_prime_hi: Value<[bool; 32]> = h_prime.map(|w| w[32..].try_into().unwrap());

            let h_prime_lo =
                AssignedBits::<32>::assign_bits(region, || "h_prime_lo", a_7, row + 1, h_prime_lo)?;
            let h_prime_hi =
                AssignedBits::<32>::assign_bits(region, || "h_prime_hi", a_8, row + 1, h_prime_hi)?;

            Ok((h_prime_lo, h_prime_hi).into())
        }
    }

    // s_e_new to get E_new = H' + D
    pub(super) fn assign_e_new(
        &self,
        region: &mut Region<'_, pallas::Base>,
        round_idx: MainRoundIdx,
        d: &RoundWordDense,
        h_prime: &RoundWordDense,
    ) -> Result<RoundWordDense, Error> {
        let row = get_e_new_row(round_idx);

        self.s_e_new.enable(region, row)?;

        let a_7 = self.extras[3];
        let a_8 = self.extras[4];
        let a_9 = self.extras[5];

        // Assign and copy d_lo, d_hi
        d.0.copy_advice(|| "d_lo", region, a_7, row)?;
        d.1.copy_advice(|| "d_hi", region, a_7, row + 1)?;

        // Assign e_new, e_new_carry
        let (e_new, e_new_carry) = sum_with_carry(vec![
            (h_prime.0.value_u32(), h_prime.1.value_u32()),
            (d.0.value_u32(), d.1.value_u32()),
        ]);

        let e_new_dense = self.assign_word_halves_dense(region, row, a_8, row + 1, a_8, e_new)?;
        region.assign_advice(
            || "e_new_carry",
            a_9,
            row + 1,
            || e_new_carry.map(pallas::Base::from),
        )?;

        Ok(e_new_dense)
    }

    // s_a_new to get A_new = H' + Maj(A, B, C) + s_upper_sigma_0(A)
    pub(super) fn assign_a_new(
        &self,
        region: &mut Region<'_, pallas::Base>,
        round_idx: MainRoundIdx,
        maj: (AssignedBits<32>, AssignedBits<32>),
        sigma_0: (AssignedBits<32>, AssignedBits<32>),
        h_prime: RoundWordDense,
    ) -> Result<RoundWordDense, Error> {
        let row = get_a_new_row(round_idx);

        self.s_a_new.enable(region, row)?;

        let a_3 = self.extras[0];
        let a_6 = self.extras[2];
        let a_7 = self.extras[3];
        let a_8 = self.extras[4];
        let a_9 = self.extras[5];

        // Assign and copy maj_1
        // maj.0.copy_advice(|| "maj_1_lo", region, a_7, row)?;
        maj.1.copy_advice(|| "maj_1_hi", region, a_3, row - 1)?;

        // Assign and copy sigma_0
        sigma_0.0.copy_advice(|| "sigma_0_lo", region, a_6, row)?;
        sigma_0
            .1
            .copy_advice(|| "sigma_0_hi", region, a_6, row + 1)?;

        // Assign and copy h_prime
        h_prime
            .0
            .copy_advice(|| "h_prime_lo", region, a_7, row - 1)?;
        h_prime
            .1
            .copy_advice(|| "h_prime_hi", region, a_8, row - 1)?;

        // Assign a_new, a_new_carry
        let (a_new, a_new_carry) = sum_with_carry(vec![
            (h_prime.0.value_u32(), h_prime.1.value_u32()),
            (sigma_0.0.value_u32(), sigma_0.1.value_u32()),
            (maj.0.value_u32(), maj.1.value_u32()),
        ]);

        let a_new_dense = self.assign_word_halves_dense(region, row, a_8, row + 1, a_8, a_new)?;
        region.assign_advice(
            || "a_new_carry",
            a_9,
            row,
            || a_new_carry.map(pallas::Base::from),
        )?;

        Ok(a_new_dense)
    }

    pub fn assign_word_halves_dense(
        &self,
        region: &mut Region<'_, pallas::Base>,
        lo_row: usize,
        lo_col: Column<Advice>,
        hi_row: usize,
        hi_col: Column<Advice>,
        word: Value<u64>,
    ) -> Result<RoundWordDense, Error> {
        let word: Value<[bool; 64]> = word.map(|w| i2lebsp(w.into()));

        let lo = {
            let lo: Value<[bool; 32]> = word.map(|w| w[..32].try_into().unwrap());
            AssignedBits::<32>::assign_bits(region, || "lo", lo_col, lo_row, lo)?
        };

        let hi = {
            let hi: Value<[bool; 32]> = word.map(|w| w[32..].try_into().unwrap());
            AssignedBits::<32>::assign_bits(region, || "hi", hi_col, hi_row, hi)?
        };

        Ok((lo, hi).into())
    }

    fn joindense(x :&SpreadVar<16,32>,y : &SpreadVar<16,32>) -> Value<[bool; 32]> {
        x
        .dense
        .value()
        .zip(y.dense.value())
        .map(|(x, y)| {
            x.iter()
                .chain(y.iter())
                .copied()
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
    }

    fn joinspread(x :&SpreadVar<16,32>,y : &SpreadVar<16,32>) -> Value<[bool; 64]> {
        x
        .spread
        .value()
        .zip(y.spread.value())
        .map(|(x, y)| {
            x.iter()
                .chain(y.iter())
                .copied()
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
    }

    // Assign hi and lo halves for both dense and spread versions of a word
    #[allow(clippy::type_complexity)]
    pub fn assign_word_halves(
        &self,
        region: &mut Region<'_, pallas::Base>,
        row: usize,
        word: Value<u64>,
    ) -> Result<(RoundWordDense, RoundWordSpread), Error> {
        // Rename these here for ease of matching the gates to the specification.
        let a_7 = self.extras[3];
        let a_8 = self.extras[4];

        let word: Value<[bool; 64]> = word.map(|w| i2lebsp(w.into()));
        let lo_lo: Value<[bool; 16]> = word.map(|w| w[..16].try_into().unwrap());
        let lo_hi: Value<[bool; 16]> = word.map(|w| w[16..32].try_into().unwrap());
        let hi_lo: Value<[bool; 16]> = word.map(|w| w[32..48].try_into().unwrap());
        let hi_hi: Value<[bool; 16]> = word.map(|w| w[48..64].try_into().unwrap());

        let w_lo_lo = SpreadVar::without_lookup(region, a_7, row, a_8, row, lo_lo.map(SpreadWord::new))?;
        let w_lo_hi = SpreadVar::without_lookup(region, a_7, row+1, a_8, row+1, lo_hi.map(SpreadWord::new))?;
        let w_hi_lo = SpreadVar::without_lookup(region, a_7, row+2, a_8, row+2, hi_lo.map(SpreadWord::new))?;
        let w_hi_hi = SpreadVar::without_lookup(region, a_7, row + 3, a_8, row + 3, hi_hi.map(SpreadWord::new))?;

        let w_lo_dense = Self::joindense(&w_lo_lo,&w_lo_hi);
        let w_lo_d = AssignedBits::<32>::assign_bits(region, || "w_lo_d", a_7, row, w_lo_dense)?;

        let w_lo_spread = Self::joinspread(&w_lo_lo,&w_lo_hi);
        let w_lo_s = AssignedBits::<64>::assign_bits(region, || "w_lo_s", a_8, row, w_lo_spread)?;

        let w_hi_dense = Self::joindense(&w_hi_lo,&w_hi_hi);
        let w_hi_d = AssignedBits::<32>::assign_bits(region, || "w_hi_d", a_7, row + 1, w_hi_dense)?;

        let w_hi_spread = Self::joinspread(&w_hi_lo,&w_hi_hi);
        let w_hi_s = AssignedBits::<64>::assign_bits(region, || "w_hi_s", a_8, row + 1, w_hi_spread)?;

        Ok((
            (w_lo_d, w_hi_d).into(),
            (w_lo_s, w_hi_s).into(),
        ))
    }
}

#[allow(clippy::many_single_char_names)]
pub fn match_state(
    state: State,
) -> (
    RoundWordA,
    RoundWord,
    RoundWord,
    RoundWordDense,
    RoundWordE,
    RoundWord,
    RoundWord,
    RoundWordDense,
) {
    let a = match state.a {
        Some(StateWord::A(a)) => a,
        _ => unreachable!(),
    };
    let b = match state.b {
        Some(StateWord::B(b)) => b,
        _ => unreachable!(),
    };
    let c = match state.c {
        Some(StateWord::C(c)) => c,
        _ => unreachable!(),
    };
    let d = match state.d {
        Some(StateWord::D(d)) => d,
        _ => unreachable!(),
    };
    let e = match state.e {
        Some(StateWord::E(e)) => e,
        _ => unreachable!(),
    };
    let f = match state.f {
        Some(StateWord::F(f)) => f,
        _ => unreachable!(),
    };
    let g = match state.g {
        Some(StateWord::G(g)) => g,
        _ => unreachable!(),
    };
    let h = match state.h {
        Some(StateWord::H(h)) => h,
        _ => unreachable!(),
    };

    (a, b, c, d, e, f, g, h)
}