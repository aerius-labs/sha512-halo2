use super::{
    super::DIGEST_SIZE,
    util::{i2lebsp, lebs2ip},
    AssignedBits, BlockWord, SpreadInputs, SpreadVar, Table16Assignment, ROUNDS, STATE,
};

use halo2_proofs::{
    circuit::{Layouter, Value},
    pasta::pallas,
    plonk::{Advice, Column, ConstraintSystem, Error, Selector},
    poly::Rotation,
};

use std::convert::TryInto;
use std::ops::Range;

mod compression_gates;
mod compression_util;
mod subregion_digest;
mod subregion_initial;
mod subregion_main;

use compression_gates::CompressionGate;

pub trait UpperSigmaVar<
    const A_LEN: usize,
    const B_LEN: usize,
    const C_LEN: usize,
    const D_LEN: usize,
>
{
    fn spread_a(&self) -> Value<[bool; A_LEN]>;
    fn spread_b(&self) -> Value<[bool; B_LEN]>;
    fn spread_c(&self) -> Value<[bool; C_LEN]>;
    fn spread_d(&self) -> Value<[bool; D_LEN]>;

    fn xor_upper_sigma(&self) -> Value<[bool; 128]> {
        self.spread_a()
            .zip(self.spread_b())
            .zip(self.spread_c())
            .zip(self.spread_d())
            .map(|(((a, b), c), d)| {
                let xor_0 = b
                    .iter()
                    .chain(c.iter())
                    .chain(d.iter())
                    .chain(a.iter())
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

/// A variable that represents the `[A,B,C,D]` words of the SHA-512 internal state.
///
/// The structure of this variable is influenced by the following factors:
/// - In `Σ_0(A)` we need `A` to be split into pieces `(a,b,c,d)` of lengths `(28,6,5,25)`
///   bits respectively (counting from the little end), as well as their spread forms.
/// - `Maj(A,B,C)` requires having the bits of each input in spread form. For `A` we can
///   reuse the pieces from `Σ_0(A)`. Since `B` and `C` are assigned from `A` and `B`
///   respectively in each round, we therefore also have the same pieces in earlier rows.
///   We align the columns to make it efficient to copy-constrain these forms where they
///   are needed.
#[derive(Clone, Debug)]
pub struct AbcdVar {
    a: SpreadVar<28, 56>,
    b_lo: SpreadVar<3,6>,
    b_hi: SpreadVar<3,6>,
    c_lo: SpreadVar<2, 4>,
    c_hi: SpreadVar<3, 6>,
    d: SpreadVar<25, 50>,
}

impl AbcdVar {
    fn a_range() -> Range<usize> {
        0..28
    }

    fn b_lo_range() -> Range<usize> {
        28..31
    }

    fn b_hi_range() -> Range<usize> {
        31..34
    }

    fn c_lo_range() -> Range<usize> {
        34..36
    }

    fn c_hi_range() -> Range<usize> {
        36..39
    }

    fn d_range() -> Range<usize> {
        39..64
    }

    fn pieces(val: u64) -> Vec<Vec<bool>> {
        let val: [bool; 64] = i2lebsp(val.into());
        vec![
            val[Self::a_range()].to_vec(),
            val[Self::b_hi_range()].to_vec(),
            val[Self::b_lo_range()].to_vec(),
            val[Self::c_hi_range()].to_vec(),
            val[Self::c_lo_range()].to_vec(),
            val[Self::d_range()].to_vec(),
        ]
    }
}