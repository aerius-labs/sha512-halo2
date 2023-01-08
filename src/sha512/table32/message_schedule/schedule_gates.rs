use super::super::Gate;

use group::ff::{Field, PrimeField};
use halo2_proofs::plonk::Expression;
use std::marker::PhantomData;

pub struct ScheduleGate<F: Field>(PhantomData<F>);

impl<F: PrimeField> ScheduleGate<F> {
    /// s_word for W_16 to W_79
    #[allow(clippy::too_many_arguments)]
    pub fn s_word(
        s_word: Expression<F>,
        sigma_0_lo: Expression<F>,
        sigma_0_hi: Expression<F>,
        sigma_1_lo: Expression<F>,
        sigma_1_hi: Expression<F>,
        w_minus_9_lo: Expression<F>,
        w_minus_9_hi: Expression<F>,
        w_minus_16_lo: Expression<F>,
        w_minus_16_hi: Expression<F>,
        word: Expression<F>,
        carry: Expression<F>,
    ) -> impl Iterator<Item = (&'static str, Expression<F>)> {
        let lo = sigma_0_lo + sigma_1_lo + w_minus_9_lo + w_minus_16_lo;
        let hi = sigma_0_hi + sigma_1_hi + w_minus_9_hi + w_minus_16_hi;

        let word_check = lo
            + hi * F::from(1 << 32)
            + (carry.clone() * F::from_u128(1 << 64) * (-F::ONE))
            + (word * (-F::ONE));
        let carry_check = Gate::range_check(carry, 0, 3);

        [("word_check", word_check), ("carry_check", carry_check)]
            .into_iter()
            .map(move |(name, poly)| (name, s_word.clone() * poly))
    }

    /// s_decompose_0 for all words
    pub fn s_decompose_0(
        s_decompose_0: Expression<F>,
        lo: Expression<F>,
        hi: Expression<F>,
        word: Expression<F>,
    ) -> Option<(&'static str, Expression<F>)> {
        let check = lo + hi * F::from(1 << 32) - word;
        Some(("s_decompose_0", s_decompose_0 * check))
    }

    /// s_decompose_1 for W_1 to W_13
    /// (1, 6, 1, 56)-bit chunks
    #[allow(clippy::too_many_arguments)]
    pub fn s_decompose_1(
        s_decompose_1: Expression<F>,
        a: Expression<F>,
        b: Expression<F>,
        c: Expression<F>,
        d_lo: Expression<F>,
        d_hi: Expression<F>,
        tag_d_lo: Expression<F>,
        tag_d_hi: Expression<F>,
        word: Expression<F>,
    ) -> impl Iterator<Item = (&'static str, Expression<F>)> {
        let decompose_check =
            a + b * F::from(1 << 1) + c * F::from(1 << 7) + d_lo * F::from(1 << 8) + d_hi * F::from(1 << 36) + word * (-F::ONE);
        let range_check_tag_d_lo = Gate::range_check(tag_d_lo, 0, 6);
        let range_check_tag_d_hi = Gate::range_check(tag_d_hi, 0, 6);

        [
            ("decompose_check", decompose_check),
            ("range_check_tag_d_lo", range_check_tag_d_lo),
            ("range_check_tag_d_hi", range_check_tag_d_hi),
        ]
        .into_iter()
        .map(move |(name, poly)| (name, s_decompose_1.clone() * poly))
    }

    /// s_decompose_2 for W_14 to W_64
    /// (1, 5, 1, 1, 11, 42, 3)-bit chunks
    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::too_many_arguments)]
    pub fn s_decompose_2(
        s_decompose_2: Expression<F>,
        a: Expression<F>,
        b: Expression<F>,
        c: Expression<F>,
        d: Expression<F>,
        e: Expression<F>,
        tag_e: Expression<F>,
        f_lo: Expression<F>,
        f_hi: Expression<F>,
        tag_f_lo: Expression<F>,
        tag_f_hi: Expression<F>,
        g: Expression<F>,
        word: Expression<F>,
    ) -> impl Iterator<Item = (&'static str, Expression<F>)> {
        let decompose_check = a
            + b * F::from(1 << 1)
            + c * F::from(1 << 6)
            + d * F::from(1 << 7)
            + e * F::from(1 << 8)
            + f_lo * F::from(1 << 19)
            + f_hi * F::from(1 << 40)
            + g * F::from(1 << 61)
            + word * (-F::ONE);
        let range_check_tag_e = Gate::range_check(tag_e, 0, 0);
        let range_check_tag_f_lo = Gate::range_check(tag_f_lo, 0, 3);
        let range_check_tag_f_hi = Gate::range_check(tag_f_hi, 0, 3);

        [
            ("decompose_check", decompose_check),
            ("range_check_tag_e", range_check_tag_e),
            ("range_check_tag_f_lo", range_check_tag_f_lo),
            ("range_check_tag_f_hi", range_check_tag_f_hi),
        ]
        .into_iter()
        .map(move |(name, poly)| (name, s_decompose_2.clone() * poly))
    }

    /// s_decompose_3 for W_65 to W_77
    /// (6, 13, 42, 3)-bit chunks
    #[allow(clippy::too_many_arguments)]
    pub fn s_decompose_3(
        s_decompose_3: Expression<F>,
        a: Expression<F>,
        b: Expression<F>,
        tag_b: Expression<F>,
        c_lo: Expression<F>,
        c_hi: Expression<F>,
        tag_c_lo: Expression<F>,
        tag_c_hi: Expression<F>,
        d: Expression<F>,
        word: Expression<F>,
    ) -> impl Iterator<Item = (&'static str, Expression<F>)> {
        let decompose_check = a
            + b * F::from(1 << 6)
            + c_lo * F::from(1 << 19)
            + c_hi * F::from(1 << 40)
            + d * F::from(1 << 61)
            + word * (-F::ONE);
        let range_check_tag_b = Gate::range_check(tag_b, 0, 1);
        let range_check_tag_c_lo = Gate::range_check(tag_c_lo, 0, 3);
        let range_check_tag_c_hi = Gate::range_check(tag_c_hi, 0, 3);

        [
            ("decompose_check", decompose_check),
            ("range_check_tag_b", range_check_tag_b),
            ("range_check_tag_c_lo", range_check_tag_c_lo),
            ("range_check_tag_c_hi", range_check_tag_c_hi),
        ]
        .into_iter()
        .map(move |(name, poly)| (name, s_decompose_3.clone() * poly))
    }

     /// b_lo + 2^3 * b_hi = b, on W_[1..49]
     fn check_b(b: Expression<F>, b_lo: Expression<F>, b_hi: Expression<F>) -> Expression<F> {
        let expected_b = b_lo + b_hi * F::from(1 << 3);
        expected_b - b
    }

    /// sigma_0 v1 on W_1 to W_13
    /// (1, 6, 1, 56)-bit chunks
    #[allow(clippy::too_many_arguments)]
    pub fn s_lower_sigma_0(
        s_lower_sigma_0: Expression<F>,
        spread_r0_even: Expression<F>,
        spread_r0_odd: Expression<F>,
        spread_r1_even: Expression<F>,
        spread_r1_odd: Expression<F>,
        spread_a: Expression<F>,
        b: Expression<F>,
        b_lo: Expression<F>,
        spread_b_lo: Expression<F>,
        b_hi: Expression<F>,
        spread_b_hi: Expression<F>,
        spread_c: Expression<F>,
        spread_d_lo: Expression<F>,
        spread_d_hi: Expression<F>,
    ) -> impl Iterator<Item = (&'static str, Expression<F>)> {
        let check_spread_and_range =
            Gate::three_bit_spread_and_range(b_lo.clone(), spread_b_lo.clone())
                .chain(Gate::three_bit_spread_and_range(
                    b_hi.clone(),
                    spread_b_hi.clone(),
                ));
               
        let check_b = Self::check_b(b, b_lo, b_hi);
        let spread_witness = spread_r0_even
            + spread_r0_odd * F::from(2)
            + (spread_r1_even + spread_r1_odd * F::from(2)) * F::from_u128(1 << 64);
        let xor_0 = spread_c.clone()
            + spread_d_lo.clone() * F::from(1 << 2)
            + spread_d_hi.clone() * F::from(1 << 58);
        let xor_1 = spread_b_lo.clone()
            + spread_b_hi.clone() * F::from(1 << 6)
            + spread_c.clone() * F::from(1 << 12)
            + spread_d_lo.clone() * F::from(1 << 14)
            + spread_d_hi.clone() * F::from_u128(1 << 70)
            + spread_a.clone() * F::from_u128(1 << 126);
        let xor_2 = spread_d_lo
            + spread_d_hi * F::from(1 << 56)
            + spread_a * F::from_u128(1 << 112)
            + spread_b_lo * F::from_u128(1 << 114)
            + spread_b_hi * F::from_u128(1 << 120)
            + spread_c * F::from_u128(1 << 126);
        let xor = xor_0 + xor_1 + xor_2;

        check_spread_and_range
            .chain(Some(("check_b", check_b)))
            .chain(Some(("lower_sigma_0", spread_witness - xor)))
            .map(move |(name, poly)| (name, s_lower_sigma_0.clone() * poly))
    }

    /// sigma_1 v1 on W_49 to W_77
    /// (6, 13, 42, 3)-bit chunks
    #[allow(clippy::too_many_arguments)]
    pub fn s_lower_sigma_1(
        s_lower_sigma_1: Expression<F>,
        spread_r0_even: Expression<F>,
        spread_r0_odd: Expression<F>,
        spread_r1_even: Expression<F>,
        spread_r1_odd: Expression<F>,
        a: Expression<F>,
        a_lo: Expression<F>,
        a_hi: Expression<F>,
        spread_a_lo: Expression<F>,
        spread_a_hi: Expression<F>,
        spread_b: Expression<F>,
        spread_c_lo: Expression<F>,
        spread_c_hi: Expression<F>,
        d: Expression<F>,
        spread_d: Expression<F>,
    ) -> impl Iterator<Item = (&'static str, Expression<F>)> {
        let check_spread_and_range =
            Gate::three_bit_spread_and_range(a_lo.clone(), spread_a_lo.clone())
                .chain(Gate::three_bit_spread_and_range(
                    a_hi.clone(),
                    spread_a_hi.clone(),
                ));
        // a_lo + 2^3 * a_hi = b, on W_[49..77]
        let check_a = Self::check_b(a, a_lo, a_hi);
        let spread_witness = spread_r0_even
            + spread_r0_odd * F::from(2)
            + (spread_r1_even + spread_r1_odd * F::from(2)) * F::from_u128(1 << 64);
        let xor_0 = spread_b.clone()
            + spread_c_lo.clone() * F::from(1 << 26)
            + spread_c_hi.clone() * F::from_u128(1 << 68)
            + spread_d.clone() * F::from_u128(1 << 110);
        let xor_1 = spread_c_lo.clone()
            + spread_c_hi.clone() * F::from(1 << 42)
            + spread_d.clone() * F::from_u128(1 << 84)
            + spread_a_lo.clone() * F::from_u128(1 << 90)
            + spread_a_hi.clone() * F::from_u128(1 << 96)
            + spread_b.clone() * F::from_u128(1 << 102);
        let xor_2 = spread_d
            + spread_a_lo * F::from(1 << 6)
            + spread_a_hi * F::from(1 << 12)
            + spread_b * F::from(1 << 18)
            + spread_c_lo * F::from(1 << 44)
            + spread_c_hi * F::from_u128(1 << 86);
        let xor = xor_0 + xor_1 + xor_2;

        check_spread_and_range
            .chain(Some(("check_a", check_a)))
            .chain(Some(("lower_sigma_1", spread_witness - xor)))
            .map(move |(name, poly)| (name, s_lower_sigma_1.clone() * poly))
    }

    /// sigma_0 v2 on W_14 to W_65
    /// (1, 5, 1, 1, 11, 42, 3)-bit chunks
    #[allow(clippy::too_many_arguments)]
    pub fn s_lower_sigma_0_v2(
        s_lower_sigma_0_v2: Expression<F>,
        spread_r0_even: Expression<F>,
        spread_r0_odd: Expression<F>,
        spread_r1_even: Expression<F>,
        spread_r1_odd: Expression<F>,
        spread_a: Expression<F>,
        b: Expression<F>,
        b_lo: Expression<F>,
        spread_b_lo: Expression<F>,
        b_hi: Expression<F>,
        spread_b_hi: Expression<F>,
        spread_c: Expression<F>,
        spread_d: Expression<F>,
        spread_e: Expression<F>,
        spread_f_lo: Expression<F>,
        spread_f_hi: Expression<F>,
        g: Expression<F>,
        spread_g: Expression<F>,
    ) -> impl Iterator<Item = (&'static str, Expression<F>)> {
        let check_spread_and_range =
            Gate::three_bit_spread_and_range(b_lo.clone(), spread_b_lo.clone())
                .chain(Gate::two_bit_spread_and_range(
                    b_hi.clone(),
                    spread_b_hi.clone(),
                ))
                .chain(Gate::three_bit_spread_and_range(g, spread_g.clone()));

        let check_b = Self::check_b(b, b_lo, b_hi);
        let spread_witness = spread_r0_even
            + spread_r0_odd * F::from(2)
            + (spread_r1_even + spread_r1_odd * F::from(2)) * F::from_u128(1 << 64);
        let xor_0 = spread_d.clone()
            + spread_e.clone() * F::from(1 << 2)
            + spread_f_lo.clone() * F::from(1 << 24)
            + spread_f_hi.clone() * F::from_u128(1 << 66)
            + spread_g.clone() * F::from_u128(1 << 108);
        let xor_1 = spread_b_lo.clone()
            + spread_b_hi.clone() * F::from(1 << 6)
            + spread_c.clone() * F::from(1 << 10)
            + spread_d.clone() * F::from(1 << 12)
            + spread_e.clone() * F::from(1 << 14)
            + spread_f_lo.clone() * F::from(1 << 36)
            + spread_f_hi.clone() * F::from_u128(1 << 78)
            + spread_g.clone() * F::from_u128(1 << 120)
            + spread_a.clone() * F::from_u128(1 << 126);
        let xor_2 = spread_e
            + spread_f_lo * F::from(1 << 22)
            + spread_f_hi * F::from_u128(1 << 64)
            + spread_g * F::from_u128(1 << 106)
            + spread_a * F::from_u128(1 << 112)
            + spread_b_lo * F::from_u128(1 << 114)
            + spread_b_hi * F::from_u128(1 << 120)
            + spread_c * F::from_u128(1 << 124)
            + spread_d * F::from_u128(1 << 126);
        let xor = xor_0 + xor_1 + xor_2;

        check_spread_and_range
            .chain(Some(("check_b", check_b)))
            .chain(Some(("lower_sigma_0_v2", spread_witness - xor)))
            .map(move |(name, poly)| (name, s_lower_sigma_0_v2.clone() * poly))
    }

    /// sigma_1 v2 on W_14 to W_64
    /// (1, 5, 1, 1, 11, 42, 3)-bit chunks
    #[allow(clippy::too_many_arguments)]
    pub fn s_lower_sigma_1_v2(
        s_lower_sigma_1_v2: Expression<F>,
        spread_r0_even: Expression<F>,
        spread_r0_odd: Expression<F>,
        spread_r1_even: Expression<F>,
        spread_r1_odd: Expression<F>,
        spread_a: Expression<F>,
        b: Expression<F>,
        b_lo: Expression<F>,
        spread_b_lo: Expression<F>,
        b_hi: Expression<F>,
        spread_b_hi: Expression<F>,
        spread_c: Expression<F>,
        spread_d: Expression<F>,
        spread_e: Expression<F>,
        spread_f_lo: Expression<F>,
        spread_f_hi: Expression<F>,
        g: Expression<F>,
        spread_g: Expression<F>,
    ) -> impl Iterator<Item = (&'static str, Expression<F>)> {
        let check_spread_and_range =
            Gate::three_bit_spread_and_range(b_lo.clone(), spread_b_lo.clone())
                .chain(Gate::two_bit_spread_and_range(
                    b_hi.clone(),
                    spread_b_hi.clone(),
                ))
                .chain(Gate::three_bit_spread_and_range(g, spread_g.clone()));

        let check_b = Self::check_b(b, b_lo, b_hi);
        let spread_witness = spread_r0_even
            + spread_r0_odd * F::from(2)
            + (spread_r1_even + spread_r1_odd * F::from(2)) * F::from_u128(1 << 64);
        let xor_0 = spread_c.clone()
            + spread_d.clone() * F::from(1 << 2)
            + spread_e.clone() * F::from(1 << 4)
            + spread_f_lo.clone() * F::from(1 << 26)
            + spread_f_hi.clone() * F::from_u128(1 << 68)
            + spread_g.clone() * F::from_u128(1 << 110);
        let xor_1 = spread_f_lo.clone()
            + spread_f_hi.clone() * F::from(1 << 42)
            + spread_g.clone() * F::from_u128(1 << 84)
            + spread_a.clone() * F::from_u128(1 << 90)
            + spread_b_lo.clone() * F::from_u128(1 << 92)
            + spread_b_hi.clone() * F::from_u128(1 << 98)
            + spread_c.clone() * F::from_u128(1 << 102)
            + spread_d.clone() * F::from_u128(1 << 104)
            + spread_e.clone() * F::from_u128(1 << 106);
        let xor_2 = spread_g
            + spread_a * F::from(1 << 6)
            + spread_b_lo * F::from(1 << 8)
            + spread_b_hi * F::from(1 << 14)
            + spread_c * F::from(1 << 18)
            + spread_d * F::from(1 << 20)
            + spread_e * F::from(1 << 22)
            + spread_f_lo * F::from(1 << 44)
            + spread_f_hi * F::from_u128(1 << 86);
        let xor = xor_0 + xor_1 + xor_2;

        check_spread_and_range
            .chain(Some(("check_b", check_b)))
            .chain(Some(("lower_sigma_1_v2", spread_witness - xor)))
            .map(move |(name, poly)| (name, s_lower_sigma_1_v2.clone() * poly))
    }
}
