use super::super::{util::*, Gate};

use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::plonk::{Constraint, Constraints, Expression};
use std::marker::PhantomData;

pub struct CompressionGate<F: FieldExt>(PhantomData<F>);

impl<F: FieldExt> CompressionGate<F> {
    fn ones() -> Expression<F> {
        Expression::Constant(F::one())
    }

    // Decompose `A,B,C,D` words
    // (28, 6, 5, 25)-bit chunks
    #[allow(clippy::too_many_arguments)]
    pub fn s_decompose_abcd(
        s_decompose_abcd: Expression<F>,
        a_lo: Expression<F>,
        spread_a_lo: Expression<F>,
        tag_a_lo: Expression<F>,
        a_hi: Expression<F>,
        spread_a_hi: Expression<F>,
        tag_a_hi: Expression<F>,
        b_lo: Expression<F>,
        spread_b_lo: Expression<F>,
        b_hi: Expression<F>,
        spread_b_hi: Expression<F>,
        c_lo: Expression<F>,
        spread_c_lo: Expression<F>,
        c_hi: Expression<F>,
        spread_c_hi: Expression<F>,
        d_lo: Expression<F>,
        spread_d_lo: Expression<F>,
        tag_d_lo: Expression<F>,
        d_hi: Expression<F>,
        spread_d_hi: Expression<F>,
        tag_d_hi: Expression<F>,
        word_lo: Expression<F>,
        spread_word_lo: Expression<F>,
        word_hi: Expression<F>,
        spread_word_hi: Expression<F>,
    ) -> Constraints<
        F,
        (&'static str, Expression<F>),
        impl Iterator<Item = (&'static str, Expression<F>)>,
    > {
        let check_spread_and_range =
            Gate::three_bit_spread_and_range(b_lo.clone(), spread_b_lo.clone())
                .chain(Gate::three_bit_spread_and_range(
                    b_hi.clone(),
                    spread_b_hi.clone(),
                ))
                .chain(Gate::three_bit_spread_and_range(
                    c_hi.clone(),
                    spread_c_hi.clone(),
                ))
                .chain(Gate::two_bit_spread_and_range(
                    c_lo.clone(),
                    spread_c_lo.clone(),
                ));
        let range_check_tag_a_lo = Gate::range_check(tag_a_lo, 0, 3);
        let range_check_tag_a_hi = Gate::range_check(tag_a_hi, 0, 3);
        let range_check_tag_d_lo = Gate::range_check(tag_d_lo, 0, 3);
        let range_check_tag_d_hi = Gate::range_check(tag_d_hi, 0, 1);
        let dense_check = a_lo
            + a_hi * F::from(1 << 14)
            + b_lo * F::from(1 << 28)
            + b_hi * F::from(1 << 31)
            + c_lo * F::from(1 << 34)
            + c_hi * F::from(1 << 36)
            + d_lo * F::from(1 << 39)
            + d_hi * F::from(1 << 53)
            + word_lo * (-F::one())
            + word_hi * F::from(1 << 32) * (-F::one());
        let spread_check = spread_a_lo
            + spread_a_hi * F::from(1 << 28)
            + spread_b_lo * F::from(1 << 56)
            + spread_b_hi * F::from(1 << 62)
            + spread_c_lo * F::from_u128(1 << 68)
            + spread_c_hi * F::from_u128(1 << 72)
            + spread_d_lo * F::from_u128(1 << 78)
            + spread_d_hi * F::from_u128(1 << 106)
            + spread_word_lo * (-F::one())
            + spread_word_hi * F::from_u128(1 << 64) * (-F::one());

        Constraints::with_selector(
            s_decompose_abcd,
            check_spread_and_range
                .chain(Some(("range_check_tag_a_lo", range_check_tag_a_lo)))
                .chain(Some(("range_check_tag_a_hi", range_check_tag_a_hi)))
                .chain(Some(("range_check_tag_d_lo", range_check_tag_d_lo)))
                .chain(Some(("range_check_tag_d_hi", range_check_tag_d_hi)))
                .chain(Some(("dense_check", dense_check)))
                .chain(Some(("spread_check", spread_check))),
        )
    }

    // Decompose `E,F,G,H` words
    // (14, 4, 23, 23)-bit chunks
    #[allow(clippy::too_many_arguments)]
    pub fn s_decompose_efgh(
        s_decompose_efgh: Expression<F>,
        a: Expression<F>,
        spread_a: Expression<F>,
        tag_a: Expression<F>,
        b_lo: Expression<F>,
        spread_b_lo: Expression<F>,
        b_hi: Expression<F>,
        spread_b_hi: Expression<F>,
        c_lo: Expression<F>,
        spread_c_lo: Expression<F>,
        tag_c_lo: Expression<F>,
        c_hi: Expression<F>,
        spread_c_hi: Expression<F>,
        tag_c_hi: Expression<F>,
        d_lo: Expression<F>,
        spread_d_lo: Expression<F>,
        tag_d_lo: Expression<F>,
        d_hi: Expression<F>,
        spread_d_hi: Expression<F>,
        tag_d_hi: Expression<F>,
        word_lo: Expression<F>,
        spread_word_lo: Expression<F>,
        word_hi: Expression<F>,
        spread_word_hi: Expression<F>,
    ) -> Constraints<
        F,
        (&'static str, Expression<F>),
        impl Iterator<Item = (&'static str, Expression<F>)>,
    > {
        let check_spread_and_range =
            Gate::two_bit_spread_and_range(b_lo.clone(), spread_b_lo.clone()).chain(
                Gate::two_bit_spread_and_range(b_hi.clone(), spread_b_hi.clone()),
            );
        let range_check_tag_a = Gate::range_check(tag_a, 0, 3);
        let range_check_tag_c_lo = Gate::range_check(tag_c_lo, 0, 2);
        let range_check_tag_c_hi = Gate::range_check(tag_c_hi, 0, 0);
        let range_check_tag_d_lo = Gate::range_check(tag_d_lo, 0, 2);
        let range_check_tag_d_hi = Gate::range_check(tag_d_hi, 0, 0);
        let dense_check = a
            + b_lo * F::from(1 << 14)
            + b_hi * F::from(1 << 16)
            + c_lo * F::from(1 << 18)
            + c_hi * F::from(1 << 31)
            + d_lo * F::from(1 << 41)
            + d_hi * F::from(1 << 54)
            + word_lo * (-F::one())
            + word_hi * F::from(1 << 32) * (-F::one());
        let spread_check = spread_a
            + spread_b_lo * F::from(1 << 28)
            + spread_b_hi * F::from(1 << 32)
            + spread_c_lo * F::from(1 << 36)
            + spread_c_hi * F::from(1 << 62)
            + spread_d_lo * F::from_u128(1 << 82)
            + spread_d_hi * F::from_u128(1 << 108)
            + spread_word_lo * (-F::one())
            + spread_word_hi * F::from_u128(1 << 64) * (-F::one());

        Constraints::with_selector(
            s_decompose_efgh,
            check_spread_and_range
                .chain(Some(("range_check_tag_a", range_check_tag_a)))
                .chain(Some(("range_check_tag_c_lo", range_check_tag_c_lo)))
                .chain(Some(("range_check_tag_c_hi", range_check_tag_c_hi)))
                .chain(Some(("range_check_tag_d_lo", range_check_tag_d_lo)))
                .chain(Some(("range_check_tag_d_hi", range_check_tag_d_hi)))
                .chain(Some(("dense_check", dense_check)))
                .chain(Some(("spread_check", spread_check))),
        )
    }

    // s_upper_sigma_0 on abcd words
    // (28, 6, 5, 25)-bit chunks
    #[allow(clippy::too_many_arguments)]
    pub fn s_upper_sigma_0(
        s_upper_sigma_0: Expression<F>,
        spread_r0_even_lo: Expression<F>,
        spread_r0_even_hi: Expression<F>,
        spread_r0_odd_lo: Expression<F>,
        spread_r0_odd_hi: Expression<F>,
        spread_r1_even_lo: Expression<F>,
        spread_r1_even_hi: Expression<F>,
        spread_r1_odd_lo: Expression<F>,
        spread_r1_odd_hi: Expression<F>,
        spread_a_lo: Expression<F>,
        spread_a_hi: Expression<F>,
        spread_b_lo: Expression<F>,
        spread_b_hi: Expression<F>,
        spread_c_lo: Expression<F>,
        spread_c_hi: Expression<F>,
        spread_d_lo: Expression<F>,
        spread_d_hi: Expression<F>,
    ) -> Option<(&'static str, Expression<F>)> {
        let spread_witness = (spread_r0_even_lo + spread_r0_even_hi * F::from(1 << 32))
            + (spread_r0_odd_lo + spread_r0_odd_hi * F::from(1 << 32)) * F::from(2)
            + ((spread_r1_even_lo + spread_r1_even_hi * F::from(1 << 32))
                + (spread_r1_odd_lo + spread_r1_odd_hi * F::from(1 << 32)) * F::from(2))
                * F::from_u128(1 << 64);
        let xor_0 = spread_b_lo.clone()
            + spread_b_hi.clone() * F::from(1 << 6)
            + spread_c_lo.clone() * F::from(1 << 12)
            + spread_c_hi.clone() * F::from(1 << 16)
            + spread_d_lo.clone() * F::from(1 << 22)
            + spread_d_hi.clone() * F::from(1 << 50)
            + spread_a_lo.clone() * F::from_u128(1 << 72)
            + spread_a_hi.clone() * F::from_u128(1 << 100);
        let xor_1 = spread_c_lo.clone()
            + spread_c_hi.clone() * F::from(1 << 4)
            + spread_d_lo.clone() * F::from(1 << 10)
            + spread_d_hi.clone() * F::from(1 << 38)
            + spread_a_lo.clone() * F::from(1 << 60)
            + spread_a_hi.clone() * F::from_u128(1 << 88)
            + spread_b_lo.clone() * F::from_u128(1 << 116)
            + spread_b_hi.clone() * F::from_u128(1 << 122);
        let xor_2 = spread_d_lo
            + spread_d_hi * F::from(1 << 28)
            + spread_a_lo * F::from(1 << 50)
            + spread_a_hi * F::from_u128(1 << 78)
            + spread_b_lo * F::from_u128(1 << 106)
            + spread_b_hi * F::from_u128(1 << 112)
            + spread_c_lo * F::from_u128(1 << 118)
            + spread_c_hi * F::from_u128(1 << 122);
        let xor = xor_0 + xor_1 + xor_2;
        let check = spread_witness + (xor * -F::one());

        Some(("s_upper_sigma_0", s_upper_sigma_0 * check))
    }

    // s_upper_sigma_1 on efgh words
    // (14, 4, 23, 23)-bit chunks
    #[allow(clippy::too_many_arguments)]
    pub fn s_upper_sigma_1(
        s_upper_sigma_1: Expression<F>,
        spread_r0_even_lo: Expression<F>,
        spread_r0_even_hi: Expression<F>,
        spread_r0_odd_lo: Expression<F>,
        spread_r0_odd_hi: Expression<F>,
        spread_r1_even_lo: Expression<F>,
        spread_r1_even_hi: Expression<F>,
        spread_r1_odd_lo: Expression<F>,
        spread_r1_odd_hi: Expression<F>,
        spread_a: Expression<F>,
        spread_b_lo: Expression<F>,
        spread_b_hi: Expression<F>,
        spread_c_lo: Expression<F>,
        spread_c_hi: Expression<F>,
        spread_d_lo: Expression<F>,
        spread_d_hi: Expression<F>,
    ) -> Option<(&'static str, Expression<F>)> {
        let spread_witness = (spread_r0_even_lo + spread_r0_even_hi * F::from(1 << 32))
            + (spread_r0_odd_lo + spread_r0_odd_hi * F::from(1 << 32)) * F::from(2)
            + ((spread_r1_even_lo + spread_r1_even_hi * F::from(1 << 32))
                + (spread_r1_odd_lo + spread_r1_odd_hi * F::from(1 << 32)) * F::from(2))
                * F::from_u128(1 << 64);

        let xor_0 = spread_b_lo.clone()
            + spread_b_hi.clone() * F::from(1 << 4)
            + spread_c_lo.clone() * F::from(1 << 8)
            + spread_c_hi.clone() * F::from(1 << 34)
            + spread_d_lo.clone() * F::from(1 << 54)
            + spread_d_hi.clone() * F::from_u128(1 << 80)
            + spread_a.clone() * F::from_u128(1 << 100);
        let xor_1 = spread_c_lo.clone()
            + spread_c_hi.clone() * F::from(1 << 26)
            + spread_d_lo.clone() * F::from(1 << 46)
            + spread_d_hi.clone() * F::from_u128(1 << 72)
            + spread_a.clone() * F::from_u128(1 << 92)
            + spread_b_lo.clone() * F::from_u128(1 << 120)
            + spread_b_hi.clone() * F::from_u128(1 << 124);
        let xor_2 = spread_d_lo
            + spread_d_hi * F::from(1 << 26)
            + spread_a * F::from(1 << 46)
            + spread_b_lo * F::from_u128(1 << 74)
            + spread_b_hi * F::from_u128(1 << 78)
            + spread_c_lo * F::from_u128(1 << 82)
            + spread_c_hi * F::from_u128(1 << 108);
        let xor = xor_0 + xor_1 + xor_2;
        let check = spread_witness + (xor * -F::one());

        Some(("s_upper_sigma_1", s_upper_sigma_1 * check))
    }

    // First part of choice gate on (E, F, G), E ∧ F
    #[allow(clippy::too_many_arguments)]
    pub fn s_ch(
        s_ch: Expression<F>,
        spread_p0_even_lo: Expression<F>,
        spread_p0_even_hi: Expression<F>,
        spread_p0_odd_lo: Expression<F>,
        spread_p0_odd_hi: Expression<F>,
        spread_p1_even_lo: Expression<F>,
        spread_p1_even_hi: Expression<F>,
        spread_p1_odd_lo: Expression<F>,
        spread_p1_odd_hi: Expression<F>,
        spread_e_lo: Expression<F>,
        spread_e_hi: Expression<F>,
        spread_f_lo: Expression<F>,
        spread_f_hi: Expression<F>,
    ) -> Option<(&'static str, Expression<F>)> {
        let lhs_lo = spread_e_lo + spread_f_lo;
        let lhs_hi = spread_e_hi + spread_f_hi;
        let lhs = lhs_lo + lhs_hi * F::from_u128(1 << 64);

        let spread_p0_even = spread_p0_even_lo + spread_p0_even_hi * F::from(1 << 32);
        let spread_p1_even = spread_p1_even_lo + spread_p1_even_hi * F::from(1 << 32);
        let rhs_even = spread_p0_even + spread_p1_even * F::from_u128(1 << 64);
        let spread_p0_odd = spread_p0_odd_lo + spread_p0_odd_hi * F::from(1 << 32);
        let spread_p1_odd = spread_p1_odd_lo + spread_p1_odd_hi * F::from(1 << 32);
        let rhs_odd = spread_p0_odd + spread_p1_odd * F::from_u128(1 << 64);
        let rhs = rhs_even + rhs_odd * F::from(2);

        let check = lhs + rhs * -F::one();

        Some(("s_ch", s_ch * check))
    }

    // Second part of Choice gate on (E, F, G), ¬E ∧ G
    #[allow(clippy::too_many_arguments)]
    pub fn s_ch_neg(
        s_ch_neg: Expression<F>,
        spread_q0_even_lo: Expression<F>,
        spread_q0_even_hi: Expression<F>,
        spread_q0_odd_lo: Expression<F>,
        spread_q0_odd_hi: Expression<F>,
        spread_q1_even_lo: Expression<F>,
        spread_q1_even_hi: Expression<F>,
        spread_q1_odd_lo: Expression<F>,
        spread_q1_odd_hi: Expression<F>,
        spread_e_lo: Expression<F>,
        spread_e_hi: Expression<F>,
        spread_e_neg_lo: Expression<F>,
        spread_e_neg_hi: Expression<F>,
        spread_g_lo: Expression<F>,
        spread_g_hi: Expression<F>,
    ) -> Constraints<
        F,
        (&'static str, Expression<F>),
        impl Iterator<Item = (&'static str, Expression<F>)>,
    > {
        let neg_check = {
            let evens = Self::ones() * F::from_u128(MASK_EVEN_64 as u128);
            // evens - spread_e_lo = spread_e_neg_lo
            let lo_check = spread_e_neg_lo.clone() + spread_e_lo + (evens.clone() * (-F::one()));
            // evens - spread_e_hi = spread_e_neg_hi
            let hi_check = spread_e_neg_hi.clone() + spread_e_hi + (evens * (-F::one()));

            std::iter::empty()
                .chain(Some(("lo_check", lo_check)))
                .chain(Some(("hi_check", hi_check)))
        };

        let lhs_lo = spread_e_neg_lo + spread_g_lo;
        let lhs_hi = spread_e_neg_hi + spread_g_hi;
        let lhs = lhs_lo + lhs_hi * F::from_u128(1 << 64);
        let spread_q0_even = spread_q0_even_lo + spread_q0_even_hi * F::from(1 << 32);
        let spread_q1_even = spread_q1_even_lo + spread_q1_even_hi * F::from(1 << 32);
        let rhs_even = spread_q0_even + spread_q1_even * F::from_u128(1 << 64);
        let spread_q0_odd = spread_q0_odd_lo + spread_q0_odd_hi * F::from(1 << 32);
        let spread_q1_odd = spread_q1_odd_lo + spread_q1_odd_hi * F::from(1 << 32);
        let rhs_odd = spread_q0_odd + spread_q1_odd * F::from_u128(1 << 64);
        let rhs = rhs_even + rhs_odd * F::from(2);

        Constraints::with_selector(s_ch_neg, neg_check.chain(Some(("s_ch_neg", lhs - rhs))))
    }

    // Majority gate on (A, B, C)
    #[allow(clippy::too_many_arguments)]
    pub fn s_maj(
        s_maj: Expression<F>,
        spread_m0_even_lo: Expression<F>,
        spread_m0_even_hi: Expression<F>,
        spread_m0_odd_lo: Expression<F>,
        spread_m0_odd_hi: Expression<F>,
        spread_m1_even_lo: Expression<F>,
        spread_m1_even_hi: Expression<F>,
        spread_m1_odd_lo: Expression<F>,
        spread_m1_odd_hi: Expression<F>,
        spread_a_lo: Expression<F>,
        spread_a_hi: Expression<F>,
        spread_b_lo: Expression<F>,
        spread_b_hi: Expression<F>,
        spread_c_lo: Expression<F>,
        spread_c_hi: Expression<F>,
    ) -> Option<(&'static str, Expression<F>)> {
        let spread_m0_even = spread_m0_even_lo + spread_m0_even_hi * F::from(1 << 32);
        let spread_m1_even = spread_m1_even_lo + spread_m1_even_hi * F::from(1 << 32);
        let maj_even = spread_m0_even + spread_m1_even * F::from_u128(1 << 64);
        let spread_m0_odd = spread_m0_odd_lo + spread_m0_odd_hi * F::from(1 << 32);
        let spread_m1_odd = spread_m1_odd_lo + spread_m1_odd_hi * F::from(1 << 32);
        let maj_odd = spread_m0_odd + spread_m1_odd * F::from_u128(1 << 64);
        let maj = maj_even + maj_odd * F::from(2);

        let a = spread_a_lo + spread_a_hi * F::from_u128(1 << 64);
        let b = spread_b_lo + spread_b_hi * F::from_u128(1 << 64);
        let c = spread_c_lo + spread_c_hi * F::from_u128(1 << 64);
        let sum = a + b + c;

        Some(("maj", s_maj * (sum - maj)))
    }

    // s_h_prime to get H' = H + Ch(E, F, G) + s_upper_sigma_1(E) + K + W
    #[allow(clippy::too_many_arguments)]
    pub fn s_h_prime(
        s_h_prime: Expression<F>,
        h_prime_lo: Expression<F>,
        h_prime_hi: Expression<F>,
        h_prime_carry: Expression<F>,
        sigma_e_lo: Expression<F>,
        sigma_e_hi: Expression<F>,
        ch_lo: Expression<F>,
        ch_hi: Expression<F>,
        ch_neg_lo: Expression<F>,
        ch_neg_hi: Expression<F>,
        h_lo: Expression<F>,
        h_hi: Expression<F>,
        k_lo: Expression<F>,
        k_hi: Expression<F>,
        w_lo: Expression<F>,
        w_hi: Expression<F>,
    ) -> Option<(&'static str, Expression<F>)> {
        let lo = h_lo + ch_lo + ch_neg_lo + sigma_e_lo + k_lo + w_lo;
        let hi = h_hi + ch_hi + ch_neg_hi + sigma_e_hi + k_hi + w_hi;

        let sum = lo + hi * F::from(1 << 32);
        let h_prime = h_prime_lo + h_prime_hi * F::from(1 << 32);

        let check = sum - (h_prime_carry * F::from_u128(1 << 64)) - h_prime;

        Some(("s_h_prime", s_h_prime * check))
    }

    // s_a_new to get A_new = H' + Maj(A, B, C) + s_upper_sigma_0(A)
    #[allow(clippy::too_many_arguments)]
    pub fn s_a_new(
        s_a_new: Expression<F>,
        a_new_lo: Expression<F>,
        a_new_hi: Expression<F>,
        a_new_carry: Expression<F>,
        sigma_a_lo: Expression<F>,
        sigma_a_hi: Expression<F>,
        maj_abc_lo: Expression<F>,
        maj_abc_hi: Expression<F>,
        h_prime_lo: Expression<F>,
        h_prime_hi: Expression<F>,
    ) -> Option<(&'static str, Expression<F>)> {
        let lo = sigma_a_lo + maj_abc_lo + h_prime_lo;
        let hi = sigma_a_hi + maj_abc_hi + h_prime_hi;
        let sum = lo + hi * F::from(1 << 32);
        let a_new = a_new_lo + a_new_hi * F::from(1 << 32);

        let check = sum - (a_new_carry * F::from_u128(1 << 64)) - a_new;

        Some(("s_a_new", s_a_new * check))
    }

    // s_e_new to get E_new = H' + D
    #[allow(clippy::too_many_arguments)]
    pub fn s_e_new(
        s_e_new: Expression<F>,
        e_new_lo: Expression<F>,
        e_new_hi: Expression<F>,
        e_new_carry: Expression<F>,
        d_lo: Expression<F>,
        d_hi: Expression<F>,
        h_prime_lo: Expression<F>,
        h_prime_hi: Expression<F>,
    ) -> Option<(&'static str, Expression<F>)> {
        let lo = h_prime_lo + d_lo;
        let hi = h_prime_hi + d_hi;
        let sum = lo + hi * F::from(1 << 32);
        let e_new = e_new_lo + e_new_hi * F::from(1 << 32);

        let check = sum - (e_new_carry * F::from_u128(1 << 64)) - e_new;

        Some(("s_e_new", s_e_new * check))
    }

    // s_digest on final round
    #[allow(clippy::too_many_arguments)]
    pub fn s_digest(
        s_digest: Expression<F>,
        lo_0: Expression<F>,
        hi_0: Expression<F>,
        word_0: Expression<F>,
        lo_1: Expression<F>,
        hi_1: Expression<F>,
        word_1: Expression<F>,
        lo_2: Expression<F>,
        hi_2: Expression<F>,
        word_2: Expression<F>,
        lo_3: Expression<F>,
        hi_3: Expression<F>,
        word_3: Expression<F>,
    ) -> impl IntoIterator<Item = Constraint<F>> {
        let check_lo_hi = |lo: Expression<F>, hi: Expression<F>, word: Expression<F>| {
            lo + hi * F::from(1 << 32) - word
        };

        Constraints::with_selector(
            s_digest,
            [
                ("check_lo_hi_0", check_lo_hi(lo_0, hi_0, word_0)),
                ("check_lo_hi_1", check_lo_hi(lo_1, hi_1, word_1)),
                ("check_lo_hi_2", check_lo_hi(lo_2, hi_2, word_2)),
                ("check_lo_hi_3", check_lo_hi(lo_3, hi_3, word_3)),
            ],
        )
    }
}
