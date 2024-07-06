use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

/// Trait for coordinate types used in Point
pub trait Coordinate:
    Copy + Clone + PartialOrd + PartialEq +
    Add<Output = Self> + Sub<Output = Self> + Mul<Self, Output = Self> + Div<Self, Output = Self> +
    AddAssign + Neg<Output = Self>
{
    /// Returns the zero value for this coordinate type
    fn zero() -> Self;

    /// Returns the one value for this coordinate type
    fn one() -> Self;

    /// Converts the coordinate to a float representation
    fn to_float(&self) -> f64;

    /// Creates a coordinate from a float representation
    fn from_float(value: f64) -> Self;

    /// Computes the absolute difference between two coordinates
    fn abs_diff(&self, other: &Self) -> Self;

    /// Checks if the coordinate is less than zero
    fn is_sign_negative(&self) -> bool;

    /// Computes the average of two coordinates
    fn average(&self, other: &Self) -> Self;
}

impl Coordinate for f64 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn to_float(&self) -> f64 {
        *self
    }

    fn from_float(value: f64) -> Self {
        value
    }

    fn abs_diff(&self, other: &Self) -> Self {
        (self - other).abs()
    }

    fn is_sign_negative(&self) -> bool {
        self.is_sign_negative()
    }

    fn average(&self, other: &Self) -> Self {
        (self + other) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f64_coordinate() {
        let a: f64 = 2.5;
        let b: f64 = -1.5;

        assert_eq!(f64::zero(), 0.0);
        assert_eq!(f64::one(), 1.0);
        assert_eq!(a.to_float(), 2.5);
        assert_eq!(f64::from_float(3.14), 3.14);
        assert_eq!(a.abs_diff(&b), 4.0);
        assert!(!a.is_sign_negative());
        assert!(b.is_sign_negative());
        assert_eq!(a.average(&b), 0.5);
    }
}
