use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

/// Trait for coordinate types used in Point
pub trait Coordinate:
    Copy + Clone + PartialOrd + PartialEq +
    Add<Output = Self> + Sub<Output = Self> + Mul<Self, Output = Self> + Div<Self, Output = Self> +
    AddAssign + Neg<Output = Self>
{
    /// Returns the zero value for this coordinate type.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::coordinate::Coordinate;
    ///
    /// let zero = f64::zero();
    /// assert_eq!(zero, 0.0);
    /// ```
    fn zero() -> Self;

    /// Returns the one value for this coordinate type.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::coordinate::Coordinate;
    ///
    /// let one = f64::one();
    /// assert_eq!(one, 1.0);
    /// ```
    fn one() -> Self;

    /// Converts the coordinate to a float representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::coordinate::Coordinate;
    ///
    /// let value: f64 = 3.14;
    /// assert_eq!(value.to_float(), 3.14);
    /// ```
    fn to_float(&self) -> f64;

    /// Creates a coordinate from a float representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::coordinate::Coordinate;
    ///
    /// let value = f64::from_float(3.14);
    /// assert_eq!(value, 3.14);
    /// ```
    fn from_float(value: f64) -> Self;

    /// Computes the absolute difference between two coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::coordinate::Coordinate;
    ///
    /// let a: f64 = 2.5;
    /// let b: f64 = -1.5;
    /// assert_eq!(a.abs_diff(&b), 4.0);
    /// ```
    fn abs_diff(&self, other: &Self) -> Self;

    /// Checks if the coordinate is less than zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::coordinate::Coordinate;
    ///
    /// let a: f64 = -1.0;
    /// assert!(a.is_sign_negative());
    ///
    /// let b: f64 = 1.0;
    /// assert!(!b.is_sign_negative());
    /// ```
    fn is_sign_negative(&self) -> bool;

    /// Computes the average of two coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::coordinate::Coordinate;
    ///
    /// let a: f64 = 2.0;
    /// let b: f64 = 4.0;
    /// assert_eq!(a.average(&b), 3.0);
    /// ```
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
        *self < Self::zero()
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
