use serde::Serialize;
use std::fmt::Display;
use super::coordinate::Coordinate;

/// A point in 2D cartesian space
#[derive(Debug, PartialEq, Copy, Clone, Serialize)]
pub struct Point<T: Coordinate> {
    pub(crate) x: T,
    pub(crate) y: T,
    pub(crate) weight: f64,
}

impl<T: Coordinate> Default for Point<T> {
    fn default() -> Self {
        Point {
            x: T::zero(),
            y: T::zero(),
            weight: 1.0,
        }
    }
}

impl<T: Coordinate> Point<T> {
    /// Create a new Point
    pub fn new(x: T, y: T) -> Point<T> {
        Point { x, y, weight: 1.0 }
    }

    /// Create a new Point with a specified weight
    pub fn new_with_weight(x: T, y: T, weight: f64) -> Point<T> {
        Point { x, y, weight }
    }

    /// The x position of the point
    pub fn x(&self) -> &T {
        &self.x
    }

    /// The y position of the point
    pub fn y(&self) -> &T {
        &self.y
    }

    /// The weight of the point (initially 1.0)
    pub fn weight(&self) -> f64 {
        self.weight
    }

    pub(crate) fn merge_with(&mut self, other: &Point<T>) {
        let total_weight = self.weight + other.weight;
        self.x = (self.x * T::from_float(self.weight) + other.x * T::from_float(other.weight)) / T::from_float(total_weight);
        self.y = (self.y * T::from_float(self.weight) + other.y * T::from_float(other.weight)) / T::from_float(total_weight);
        self.weight = total_weight;
    }
}

impl<T: Coordinate> From<(T, T)> for Point<T> {
    fn from(tuple: (T, T)) -> Self {
        Point::new(tuple.0, tuple.1)
    }
}

pub(crate) fn interpolate_two_points<T: Coordinate>(a: &Point<T>, b: &Point<T>, at_x: T) -> T {
    let prop = (at_x - a.x) / (b.x - a.x);
    a.y + (b.y - a.y) * prop
}
