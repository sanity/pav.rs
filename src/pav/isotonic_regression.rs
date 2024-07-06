use serde::Serialize;
use std::fmt::{Display, Formatter};
use thiserror::Error;
use super::coordinate::Coordinate;
use super::point::{Point, interpolate_two_points};

/// Errors that can occur during isotonic regression
#[derive(Error, Debug)]
pub enum IsotonicRegressionError {
    /// Error when a negative point is encountered with intersect_origin set to true
    #[error("With intersect_origin = true, all points must be >= 0 on both x and y axes")]
    NegativePointWithIntersectOrigin,
}

/// A vector of points forming an isotonic regression, along with the
/// centroid point of the original set.
/// # Example
/// ```
/// use crate::pav::point::Point;
/// use crate::pav::isotonic_regression::IsotonicRegression;
///
/// let points = &[
///     Point::new(0.0, 1.0),
///     Point::new(1.0, 2.0),
///     Point::new(2.0, 1.5),
/// ];
///
/// let regression = IsotonicRegression::new_ascending(points).unwrap();
/// assert_eq!(regression.interpolate(1.5).unwrap(), 1.75);
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct IsotonicRegression<T: Coordinate> {
    direction: Direction,
    points: Vec<Point<T>>,
    centroid_point: Centroid<T>,
    intersect_origin: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct Centroid<T: Coordinate> {
    sum_x: T,
    sum_y: T,
    sum_weight: f64,
}

#[derive(Debug, Clone, Serialize)]
#[derive(PartialEq)]
enum Direction {
    Ascending,
    Descending,
}

impl<T: Coordinate + Display> Display for IsotonicRegression<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "IsotonicRegression {{")?;
        writeln!(f, "\tdirection: {:?},", self.direction)?;
        writeln!(f, "\tpoints:")?;
        for point in &self.points {
            writeln!(f, "\t\t{}\t{:.2}\t{:.2}", point.x(), point.y(), point.weight())?;
        }
        writeln!(f, "\tcentroid_point:")?;
        writeln!(
            f,
            "\t\t{}\t{:.2}\t{:.2}",
            self.centroid_point.sum_x, self.centroid_point.sum_y, self.centroid_point.sum_weight
        )?;
        write!(f, "}}")
    }
}

impl<T: Coordinate> IsotonicRegression<T> {
    /// Find an ascending isotonic regression from a set of points
    pub fn new_ascending(points: &[Point<T>]) -> Result<IsotonicRegression<T>, IsotonicRegressionError> {
        IsotonicRegression::new(points, Direction::Ascending, false)
    }

    /// Find a descending isotonic regression from a set of points
    pub fn new_descending(points: &[Point<T>]) -> Result<IsotonicRegression<T>, IsotonicRegressionError> {
        IsotonicRegression::new(points, Direction::Descending, false)
    }

    /// Find an isotonic regression in the specified direction. If `intersect_origin` is true, the
    /// regression will intersect the origin (0,0) and all points must be >= 0 on both axes.
    fn new(points: &[Point<T>], direction: Direction, intersect_origin: bool) -> Result<IsotonicRegression<T>, IsotonicRegressionError> {
        let (sum_x, sum_y, sum_weight) = points.iter().try_fold((T::zero(), T::zero(), 0.0), |(sx, sy, sw), point| {
            if intersect_origin && (point.x().is_sign_negative() || point.y().is_sign_negative()) {
                Err(IsotonicRegressionError::NegativePointWithIntersectOrigin)
            } else {
                Ok((sx + *point.x() * T::from_float(point.weight()), sy + *point.y() * T::from_float(point.weight()), sw + point.weight()))
            }
        })?;

        let mut regression = IsotonicRegression {
            direction: direction.clone(),
            points: Vec::new(),
            centroid_point: Centroid {
                sum_x,
                sum_y,
                sum_weight,
            },
            intersect_origin,
        };
        regression.add_points(points);
        Ok(regression)
    }

    /// Find the _y_ point at position `at_x` or None if the regression is empty
    #[must_use]
    pub fn interpolate(&self, at_x: T) -> Option<T> {
        if self.points.is_empty() {
            return None;
        }

        let interpolation = if self.points.len() == 1 {
            *self.points[0].y()
        } else {
            let pos = self
                .points
                .binary_search_by(|p| p.x().partial_cmp(&at_x).unwrap());
            match pos {
                Ok(ix) => *self.points[ix].y(),
                Err(ix) => {
                    if ix < 1 {
                        if self.intersect_origin {
                            interpolate_two_points(
                                &Point::new(T::zero(), T::zero()),
                                self.points.first().unwrap(),
                                at_x,
                            )
                        } else {
                            interpolate_two_points(
                                self.points.first().unwrap(),
                                &self.get_centroid_point().unwrap(),
                                at_x,
                            )
                        }
                    } else if ix >= self.points.len() {
                        interpolate_two_points(
                            &self.get_centroid_point().unwrap(),
                            self.points.last().unwrap(),
                            at_x,
                        )
                    } else {
                        interpolate_two_points(&self.points[ix - 1], &self.points[ix], at_x)
                    }
                }
            }
        };

        Some(interpolation)
    }

    /// Retrieve the points that make up the isotonic regression
    pub fn get_points(&self) -> &[Point<T>] {
        &self.points
    }

    /// Retrieve the mean point of the original point set
    pub fn get_centroid_point(&self) -> Option<Point<T>> {
        if self.centroid_point.sum_weight == 0.0 {
            None
        } else {
            Some(Point::new_with_weight(
                self.centroid_point.sum_x / T::from_float(self.centroid_point.sum_weight),
                self.centroid_point.sum_y / T::from_float(self.centroid_point.sum_weight),
                1.0,
            ))
        }
    }

    /// Add new points to the regression
    pub fn add_points(&mut self, points: &[Point<T>]) {
        for point in points {
            assert!(!self.intersect_origin || 
                (!point.x().is_sign_negative() && !point.y().is_sign_negative()), "With intersect_origin = true, all points must be >= 0 on both x and y axes" );
            self.centroid_point.sum_x = self.centroid_point.sum_x + *point.x() * T::from_float(point.weight());
            self.centroid_point.sum_y = self.centroid_point.sum_y + *point.y() * T::from_float(point.weight());
            self.centroid_point.sum_weight = self.centroid_point.sum_weight + point.weight();
        }

        let mut new_points = self.points.clone();
        new_points.extend_from_slice(points);
        self.points = isotonic(&new_points, self.direction.clone());
    }


    /// Remove points from the regression
    pub fn remove_points(&mut self, points: &[Point<T>]) {
        for point in points {
            assert!(!self.intersect_origin || 
                (!point.x().is_sign_negative() && !point.y().is_sign_negative()), "With intersect_origin = true, all points must be >= 0 on both x and y axes" );
            self.centroid_point.sum_x = self.centroid_point.sum_x - *point.x() * T::from_float(point.weight());
            self.centroid_point.sum_y = self.centroid_point.sum_y - *point.y() * T::from_float(point.weight());
            self.centroid_point.sum_weight = self.centroid_point.sum_weight - point.weight();
        }

        let mut new_points = self.points.clone();
        for point in points {
            if let Some(pos) = new_points.iter().position(|p| p.x() == point.x() && p.y() == point.y() && p.weight() == point.weight()) {
                new_points.remove(pos);
            }
        }
        self.points = isotonic(&new_points, self.direction.clone());
    }

    /// How many points?
    pub fn len(&self) -> usize {
        self.centroid_point.sum_weight.round() as usize
    }

    /// Are there any points?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.centroid_point.sum_weight == 0.0
    }
}

fn isotonic<T: Coordinate>(points: &[Point<T>], direction: Direction) -> Vec<Point<T>> {
    let mut merged_points: Vec<Point<T>> = match direction {
        Direction::Ascending => points.to_vec(),
        Direction::Descending => points.iter().map(|p| Point::new_with_weight(*p.x(), *p.y(), p.weight())).collect(),
    };

    // Sort the points by x, and if x is equal, sort by y descending to ensure that points with the same x
    // get merged.
    merged_points.sort_by(|a, b| {
        a.x().partial_cmp(b.x())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.y().partial_cmp(a.y()).unwrap_or(std::cmp::Ordering::Equal))
    });

    let iso_points = merged_points.into_iter().fold(Vec::new(), |mut acc: Vec<Point<T>>, mut point| {
        while let Some(last) = acc.last() {
            if (direction == Direction::Ascending && last.y() >= point.y()) ||
               (direction == Direction::Descending && last.y() <= point.y()) {
                point.merge_with(&acc.pop().unwrap());
            } else {
                break;
            }
        }
        acc.push(point);
        acc
    });

    match direction {
        Direction::Ascending => iso_points,
        Direction::Descending => iso_points,
    }
}

#[cfg(test)]
mod tests {
    use super.*;

    #[test]
    fn test_ascending_regression() {
        let points = &[
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 1.5),
            Point::new(3.0, 3.0),
        ];

        let regression = IsotonicRegression::new_ascending(points).unwrap();
        assert_eq!(regression.get_points().len(), 3);
        assert_eq!(*regression.get_points()[0].y(), 1.0);
        assert_eq!(*regression.get_points()[1].y(), 1.75);
        assert_eq!(*regression.get_points()[2].y(), 3.0);
    }

    #[test]
    fn test_descending_regression() {
        let points = &[
            Point::new(0.0, 3.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 2.5),
            Point::new(3.0, 1.0),
        ];

        let regression = IsotonicRegression::new_descending(points).unwrap();
        assert_eq!(regression.get_points().len(), 3);
        assert_eq!(*regression.get_points()[0].y(), 3.0);
        assert_eq!(*regression.get_points()[1].y(), 2.25);
        assert_eq!(*regression.get_points()[2].y(), 1.0);
    }

    #[test]
    fn test_add_points() {
        let mut regression = IsotonicRegression::new_ascending(&[Point::new(0.0, 1.0), Point::new(2.0, 2.0)]).unwrap();
        regression.add_points(&[Point::new(1.0, 1.5)]);
        assert_eq!(regression.get_points().len(), 3);
        assert_eq!(*regression.get_points()[1].x(), 1.0);
        assert_eq!(*regression.get_points()[1].y(), 1.5);
    }

    #[test]
    fn test_remove_points() {
        let mut regression = IsotonicRegression::new_ascending(&[
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 3.0),
        ]).unwrap();
        regression.remove_points(&[Point::new(1.0, 2.0)]);
        assert_eq!(regression.get_points().len(), 2);
        assert_eq!(*regression.get_points()[0].x(), 0.0);
        assert_eq!(*regression.get_points()[1].x(), 2.0);
    }

    #[test]
    fn test_centroid_point() {
        let points = &[
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 3.0),
        ];
        let regression = IsotonicRegression::new_ascending(points).unwrap();
        let centroid = regression.get_centroid_point().unwrap();
        assert_eq!(*centroid.x(), 1.0);
        assert_eq!(*centroid.y(), 2.0);
    }

    #[test]
    fn test_empty_regression() {
        let regression: IsotonicRegression<f64> = IsotonicRegression::new_ascending(&[]).unwrap();
        assert!(regression.is_empty());
        assert_eq!(regression.len(), 0);
        assert!(regression.interpolate(1.0).is_none());
    }
}
