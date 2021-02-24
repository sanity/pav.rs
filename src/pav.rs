use ordered_float::OrderedFloat;

pub struct IsotonicRegression {
    points: Vec<Point>,
    mean_point: Point,
}

enum Direction {
    Ascending,
    Descending,
}

impl IsotonicRegression {
    pub fn new_ascending(points: &[Point]) -> IsotonicRegression {
        IsotonicRegression::new(points, Direction::Ascending)
    }

    pub fn new_descending(points: &[Point]) -> IsotonicRegression {
        IsotonicRegression::new(points, Direction::Descending)
    }

    fn new(points: &[Point], direction: Direction) -> IsotonicRegression {
        let point_count: f64 = points.len() as f64;
        let mut sum_x: f64 = 0.0;
        let mut sum_y: f64 = 0.0;
        for point in points {
            sum_x += point.x;
            sum_y += point.y;
        }

        IsotonicRegression {
            points: isotonic(points, direction),
            mean_point: Point {
                x: sum_x / point_count,
                y: sum_y / point_count,
            },
        }
    }

    pub fn interpolate(&self, at_x: f64) -> f64 {
        let pos = self
            .points
            .binary_search_by_key(&OrderedFloat(at_x), |p| OrderedFloat(p.x));
        return match pos {
            Ok(ix) => self.points[ix].y,
            Err(ix) => {
                if ix < 1 {
                    interpolate_two_points(&self.points.first().unwrap(), &self.mean_point, at_x)
                } else if ix >= self.points.len() {
                    interpolate_two_points(&self.mean_point, self.points.last().unwrap(), at_x)
                } else {
                    interpolate_two_points(&self.points[ix - 1], &self.points[ix], at_x)
                }
            }
        };
    }

    pub fn get_points(&self) -> &[Point] {
        return &self.points;
    }
}

#[derive(Debug, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    fn as_weighted_point(&self) -> WeightedPoint {
        WeightedPoint {
            x: self.x,
            y: self.y,
            weight: 1.0,
        }
    }
}

#[derive(Clone, Copy)]
struct WeightedPoint {
    x: f64,
    y: f64,
    weight: f64,
}

impl WeightedPoint {
    fn merge_with(&mut self, other: &WeightedPoint) {
        self.x = ((self.x * self.weight) + (other.x * other.weight)) / (self.weight + other.weight);

        self.y = ((self.y * self.weight) + (other.y * other.weight)) / (self.weight + other.weight);

        self.weight += other.weight;
    }

    fn as_point(&self) -> Point {
        Point {
            x: self.x,
            y: self.y,
        }
    }
}

fn interpolate_two_points(a: &Point, b: &Point, at_x: f64) -> f64 {
    let prop = (at_x - (a.x)) / (b.x - a.x);
    (b.y - a.y) * prop + a.y
}

fn isotonic(points: &[Point], direction: Direction) -> Vec<Point> {
    let mut weighted_points: Vec<WeightedPoint> =
        points.iter().map(|p| p.as_weighted_point()).collect();

    weighted_points.sort_by_key(|point| {
        OrderedFloat(match direction {
            Direction::Ascending => point.x,
            Direction::Descending => -point.x,
        })
    });

    // By xoring it this will effectively invert the conditional
    let dir_bool = match direction {
        Direction::Ascending => false,
        Direction::Descending => true,
    };

    let mut iso_points: Vec<WeightedPoint> = Vec::new();
    for weighted_point in &mut weighted_points.iter() {
        if iso_points.is_empty() || (dir_bool ^ (weighted_point.y > iso_points.last().unwrap().y)) {
            iso_points.push(*weighted_point)
        } else {
            let mut new_point = *weighted_point;
            loop {
                if iso_points.is_empty()
                    || (dir_bool ^ (iso_points.last().unwrap().y < (new_point).y))
                {
                    iso_points.push(new_point);
                    break;
                } else {
                    let last_to_repl = iso_points.pop();
                    new_point.merge_with(&last_to_repl.unwrap());
                }
            }
        }
    }

    return iso_points.iter().map(|pw| pw.as_point()).collect();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isotonic_no_points() {
        assert_eq!(isotonic(&[], Direction::Ascending).is_empty(), true);
    }

    #[test]
    fn isotonic_one_point() {
        assert_eq!(
            isotonic(&[Point { x: 1.0, y: 2.0 }], Direction::Ascending)
                .pop()
                .unwrap(),
            Point { x: 1.0, y: 2.0 }
        );
    }

    #[test]
    fn isotonic_simple_merge() {
        assert_eq!(
            isotonic(
                &[Point { x: 1.0, y: 2.0 }, Point { x: 2.0, y: 0.0 }],
                Direction::Ascending
            )
            .pop()
            .unwrap(),
            Point { x: 1.5, y: 1.0 }
        );
    }

    #[test]
    fn isotonic_one_not_merged() {
        assert_eq!(
            isotonic(
                &[
                    Point { x: 0.5, y: -0.5 },
                    Point { x: 1.0, y: 2.0 },
                    Point { x: 2.0, y: 0.0 }
                ],
                Direction::Ascending
            ),
            [Point { x: 0.5, y: -0.5 }, Point { x: 1.5, y: 1.0 }]
        );
    }

    #[test]
    fn isotonic_merge_three() {
        assert_eq!(
            isotonic(
                &[
                    Point { x: 0.0, y: 1.0 },
                    Point { x: 1.0, y: 2.0 },
                    Point { x: 2.0, y: -1.0 }
                ],
                Direction::Ascending
            ),
            [Point {
                x: 1.0,
                y: 2.0 / 3.0
            }]
        );
    }

    #[test]
    fn test_interpolate() {
        let regression = IsotonicRegression::new_ascending(&[
            Point { x: 1.0, y: 5.0 },
            Point { x: 2.0, y: 7.0 },
        ]);
        assert!((regression.interpolate(1.5) - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_isotonic_ascending() {
        let points = &[
            Point { x: 0.0, y: 1.0 },
            Point { x: 1.0, y: 2.0 },
            Point { x: 2.0, y: -1.0 },
        ];
        let regression = IsotonicRegression::new_ascending(points);
        assert_eq!(
            regression.get_points(),
            &[Point {
                x: (0.0 + 1.0 + 2.0) / 3.0,
                y: (1.0 + 2.0 - 1.0) / 3.0
            }]
        )
    }

    #[test]
    fn test_isotonic_descending() {
        let points = &[
            Point { x: 0.0, y: -1.0 },
            Point { x: 1.0, y: 2.0 },
            Point { x: 2.0, y: 1.0 },
        ];
        let regression = IsotonicRegression::new_descending(points);
        assert_eq!(
            regression.get_points(),
            &[Point { x: 1.5, y: 1.5 }, Point { x: 0.0, y: -1.0 }]
        )
    }
}
