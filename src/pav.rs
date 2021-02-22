use ordered_float::OrderedFloat;

#[derive(Debug, PartialEq)]
pub struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn as_weighted_point(&self) -> WeightedPoint {
        return WeightedPoint { x : self.x, y : self.y, weight : 1.0 };
    }
}

#[derive(Clone, Copy)]
pub struct WeightedPoint {
    x: f64,
    y: f64,
    weight: f64,
}

impl WeightedPoint {
    pub fn merge_with(&mut self, other: &WeightedPoint) {
        self.x = ((self.x * self.weight) + (other.x * other.weight)) / (self.weight + other.weight);

        self.y = ((self.y * self.weight) + (other.y * other.weight)) / (self.weight + other.weight);

        self.weight = self.weight + other.weight;
    }

    pub fn as_point(&self) -> Point {
        return Point { x: self.x, y: self.y };
    }
}

pub fn interpolate(points : &Vec<Point>, at_x : f64) -> f64 {
    let pos = points.binary_search_by_key(&OrderedFloat(at_x), |p| OrderedFloat(p.x));
    return match pos {
        Ok(ix) => points[ix].y,
        Err(ix) => {
            let below = &points[ix-1];
            let above = &points[ix];
            let prop = (at_x-(below.x))/(above.x - below.x);
            (above.y-below.y)*prop + below.y
        }
    };
}

pub fn isotonic(points: &Vec<Point>) -> Vec<Point> {
    let mut weighted_points: Vec<WeightedPoint> = points
        .iter()
        .map(|p| p.as_weighted_point())
        .collect();

    weighted_points.sort_by_key(|point| OrderedFloat(point.x));

    let mut iso_points: Vec<WeightedPoint> = Vec::new();
    for weighted_point in &mut weighted_points.iter() {
        if iso_points.is_empty() || weighted_point.y > iso_points.last().unwrap().y {
            iso_points.push(weighted_point.clone())
        } else {
            let mut new_point = weighted_point.clone();
            loop {
                if iso_points.is_empty() || iso_points.last().unwrap().y < (new_point).y {
                    iso_points.push(new_point);
                    break;
                } else {
                    let last_to_repl = iso_points.pop();
                    new_point.merge_with(&last_to_repl.unwrap());
                }
            }
        }
    }

    return iso_points
        .iter()
        .map(|pw| pw.as_point())
        .collect();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_points() {
        assert_eq!(isotonic(&vec![]).is_empty(), true);
    }

    #[test]
    fn one_point() {
        assert_eq!(
            isotonic(&vec![Point { x: 1.0, y: 2.0 }]).pop().unwrap(),
            Point { x: 1.0, y: 2.0 }
        );
    }

    #[test]
    fn simple_merge() {
        assert_eq!(
            isotonic(&vec![Point { x: 1.0, y: 2.0 }, Point { x: 2.0, y: 0.0 },])
                .pop()
                .unwrap(),
            Point { x: 1.5, y: 1.0 }
        );
    }

    #[test]
    fn one_not_merged() {
        assert_eq!(
            isotonic(&vec![Point {x : 0.5, y : -0.5}, Point { x: 1.0, y: 2.0 }, Point { x: 2.0, y: 0.0 },])
                .pop()
                .unwrap(),
            Point { x: 1.5, y: 1.0 }
        );
    }
}
