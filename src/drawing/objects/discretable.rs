use crate::{
    geometry::vertex::{Vertex2D, Vertex2DBuilder},
    math::{Rect, Vec2},
};

use super::{Light, Wall};

pub fn discrete_vec2(vec: &Vec2) -> [Vertex2D; 1] {
    [Vertex2DBuilder::new([vec.x(), vec.y()]).build()]
}

pub fn discrete_rect(rect: &Rect) -> [Vertex2D; 4] {
    [
        Vertex2DBuilder::new([0.0, 0.0]).build(),
        Vertex2DBuilder::new([rect.size.x(), 0.0]).build(),
        Vertex2DBuilder::new([0.0, rect.size.y()]).build(),
        Vertex2DBuilder::new([rect.size.x(), rect.size.y()]).build(),
    ]
}

pub fn discrete_light(light: &Light) -> [Vertex2D; 4] {
    discrete_rect(&light.rect)
}

pub fn discrete_wall(wall: &Wall) -> [Vertex2D; 4] {
    discrete_rect(&wall.rect)
}
