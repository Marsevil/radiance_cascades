use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

type Position = [f32; 2];
type Color = [f32; 3];

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Vertex2D {
    #[format(R32G32_SFLOAT)]
    position: Position,
    #[format(R32G32B32_SFLOAT)]
    color: Color,
}
impl From<Vertex2DBuilder> for Vertex2D {
    fn from(value: Vertex2DBuilder) -> Self {
        value.build()
    }
}

pub struct Vertex2DBuilder {
    position: Position,
    color: Option<Color>,
}
impl Vertex2DBuilder {
    pub fn new(position: Position) -> Self {
        Self {
            position,
            color: None,
        }
    }

    pub fn color(self, color: Color) -> Self {
        Self {
            color: Some(color),
            ..self
        }
    }

    pub fn build(self) -> Vertex2D {
        let color = self.color.unwrap_or([1.0, 1.0, 1.0]);
        Vertex2D {
            position: self.position,
            color,
        }
    }
}
