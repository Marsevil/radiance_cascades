use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Vertex2D {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}
impl From<[f32; 2]> for Vertex2D {
    fn from(value: [f32; 2]) -> Self {
        Self { position: value }
    }
}
