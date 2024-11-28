pub struct Vec2(pub [f32; 2]);

impl From<[f32; 2]> for Vec2 {
    fn from(value: [f32; 2]) -> Self {
        Self(value)
    }
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self([x, y])
    }

    pub fn x(&self) -> f32 {
        self.0[0]
    }

    pub fn y(&self) -> f32 {
        self.0[1]
    }
}
