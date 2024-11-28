#version 460

layout(location = 0) in vec2 position;
layout(location = 1) in vec3 color;

layout(set = 0, binding = 0) uniform Data {
    vec2 viewport;
    vec2 position;
} uniforms;

layout(location = 0) out vec3 fragColor;

vec4 get_frag_pos(vec2 obj_pos, vec2 vert_pos, vec2 viewport) {
    vec4 obj_pos4 = vec4(obj_pos, 0.0, 1.0);
    mat4 obj_trans = mat4(1.0);
    obj_trans[3] = obj_pos4;

    vec4 vert_pos4 = vec4(vert_pos.x / viewport.x, vert_pos.y / viewport.y, 0.0, 1.0);

    return obj_trans * vert_pos4;
}

void main() {
    gl_Position = get_frag_pos(uniforms.position, position, uniforms.viewport);
    fragColor = color;
}
