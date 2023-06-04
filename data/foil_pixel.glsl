uniform sampler2D al_tex;
uniform sampler2D time_bias_tex;
varying vec4 varying_color;
varying vec2 varying_texcoord;
uniform float time;
uniform float draw_scale;
uniform vec2 bitmap_dims;

void main()
{
	vec2 uv = varying_texcoord;
    float c1 = 0.5 + 0.5 * cos(time / 10.);
    float s1 = 0.5 + 0.5 * sin(time / 10.);
    float time_bias = 2. * 3.1415 * texture2D(time_bias_tex, mod(draw_scale * uv * bitmap_dims / bitmap_dims.y, 1.)).r;
    float c = cos(time + time_bias);
    float s = sin(time + time_bias);
    vec3 u = vec3(1.0, 1.0, 1.0) / 1.73;
    // This is transposed...
    mat3 rot = mat3(
        c + u.x * u.x * (1. - c), u.x * u.y * (1. - c) - u.z * s, u.x * u.z * (1. - c) + u.y * s,
        u.y * u.x * (1. - c) + u.z * s, c + u.y * u.y * (1. - c), u.y * u.z * (1. - c) - u.x * s,
        u.z * u.x * (1. - c) - u.y * s, u.z * u.y * (1. - c) + u.x * s, c + u.z * u.z * (1. - c)
    );
    vec4 color = varying_color * texture2D(al_tex, uv);

    float mask = float(
        (color.xyz == vec3(153., 229., 80.) / 255.) ||
        (color.xyz == vec3(76., 137., 17.) / 255.) ||
        (color.xyz == vec3(170., 243., 100.) / 255.) ||
        (color.xyz == vec3(208., 248., 171.) / 255.)
    );
	gl_FragColor = mask * vec4(rot * color.xyz, color.w) + (1. - mask) * color;
}

