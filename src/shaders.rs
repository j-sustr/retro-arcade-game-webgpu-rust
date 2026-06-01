pub const GRID_SHADER: &str = r#"
fn PristineGrid(uv: vec2f, lineWidth: vec2f) -> f32 {
    let uvDDXY = vec4f(dpdx(uv), dpdy(uv));
    let uvDeriv = vec2f(length(uvDDXY.xz), length(uvDDXY.yw));
    let invertLine: vec2<bool> = lineWidth > vec2f(0.5);
    let targetWidth: vec2f = select(lineWidth, 1 - lineWidth, invertLine);
    let drawWidth: vec2f = clamp(targetWidth, uvDeriv, vec2f(0.5));
    let lineAA: vec2f = uvDeriv * 1.5;
    var gridUV: vec2f = abs(fract(uv) * 2.0 - 1.0);
    gridUV = select(1 - gridUV, gridUV, invertLine);
    var grid2: vec2f = smoothstep(drawWidth + lineAA, drawWidth - lineAA, gridUV);
    grid2 *= saturate(targetWidth / drawWidth);
    grid2 = mix(grid2, targetWidth, saturate(uvDeriv * 2.0 - 1.0));
    grid2 = select(grid2, 1.0 - grid2, invertLine);
    return mix(grid2.x, 1.0, grid2.y);
}

struct VertexIn {
  @location(0) pos: vec4f,
  @location(1) uv: vec2f,
}

struct VertexOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) fog: f32,
}

struct Camera {
  projection: mat4x4f,
  view: mat4x4f,
}
@group(0) @binding(0) var<uniform> camera: Camera;

struct GridArgs {
  lineColor: vec4f,
  baseColor: vec4f,
  lineWidth: vec2f,
}
@group(1) @binding(0) var<uniform> gridArgs: GridArgs;

const FOG_COLOR = vec4f(0.115, 0.02, 0.25, 1.0);
const FOG_START = 72.0;
const FOG_END = 210.0;

@vertex
fn vertexMain(in: VertexIn) -> VertexOut {
  var out: VertexOut;
  let viewPos = camera.view * in.pos;
  out.pos = camera.projection * viewPos;
  out.uv = in.uv;
  out.fog = smoothstep(FOG_START, FOG_END, -viewPos.z);
  return out;
}

@fragment
fn fragmentMain(in: VertexOut) -> @location(0) vec4f {
  let grid = PristineGrid(in.uv, gridArgs.lineWidth);
  let color = mix(gridArgs.baseColor, gridArgs.lineColor, grid * gridArgs.lineColor.a);
  return mix(color, FOG_COLOR, in.fog);
}
"#;

pub const OBJECT_SHADER: &str = r#"
struct VertexIn {
  @location(0) pos: vec4f,
  @location(1) color: vec4f,
}

struct VertexOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec4f,
  @location(1) fog: f32,
}

struct Camera {
  projection: mat4x4f,
  view: mat4x4f,
}
@group(0) @binding(0) var<uniform> camera: Camera;

const FOG_COLOR = vec4f(0.115, 0.02, 0.25, 1.0);
const FOG_START = 58.0;
const FOG_END = 205.0;

@vertex
fn vertexMain(in: VertexIn) -> VertexOut {
  var out: VertexOut;
  let viewPos = camera.view * in.pos;
  out.pos = camera.projection * viewPos;
  out.color = in.color;
  out.fog = smoothstep(FOG_START, FOG_END, -viewPos.z);
  return out;
}

@fragment
fn fragmentMain(in: VertexOut) -> @location(0) vec4f {
  return mix(in.color, FOG_COLOR, in.fog);
}
"#;
